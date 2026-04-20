import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from timeclaw.memory import hierarchical as hm
from timeclaw.memory.hierarchical import DistillStateStore, DistillationAgent, MemoryStore
from timeclaw.pipeline.runner import PipelineRunner


class _FakeNotes:
    def __init__(self, total=0, scoped=None, notes=None):
        self.total = total
        self._scoped = scoped or {}
        self._notes = notes or []

    def scoped_total(self, scope):
        return self._scoped.get(scope, 0)

    def get_recent_notes(self, n=10):
        return self._notes[:n]

    def get_recent_note_items(self, n=10):
        return [
            {"id": f"note_{idx}", "text": text}
            for idx, text in enumerate(self._notes[:n])
        ]

    def get_scope_note_items(self, scope):
        return [
            {"id": f"note_{idx}", "text": text}
            for idx, text in enumerate(self._notes_for_scope(scope))
        ]

    def _notes_for_scope(self, scope):
        marker = f"\nScope:\n{scope}\n"
        return [text for text in self._notes if marker in text]

    async def resolve_conflicts(self, query, domain=None, task=None, scope=None):
        return []


class _FakeMemory:
    def _dedupe_rule_ids(self, entries):
        return entries

    def _normalize_rule_payload(self, scope, payload, index):
        normalized = dict(payload)
        normalized.setdefault("rule_id", f"{scope}::{index + 1:03d}")
        normalized.setdefault("conditions", [])
        normalized.setdefault("preferred_tools", [])
        normalized.setdefault("avoid_tools", [])
        normalized.setdefault("preferred_tool_chains", [])
        normalized.setdefault("avoid_tool_chains", [])
        normalized.setdefault("rationale", "")
        normalized.setdefault("evidence", {"sample_refs": []})
        return normalized

    def _entry_note_refs(self, entry):
        return MemoryStore._entry_note_refs(entry)

    def _entry_support_count(self, entry):
        return MemoryStore._entry_support_count(entry)

    def get_entries(self, scope):
        return []

    def get_all_entries(self):
        return []


class _FakeTools:
    def update(self, scope, notes):
        return None

    def get_notes(self, scope=""):
        return {}

    def total(self):
        return 0


class _FakeSkills:
    def get_skill(self, scope):
        return ""

    def delete_skill(self, scope):
        return None


class _FakeSkillsDecision:
    def get_skill_decision(self, scope):
        return ""

    def delete_skill_decision(self, scope):
        return None


def _make_agent(*, notes=None, memory=None, tools=None, skills=None, skills_decision=None):
    return DistillationAgent(
        notes=notes or _FakeNotes(),
        memory=memory or _FakeMemory(),
        tools=tools or _FakeTools(),
        skills=skills or _FakeSkills(),
        skills_decision=skills_decision or _FakeSkillsDecision(),
    )


def _note(scope: str) -> str:
    return f"""### [00:00:00] {scope} — SCORED
[{scope}] SCORED

Scope:
{scope}

Prediction:
ok
"""


def test_distill_state_store_writes_atomically_without_tmp_leftovers(tmp_path, monkeypatch):
    monkeypatch.setattr(hm, "MEMORY_ROOT", tmp_path / "hierarchical_memory")
    monkeypatch.setattr(hm, "STATE_DIR", hm.MEMORY_ROOT / "state")
    monkeypatch.setattr(hm, "DISTILL_STATE_FILE", hm.STATE_DIR / "distill_state.json")
    monkeypatch.setattr(hm, "DISTILL_STATE_LOCK_FILE", hm.STATE_DIR / ".distill_state.lock")

    store = DistillStateStore()
    store.update_scope("scope_a", notes_total_seq=3, memory_fingerprint="fp")

    assert hm.DISTILL_STATE_FILE.exists()
    assert json.loads(hm.DISTILL_STATE_FILE.read_text())["scopes"]["scope_a"]["memory_fingerprint"] == "fp"
    assert list(hm.STATE_DIR.glob(".*.tmp")) == []


def test_memory_to_skills_prompt_requires_judgment_target_before_tools():
    prompt = hm.MEMORY_TO_SKILLS_PROMPT

    assert "real judgment target or comparison target" in prompt
    assert "state what must be judged or compared before naming tools" in prompt
    assert "`Core Idea` should name the task's real judgment/comparison target" in prompt


def test_memory_to_skills_decision_prompt_requires_judgment_target_before_evidence_logic():
    prompt = hm.MEMORY_TO_SKILLS_DECISION_PROMPT

    assert "real judgment target or comparison target" in prompt
    assert "explain what is being judged first" in prompt
    assert "`Why This Usually Supports The Judgment` should name the actual judgment/comparison target" in prompt


@pytest.mark.asyncio
async def test_notes_scope_storage_is_reversible_for_custom_scope_names(tmp_path, monkeypatch):
    monkeypatch.setattr(hm, "MEMORY_ROOT", tmp_path / "hierarchical_memory")
    monkeypatch.setattr(hm, "NOTES_DIR", hm.MEMORY_ROOT / "notes")
    hm.NOTES_DIR.mkdir(parents=True, exist_ok=True)

    notes = hm.Notes()
    scope = "custom scope/[v1]"
    await notes.add(
        {
            "scope_key": scope,
            "domain": "demo",
            "task": "forecast",
            "predicted": "ok",
            "metrics": {"mae": 0.1},
        }
    )

    items = notes.get_scope_note_items(scope)

    assert len(items) == 1
    assert f"\nScope:\n{scope}\n" in items[0]["text"]
    stored_files = list(hm.NOTES_DIR.glob("*.md"))
    assert len(stored_files) == 1
    assert stored_files[0].name != f"{scope}.md"


@pytest.mark.asyncio
async def test_notes_to_memory_uses_scoped_note_threshold():
    agent = _make_agent(notes=_FakeNotes(total=100, scoped={"scope_a": 4}))

    assert await agent.maybe_distill_notes_to_memory("scope_a") is False


def test_notes_for_scope_collects_unique_scoped_notes_only():
    notes = _FakeNotes(
        notes=[
            _note("scope_a"),
            _note("scope_b"),
            _note("scope_a"),
        ]
    )
    agent = _make_agent(notes=notes)

    assert agent._notes_for_scope("scope_a") == [
        {"id": "note_0", "text": _note("scope_a")},
        {"id": "note_2", "text": _note("scope_a")},
    ]


@pytest.mark.asyncio
async def test_shutdown_flushes_pending_memory_distillation():
    runner = PipelineRunner.__new__(PipelineRunner)
    called = {"distill_pending": False, "distill_pending_scopes": None}

    async def fake_distill_pending():
        called["distill_pending"] = True

    async def fake_distill_pending_scopes(scope_keys):
        called["distill_pending_scopes"] = list(scope_keys)

    runner._disable_memory = False
    runner._touched_memory_scopes = {"scope_b", "scope_a"}
    runner.memory = SimpleNamespace(
        distill_pending=fake_distill_pending,
        distill_pending_scopes=fake_distill_pending_scopes,
    )
    runner._mcp_manager = None
    runner._mcp_started = False

    await PipelineRunner.shutdown(runner)

    assert called["distill_pending"] is False
    assert called["distill_pending_scopes"] == ["scope_a", "scope_b"]


@pytest.mark.asyncio
async def test_shutdown_falls_back_to_global_flush_when_no_scopes_were_touched():
    runner = PipelineRunner.__new__(PipelineRunner)
    called = {"distill_pending": False, "distill_pending_scopes": False}

    async def fake_distill_pending():
        called["distill_pending"] = True

    async def fake_distill_pending_scopes(scope_keys):
        called["distill_pending_scopes"] = True

    runner._disable_memory = False
    runner._touched_memory_scopes = set()
    runner.memory = SimpleNamespace(
        distill_pending=fake_distill_pending,
        distill_pending_scopes=fake_distill_pending_scopes,
    )
    runner._mcp_manager = None
    runner._mcp_started = False

    await PipelineRunner.shutdown(runner)

    assert called["distill_pending"] is True
    assert called["distill_pending_scopes"] is False


@pytest.mark.asyncio
async def test_shutdown_skips_memory_drain_when_disabled_for_test_mode():
    runner = PipelineRunner.__new__(PipelineRunner)
    called = {"distill_pending": False, "distill_pending_scopes": False}

    async def fake_distill_pending():
        called["distill_pending"] = True

    async def fake_distill_pending_scopes(scope_keys):
        called["distill_pending_scopes"] = True

    runner._disable_memory = False
    runner._drain_memory_on_shutdown = False
    runner._touched_memory_scopes = {"scope_a"}
    runner.memory = SimpleNamespace(
        distill_pending=fake_distill_pending,
        distill_pending_scopes=fake_distill_pending_scopes,
    )
    runner.router = SimpleNamespace(shutdown=AsyncMock())
    runner._mcp_manager = None
    runner._mcp_started = False

    await PipelineRunner.shutdown(runner)

    assert called["distill_pending"] is False
    assert called["distill_pending_scopes"] is False
    runner.router.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_distill_pending_scopes_limits_flush_to_selected_scope_and_bucket(monkeypatch):
    agent = _make_agent(notes=_FakeNotes(scoped={"scope_a": 3, "scope_b": 4}))
    seen_scopes: list[str] = []
    seen_buckets: list[str] = []

    async def fake_update_scope_state(scope, **kwargs):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_skills_decision_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_tools_done_fingerprint": agent._scope_memory_fingerprint(scope),
        }

    async def fake_get_scope_state(scope):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_skills_decision_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_tools_done_fingerprint": agent._scope_memory_fingerprint(scope),
        }

    async def fake_get_bucket_state(bucket):
        return {
            "memory_to_cross_done_fingerprint": agent._bucket_memory_fingerprint(bucket),
        }

    monkeypatch.setattr(agent, "_ensure_scope_worker", lambda scope, force=False: seen_scopes.append(scope))
    monkeypatch.setattr(agent, "_ensure_bucket_worker", lambda bucket, force=False: seen_buckets.append(bucket))
    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "_get_bucket_state", fake_get_bucket_state)

    await agent.distill_pending_scopes(["scope_b"])

    assert seen_scopes == ["scope_b"]
    assert seen_buckets == ["scope_b"]


@pytest.mark.asyncio
async def test_distill_pending_scopes_waits_for_scope_flush_before_bucket_flush(monkeypatch):
    agent = _make_agent(notes=_FakeNotes(scoped={"scope_b": 4}))
    scope_state = {
        "scope_b": {
            "notes_total_seq": 0,
            "notes_to_memory_done_seq": 0,
            "memory_to_skills_done_fingerprint": "",
            "memory_to_skills_decision_done_fingerprint": "",
            "memory_to_tools_done_fingerprint": "",
        }
    }
    bucket_state = {
        "scope_b": {
            "memory_to_cross_done_fingerprint": "",
        }
    }
    events: list[str] = []

    async def fake_update_scope_state(scope, **kwargs):
        scope_state.setdefault(scope, {}).update(kwargs)
        return dict(scope_state[scope])

    async def fake_get_scope_state(scope):
        return dict(scope_state.get(scope, {}))

    async def fake_update_bucket_state(bucket, **kwargs):
        bucket_state.setdefault(bucket, {}).update(kwargs)
        return dict(bucket_state[bucket])

    async def fake_get_bucket_state(bucket):
        return dict(bucket_state.get(bucket, {}))

    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "_update_bucket_state", fake_update_bucket_state)
    monkeypatch.setattr(agent, "_get_bucket_state", fake_get_bucket_state)
    monkeypatch.setattr(agent, "_scope_memory_fingerprint", lambda scope: "scope-fp")
    monkeypatch.setattr(agent, "_bucket_memory_fingerprint", lambda bucket: "bucket-fp")

    def fake_ensure_scope_worker(scope, force=False):
        if scope in agent._scope_workers and not agent._scope_workers[scope].done():
            return

        async def _run():
            events.append(f"scope_start:{scope}")
            await fake_update_scope_state(
                scope,
                notes_to_memory_done_seq=agent.notes.scoped_total(scope),
                memory_to_skills_done_fingerprint="scope-fp",
                memory_to_skills_decision_done_fingerprint="scope-fp",
                memory_to_tools_done_fingerprint="scope-fp",
            )
            events.append(f"scope_end:{scope}")

        agent._scope_workers[scope] = asyncio.create_task(_run())

    def fake_ensure_bucket_worker(bucket, force=False):
        if bucket in agent._bucket_workers and not agent._bucket_workers[bucket].done():
            return

        async def _run():
            events.append(
                f"bucket_seen_done_seq:{scope_state['scope_b']['notes_to_memory_done_seq']}"
            )
            assert scope_state["scope_b"]["notes_to_memory_done_seq"] == 4
            await fake_update_bucket_state(
                bucket,
                memory_to_cross_done_fingerprint="bucket-fp",
            )
            events.append(f"bucket_end:{bucket}")

        agent._bucket_workers[bucket] = asyncio.create_task(_run())

    monkeypatch.setattr(agent, "_ensure_scope_worker", fake_ensure_scope_worker)
    monkeypatch.setattr(agent, "_ensure_bucket_worker", fake_ensure_bucket_worker)

    await agent.distill_pending_scopes(["scope_b"])

    assert events == [
        "scope_start:scope_b",
        "scope_end:scope_b",
        "bucket_seen_done_seq:4",
        "bucket_end:scope_b",
    ]


@pytest.mark.asyncio
async def test_distill_pending_scopes_does_not_wait_for_unrelated_scope_worker(monkeypatch):
    agent = _make_agent(notes=_FakeNotes(scoped={"scope_a": 4, "scope_b": 4}))

    async def fake_update_scope_state(scope, **kwargs):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_skills_decision_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_tools_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "pending_snapshot_notes_to_seq": 0,
        }

    async def fake_get_scope_state(scope):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_skills_decision_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_tools_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "pending_snapshot_notes_to_seq": 0,
        }

    async def fake_get_bucket_state(bucket):
        return {
            "memory_to_cross_done_fingerprint": agent._bucket_memory_fingerprint(bucket),
        }

    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "_get_bucket_state", fake_get_bucket_state)

    unrelated_gate = asyncio.Event()
    agent._scope_workers["scope_a"] = asyncio.create_task(unrelated_gate.wait())

    await asyncio.wait_for(agent.distill_pending_scopes(["scope_b"]), timeout=0.2)

    assert not unrelated_gate.is_set()
    agent._scope_workers["scope_a"].cancel()
    with pytest.raises(asyncio.CancelledError):
        await agent._scope_workers["scope_a"]


@pytest.mark.asyncio
async def test_distill_pending_scopes_does_not_wait_for_unrelated_bucket_worker(monkeypatch):
    agent = _make_agent(notes=_FakeNotes(scoped={"scope_b": 4}))

    async def fake_update_scope_state(scope, **kwargs):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_skills_decision_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_tools_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "pending_snapshot_notes_to_seq": 0,
        }

    async def fake_get_scope_state(scope):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_skills_decision_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_tools_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "pending_snapshot_notes_to_seq": 0,
        }

    async def fake_get_bucket_state(bucket):
        return {
            "memory_to_cross_done_fingerprint": agent._bucket_memory_fingerprint(bucket),
        }

    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "_get_bucket_state", fake_get_bucket_state)

    unrelated_gate = asyncio.Event()
    agent._bucket_workers["other_bucket"] = asyncio.create_task(unrelated_gate.wait())

    await asyncio.wait_for(agent.distill_pending_scopes(["scope_b"]), timeout=0.2)

    assert not unrelated_gate.is_set()
    agent._bucket_workers["other_bucket"].cancel()
    with pytest.raises(asyncio.CancelledError):
        await agent._bucket_workers["other_bucket"]


@pytest.mark.asyncio
async def test_force_flush_upgrades_existing_scope_worker(monkeypatch):
    initial_note_count = hm._DISTILL_L1_TO_L2
    notes = _FakeNotes(scoped={"scope_b": initial_note_count})
    agent = _make_agent(notes=notes)
    scope_state = {
        "scope_b": {
            "notes_total_seq": 0,
            "notes_to_memory_done_seq": 0,
            "memory_fingerprint": "",
            "memory_to_skills_done_fingerprint": "",
            "memory_to_skills_decision_done_fingerprint": "",
            "memory_to_tools_done_fingerprint": "",
            "pending_snapshot_notes_from_seq": 0,
            "pending_snapshot_notes_to_seq": 0,
            "pending_snapshot_memory_fingerprint": "",
        }
    }
    note_force_calls: list[bool] = []
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    async def fake_update_scope_state(scope, **kwargs):
        scope_state.setdefault(scope, {}).update(kwargs)
        return dict(scope_state[scope])

    async def fake_get_scope_state(scope):
        return dict(scope_state.get(scope, {}))

    async def fake_distill_notes_to_memory(scope, target_seq=None, force=False):
        note_force_calls.append(bool(force))
        if len(note_force_calls) == 1:
            first_call_started.set()
            await release_first_call.wait()
        return True

    async def fake_distill_memory_to_skills(scope):
        return True

    async def fake_distill_memory_to_tools(scope):
        return True

    async def fake_append_memory_history_snapshot(**kwargs):
        return None

    def fake_ensure_bucket_worker(bucket, force=False):
        if bucket in agent._bucket_workers and not agent._bucket_workers[bucket].done():
            return
        agent._bucket_workers[bucket] = asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "maybe_distill_notes_to_memory", fake_distill_notes_to_memory)
    monkeypatch.setattr(agent, "maybe_distill_memory_to_skills", fake_distill_memory_to_skills)
    monkeypatch.setattr(agent, "maybe_distill_memory_to_tools", fake_distill_memory_to_tools)
    monkeypatch.setattr(agent, "_append_memory_history_snapshot", fake_append_memory_history_snapshot)
    monkeypatch.setattr(agent, "_scope_memory_fingerprint", lambda scope: "scope-fp")
    monkeypatch.setattr(agent, "_ensure_bucket_worker", fake_ensure_bucket_worker)

    agent._ensure_scope_worker("scope_b", force=False)
    await first_call_started.wait()

    notes._scoped["scope_b"] = initial_note_count + 1
    agent._ensure_scope_worker("scope_b", force=True)
    release_first_call.set()

    worker = agent._scope_workers["scope_b"]
    await asyncio.wait_for(worker, timeout=1.0)

    assert note_force_calls == [False, True]
    assert scope_state["scope_b"]["notes_to_memory_done_seq"] == initial_note_count + 1


@pytest.mark.asyncio
async def test_force_bucket_worker_rebuilds_only_once(monkeypatch):
    agent = _make_agent()
    bucket_state = {
        "scope_b": {
            "memory_fingerprint": "",
            "memory_to_cross_done_fingerprint": "",
            "memory_to_cross_inflight": False,
        }
    }
    cross_calls: list[str] = []

    async def fake_update_bucket_state(bucket, **kwargs):
        bucket_state.setdefault(bucket, {}).update(kwargs)
        return dict(bucket_state[bucket])

    monkeypatch.setattr(agent, "_update_bucket_state", fake_update_bucket_state)
    monkeypatch.setattr(agent, "_bucket_memory_fingerprint", lambda bucket: "bucket-fp")

    async def fake_distill_memory_to_cross_domain(bucket):
        cross_calls.append(bucket)
        return True

    monkeypatch.setattr(agent, "maybe_distill_memory_to_cross_domain", fake_distill_memory_to_cross_domain)

    await asyncio.wait_for(agent._bucket_worker("scope_b", force=True), timeout=1.0)

    assert cross_calls == ["scope_b"]
    assert bucket_state["scope_b"]["memory_to_cross_done_fingerprint"] == "bucket-fp"


def test_entry_support_count_uses_unique_note_refs():
    entry = {
        "evidence": {
            "support_count": 99,
            "sample_refs": ["n1", "n2", "n1"],
        }
    }

    assert MemoryStore._entry_support_count(entry) == 2


def test_entry_support_count_ignores_legacy_counter_without_note_refs():
    entry = {
        "evidence": {
            "support_count": 99,
            "sample_refs": [],
        }
    }

    assert MemoryStore._entry_support_count(entry) == 0


def test_memory_prompt_section_only_injects_injectable_rules():
    store = MemoryStore()
    store._entries = {
        "finance_aligned_long_indicator_upper_bb": [
            {
                "summary": "Prefer ets_forecast on smooth downtrends.",
                "conditions": ["smooth downtrend"],
                "preferred_tools": ["ets_forecast"],
                "avoid_tools": [],
                "rationale": "It matched repeated winning runs.",
                "evidence": {"sample_refs": ["n1", "n2"]},
                "injectable": True,
            },
            {
                "summary": "Old contradicted rule.",
                "conditions": ["same setting"],
                "preferred_tools": ["llm_reasoning_indicator"],
                "avoid_tools": [],
                "rationale": "Should stay stored but not injected.",
                "evidence": {"sample_refs": ["n3"]},
                "injectable": False,
            },
        ]
    }

    prompt = store.get_prompt_section(scope="finance_aligned_long_indicator_upper_bb")

    assert "Rule: Prefer ets_forecast on smooth downtrends." in prompt
    assert "Evidence: 2 note(s)" in prompt
    assert "Old contradicted rule." not in prompt


def test_memory_prompt_section_renders_path_rules_without_flattening_to_single_tools():
    store = MemoryStore()
    store._entries = {
        "weather_aligned_long_trend": [
            {
                "summary": "Daily aggregation should precede trend classification on hourly weather trend tasks.",
                "conditions": ["hourly weather trend classification"],
                "preferred_tools": ["segment"],
                "avoid_tools": [],
                "preferred_tool_chains": [["segment", "llm_reasoning_classification"]],
                "avoid_tool_chains": [["detect_trend", "llm_reasoning_classification"]],
                "rationale": "Winning runs first aggregate to daily structure, then classify.",
                "evidence": {"sample_refs": ["n1", "n2", "n3"]},
                "injectable": True,
            }
        ]
    }

    prompt = store.get_prompt_section(scope="weather_aligned_long_trend")

    assert "Prefer Path: segment -> llm_reasoning_classification" in prompt
    assert "Avoid Path: detect_trend -> llm_reasoning_classification" in prompt
    assert "Prefer: segment" in prompt


def test_notes_to_memory_prompt_explicitly_distinguishes_single_tools_from_tool_chains():
    assert "A multi-item `preferred_tools` list should read as \"any of these single tools may be suitable\"" in hm.NOTES_TO_MEMORY_PROMPT
    assert "If the evidence says \"tool A before tool B\"" in hm.NOTES_TO_MEMORY_PROMPT
    assert "verify that every non-empty `preferred_tools` / `avoid_tools`" in hm.NOTES_TO_MEMORY_PROMPT


def test_memory_prompt_section_orders_top_ranked_rules_first_without_truncating_injectable_rules():
    store = MemoryStore()
    store._entries = {
        "weather_aligned_long_trend": [
            {
                "rule_id": f"weather_aligned_long_trend::{idx:03d}",
                "summary": f"rule_{idx}",
                "conditions": [],
                "preferred_tools": [],
                "avoid_tools": [],
                "rationale": "",
                "evidence": {"sample_refs": [f"n{idx}"] * max(1, 12 - idx)},
                "injectable": True,
                "confidence": 1.0 - (idx * 0.01),
                "timestamp": 1000 - idx,
            }
            for idx in range(10)
        ]
    }

    prompt = store.get_prompt_section(scope="weather_aligned_long_trend")

    assert "Rule: rule_0" in prompt
    assert "Rule: rule_7" in prompt
    assert "Rule: rule_8" in prompt
    assert "Rule: rule_9" in prompt
    assert prompt.index("Rule: rule_0") < prompt.index("Rule: rule_8")
    assert prompt.index("Rule: rule_8") < prompt.index("Rule: rule_9")


def test_compute_statistics_treats_scored_separately_and_does_not_claim_success_rates():
    notes = [
        """### [00:00:00] s1 — SCORED
[s1] SCORED

Winner Tools:
- compute_macd

Prediction:
1.0

Metrics:
- macd_mae: 0.100
""",
        """### [00:00:01] s2 — FAIL
[s2] FAIL

Winner Tools:
- compute_macd
""",
        """### [00:00:02] s3 — OK
[s3] OK

Winner Tools:
- compute_bollinger
""",
    ]

    statistics = DistillationAgent._compute_statistics(notes)

    assert "Total runs: 3 (OK=1, FAIL=1, SCORED=1)" in statistics
    assert "Tool success rates:" not in statistics
    assert "Winner-tool occurrence by note outcome:" in statistics
    assert "compute_macd: total=2 (FAIL=1, SCORED=1)" in statistics


def test_source_rule_note_refs_dedupes_across_source_rules():
    memory = _FakeMemory()
    memory.get_all_entries = lambda: [
        {"rule_id": "r1", "evidence": {"sample_refs": ["n1", "n2"]}},
        {"rule_id": "r2", "evidence": {"sample_refs": ["n2", "n3"]}},
    ]
    agent = _make_agent(memory=memory)

    assert agent._source_rule_note_refs(["r1", "r2"]) == ["n1", "n2", "n3"]


def test_apply_memory_update_decision_merges_near_duplicate_result_constraint():
    agent = _make_agent()
    existing_entries = [
        {
            "rule_id": "scope_a::001",
            "kind": "result_constraint",
            "summary": "Indicator outputs should remain internally consistent.",
            "conditions": ["indicator task", "structured numeric output"],
            "preferred_tools": [],
            "avoid_tools": [],
            "rationale": "Repeated successful runs preserved numeric consistency.",
            "evidence": {"sample_refs": ["n0"]},
            "injectable": True,
            "timestamp": 1.0,
            "confidence": 0.6,
        }
    ]
    decision = {
        "support_updates": [],
        "merge_groups": [],
        "conflict_groups": [],
        "non_injectable_rule_ids": [],
        "new_rules": [
            {
                "kind": "result_constraint",
                "summary": "Indicator results must preserve numeric self consistency.",
                "conditions": ["indicator task", "structured numeric output"],
                "preferred_tools": [],
                "avoid_tools": [],
                "rationale": "Winning runs keep the indicator fields aligned and non-contradictory.",
                "support_note_ids": ["n1", "n2"],
                "closest_existing_rule_id": "scope_a::001",
                "novelty_reason": "same rule restated more concretely",
            }
        ],
    }

    updated = agent._apply_memory_update_decision(
        "scope_a",
        existing_entries,
        decision,
        available_note_ids={"n1", "n2"},
    )

    assert len(updated) == 1
    assert updated[0]["rule_id"] == "scope_a::001"
    assert updated[0]["evidence"]["sample_refs"] == ["n0", "n1", "n2"]


def test_apply_memory_update_decision_keeps_distinct_domain_heuristic():
    agent = _make_agent()
    existing_entries = [
        {
            "rule_id": "scope_a::001",
            "kind": "domain_heuristic",
            "summary": "Short-horizon weather series with stable daily cycles favor seasonal forecasting tools.",
            "conditions": ["short horizon", "stable daily cycle"],
            "preferred_tools": ["chronos2_forecast"],
            "avoid_tools": [],
            "rationale": "Seasonal structure is strong in repeated winning runs.",
            "evidence": {"sample_refs": ["n0"]},
            "injectable": True,
            "timestamp": 1.0,
            "confidence": 0.6,
        }
    ]
    decision = {
        "support_updates": [],
        "merge_groups": [],
        "conflict_groups": [],
        "non_injectable_rule_ids": [],
        "new_rules": [
            {
                "kind": "domain_heuristic",
                "summary": "Text-heavy finance QA tasks benefit from text-grounded reasoning.",
                "conditions": ["finance QA", "rich text context"],
                "preferred_tools": ["llm_reasoning_qa"],
                "avoid_tools": [],
                "rationale": "The decision depends on text evidence rather than seasonal numeric structure.",
                "support_note_ids": ["n1", "n2"],
                "closest_existing_rule_id": "scope_a::001",
                "novelty_reason": "Different task structure, conditions, and preferred tool family.",
            }
        ],
    }

    updated = agent._apply_memory_update_decision(
        "scope_a",
        existing_entries,
        decision,
        available_note_ids={"n1", "n2"},
    )

    assert len(updated) == 2
    assert {entry["rule_id"] for entry in updated} == {"scope_a::001", "scope_a::002"}


@pytest.mark.asyncio
async def test_record_serializes_sample_commit_when_distillation_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    memory = hm.HierarchicalMemory(enabled=True)
    order: list[str] = []
    call_index = {"value": 0}

    async def fake_on_commit(scope_keys):
        call_index["value"] += 1
        idx = call_index["value"]
        order.append(f"start_{idx}")
        await asyncio.sleep(0.01)
        order.append(f"end_{idx}")

    memory.distiller.on_commit = fake_on_commit

    await asyncio.gather(
        memory.record(
            domain="weather",
            task="forecast",
            scope_keys=["weather_aligned_short_forecast"],
            description="sample-a",
            predicted="[1,2,3]",
            correct=True,
            tool_chain=["chronos2_forecast"],
        ),
        memory.record(
            domain="weather",
            task="forecast",
            scope_keys=["weather_aligned_short_forecast"],
            description="sample-b",
            predicted="[4,5,6]",
            correct=False,
            tool_chain=["chronos2_forecast"],
        ),
    )

    assert order == ["start_1", "end_1", "start_2", "end_2"]


@pytest.mark.asyncio
async def test_record_allows_parallel_scope_local_commits_for_different_scopes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    memory = hm.HierarchicalMemory(enabled=True)
    order: list[str] = []

    async def fake_on_commit(scope_keys):
        scope = scope_keys[0]
        order.append(f"start_{scope}")
        await asyncio.sleep(0.05)
        order.append(f"end_{scope}")

    memory.distiller.on_commit = fake_on_commit

    started = asyncio.get_running_loop().time()
    await asyncio.gather(
        memory.record(
            domain="weather",
            task="forecast",
            scope_keys=["weather_aligned_short_forecast"],
            description="sample-a",
            predicted="[1,2,3]",
            correct=True,
            tool_chain=["chronos2_forecast"],
        ),
        memory.record(
            domain="finance",
            task="mcqa",
            scope_keys=["finance_qa_short_mcqa"],
            description="sample-b",
            predicted="[4,5,6]",
            correct=False,
            tool_chain=["chronos2_forecast"],
        ),
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert "start_weather_aligned_short_forecast" in order
    assert "start_finance_qa_short_mcqa" in order
    assert elapsed < 0.09


@pytest.mark.asyncio
async def test_scope_worker_clears_stale_done_fingerprints_when_memory_is_empty():
    agent = _make_agent(notes=_FakeNotes(scoped={"scope_a": 5}))
    await agent._update_scope_state(
        "scope_a",
        notes_total_seq=5,
        notes_to_memory_done_seq=15,
        memory_fingerprint="",
        memory_to_skills_done_fingerprint="stale-skill-fp",
        memory_to_skills_decision_done_fingerprint="stale-skill-decision-fp",
        memory_to_tools_done_fingerprint="stale-tool-fp",
    )

    await agent._scope_worker("scope_a", force=True)

    state = await agent._get_scope_state("scope_a")
    assert state["memory_to_skills_done_fingerprint"] == ""
    assert state["memory_to_skills_decision_done_fingerprint"] == ""
    assert state["memory_to_tools_done_fingerprint"] == ""


@pytest.mark.asyncio
async def test_bucket_worker_clears_stale_cross_domain_done_fingerprint_when_memory_is_empty():
    agent = _make_agent()
    await agent._update_bucket_state(
        "bucket_a",
        memory_fingerprint="",
        memory_to_cross_done_fingerprint="stale-cross-fp",
    )

    await agent._bucket_worker("bucket_a", force=True)

    state = await agent._get_bucket_state("bucket_a")
    assert state["memory_to_cross_done_fingerprint"] == ""
