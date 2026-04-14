import asyncio
from types import SimpleNamespace

import pytest

from timeclaw.memory import hierarchical as hm
from timeclaw.memory.hierarchical import DistillationAgent, MemoryStore
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


def _note(scope: str) -> str:
    return f"""### [00:00:00] {scope} — SCORED
[{scope}] SCORED

Scope:
{scope}

Prediction:
ok
"""


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
    agent = DistillationAgent(
        notes=_FakeNotes(total=100, scoped={"scope_a": 4}),
        memory=_FakeMemory(),
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )

    assert await agent.maybe_distill_notes_to_memory("scope_a") is False


def test_notes_for_scope_collects_unique_scoped_notes_only():
    notes = _FakeNotes(
        notes=[
            _note("scope_a"),
            _note("scope_b"),
            _note("scope_a"),
        ]
    )
    agent = DistillationAgent(
        notes=notes,
        memory=_FakeMemory(),
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )

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
async def test_distill_pending_scopes_limits_flush_to_selected_scope_and_bucket(monkeypatch):
    agent = DistillationAgent(
        notes=_FakeNotes(scoped={"scope_a": 3, "scope_b": 4}),
        memory=_FakeMemory(),
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )
    seen_scopes: list[str] = []
    seen_buckets: list[str] = []

    async def fake_update_scope_state(scope, **kwargs):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
            "memory_to_tools_done_fingerprint": agent._scope_memory_fingerprint(scope),
        }

    async def fake_get_scope_state(scope):
        return {
            "notes_to_memory_done_seq": agent.notes.scoped_total(scope),
            "memory_to_skills_done_fingerprint": agent._scope_memory_fingerprint(scope),
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


def test_source_rule_note_refs_dedupes_across_source_rules():
    memory = _FakeMemory()
    memory.get_all_entries = lambda: [
        {"rule_id": "r1", "evidence": {"sample_refs": ["n1", "n2"]}},
        {"rule_id": "r2", "evidence": {"sample_refs": ["n2", "n3"]}},
    ]
    agent = DistillationAgent(
        notes=_FakeNotes(),
        memory=memory,
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )

    assert agent._source_rule_note_refs(["r1", "r2"]) == ["n1", "n2", "n3"]


def test_apply_memory_update_decision_merges_near_duplicate_result_constraint():
    agent = DistillationAgent(
        notes=_FakeNotes(),
        memory=_FakeMemory(),
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )
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
    agent = DistillationAgent(
        notes=_FakeNotes(),
        memory=_FakeMemory(),
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )
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
    agent = DistillationAgent(
        notes=_FakeNotes(scoped={"scope_a": 5}),
        memory=_FakeMemory(),
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )
    await agent._update_scope_state(
        "scope_a",
        notes_total_seq=5,
        notes_to_memory_done_seq=15,
        memory_fingerprint="",
        memory_to_skills_done_fingerprint="stale-skill-fp",
        memory_to_tools_done_fingerprint="stale-tool-fp",
    )

    await agent._scope_worker("scope_a", force=True)

    state = await agent._get_scope_state("scope_a")
    assert state["memory_to_skills_done_fingerprint"] == ""
    assert state["memory_to_tools_done_fingerprint"] == ""


@pytest.mark.asyncio
async def test_bucket_worker_clears_stale_cross_domain_done_fingerprint_when_memory_is_empty():
    agent = DistillationAgent(
        notes=_FakeNotes(),
        memory=_FakeMemory(),
        tools=_FakeTools(),
        skills=_FakeSkills(),
    )
    await agent._update_bucket_state(
        "bucket_a",
        memory_fingerprint="",
        memory_to_cross_done_fingerprint="stale-cross-fp",
    )

    await agent._bucket_worker("bucket_a", force=True)

    state = await agent._get_bucket_state("bucket_a")
    assert state["memory_to_cross_done_fingerprint"] == ""
