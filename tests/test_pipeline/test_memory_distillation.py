import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from timeclaw.memory import hierarchical as hm
from timeclaw.agents.executor import ExecutorAgent
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


class _MutableFakeMemory(_FakeMemory):
    def __init__(self):
        self.entries = {}

    def get_entries(self, scope):
        return list(self.entries.get(scope, []))

    def get_all_entries(self):
        out = []
        for entries in self.entries.values():
            out.extend(entries)
        return out

    def replace_entries(self, scope, entries):
        self.entries[scope] = list(entries)


class _FakeTools:
    def update(self, scope, notes):
        return None

    def get_notes(self, scope=""):
        return {}

    def total(self):
        return 0


class _FakeSkills:
    def __init__(self):
        self.skills = {}

    def get_skill(self, scope):
        return self.skills.get(scope, "")

    def update_skill(self, scope, content):
        self.skills[scope] = content

    def get_skill_metadata(self, scope):
        skill_markdown = self.skills.get(scope, "")
        if not skill_markdown:
            return {}
        return {
            "scope": scope,
            "skills": [
                {
                    "title": str(block.get("title") or "").strip(),
                    "source_rule_ids": list(block.get("source_rule_ids") or []),
                }
                for block in hm._parse_skill_markdown_blocks(skill_markdown)
            ],
        }

    def delete_skill(self, scope):
        self.skills.pop(scope, None)
        return None

    def delete_skill_metadata(self, scope):
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


def _make_executor_with_skill_selector(
    selected_titles,
    *,
    selected_decision_titles=None,
    selected_rule_ids=None,
    prioritized_tool_names=None,
    primary_path_tools=None,
    challenger_path_tools=None,
    tool_prep_tools=None,
    selector_guidance="",
    selection_audit=None,
):
    class _FakeSelectorLLM:
        async def complete(self, *_args, **_kwargs):
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "selected_titles": list(selected_titles),
                        "selected_decision_titles": list(selected_decision_titles or []),
                        "selected_rule_ids": list(selected_rule_ids or []),
                        "primary_path_tools": list(primary_path_tools or prioritized_tool_names or []),
                        "challenger_path_tools": list(challenger_path_tools or []),
                        "tool_prep_tools": list(tool_prep_tools or []),
                        "selector_guidance": selector_guidance,
                        "selection_audit": selection_audit
                        or {
                            "why_skills": "picked matched skills",
                            "why_decision_skills": "picked judgment guardrails",
                            "why_rules": "picked linked rules",
                            "why_tools": "picked needed tools",
                        },
                    }
                )
            )

    return ExecutorAgent(llm=_FakeSelectorLLM(), config={})


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


def test_memory_to_skills_prompt_requires_embedded_source_rule_ids():
    prompt = hm.MEMORY_TO_SKILLS_PROMPT

    assert "Source Rule IDs: rule_id_1, rule_id_2" in prompt
    assert "`Source Rule IDs` is required" in prompt
    assert "Every `Source Rule IDs` item must be an exact MEMORY `rule_id`" in prompt


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


@pytest.mark.asyncio
async def test_memory_to_skills_normalizes_embedded_source_rule_ids():
    skills = _FakeSkills()
    scope = "scope_a"
    memory = _MutableFakeMemory()
    memory.replace_entries(
        scope,
        [
            {
                "rule_id": "scope_a::001",
                "summary": "rule one",
                "conditions": ["a"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "r1",
                "injectable": True,
                "evidence": {"sample_refs": ["n1", "n2"]},
            },
            {
                "rule_id": "scope_a::002",
                "summary": "rule two",
                "conditions": ["b"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "r2",
                "injectable": True,
                "evidence": {"sample_refs": ["n3"]},
            },
            {
                "rule_id": "scope_a::003",
                "summary": "rule three",
                "conditions": ["c"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "r3",
                "injectable": True,
                "evidence": {"sample_refs": ["n4"]},
            },
        ],
    )
    agent = _make_agent(memory=memory, skills=skills)

    class _FakeLLM:
        async def complete(self, *_args, **_kwargs):
            return SimpleNamespace(
                text="\n".join(
                    [
                        "### Skill 1",
                        "Source Rule IDs: scope_a::002, bad_rule",
                        "",
                        "Trigger:",
                        "x",
                        "",
                        "Core Idea:",
                        "y",
                        "",
                        "Workflow:",
                        "1. z",
                        "",
                        "Evidence:",
                        "- Support: strong (2 rules)",
                    ]
                )
            )

    async def fake_get_llm():
        return _FakeLLM()

    agent._get_llm = fake_get_llm

    success = await agent.maybe_distill_memory_to_skills(scope)

    assert success is True
    stored = skills.get_skill(scope)
    assert "Source Rule IDs: scope_a::002" in stored
    assert "bad_rule" not in stored
    payload = skills.get_skill_metadata(scope)
    assert payload["skills"] == [{"title": "### Skill 1", "source_rule_ids": ["scope_a::002"]}]


def test_skills_prompt_section_strips_embedded_source_rule_ids():
    store = hm.SkillsStore.__new__(hm.SkillsStore)
    store._skills = {
        "scope_a": "\n".join(
        [
            "### Skill 1",
            "Source Rule IDs: scope_a::001, scope_a::002",
            "",
            "Trigger:",
            "- x",
            "",
            "Core Idea:",
            "- y",
            "",
            "Workflow:",
            "1. z",
            "",
            "Evidence:",
            "- Support: strong (2 rules)",
        ]
    )
    }

    prompt = store.get_prompt_section(scope="scope_a")

    assert "Source Rule IDs:" not in prompt
    assert "### Skill 1" in prompt


@pytest.mark.asyncio
async def test_matched_skill_package_attaches_source_rule_ids():
    skill_prompt = "\n".join(
        [
            "### Skill 1",
            "Trigger:",
            "recent regime is flat",
            "Core Idea:",
            "prefer `tool_a`",
            "Workflow:",
            "1. inspect",
            "2. run `tool_a`",
            "Evidence:",
            "- Support: strong (2 rules)",
            "",
            "### Skill 2",
            "Trigger:",
            "recent regime is noisy",
            "Core Idea:",
            "prefer `tool_b`",
            "Workflow:",
            "1. inspect",
            "2. run `tool_b`",
            "Evidence:",
            "- Support: medium (1 rule)",
        ]
    )
    executor = _make_executor_with_skill_selector(["### Skill 1", "### Skill 2"])
    pkg = await executor._build_matched_skills_package(
        skill_prompt=skill_prompt,
        skill_metadata={
            "skills": [
                {"title": "### Skill 1", "source_rule_ids": ["r1", "r2"]},
                {"title": "### Skill 2", "source_rule_ids": ["r3"]},
            ]
        },
        task_description="flat regime with recent persistence",
        profiling_section="",
        candidates=["tool_a", "tool_b"],
    )

    assert pkg["selected_rule_ids"][:2] == ["r1", "r2"]
    assert pkg["selected_blocks"][0]["source_rule_ids"] == ["r1", "r2"]


@pytest.mark.asyncio
async def test_matched_skill_package_keeps_two_by_default():
    skill_prompt = "\n".join(
        [
            "### Skill 1",
            "Trigger:",
            "flat regime with persistence",
            "Core Idea:",
            "prefer `tool_a`",
            "Workflow:",
            "1. inspect flat tail",
            "2. run `tool_a`",
            "Evidence:",
            "- Support: strong (2 rules)",
            "",
            "### Skill 2",
            "Trigger:",
            "flat regime with persistence",
            "Core Idea:",
            "double-check with `tool_b`",
            "Workflow:",
            "1. inspect flat tail",
            "2. run `tool_b`",
            "Evidence:",
            "- Support: strong (2 rules)",
            "",
            "### Skill 3",
            "Trigger:",
            "flat regime with persistence",
            "Core Idea:",
            "also inspect flat tail before `tool_b`",
            "Workflow:",
            "1. inspect flat tail",
            "2. run `tool_b`",
            "Evidence:",
            "- Support: medium (1 rule)",
        ]
    )

    executor = _make_executor_with_skill_selector(["### Skill 1", "### Skill 2"])
    pkg = await executor._build_matched_skills_package(
        skill_prompt=skill_prompt,
        skill_metadata={"skills": []},
        task_description="flat regime with recent persistence",
        profiling_section="",
        candidates=["tool_a", "tool_b"],
    )

    assert pkg["selected_count"] == 2
    assert pkg["selected_titles"] == ["### Skill 1", "### Skill 2"]


@pytest.mark.asyncio
async def test_matched_skill_package_allows_third_when_it_adds_new_coverage():
    skill_prompt = "\n".join(
        [
            "### Skill 1",
            "Trigger:",
            "flat regime with persistence",
            "Core Idea:",
            "prefer `tool_a`",
            "Workflow:",
            "1. inspect flat tail",
            "2. run `tool_a`",
            "Evidence:",
            "- Support: strong (2 rules)",
            "",
            "### Skill 2",
            "Trigger:",
            "flat regime with persistence",
            "Core Idea:",
            "double-check with `tool_b`",
            "Workflow:",
            "1. inspect flat tail",
            "2. run `tool_b`",
            "Evidence:",
            "- Support: strong (2 rules)",
            "",
            "### Skill 3",
            "Trigger:",
            "flat regime with persistence and anomaly spike",
            "Core Idea:",
            "if anomaly spike appears, also use `tool_c`",
            "Workflow:",
            "1. inspect anomaly spike",
            "2. run `tool_c`",
            "Evidence:",
            "- Support: medium (1 rule)",
        ]
    )

    executor = _make_executor_with_skill_selector(
        ["### Skill 1", "### Skill 2", "### Skill 3"]
    )
    pkg = await executor._build_matched_skills_package(
        skill_prompt=skill_prompt,
        skill_metadata={"skills": []},
        task_description="flat regime with recent persistence and anomaly spike",
        profiling_section="",
        candidates=["tool_a", "tool_b", "tool_c"],
    )

    assert pkg["selected_count"] == 3
    assert sorted(pkg["selected_titles"]) == ["### Skill 1", "### Skill 2", "### Skill 3"]


@pytest.mark.asyncio
async def test_sample_experience_packet_selects_skills_rules_and_priority_tools_together():
    skill_prompt = "\n".join(
        [
            "### Skill 1",
            "Trigger:",
            "flat regime with persistence",
            "Core Idea:",
            "prefer `tool_a`",
            "Workflow:",
            "1. inspect flat tail",
            "2. run `tool_a`",
            "Evidence:",
            "- Support: strong (2 rules)",
            "",
            "### Skill 2",
            "Trigger:",
            "noisy regime",
            "Core Idea:",
            "prefer `tool_b`",
            "Workflow:",
            "1. inspect noise",
            "2. run `tool_b`",
            "Evidence:",
            "- Support: medium (1 rule)",
        ]
    )
    skill_decision_prompt = "\n".join(
        [
            "### Decision Skill 1",
            "When:",
            "flat regime and local persistence drive the call",
            "Why This Usually Supports The Judgment:",
            "Prefer judging from the recent regime before overreading weak context.",
            "How To Explain It To A User:",
            "- The latest regime is the most time-relevant signal here.",
            "Do Not Overclaim:",
            "- Do not imply weak context is irrelevant.",
            "Evidence Strength:",
            "- Support: strong (2 rules)",
        ]
    )
    executor = _make_executor_with_skill_selector(
        ["### Skill 1"],
        selected_decision_titles=["### Decision Skill 1"],
        selected_rule_ids=["scope::002"],
        primary_path_tools=["tool_a"],
        challenger_path_tools=["tool_b"],
        selector_guidance="Start from the flat-regime skill and keep the packet narrow.",
    )

    packet = await executor._build_sample_experience_packet(
        task_type="forecast",
        skill_prompt=skill_prompt,
        skill_decision_prompt=skill_decision_prompt,
        skill_metadata={
            "skills": [
                {"title": "### Skill 1", "source_rule_ids": ["scope::001", "scope::002"]},
                {"title": "### Skill 2", "source_rule_ids": ["scope::003"]},
            ]
        },
        memory_entries=[
            {
                "rule_id": "scope::001",
                "summary": "general flat-regime reminder",
                "conditions": ["flat regime"],
                "preferred_tools": ["tool_a"],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "general",
                "injectable": True,
                "confidence": 0.4,
                "evidence": {"sample_refs": ["n1"]},
            },
            {
                "rule_id": "scope::002",
                "summary": "linked flat-regime execution rule",
                "conditions": ["flat regime", "recent persistence"],
                "preferred_tools": ["tool_a"],
                "avoid_tools": [],
                "preferred_tool_chains": [["inspect", "tool_a"]],
                "avoid_tool_chains": [],
                "rationale": "linked",
                "injectable": True,
                "confidence": 0.9,
                "evidence": {"sample_refs": ["n2", "n3"]},
            },
        ],
        task_description="flat regime with recent persistence",
        profiling_section="Profile: flat recent tail",
        candidates=["tool_a", "tool_b"],
        tool_notes_section="Tool Notes:\n- tool_a: good for flat persistent tails\n",
    )

    assert packet["matched_skills_package"]["selected_titles"] == ["### Skill 1"]
    assert packet["decision_skills_package"]["selected_titles"] == ["### Decision Skill 1"]
    assert packet["active_experience_package"]["selected_rule_ids"] == ["scope::002"]
    assert packet["primary_path_tools"] == ["tool_a"]
    assert packet["challenger_path_tools"] == ["tool_b"]
    assert packet["prioritized_tool_names"] == ["tool_a", "tool_b"]
    assert "Experience Role Discipline" in packet["active_experience_package"]["text"]
    assert "Suggested Execution Lanes" in packet["active_experience_package"]["text"]


def test_active_experience_package_prefers_skill_linked_rules():
    pkg = ExecutorAgent._build_active_experience_package(
        memory_entries=[
            {
                "rule_id": "scope::001",
                "summary": "weakly related general rule",
                "conditions": ["generic"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "general",
                "injectable": True,
                "confidence": 0.0,
                "evidence": {"sample_refs": ["n1"]},
            },
            {
                "rule_id": "scope::002",
                "summary": "linked rule about flat regime",
                "conditions": ["flat regime", "recent persistence"],
                "preferred_tools": ["tool_a"],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "linked",
                "injectable": True,
                "confidence": 0.0,
                "evidence": {"sample_refs": ["n2", "n3"]},
            },
        ],
        task_description="flat regime with recent persistence",
        profiling_section="",
        candidates=["tool_a"],
        skill_rule_ids=["scope::002"],
    )

    assert pkg["selected_rule_ids"][0] == "scope::002"


def test_active_experience_package_deduplicates_reselected_skill_linked_rules():
    pkg = ExecutorAgent._build_active_experience_package(
        memory_entries=[
            {
                "rule_id": "scope::001",
                "summary": "linked rule one",
                "conditions": ["boundary shock"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "linked one",
                "injectable": True,
                "confidence": 0.0,
                "evidence": {"sample_refs": ["n1"]},
            },
            {
                "rule_id": "scope::002",
                "summary": "linked rule two",
                "conditions": ["daily mean comparison"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [],
                "rationale": "linked two",
                "injectable": True,
                "confidence": 0.0,
                "evidence": {"sample_refs": ["n2"]},
            },
        ],
        task_description="boundary shock with daily mean comparison",
        profiling_section="",
        candidates=[],
        skill_rule_ids=["scope::001", "scope::002"],
    )

    assert sorted(pkg["selected_rule_ids"]) == ["scope::001", "scope::002"]
    assert len(pkg["selected_rule_ids"]) == len(set(pkg["selected_rule_ids"]))


@pytest.mark.asyncio
async def test_notes_to_memory_retries_semantically_empty_distill_response(monkeypatch):
    notes = _FakeNotes(
        total=10,
        scoped={"scope_a": 10},
        notes=[_note("scope_a") for _ in range(10)],
    )
    memory = _MutableFakeMemory()
    agent = _make_agent(notes=notes, memory=memory)

    calls = {"count": 0}

    class _FakeLLM:
        async def complete(self, *_args, **_kwargs):
            calls["count"] += 1
            return SimpleNamespace(parsed={"new_rules": []}, text='{"new_rules": []}')

    async def fake_get_llm():
        return _FakeLLM()

    async def fake_get_scope_state(_scope):
        return {"notes_to_memory_done_seq": 0}

    monkeypatch.setattr(agent, "_get_llm", fake_get_llm)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "_apply_memory_update_decision", lambda *_args, **_kwargs: [])

    success = await agent.maybe_distill_notes_to_memory("scope_a", target_seq=10, force=True)

    assert success is False
    assert calls["count"] == hm._NOTES_TO_MEMORY_EMPTY_RETRY_ATTEMPTS
    assert memory.get_entries("scope_a") == []


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
async def test_training_shutdown_waits_for_memory_instead_of_cleanup_deadline(monkeypatch):
    from timeclaw.pipeline import runner as runner_module

    # Shrink the old 30-second cancellation window to reproduce it quickly.
    monkeypatch.setattr(runner_module, "_SHUTDOWN_MEMORY_DRAIN_TIMEOUT_S", 0.01, raising=False)
    finished = []

    async def distill(scopes):
        await asyncio.sleep(0.03)
        finished.extend(scopes)

    runner = PipelineRunner.__new__(PipelineRunner)
    runner._disable_memory = False
    runner._touched_memory_scopes = {"finance_aligned_short_forecast"}
    runner.memory = SimpleNamespace(distill_pending_scopes=distill)
    runner.router = SimpleNamespace(shutdown=AsyncMock())
    runner._mcp_manager = None
    runner._mcp_started = False
    await runner.shutdown()
    assert finished == ["finance_aligned_short_forecast"]
    runner.router.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_training_shutdown_surfaces_memory_failure_and_closes_router():
    runner = PipelineRunner.__new__(PipelineRunner)
    runner._disable_memory = False
    runner._touched_memory_scopes = {"finance_aligned_short_forecast"}
    runner.memory = SimpleNamespace(
        distill_pending_scopes=AsyncMock(side_effect=RuntimeError("unresolved notes")),
        cancel_pending_distillation=AsyncMock(),
    )
    runner.router = SimpleNamespace(shutdown=AsyncMock())
    runner._mcp_manager = None
    runner._mcp_started = False
    with pytest.raises(RuntimeError, match="unresolved notes"):
        await runner.shutdown()
    runner.memory.cancel_pending_distillation.assert_awaited_once()
    runner.router.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_distill_pending_scopes_limits_flush_to_selected_scope_and_bucket(monkeypatch):
    agent = _make_agent(notes=_FakeNotes(scoped={"scope_a": 3, "scope_b": 4}))
    touched_scope_notes: list[str] = []
    touched_scope_post: list[str] = []
    touched_scope_snapshot: list[str] = []
    touched_buckets: list[str] = []

    scope_state = {
        "scope_a": {
            "notes_total_seq": 3,
            "notes_to_memory_done_seq": 3,
            "memory_fingerprint": "scope-a-fp",
            "memory_to_skills_done_fingerprint": "scope-a-fp",
            "memory_to_skills_decision_done_fingerprint": "scope-a-fp",
            "memory_to_tools_done_fingerprint": "scope-a-fp",
            "pending_snapshot_notes_from_seq": 0,
            "pending_snapshot_notes_to_seq": 0,
            "pending_snapshot_memory_fingerprint": "",
        },
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
        },
    }
    bucket_state = {"scope_b": {"memory_fingerprint": "", "memory_to_cross_done_fingerprint": ""}}

    async def fake_update_scope_state(scope, **kwargs):
        scope_state.setdefault(scope, {}).update(kwargs)
        return dict(scope_state[scope])

    async def fake_get_scope_state(scope):
        return dict(scope_state.get(scope, {}))

    async def fake_get_bucket_state(bucket):
        return dict(bucket_state.get(bucket, {}))

    async def fake_drain_scope_notes_once(scope):
        touched_scope_notes.append(scope)
        assert scope == "scope_b"
        await fake_update_scope_state(
            scope,
            notes_to_memory_done_seq=agent.notes.scoped_total(scope),
            memory_fingerprint="scope-b-fp",
            pending_snapshot_notes_from_seq=1,
            pending_snapshot_notes_to_seq=agent.notes.scoped_total(scope),
            pending_snapshot_memory_fingerprint="scope-b-fp",
        )
        return True

    async def fake_drain_scope_post_memory_once(scope):
        touched_scope_post.append(scope)
        assert scope == "scope_b"
        await fake_update_scope_state(
            scope,
            memory_to_skills_done_fingerprint="scope-b-fp",
            memory_to_skills_decision_done_fingerprint="scope-b-fp",
            memory_to_tools_done_fingerprint="scope-b-fp",
        )
        return True

    async def fake_drain_bucket_cross_once(bucket):
        touched_buckets.append(bucket)
        assert bucket == "scope_b"
        bucket_state[bucket]["memory_to_cross_done_fingerprint"] = "bucket-fp"
        return True

    async def fake_finalize_scope_snapshot_once(scope):
        touched_scope_snapshot.append(scope)
        assert scope == "scope_b"
        await fake_update_scope_state(
            scope,
            pending_snapshot_notes_from_seq=0,
            pending_snapshot_notes_to_seq=0,
            pending_snapshot_memory_fingerprint="",
        )
        return True

    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "_get_bucket_state", fake_get_bucket_state)
    monkeypatch.setattr(agent, "_drain_scope_notes_once", fake_drain_scope_notes_once)
    monkeypatch.setattr(agent, "_drain_scope_post_memory_once", fake_drain_scope_post_memory_once)
    monkeypatch.setattr(agent, "_drain_bucket_cross_once", fake_drain_bucket_cross_once)
    monkeypatch.setattr(agent, "_finalize_scope_snapshot_once", fake_finalize_scope_snapshot_once)
    monkeypatch.setattr(agent, "_scope_memory_fingerprint", lambda scope: "scope-b-fp" if scope == "scope_b" else "scope-a-fp")
    monkeypatch.setattr(agent, "_bucket_memory_fingerprint", lambda bucket: "bucket-fp")

    await agent.distill_pending_scopes(["scope_b"])

    assert touched_scope_notes == ["scope_b"]
    assert touched_scope_post == ["scope_b"]
    assert touched_scope_snapshot == ["scope_b"]
    assert touched_buckets == ["scope_b"]


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

    async def fake_drain_scope_notes_once(scope):
        events.append(f"scope_start:{scope}")
        await fake_update_scope_state(
            scope,
            notes_to_memory_done_seq=agent.notes.scoped_total(scope),
            memory_fingerprint="scope-fp",
            pending_snapshot_notes_from_seq=1,
            pending_snapshot_notes_to_seq=agent.notes.scoped_total(scope),
            pending_snapshot_memory_fingerprint="scope-fp",
        )
        events.append(f"scope_end:{scope}")
        return True

    async def fake_drain_scope_post_memory_once(scope):
        await fake_update_scope_state(
            scope,
            memory_to_skills_done_fingerprint="scope-fp",
            memory_to_skills_decision_done_fingerprint="scope-fp",
            memory_to_tools_done_fingerprint="scope-fp",
        )
        return True

    async def fake_drain_bucket_cross_once(bucket):
        events.append(
            f"bucket_seen_done_seq:{scope_state['scope_b']['notes_to_memory_done_seq']}"
        )
        assert scope_state["scope_b"]["notes_to_memory_done_seq"] == 4
        await fake_update_bucket_state(
            bucket,
            memory_to_cross_done_fingerprint="bucket-fp",
        )
        events.append(f"bucket_end:{bucket}")
        return True

    async def fake_finalize_scope_snapshot_once(scope):
        await fake_update_scope_state(
            scope,
            pending_snapshot_notes_from_seq=0,
            pending_snapshot_notes_to_seq=0,
            pending_snapshot_memory_fingerprint="",
        )
        return True

    monkeypatch.setattr(agent, "_drain_scope_notes_once", fake_drain_scope_notes_once)
    monkeypatch.setattr(agent, "_drain_scope_post_memory_once", fake_drain_scope_post_memory_once)
    monkeypatch.setattr(agent, "_drain_bucket_cross_once", fake_drain_bucket_cross_once)
    monkeypatch.setattr(agent, "_finalize_scope_snapshot_once", fake_finalize_scope_snapshot_once)

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


@pytest.mark.asyncio
async def test_scope_worker_does_not_auto_finalize_snapshot_before_30_notes(monkeypatch):
    notes = _FakeNotes(scoped={"scope_b": 20})
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
    appended_snapshots: list[dict[str, int | str]] = []

    async def fake_update_scope_state(scope, **kwargs):
        scope_state.setdefault(scope, {}).update(kwargs)
        return dict(scope_state[scope])

    async def fake_get_scope_state(scope):
        return dict(scope_state.get(scope, {}))

    async def fake_distill_notes_to_memory(scope, target_seq=None, force=False):
        return True

    async def fake_distill_memory_to_skills(scope):
        return True

    async def fake_distill_memory_to_skills_decision(scope):
        return True

    async def fake_distill_memory_to_tools(scope):
        return True

    async def fake_append_memory_history_snapshot(**kwargs):
        appended_snapshots.append(dict(kwargs))

    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "maybe_distill_notes_to_memory", fake_distill_notes_to_memory)
    monkeypatch.setattr(agent, "maybe_distill_memory_to_skills", fake_distill_memory_to_skills)
    monkeypatch.setattr(agent, "maybe_distill_memory_to_skills_decision", fake_distill_memory_to_skills_decision)
    monkeypatch.setattr(agent, "maybe_distill_memory_to_tools", fake_distill_memory_to_tools)
    monkeypatch.setattr(agent, "_append_memory_history_snapshot", fake_append_memory_history_snapshot)
    monkeypatch.setattr(agent, "_scope_memory_fingerprint", lambda scope: "scope-fp")

    await asyncio.wait_for(agent._scope_worker("scope_b", force=True), timeout=1.0)

    assert appended_snapshots == []
    assert scope_state["scope_b"]["notes_to_memory_done_seq"] == 20
    assert scope_state["scope_b"]["pending_snapshot_notes_from_seq"] == 1
    assert scope_state["scope_b"]["pending_snapshot_notes_to_seq"] == 20
    assert scope_state["scope_b"]["pending_snapshot_memory_fingerprint"] == "scope-fp"


@pytest.mark.asyncio
async def test_distill_pending_scopes_flushes_short_snapshot_tail(monkeypatch):
    notes = _FakeNotes(scoped={"scope_b": 20})
    agent = _make_agent(notes=notes)
    scope_state = {
        "scope_b": {
            "notes_total_seq": 20,
            "notes_to_memory_done_seq": 20,
            "memory_fingerprint": "scope-fp",
            "memory_to_skills_done_fingerprint": "scope-fp",
            "memory_to_skills_decision_done_fingerprint": "scope-fp",
            "memory_to_tools_done_fingerprint": "scope-fp",
            "pending_snapshot_notes_from_seq": 1,
            "pending_snapshot_notes_to_seq": 20,
            "pending_snapshot_memory_fingerprint": "scope-fp",
        }
    }
    bucket_state = {"scope_b": {"memory_to_cross_done_fingerprint": "bucket-fp"}}
    appended_snapshots: list[dict[str, int | str]] = []

    async def fake_update_scope_state(scope, **kwargs):
        scope_state.setdefault(scope, {}).update(kwargs)
        return dict(scope_state[scope])

    async def fake_get_scope_state(scope):
        return dict(scope_state.get(scope, {}))

    async def fake_get_bucket_state(bucket):
        return dict(bucket_state.get(bucket, {}))

    async def fake_append_memory_history_snapshot(**kwargs):
        appended_snapshots.append(dict(kwargs))

    monkeypatch.setattr(agent, "_update_scope_state", fake_update_scope_state)
    monkeypatch.setattr(agent, "_get_scope_state", fake_get_scope_state)
    monkeypatch.setattr(agent, "_get_bucket_state", fake_get_bucket_state)
    monkeypatch.setattr(agent, "_append_memory_history_snapshot", fake_append_memory_history_snapshot)
    monkeypatch.setattr(agent, "_scope_memory_fingerprint", lambda scope: "scope-fp")
    monkeypatch.setattr(agent, "_bucket_memory_fingerprint", lambda bucket: "bucket-fp")

    await agent.distill_pending_scopes(["scope_b"])

    assert appended_snapshots == [
        {
            "scope": "scope_b",
            "notes_from_seq": 1,
            "notes_to_seq": 20,
            "memory_fingerprint": "scope-fp",
        }
    ]
    assert scope_state["scope_b"]["pending_snapshot_notes_from_seq"] == 0
    assert scope_state["scope_b"]["pending_snapshot_notes_to_seq"] == 0
    assert scope_state["scope_b"]["pending_snapshot_memory_fingerprint"] == ""


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


def test_build_tool_evidence_from_memory_includes_tool_chains():
    memory = _MutableFakeMemory()
    memory.entries = {
        "weather_aligned_long_trend": [
            {
                "scope": "weather_aligned_long_trend",
                "kind": "tool_selection",
                "summary": "Daily aggregation should precede trend classification.",
                "conditions": ["hourly weather trend classification"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [["segment", "llm_reasoning_classification"]],
                "avoid_tool_chains": [],
                "evidence": {"sample_refs": ["n1", "n2"]},
            },
            {
                "scope": "weather_aligned_long_trend",
                "kind": "tool_failure",
                "summary": "Direct trend detection overreacted to intra-day noise.",
                "conditions": ["noisy hourly weather trend classification"],
                "preferred_tools": [],
                "avoid_tools": [],
                "preferred_tool_chains": [],
                "avoid_tool_chains": [["detect_trend", "llm_reasoning_classification"]],
                "evidence": {"sample_refs": ["n3"]},
            },
        ]
    }

    agent = _make_agent(memory=memory)
    evidence = agent._build_tool_evidence_from_memory("weather_aligned_long_trend")

    assert "segment" in evidence
    assert "llm_reasoning_classification" in evidence
    assert any(
        "segment -> llm_reasoning_classification" in item
        for item in evidence["segment"]["when_it_helps"]
    )
    assert "detect_trend" in evidence
    assert any(
        "detect_trend -> llm_reasoning_classification" in item
        for item in evidence["detect_trend"]["when_to_avoid"]
    )
    assert any(
        "detect_trend -> llm_reasoning_classification" in item
        for item in evidence["detect_trend"]["output_risks"]
    )


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


def test_apply_memory_update_decision_persists_conflicts_across_rounds():
    agent = _make_agent()
    round0_entries = [
        {
            "rule_id": "scope_a::001",
            "kind": "domain_heuristic",
            "summary": "Use the trend-preserving rule.",
            "conditions": ["scope a"],
            "preferred_tools": ["tool_a"],
            "avoid_tools": [],
            "rationale": "Supported by repeated wins.",
            "evidence": {"sample_refs": ["n0", "n1"]},
            "injectable": True,
            "timestamp": 10.0,
            "confidence": 0.7,
        },
        {
            "rule_id": "scope_a::002",
            "kind": "domain_heuristic",
            "summary": "Use the contradictory rule.",
            "conditions": ["scope a"],
            "preferred_tools": ["tool_b"],
            "avoid_tools": [],
            "rationale": "Older weaker evidence.",
            "evidence": {"sample_refs": ["n2"]},
            "injectable": True,
            "timestamp": 11.0,
            "confidence": 0.5,
        },
    ]

    round1 = agent._apply_memory_update_decision(
        "scope_a",
        round0_entries,
        {
            "support_updates": [],
            "merge_groups": [],
            "conflict_groups": [["scope_a::001", "scope_a::002"]],
            "non_injectable_rule_ids": ["scope_a::002"],
            "new_rules": [],
        },
        available_note_ids=set(),
    )
    by_id_round1 = {entry["rule_id"]: entry for entry in round1}

    assert by_id_round1["scope_a::001"]["conflict_rule_ids"] == ["scope_a::002"]
    assert by_id_round1["scope_a::002"]["conflict_rule_ids"] == ["scope_a::001"]
    assert by_id_round1["scope_a::001"]["injectable"] is True
    assert by_id_round1["scope_a::002"]["injectable"] is False

    round2 = agent._apply_memory_update_decision(
        "scope_a",
        round1,
        {
            "support_updates": [
                {"rule_id": "scope_a::001", "support_note_ids": ["n3"]},
            ],
            "merge_groups": [],
            "conflict_groups": [],
            "non_injectable_rule_ids": [],
            "new_rules": [],
        },
        available_note_ids={"n3"},
    )
    by_id_round2 = {entry["rule_id"]: entry for entry in round2}

    assert by_id_round2["scope_a::001"]["conflict_rule_ids"] == ["scope_a::002"]
    assert by_id_round2["scope_a::002"]["conflict_rule_ids"] == ["scope_a::001"]
    assert by_id_round2["scope_a::001"]["evidence"]["sample_refs"] == ["n0", "n1", "n3"]
    assert by_id_round2["scope_a::001"]["injectable"] is True
    assert by_id_round2["scope_a::002"]["injectable"] is False


def test_apply_memory_update_decision_redirects_conflicts_when_rules_merge():
    agent = _make_agent()
    existing_entries = [
        {
            "rule_id": "scope_a::001",
            "kind": "domain_heuristic",
            "summary": "Primary rule.",
            "conditions": ["scope a"],
            "preferred_tools": ["tool_a"],
            "avoid_tools": [],
            "rationale": "Primary evidence.",
            "evidence": {"sample_refs": ["n0", "n1"]},
            "conflict_rule_ids": ["scope_a::002"],
            "injectable": True,
            "timestamp": 10.0,
            "confidence": 0.7,
        },
        {
            "rule_id": "scope_a::002",
            "kind": "domain_heuristic",
            "summary": "Merged-away variant.",
            "conditions": ["scope a"],
            "preferred_tools": ["tool_a"],
            "avoid_tools": [],
            "rationale": "Weaker duplicate evidence.",
            "evidence": {"sample_refs": ["n2"]},
            "conflict_rule_ids": ["scope_a::001", "scope_a::003"],
            "injectable": False,
            "timestamp": 9.0,
            "confidence": 0.4,
        },
        {
            "rule_id": "scope_a::003",
            "kind": "domain_heuristic",
            "summary": "Still conflicting rule.",
            "conditions": ["scope a"],
            "preferred_tools": ["tool_c"],
            "avoid_tools": [],
            "rationale": "Competing strategy.",
            "evidence": {"sample_refs": ["n3", "n4", "n5"]},
            "conflict_rule_ids": ["scope_a::002"],
            "injectable": True,
            "timestamp": 12.0,
            "confidence": 0.8,
        },
    ]

    updated = agent._apply_memory_update_decision(
        "scope_a",
        existing_entries,
        {
            "support_updates": [],
            "merge_groups": [["scope_a::001", "scope_a::002"]],
            "conflict_groups": [],
            "non_injectable_rule_ids": [],
            "new_rules": [],
        },
        available_note_ids=set(),
    )
    by_id = {entry["rule_id"]: entry for entry in updated}

    assert set(by_id) == {"scope_a::001", "scope_a::003"}
    assert by_id["scope_a::001"]["conflict_rule_ids"] == ["scope_a::003"]
    assert by_id["scope_a::003"]["conflict_rule_ids"] == ["scope_a::001"]


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
