import json

import pytest

from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
from timeclaw.benchmark.mtbench_support import (
    expand_dataset_tasks,
    scope_keys_for_dataset_task,
)
from timeclaw.memory import hierarchical as hm


CANONICAL_SCOPES = {
    "weather_aligned_short_forecast",
    "weather_aligned_short_trend",
    "weather_aligned_short_trend_past",
    "weather_aligned_long_forecast",
    "weather_aligned_long_trend",
    "weather_aligned_long_trend_past",
    "finance_aligned_short_forecast",
    "finance_aligned_short_trend",
    "finance_aligned_short_indicator_macd",
    "finance_aligned_short_indicator_upper_bb",
    "finance_aligned_long_forecast",
    "finance_aligned_long_trend",
    "finance_aligned_long_indicator_macd",
    "finance_aligned_long_indicator_upper_bb",
    "finance_qa_short_mcqa",
    "finance_qa_short_correlation",
    "finance_qa_long_mcqa",
    "finance_qa_long_correlation",
}


def test_scope_keys_cover_active_registry_tasks():
    active_scopes = set()
    for dataset_name, spec in DATASET_REGISTRY.items():
        for task in expand_dataset_tasks(dataset_name, list(spec["tasks"])):
            active_scopes.update(scope_keys_for_dataset_task(dataset_name, task))

    assert active_scopes <= CANONICAL_SCOPES
    assert "weather_aligned_short_trend_past" in active_scopes
    assert "weather_aligned_long_trend_past" in active_scopes
    assert len(active_scopes) == 18


def test_expand_dataset_tasks_splits_finance_indicator_scopes():
    assert expand_dataset_tasks(
        "finance_aligned_long",
        ["forecast", "indicator", "trend"],
    ) == ["forecast", "indicator_macd", "indicator_upper_bb", "trend"]
    assert expand_dataset_tasks(
        "weather_aligned_long",
        ["forecast", "indicator", "trend"],
    ) == ["forecast", "indicator", "trend"]


@pytest.mark.asyncio
async def test_record_writes_one_note_per_scope_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    memory = hm.HierarchicalMemory(enabled=True)
    scope_keys = [
        "finance_aligned_short_indicator_macd",
        "finance_aligned_short_indicator_upper_bb",
    ]

    await memory.record(
        domain="finance",
        task="indicator",
        scope_keys=scope_keys,
        description="indicator training sample",
        predicted='{"macd": 1.2, "upper_bb": 99.0}',
        ground_truth='{"macd": 1.0, "upper_bb": 100.0}',
        correct=True,
        tool_chain=["compute_macd", "compute_bollinger"],
    )

    notes = memory.notes.get_recent_notes(10)
    assert len(notes) == 2
    assert any("[finance_aligned_short_indicator_macd] OK" in note for note in notes)
    assert any("[finance_aligned_short_indicator_upper_bb] OK" in note for note in notes)
    assert all("[finance_indicator]" not in note for note in notes)


def test_update_scope_counts_and_tool_prompt_respect_canonical_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    memory = hm.HierarchicalMemory(enabled=True)
    memory.tools.update(
        "finance_aligned_short_indicator_macd",
        {"compute_macd": "Use for MACD."},
    )
    memory.tools.update(
        "finance_aligned_short_indicator_upper_bb",
        {"compute_bollinger": "Use for Bollinger Band."},
    )

    notes = [
        "[finance_aligned_short_indicator_macd] OK | Tools: compute_macd | Pred: 1.0",
        "[finance_aligned_short_indicator_macd] FAIL | Tools: compute_macd | Pred: 2.0",
        "[finance_aligned_short_indicator_upper_bb] OK | Tools: compute_bollinger | Pred: 3.0",
    ]
    memory.distiller._update_scope_counts(notes)

    tool_notes = memory.get_tool_notes(
        scope_keys=[
            "finance_aligned_short_indicator_macd",
            "finance_aligned_short_indicator_upper_bb",
        ]
    )

    assert "compute_macd" in tool_notes
    assert "Use for MACD." in tool_notes["compute_macd"]
    assert "compute_bollinger" in tool_notes
    assert "Use for Bollinger Band." in tool_notes["compute_bollinger"]
    assert "finance_indicator" not in json.dumps(tool_notes)


def test_daily_note_filters_framework_tools_from_visible_tool_chain():
    note = hm.Notes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": [
                "spawn_subagent",
                "chronos2_forecast",
                "evaluate_batch_against_gt",
                "evaluate_against_gt",
            ],
            "metrics": {"mae": 1.0},
        }
    )

    assert "Winner Tools:\n- chronos2_forecast" in note
    assert "spawn_subagent" not in note
    assert "evaluate_batch_against_gt" not in note
    assert "evaluate_against_gt" not in note
    assert hm.DistillationAgent._extract_tools_from_note(note) == ["chronos2_forecast"]


@pytest.mark.asyncio
async def test_train_run_updates_scope_tool_usage_immediately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    from timeclaw.pipeline.runner import PipelineRunner

    runner = PipelineRunner()

    async def fake_ensure_tools():
        return None

    async def fake_preprocess(_envelope, _data, _text_data, _images):
        return {
            "fingerprint": {"length": 3},
            "modalities": ["timeseries"],
            "profiling_results": {},
        }

    class _Executor:
        evidence_log = type("E", (), {"summary": staticmethod(lambda: "ok")})()

        async def run(self, _envelope):
            return {
                "result": {
                    "answer": [1.0, 2.0],
                    "answer_type": "forecast",
                    "confidence": 0.8,
                    "reasoning": "Reason: x. Action: y.",
                    "execution_log": [
                        {
                            "tool": "spawn_subagent",
                            "success": True,
                            "output": [
                                {
                                    "success": True,
                                    "result": {
                                        "execution_log": [
                                            {"tool": "chronos2_forecast", "success": True},
                                            {"tool": "evaluate_against_gt", "success": True},
                                        ]
                                    },
                                }
                            ],
                        }
                    ],
                }
            }

    monkeypatch.setattr(runner, "_ensure_tools", fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _Executor())
    monkeypatch.setattr(
        runner._quality_gate,
        "check",
        lambda prediction, evidence, constraints: type(
            "Q", (), {"accepted": True, "rejections": []}
        )(),
    )

    await runner.run(
        data=[1.0, 2.0, 3.0],
        task_type="forecast",
        description="forecast sample",
        domain="weather",
        scope_keys=["weather_aligned_short_forecast"],
        mode="train",
        ground_truth=[1.0, 2.0],
    )

    counts = runner.memory.tools.get_scope_usage("weather_aligned_short_forecast")
    assert counts["chronos2_forecast"] == 1
    assert "evaluate_against_gt" not in counts


def test_daily_note_includes_candidate_summary_and_sample_stats():
    note = hm.Notes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": ["chronos2_forecast"],
            "metrics": {"mae": 1.0},
            "data_fingerprint": "horizon=24; length=168; mean=12.3",
            "candidate_traces": [
                {
                    "name": "chronos2_forecast",
                    "metrics": {"mae": 1.0},
                    "tool_chain": ["chronos2_forecast"],
                    "steps": [
                        {
                            "tool": "chronos2_forecast",
                            "thought": "Reason: daily cycle is stable. Action: run chronos2_forecast.",
                            "observation": '{"forecast":[1,2,3]}',
                        }
                    ],
                },
                {"name": "ets_forecast", "metrics": {"mae": 2.5}},
            ],
        }
    )

    assert "Winner Tools:\n- chronos2_forecast" in note
    assert "Fingerprint:\nhorizon=24; length=168; mean=12.3" in note
    assert "Candidates:" in note
    assert "- Candidate: chronos2_forecast (mae=1.000)" in note
    assert "Tools: chronos2_forecast" in note
    assert "Reasoning: Daily cycle is stable." in note
    assert "- Candidate: ets_forecast (mae=2.500)" in note


def test_daily_note_source_section_is_parser_compatible():
    note = hm.Notes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "trace_id": "trace-note-123",
            "source_metadata": {
                "dataset_name": "weather_aligned_short",
                "sample_ref": "weather_aligned_short:file.json:2:forecast",
                "source_file": "file.json",
                "sample_idx": 2,
                "task_name": "forecast",
            },
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": ["chronos2_forecast"],
            "metrics": {"mae": 1.0},
        }
    )

    parsed = hm.DistillationAgent._parse_note_sections(note)
    assert "Source" in parsed["sections"]
    assert "Sample Ref: weather_aligned_short:file.json:2:forecast" in parsed["sections"]["Source"]
    assert "Trace Path: .timeclaw/traces/trace-note-123.jsonl" in parsed["sections"]["Source"]


def test_max_conflict_support_uses_memory_store_without_self():
    entries_by_id = {
        "rule_a": {"rule_id": "rule_a", "evidence": {"sample_refs": ["n1", "n2", "n3"]}},
        "rule_b": {"rule_id": "rule_b", "evidence": {"sample_refs": ["n4"]}},
        "rule_c": {"rule_id": "rule_c", "evidence": {"sample_refs": ["n5", "n6", "n7", "n8", "n9"]}},
    }

    support = hm.DistillationAgent._max_conflict_support(
        "rule_b",
        [["rule_a", "rule_b", "rule_c"]],
        entries_by_id,
    )

    assert support == 5


def test_daily_note_approaches_use_candidate_trace_names_not_approach_ids():
    note = hm.Notes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": ["chronos2_forecast"],
            "metrics": {"mae": 1.0},
            "candidate_traces": [
                {
                    "name": "approach_0",
                    "tool_chain": ["chronos2_forecast"],
                    "metrics": {"mae": 1.0},
                },
                {
                    "name": "approach_1",
                    "tool_chain": ["ets_forecast"],
                    "metrics": {"mae": 2.5},
                },
            ],
        }
    )

    assert "Approaches:\n- chronos2_forecast(mae=1.000)\n- ets_forecast(mae=2.500)" in note
    assert "approach_0" not in note
    assert "approach_1" not in note


def test_daily_note_analysis_keeps_full_pipeline_summary():
    analysis = (
        "Reason: " + "A" * 700 + "\n"
        "Action: " + "B" * 700
    )
    note = hm.Notes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": ["chronos2_forecast"],
            "metrics": {"mae": 1.0},
            "pipeline_summary": analysis,
        }
    )

    assert "Analysis:\n[ANALYSIS_BEGIN]\n" + analysis in note


@pytest.mark.asyncio
async def test_force_distill_all_discovers_canonical_scope_headers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    memory = hm.HierarchicalMemory(enabled=True)

    await memory.record(
        domain="weather",
        task="forecast",
        scope_keys=["weather_aligned_short_forecast"],
        description="forecast sample",
        predicted="[1,2,3]",
        ground_truth="[1,2,3]",
        correct=True,
        tool_chain=["seasonal_baseline"],
    )
    await memory.record(
        domain="finance",
        task="indicator",
        scope_keys=[
            "finance_aligned_long_indicator_macd",
            "finance_aligned_long_indicator_upper_bb",
        ],
        description="indicator sample",
        predicted='{"macd": 1.0, "upper_bb": 2.0}',
        ground_truth='{"macd": 1.0, "upper_bb": 2.0}',
        correct=True,
        tool_chain=["compute_macd", "compute_bollinger"],
    )

    seen_memory_scopes = []
    seen_skill_scopes = []

    async def fake_notes_to_memory(scope: str) -> bool:
        seen_memory_scopes.append(scope)
        return False

    async def fake_memory_to_skills(scope: str) -> bool:
        seen_skill_scopes.append(scope)
        return False

    async def fake_memory_to_cross_domain(bucket: str) -> bool:
        return False

    async def fake_notes_to_tools(scope: str) -> bool:
        return False

    monkeypatch.setattr(memory.distiller, "maybe_distill_notes_to_memory", fake_notes_to_memory)
    monkeypatch.setattr(memory.distiller, "maybe_distill_memory_to_skills", fake_memory_to_skills)
    monkeypatch.setattr(memory.distiller, "maybe_distill_memory_to_cross_domain", fake_memory_to_cross_domain)
    monkeypatch.setattr(memory.distiller, "maybe_distill_notes_to_tools", fake_notes_to_tools)

    await memory.distiller.force_distill_all()

    assert set(seen_memory_scopes) == {
        "weather_aligned_short_forecast",
        "finance_aligned_long_indicator_macd",
        "finance_aligned_long_indicator_upper_bb",
    }
    assert set(seen_skill_scopes) == set(seen_memory_scopes)
