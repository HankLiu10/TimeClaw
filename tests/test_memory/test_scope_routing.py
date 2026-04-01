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
    "weather_aligned_short_indicator",
    "weather_aligned_long_forecast",
    "weather_aligned_long_trend",
    "weather_aligned_long_trend_past",
    "weather_aligned_long_indicator",
    "weather_qa_short_mcqa",
    "weather_qa_long_forecast",
    "weather_qa_long_mcqa",
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
    assert len(active_scopes) == 23


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
async def test_record_writes_one_daily_note_per_scope_key(tmp_path, monkeypatch):
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

    notes = memory.daily.get_recent_notes(10)
    assert len(notes) == 2
    assert any("[finance_aligned_short_indicator_macd] OK" in note for note in notes)
    assert any("[finance_aligned_short_indicator_upper_bb] OK" in note for note in notes)
    assert all("[finance_indicator]" not in note for note in notes)


def test_update_scope_counts_and_tool_prompt_respect_canonical_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    memory = hm.HierarchicalMemory(enabled=True)
    memory.tools.update({"compute_macd": "Use for MACD.", "compute_bollinger": "Use for Bollinger Band."})

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
    note = hm.DailyNotes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": [
                "spawn_subagent",
                "chronos_forecast",
                "evaluate_batch_against_gt",
                "evaluate_against_gt",
            ],
            "metrics": {"mae": 1.0},
        }
    )

    assert "Winner Tool:\n- chronos_forecast" in note
    assert "spawn_subagent" not in note
    assert "evaluate_batch_against_gt" not in note
    assert "evaluate_against_gt" not in note
    assert hm.DistillationAgent._extract_tools_from_note(note) == ["chronos_forecast"]


def test_daily_note_includes_candidate_summary_and_sample_stats():
    note = hm.DailyNotes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": ["chronos_forecast"],
            "metrics": {"mae": 1.0},
            "data_fingerprint": "horizon=24; length=168; mean=12.3",
            "candidate_traces": [
                {
                    "name": "chronos_forecast",
                    "metrics": {"mae": 1.0},
                    "tool_chain": ["chronos_forecast"],
                    "steps": [
                        {
                            "tool": "chronos_forecast",
                            "thought": "Reason: daily cycle is stable. Action: run chronos_forecast.",
                            "observation": '{"forecast":[1,2,3]}',
                        }
                    ],
                },
                {"name": "ets_forecast", "metrics": {"mae": 2.5}},
            ],
        }
    )

    assert "Winner Tool:\n- chronos_forecast" in note
    assert "Fingerprint:\nhorizon=24; length=168; mean=12.3" in note
    assert "Candidates:" in note
    assert "- Candidate: chronos_forecast (mae=1.00)" in note
    assert "Tool chain: chronos_forecast" in note
    assert "- Candidate: ets_forecast (mae=2.50)" in note


def test_daily_note_approaches_use_candidate_trace_names_not_approach_ids():
    note = hm.DailyNotes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": ["chronos_forecast"],
            "metrics": {"mae": 1.0},
            "candidate_traces": [
                {
                    "name": "approach_0",
                    "tool_chain": ["chronos_forecast"],
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

    assert "Approaches:\n- chronos_forecast(mae=1.00)\n- ets_forecast(mae=2.50)" in note
    assert "approach_0" not in note
    assert "approach_1" not in note


def test_daily_note_analysis_keeps_full_pipeline_summary():
    analysis = (
        "Reason: " + "A" * 700 + "\n"
        "Action: " + "B" * 700
    )
    note = hm.DailyNotes()._build_note(
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "description": "forecast sample",
            "predicted": "[1, 2, 3]",
            "tool_chain": ["chronos_forecast"],
            "metrics": {"mae": 1.0},
            "pipeline_summary": analysis,
        }
    )

    assert "Analysis:\n" + analysis in note


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

    async def fake_memory_to_cross_domain() -> bool:
        return False

    async def fake_notes_to_tools() -> bool:
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
