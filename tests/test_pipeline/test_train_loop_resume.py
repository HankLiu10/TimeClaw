import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark import train_loop as tl
from timeclaw.benchmark.mtbench_support.sample_selection import (
    select_samples_with_original_indices,
)
from timeclaw.benchmark.train_loop import (
    _process_one_task_sample,
    _build_train_result_trace_payload,
    _build_train_result_row,
    _build_train_result_evidence_preview,
    _build_failed_result_payload,
    _extract_train_result_reasoning_preview,
    _extract_train_result_tool_chain,
    _load_checkpoint_rows,
    _normalize_train_result_row,
    _recover_failed_result_from_trace,
    _run_pipeline_train_call_with_timeout,
    _sanitize_learning_artifact_preview,
    _summarize_results,
    _train_done_keys,
)
from timeclaw.agents.executor import MaxStepsExhaustedError


def test_train_done_keys_skip_failed_checkpoint_rows():
    rows = [
        {"idx": 0, "task": "forecast", "status": "completed"},
        {"idx": 1, "task": "trend", "status": "timeout"},
        {"idx": 2, "task": "mcqa", "status": "error", "error": "insufficient_quota"},
        {"idx": 3, "task": "indicator_macd", "error": "rate limit"},
        {"idx": 6, "task": "trend_past", "status": "max_steps_exhausted"},
        {"idx": 7, "task": "indicator", "status": "cancelled"},
        {"idx": 5, "task": "forecast", "answer_type": "", "has_prediction": False, "tools_run": 0},
        {"idx": 4, "task": "correlation"},
    ]

    assert _train_done_keys(rows) == {
        (0, "forecast"),
        (4, "correlation"),
    }


def test_select_samples_with_original_indices_preserves_requested_indices():
    samples = ["s0", "s1", "s2", "s3", "s4"]
    selected = select_samples_with_original_indices(
        samples,
        "demo_ds",
        {"demo_ds": [3, 1]},
    )

    assert selected == [(3, "s3"), (1, "s1")]


def test_selected_original_indices_do_not_collide_with_checkpoint_prefix():
    existing_rows = [
        {"idx": i, "task": "forecast", "status": "completed"}
        for i in range(10)
    ]
    done = _train_done_keys(existing_rows)
    selected = select_samples_with_original_indices(
        [f"s{i}" for i in range(20)],
        "demo_ds",
        {"demo_ds": [10, 11]},
    )

    pending = [
        (idx, sample)
        for idx, sample in selected
        if (idx, "forecast") not in done
    ]

    assert pending == [(10, "s10"), (11, "s11")]


def test_build_train_result_row_includes_trace_and_audit_fields():
    sample = MTBenchSample(
        file="00000.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    row = _build_train_result_row(
        ds_name="finance_aligned_short",
        sample_idx=3,
        sample=sample,
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["finance_aligned_short_forecast"],
        result={
            "status": "completed",
            "answer_type": "learning_summary",
            "prediction": [3.0],
            "trace_id": "trace-123",
            "gate_accepted": False,
            "confidence": 0.7,
            "execution_log": [{"tool": "timesfm2_forecast"}],
            "tool_chain": ["timesfm2_forecast", "evaluate_batch_against_gt"],
            "total_ms": 1234,
            "raw_answer": {"insight": "i", "recommendation": "r"},
            "reasoning": "why",
            "evidence_summary": "summary",
        },
    )

    assert row["dataset_name"] == "finance_aligned_short"
    assert row["dataset"] == "finance_aligned_short"
    assert row["abs_idx"] == 3
    assert row["sample_ref"] == "finance_aligned_short:00000.parquet_row:3:forecast"
    assert row["trace_id"] == "trace-123"
    assert row["gate"] == "FAIL"
    assert row["gate_accepted"] is False
    assert row["scope_keys"] == ["finance_aligned_short_forecast"]


def test_build_train_result_row_promotes_learning_summary_candidates_to_top_level():
    sample = MTBenchSample(
        file="00000.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    row = _build_train_result_row(
        ds_name="finance_aligned_short",
        sample_idx=3,
        sample=sample,
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["finance_aligned_short_forecast"],
        result={
            "status": "completed",
            "answer_type": "learning_summary",
            "prediction": [3.0],
            "trace_id": "trace-123",
            "gate_accepted": True,
            "confidence": 0.7,
            "execution_log": [
                {"tool": "timesfm2_forecast", "success": True, "output": {"forecast": [3.0]}},
                {
                    "tool": "evaluate_batch_against_gt",
                    "observation": {
                        "results": [
                            {
                                "name": "timesfm2_forecast",
                                "tool_chain": ["timesfm2_forecast"],
                                "metrics": {"mae": 0.1},
                                "prediction": [3.0],
                            },
                            {
                                "name": "llm_reasoning_forecast",
                                "tool_chain": ["llm_reasoning_forecast"],
                                "metrics": {"mae": 0.2},
                                "prediction": [3.2],
                            },
                        ]
                    },
                },
            ],
            "tool_chain": ["timesfm2_forecast", "evaluate_batch_against_gt"],
            "raw_answer": {
                "insight": "i",
                "recommendation": "r",
                "winner_candidate": "timesfm2_forecast",
                "winner_prediction": [3.0],
                "winner_tool_chain": ["timesfm2_forecast"],
                "winner_metrics": {"mae": 0.1},
                "compared_candidates": [
                    {
                        "name": "timesfm2_forecast",
                        "tool_chain": ["timesfm2_forecast"],
                        "prediction": [3.0],
                        "metrics": {"mae": 0.1},
                    },
                    {
                        "name": "llm_reasoning_forecast",
                        "tool_chain": ["llm_reasoning_forecast"],
                        "prediction": [3.2],
                        "metrics": {"mae": 0.2},
                    },
                ],
            },
        },
    )

    assert row["winner_candidate"] == "timesfm2_forecast"
    assert row["winner_prediction"] == [3.0]
    assert row["winner_metrics"] == {"mae": 0.1}
    assert row["winner_tool_chain"] == ["timesfm2_forecast"]
    assert [item["name"] for item in row["compared_candidates"]] == [
        "timesfm2_forecast",
    ]
    assert row["prediction"] == [3.0]
    assert row["metrics"] == {"mae": 0.1}
    assert row["learning_summary"]["winner_prediction"] == [3.0]


def test_normalize_train_result_row_backfills_legacy_learning_summary_fields():
    normalized = _normalize_train_result_row(
        {
            "dataset": "finance_qa_short",
            "dataset_name": "finance_qa_short",
            "idx": 0,
            "abs_idx": 0,
            "task": "correlation",
            "pipeline_task": "correlation",
            "scope_keys": ["finance_qa_short_correlation"],
            "domain": "finance",
            "status": "completed",
            "answer_type": "learning_summary",
            "raw_answer": {
                "insight": "window_stats -> llm_reasoning_correlation was strongest.",
                "recommendation": "Prefer event-window stats before correlation reasoning.",
                "winner_state": "selected",
                "summary_kind": "validated_summary",
                "winner_candidate": "window_stats -> llm_reasoning_correlation",
                "winner_prediction": "Strong Positive Correlation",
                "winner_tool_chain": ["window_stats", "llm_reasoning_correlation"],
                "winner_metrics": {"accuracy": 1.0, "correct": True},
                "compared_candidates": [
                    {
                        "name": "window_stats -> llm_reasoning_correlation",
                        "tool_chain": ["window_stats", "llm_reasoning_correlation"],
                        "prediction": "Strong Positive Correlation",
                        "metrics": {"accuracy": 1.0, "correct": True},
                    }
                ],
            },
            "execution_log": [],
        }
    )

    assert normalized["winner_prediction"] == "Strong Positive Correlation"
    assert normalized["winner_metrics"] == {"accuracy": 1.0, "correct": True}
    assert normalized["prediction"] == "Strong Positive Correlation"
    assert normalized["metrics"] == {"accuracy": 1.0, "correct": True}
    assert normalized["learning_summary"]["summary_kind"] == "validated_summary"


def test_recover_failed_result_from_trace_recovers_content_block_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace_dir = tmp_path / ".timeclaw" / "traces"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "trace-recover.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "content_block",
                        "payload": {
                            "kind": "prompt_message/assistant",
                            "content": json.dumps(
                                {
                                    "done": True,
                                    "thought": "Reason: enough evidence. Action: finish.",
                                    "answer": {
                                        "insight": "slice path isolated the decisive post-event drop.",
                                        "recommendation": "Prefer tail slicing before correlation reasoning when the event window is the real signal.",
                                    },
                                    "answer_type": "learning_summary",
                                    "confidence": "high",
                                    "reasoning": "1. `slice` isolated the local window.\n2. `llm_reasoning_correlation` matched the sign reversal.",
                                    "execution_context": {
                                        "tool_chain": ["slice", "llm_reasoning_correlation"],
                                        "public_reasoning": [
                                            "`slice` isolated the post-publication drop.",
                                        ],
                                        "evidence_summary": "The sliced path preserved the decisive local move.",
                                    },
                                    "control_context": {
                                        "completion_notes": "Recovered from trace content block.",
                                    },
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "run_end",
                        "payload": {
                            "status": "failed",
                            "error_type": "ValidationError",
                            "failure_reason": "ValidationError",
                            "error": "bad confidence type",
                        },
                    }
                ),
            ]
        )
    )

    recovered = _recover_failed_result_from_trace("trace-recover")

    assert recovered["answer_type"] == "learning_summary"
    assert recovered["raw_answer"]["insight"].startswith("slice path")
    assert recovered["execution_context"]["tool_chain"] == ["slice", "llm_reasoning_correlation"]
    assert recovered["control_context"]["completion_notes"] == "Recovered from trace content block."
    assert recovered["confidence"] is None
    assert recovered["trace_recovered"] is True


@pytest.mark.asyncio
async def test_run_pipeline_train_call_with_timeout_preserves_trace_metadata():
    async def _never_finishes():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError as exc:
            setattr(exc, "trace_id", "trace-timeout-123")
            setattr(exc, "total_ms", 4321)
            raise

    completed, payload = await _run_pipeline_train_call_with_timeout(
        _never_finishes(),
        timeout_s=0.01,
    )

    assert completed is False
    assert isinstance(payload, asyncio.TimeoutError)
    assert getattr(payload, "trace_id") == "trace-timeout-123"
    assert getattr(payload, "total_ms") == 4321
    assert getattr(payload, "failure_reason") == "timeout"


def test_build_failed_result_payload_coerces_non_numeric_confidence_to_none():
    exc = RuntimeError("boom")
    setattr(exc, "failure_reason", "error")
    setattr(exc, "error_type", "RuntimeError")
    setattr(exc, "trace_id", "trace-abc")

    payload = _build_failed_result_payload(
        exc,
        {
            "answer_type": "learning_summary",
            "raw_answer": {"insight": "x", "recommendation": "y"},
            "confidence": "high",
            "reasoning": "1. x",
            "execution_context": {"tool_chain": ["slice"]},
            "control_context": {"completion_notes": ""},
            "trace_recovered": True,
        },
    )

    assert payload["confidence"] is None
    assert payload["trace_id"] == "trace-abc"
    assert payload["raw_answer"]["insight"] == "x"
    assert payload["raw_answer"]["recommendation"] == "y"
    assert payload["raw_answer"]["summary_kind"] == "failure_summary"
    assert payload["raw_answer"]["winner_tool_chain"] == []


def test_build_failed_result_payload_normalizes_learning_summary_winner_metadata():
    exc = RuntimeError("boom")
    setattr(exc, "failure_reason", "RuntimeError")
    setattr(exc, "error_type", "RuntimeError")
    setattr(exc, "trace_id", "trace-xyz")
    setattr(exc, "partial_result", {
        "answer_type": "learning_summary",
        "answer": {
            "insight": "The winning path was summarize -> llm_reasoning_correlation.",
            "recommendation": "Keep the summarized article evidence.",
        },
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "slot_id": 0,
                            "answer": "Moderate Negative Correlation",
                            "answer_type": "correlation",
                            "reasoning": "1. summarize -> llm_reasoning_correlation captured the mild negative tone.",
                            "execution_context": {
                                "tool_chain": ["summarize", "llm_reasoning_correlation"],
                                "public_reasoning": [
                                    "summarize highlighted the mildly weaker analyst outlook.",
                                ],
                                "evidence_summary": "The summary preserved the slight negative tone before correlation reasoning.",
                            },
                            "execution_log": [
                                {"tool": "summarize", "observation": {"summary": "mildly negative"}},
                                {"tool": "llm_reasoning_correlation", "observation": {"answer": "Moderate Negative Correlation"}},
                            ],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {
                            "name": "slot_0",
                            "prediction": "Moderate Negative Correlation",
                            "metrics": {"accuracy": 1.0, "correct": True},
                        },
                    ],
                },
            },
        ],
    })

    payload = _build_failed_result_payload(exc, {})

    assert payload["answer_type"] == "learning_summary"
    assert payload["raw_answer"]["winner_tool_chain"] == ["summarize", "llm_reasoning_correlation"]
    assert payload["raw_answer"]["winner_prediction"] == "Moderate Negative Correlation"


def test_build_train_result_row_hides_learning_text_for_failed_rows():
    sample = MTBenchSample(
        file="demo.json",
        dataset="demo_ds",
        domain="finance",
        task_types=["correlation"],
        input_window=np.array([1.0, 2.0, 3.0]),
        output_window=np.array([4.0]),
        text="demo",
        timestamps=[0.0, 1.0, 2.0],
        start_year=2024,
        labels={},
    )
    row = _build_train_result_row(
        ds_name="demo_ds",
        sample_idx=0,
        sample=sample,
        task="correlation",
        pipeline_task="correlation",
        scope_keys=["demo_scope"],
        train_run_id="run-1",
        result={
            "status": "RuntimeError",
            "answer_type": "learning_summary",
            "raw_answer": {"insight": "x", "recommendation": "y"},
            "execution_context": {"tool_chain": ["slice", "llm_reasoning_correlation"]},
            "execution_log": [],
        },
    )

    assert row["learning_insight"] is None
    assert row["learning_recommendation"] is None


def test_summarize_results_counts_only_completed_learning_summaries():
    summary = _summarize_results([
        {"task": "forecast", "status": "completed", "answer_type": "learning_summary"},
        {"task": "forecast", "status": "RuntimeError", "answer_type": "learning_summary"},
        {"task": "forecast", "status": "completed", "answer_type": "qa"},
    ])

    assert summary["forecast"]["count"] == 3
    assert summary["forecast"]["learning_summaries"] == 1


def test_summarize_results_exposes_aligned_counts_without_mixing_semantics():
    summary = _summarize_results([
        {
            "task": "forecast",
            "status": "completed",
            "answer_type": "learning_summary",
            "has_prediction": True,
            "gate_accepted": True,
            "winner_state": "selected",
            "winner_metrics": {"mae": 1.2, "rmse": 1.5},
        },
        {
            "task": "forecast",
            "status": "completed",
            "answer_type": "learning_summary",
            "has_prediction": False,
            "gate_accepted": False,
            "winner_state": "none",
            "winner_metrics": {},
        },
        {
            "task": "forecast",
            "status": "timeout",
            "answer_type": "",
            "has_prediction": False,
            "gate_accepted": False,
        },
        {
            "task": "trend",
            "status": "completed",
            "answer_type": "learning_summary",
            "has_prediction": True,
            "gate_accepted": True,
            "winner_state": "selected",
            "winner_metrics": {"correct": True},
        },
    ])

    forecast_counts = summary["forecast"]["aligned_counts"]
    assert forecast_counts == {
        "execution_total_n": 3,
        "execution_completed_n": 2,
        "execution_failed_n": 1,
        "prediction_emitted_n": 1,
        "gate_pass_n": 1,
        "gate_fail_n": 2,
        "learning_summary_n": 2,
        "validated_winner_n": 1,
        "no_winner_n": 1,
        "binary_correct_n": 0,
        "binary_incorrect_n": 0,
        "continuous_scored_n": 1,
    }
    trend_counts = summary["trend"]["aligned_counts"]
    assert trend_counts["binary_correct_n"] == 1
    assert trend_counts["validated_winner_n"] == 1
    assert trend_counts["continuous_scored_n"] == 0


def test_build_train_result_row_flattens_summary_kind_from_learning_summary():
    row = tl._build_train_result_row(
        ds_name="demo_ds",
        sample_idx=0,
        sample=SimpleNamespace(
            file="demo.json",
            domain="weather",
            output_window=np.asarray([1.0, 2.0]),
        ),
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["demo_ds_forecast"],
        result={
            "status": "completed",
            "answer_type": "learning_summary",
            "prediction": [1.0, 2.0],
            "gate_accepted": True,
            "raw_answer": {
                "summary_kind": "validated_summary",
                "winner_state": "selected",
                "winner_candidate": "chronos2_forecast",
                "winner_prediction": [1.0, 2.0],
                "winner_tool_chain": ["chronos2_forecast"],
                "winner_metrics": {"mae": 1.2},
                "insight": "chronos won",
                "recommendation": "prefer chronos",
            },
            "execution_log": [{"tool": "chronos2_forecast"}],
            "execution_context": {"tool_chain": ["chronos2_forecast"]},
            "control_context": {},
        },
        train_run_id="run-1",
    )

    assert row["summary_kind"] == "validated_summary"
    assert row["winner_state"] == "selected"
    assert row["winner_tool_chain"] == ["chronos2_forecast"]


def test_build_run_manifest_uses_same_aligned_counts_as_summary():
    rows = [
        {
            "sample_ref": "demo_ds:file:0:forecast",
            "task": "forecast",
            "status": "completed",
            "answer_type": "learning_summary",
            "has_prediction": True,
            "gate_accepted": True,
            "winner_state": "selected",
            "winner_metrics": {"mae": 1.0},
        },
        {
            "sample_ref": "demo_ds:file:1:forecast",
            "task": "forecast",
            "status": "completed",
            "answer_type": "learning_summary",
            "has_prediction": False,
            "gate_accepted": False,
            "winner_state": "none",
            "winner_metrics": {},
        },
        {
            "sample_ref": "demo_ds:file:2:forecast",
            "task": "forecast",
            "status": "timeout",
            "answer_type": "",
            "has_prediction": False,
            "gate_accepted": False,
        },
    ]

    manifest = tl._build_run_manifest(
        train_run_id="run-1",
        started_at=10.0,
        finished_at=12.5,
        all_results={},
        task_rows_by_key={("demo_ds", "forecast"): rows},
        note_view_paths={},
    )

    task_info = manifest["tasks"]["demo_ds::forecast"]
    assert task_info["rows"] == 3
    assert task_info["status_counts"] == {"completed": 2, "timeout": 1}
    assert task_info["failed_sample_refs"] == [
        "demo_ds:file:1:forecast",
        "demo_ds:file:2:forecast",
    ]
    assert task_info["aligned_counts"] == tl._aligned_train_counts(rows)


def test_save_task_checkpoint_writes_complete_json_without_tmp_leftovers(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "TRAIN_RESULTS_DIR", tmp_path)
    rows = [{"idx": 0, "task": "forecast", "status": "completed"}]

    tl._save_task_checkpoint(rows, "demo_ds", "forecast")

    path = tmp_path / "train_demo_ds__forecast.json"
    assert json.loads(path.read_text()) == [{
        "idx": 0,
        "abs_idx": 0,
        "task": "forecast",
        "status": "completed",
        "dataset": "demo_ds",
        "dataset_name": "demo_ds",
        "winner_tool_chain": [],
        "tools_run": 0,
    }]
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_train_summary_replaces_stale_payload(tmp_path, monkeypatch):
    summary_path = tmp_path / "train_summary.json"
    monkeypatch.setattr(tl, "TRAIN_SUMMARY_PATH", summary_path)

    tl._save_train_summary({"finance_qa_long": {"total": 1}})
    tl._save_train_summary({"weather_aligned_short": {"total": 15}})

    assert json.loads(summary_path.read_text()) == {"weather_aligned_short": {"total": 15}}


def test_export_train_note_views_filters_to_current_run(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "TRAIN_RESULTS_DIR", tmp_path)

    class _Notes:
        def get_scope_note_items(self, scope):
            if scope != "finance_aligned_short_forecast":
                return []
            return [
                {
                    "id": "old",
                    "text": (
                        "[finance_aligned_short_forecast] SCORED\n\n"
                        "Source:\n"
                        "- Sample Ref: finance_aligned_short:old.parquet_row:0:forecast\n"
                        "- Train Run ID: old-run\n"
                    ),
                },
                {
                    "id": "new",
                    "text": (
                        "[finance_aligned_short_forecast] SCORED\n\n"
                        "Source:\n"
                        "- Sample Ref: finance_aligned_short:new.parquet_row:1:forecast\n"
                        "- Train Run ID: run-123\n"
                    ),
                },
            ]

    pipeline = type(
        "P",
        (),
        {"memory": type("M", (), {"notes": _Notes()})()},
    )()

    exports = tl._export_train_note_views(
        pipeline=pipeline,
        train_run_id="run-123",
        run_rows_by_task={
            ("finance_aligned_short", "forecast"): [
                {"sample_ref": "finance_aligned_short:new.parquet_row:1:forecast"},
            ]
        },
    )

    out_path = tmp_path / "train_notes_finance_aligned_short__forecast.md"
    assert exports["finance_aligned_short::forecast"] == str(out_path)
    content = out_path.read_text()
    assert "run-123" in content
    assert "old-run" not in content


def test_export_train_note_views_excludes_historical_same_sample_ref_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "TRAIN_RESULTS_DIR", tmp_path)

    sample_ref = "weather_aligned_long:USW00003017_0.json:0:trend_past"

    class _Notes:
        def get_scope_note_items(self, scope):
            if scope != "weather_aligned_long_trend_past":
                return []
            return [
                {
                    "id": "old",
                    "text": (
                        "[weather_aligned_long_trend_past] OK\n\n"
                        "Source:\n"
                        f"- Sample Ref: {sample_ref}\n"
                        "- Train Run ID: old-run\n"
                        "\nCandidates:\n"
                        "[CANDIDATES_BEGIN]\n"
                        "- Candidate: llm_reasoning_classification\n"
                        "  Prediction: spawn_subagent\n"
                        "[CANDIDATES_END]\n"
                    ),
                },
                {
                    "id": "new",
                    "text": (
                        "[weather_aligned_long_trend_past] OK\n\n"
                        "Source:\n"
                        f"- Sample Ref: {sample_ref}\n"
                        "- Train Run ID: run-123\n"
                        "\nCandidates:\n"
                        "[CANDIDATES_BEGIN]\n"
                        "- Candidate: llm_reasoning_classification\n"
                        "  Prediction: decreasing\n"
                        "[CANDIDATES_END]\n"
                    ),
                },
            ]

    pipeline = type(
        "P",
        (),
        {"memory": type("M", (), {"notes": _Notes()})()},
    )()

    tl._export_train_note_views(
        pipeline=pipeline,
        train_run_id="run-123",
        run_rows_by_task={
            ("weather_aligned_long", "trend_past"): [
                {"sample_ref": sample_ref},
            ],
        },
    )

    content = (tmp_path / "train_notes_weather_aligned_long__trend_past.md").read_text()
    assert "run-123" in content
    assert "old-run" not in content
    assert "Prediction: spawn_subagent" not in content
    assert "Prediction: decreasing" in content


def test_normalize_train_result_row_backfills_locator_fields_from_legacy_row():
    normalized = _normalize_train_result_row({
        "idx": 8,
        "dataset_name": "finance_aligned_long",
        "task": "forecast",
        "tool_chain": ["chronos2_forecast", "evaluate_batch_against_gt"],
    })

    assert normalized["idx"] == 8
    assert normalized["abs_idx"] == 8
    assert normalized["dataset_name"] == "finance_aligned_long"
    assert normalized["dataset"] == "finance_aligned_long"
    assert normalized["tool_chain"] == ["chronos2_forecast"]
    assert normalized["winner_tool_chain"] == ["chronos2_forecast"]


def test_normalize_train_result_row_repairs_failed_learning_summary_winner_alignment():
    normalized = _normalize_train_result_row({
        "idx": 8,
        "dataset_name": "finance_qa_short",
        "task": "correlation",
        "status": "RuntimeError",
        "answer_type": "learning_summary",
        "raw_answer": {
            "insight": "The winning path was summarize -> llm_reasoning_correlation.",
            "recommendation": "Use summarize before correlation reasoning.",
        },
        "execution_context": {
            "tool_chain": ["summarize", "llm_reasoning_correlation"],
        },
        "learning_insight": "The winning path was summarize -> llm_reasoning_correlation.",
        "learning_recommendation": "Use summarize before correlation reasoning.",
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "slot_id": 0,
                            "answer": "No Correlation",
                            "answer_type": "correlation",
                            "execution_log": [
                                {"tool": "summarize", "success": True, "output": {"summary": "slightly negative"}},
                                {
                                    "tool": "llm_reasoning_correlation",
                                    "success": True,
                                    "output": {"answer": "No Correlation"},
                                },
                            ],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "slot_id": 1,
                            "answer": "Moderate Negative Correlation",
                            "answer_type": "correlation",
                            "execution_log": [
                                {"tool": "value_at", "success": True, "output": {"value": 76.065}},
                                {
                                    "tool": "llm_reasoning_correlation",
                                    "success": True,
                                    "output": {"answer": "Moderate Negative Correlation"},
                                },
                            ],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {
                            "name": "slot_0",
                            "prediction": "No Correlation",
                            "metrics": {"accuracy": 0.0, "correct": False},
                        },
                        {
                            "name": "slot_1",
                            "prediction": "Moderate Negative Correlation",
                            "metrics": {"accuracy": 1.0, "correct": True},
                        },
                    ],
                },
            },
        ],
    })

    assert normalized["tool_chain"] == ["value_at", "llm_reasoning_correlation"]
    assert normalized["winner_tool_chain"] == ["value_at", "llm_reasoning_correlation"]
    assert normalized["raw_answer"]["winner_tool_chain"] == ["value_at", "llm_reasoning_correlation"]
    assert "value_at -> llm_reasoning_correlation" in normalized["raw_answer"]["insight"]
    assert normalized["learning_insight"] is None
    assert normalized["learning_recommendation"] is None


def test_normalize_train_result_row_backfills_top_level_winner_fields_from_raw_answer():
    normalized = _normalize_train_result_row({
        "idx": 8,
        "dataset_name": "finance_aligned_long",
        "task": "trend",
        "status": "completed",
        "answer_type": "learning_summary",
        "raw_answer": {
            "insight": "i",
            "recommendation": "r",
            "winner_candidate": "segment -> llm_reasoning_classification",
            "winner_prediction": "increasing",
            "winner_tool_chain": ["slice", "llm_reasoning_classification"],
            "winner_metrics": {"acc": 1.0},
            "compared_candidates": [
                {
                    "name": "segment -> llm_reasoning_classification",
                    "tool_chain": ["slice", "llm_reasoning_classification"],
                    "prediction": "increasing",
                    "metrics": {"acc": 1.0},
                },
                {
                    "name": "sentiment -> llm_reasoning_classification",
                    "tool_chain": ["sentiment", "llm_reasoning_classification"],
                    "prediction": "stable",
                    "metrics": {"acc": 0.0},
                },
            ],
        },
        "execution_log": [],
        "winner_candidate": None,
        "winner_prediction": None,
        "winner_metrics": None,
        "compared_candidates": None,
    })

    assert normalized["winner_candidate"] == "segment -> llm_reasoning_classification"
    assert normalized["winner_prediction"] == "increasing"
    assert normalized["winner_metrics"] == {"acc": 1.0}
    assert [item["name"] for item in normalized["compared_candidates"]] == [
        "segment -> llm_reasoning_classification",
        "sentiment -> llm_reasoning_classification",
    ]


def test_normalize_train_result_row_keeps_indicator_winner_predictions_task_aware():
    normalized = _normalize_train_result_row({
        "idx": 11,
        "dataset_name": "finance_aligned_short",
        "dataset": "finance_aligned_short",
        "domain": "finance",
        "task": "indicator_upper_bb",
        "pipeline_task": "indicator",
        "scope_keys": ["finance_aligned_short_indicator_upper_bb"],
        "status": "completed",
        "answer_type": "learning_summary",
        "raw_answer": {
            "insight": "i",
            "recommendation": "r",
        },
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": [101.5, 102.0],
                            "execution_log": [
                                {
                                    "tool": "compute_bollinger",
                                    "observation": {"forecast": [101.5, 102.0]},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {
                            "name": "compute_bollinger",
                            "prediction": [101.5, 102.0],
                            "metrics": {"upper_bb_mae": 0.2},
                        }
                    ]
                },
            },
        ],
    })

    assert normalized["winner_prediction"] == {"upper_bb": [101.5, 102.0]}
    assert normalized["winner_tool_chain"] == ["compute_bollinger"]
    assert normalized["raw_answer"]["winner_prediction"] == {"upper_bb": [101.5, 102.0]}
    assert normalized["raw_answer"]["compared_candidates"][0]["prediction"] == {
        "upper_bb": [101.5, 102.0]
    }


def test_build_train_result_trace_payload_keeps_indicator_candidates_task_aware():
    payload = _build_train_result_trace_payload(
        {
            "trace_id": "trace-indicator-1",
            "dataset_name": "finance_aligned_short",
            "dataset": "finance_aligned_short",
            "domain": "finance",
            "task": "indicator_upper_bb",
            "pipeline_task": "indicator",
            "scope_keys": ["finance_aligned_short_indicator_upper_bb"],
            "raw_answer": {
                "winner_candidate": "compute_bollinger",
                "winner_prediction": {"upper_bb": [101.5, 102.0]},
                "winner_tool_chain": ["compute_bollinger"],
                "winner_metrics": {"upper_bb_mae": 0.2},
            },
            "execution_log": [
                {
                    "tool": "spawn_subagent",
                    "output": [
                        {
                            "success": True,
                            "result": {
                                "answer": [101.5, 102.0],
                                "execution_log": [
                                    {
                                        "tool": "compute_bollinger",
                                        "observation": {"forecast": [101.5, 102.0]},
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "tool": "evaluate_batch_against_gt",
                    "observation": {
                        "results": [
                            {
                                "name": "compute_bollinger",
                                "prediction": [101.5, 102.0],
                                "metrics": {"upper_bb_mae": 0.2},
                            }
                        ]
                    },
                },
            ],
        },
        substantive_tool_chain=["compute_bollinger"],
    )

    assert payload["candidate_traces"][0]["prediction"] == {"upper_bb": [101.5, 102.0]}
    assert payload["winner_trace"]["prediction"] == {"upper_bb": [101.5, 102.0]}


def test_normalize_train_result_row_forces_fail_when_learning_summary_has_no_winner():
    normalized = _normalize_train_result_row({
        "idx": 9,
        "dataset_name": "weather_aligned_long",
        "task": "trend_past",
        "status": "completed",
        "answer_type": "learning_summary",
        "has_prediction": True,
        "gate": "PASS",
        "gate_accepted": True,
        "gate_rejections": [],
        "tool_chain": ["segment", "llm_reasoning_classification"],
        "winner_tool_chain": ["segment", "llm_reasoning_classification"],
        "tools_run": 2,
        "execution_context": {"tool_chain": ["wrong -> path"]},
        "raw_answer": {
            "insight": "No validated path exists.",
            "recommendation": "Recheck the numeric rule.",
            "winner_state": "none",
            "winner_candidate": "",
            "winner_prediction": None,
            "winner_tool_chain": [],
            "winner_metrics": {},
            "execution_context": {
                "tool_chain": [
                    "segment -> llm_reasoning_classification",
                    "llm_reasoning_classification",
                ],
                "public_reasoning": [
                    "Both explored paths returned stable.",
                ],
                "evidence_summary": "Both explored paths failed against ground truth.",
            },
            "compared_candidates": [
                {
                    "name": "segment -> llm_reasoning_classification",
                    "tool_chain": ["segment", "llm_reasoning_classification"],
                    "prediction": "stable",
                    "metrics": {"accuracy": 0.0, "correct": False},
                },
                {
                    "name": "llm_reasoning_classification",
                    "tool_chain": ["llm_reasoning_classification"],
                    "prediction": "stable",
                    "metrics": {"accuracy": 0.0, "correct": False},
                },
            ],
        },
        "execution_log": [],
    })

    assert normalized["gate"] == "FAIL"
    assert normalized["gate_accepted"] is False
    assert normalized["has_prediction"] is False
    assert normalized["tool_chain"] == []
    assert normalized["winner_tool_chain"] == []
    assert normalized["winner_state"] == "none"
    assert normalized["tools_run"] == 0
    assert "prediction is None" in normalized["gate_rejections"]
    assert normalized["execution_context"]["tool_chain"] == [
        "segment -> llm_reasoning_classification",
        "llm_reasoning_classification",
    ]
    assert normalized["execution_context"]["public_reasoning"] == [
        "Both explored paths returned stable.",
    ]
    assert normalized["execution_context"]["evidence_summary"] == (
        "Both explored paths failed against ground truth."
    )


def test_normalize_train_result_row_rebuilds_stale_trace_from_execution_log():
    normalized = _normalize_train_result_row({
        "idx": 10,
        "dataset_name": "weather_aligned_long",
        "task": "forecast",
        "status": "completed",
        "answer_type": "learning_summary",
        "trace": {
            "trace_id": "stale-trace",
            "candidate_traces": [
                {
                    "tool_chain": ["slice"],
                    "prediction": [1.0, 2.0],
                    "metrics": {"mae": 9.9},
                }
            ],
            "winner_trace": {
                "tool_chain": ["slice"],
                "prediction": [1.0, 2.0],
                "metrics": {"mae": 9.9},
            },
        },
        "raw_answer": {
            "insight": "TimesFM won over the slice baseline.",
            "recommendation": "Use TimesFM directly for this forecast.",
            "winner_candidate": "timesfm2_forecast",
            "winner_prediction": [3.0, 4.0],
            "winner_tool_chain": ["timesfm2_forecast"],
            "winner_metrics": {"mae": 0.1},
            "compared_candidates": [
                {
                    "name": "timesfm2_forecast",
                    "tool_chain": ["timesfm2_forecast"],
                    "prediction": [3.0, 4.0],
                    "metrics": {"mae": 0.1},
                },
                {
                    "name": "slice",
                    "tool_chain": ["slice"],
                    "prediction": [1.0, 2.0],
                    "metrics": {"mae": 1.2},
                },
            ],
        },
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "slot_id": 0,
                            "answer": [3.0, 4.0],
                            "answer_type": "forecast",
                            "execution_log": [
                                {
                                    "tool": "timesfm2_forecast",
                                    "success": True,
                                    "output": {"forecast": [3.0, 4.0]},
                                },
                            ],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "slot_id": 1,
                            "answer": [1.0, 2.0],
                            "answer_type": "forecast",
                            "execution_log": [
                                {
                                    "tool": "slice",
                                    "success": True,
                                    "output": {"window": [1.0, 2.0]},
                                },
                            ],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {
                            "name": "slot_0",
                            "prediction": [3.0, 4.0],
                            "metrics": {"mae": 0.1},
                        },
                        {
                            "name": "slot_1",
                            "prediction": [1.0, 2.0],
                            "metrics": {"mae": 1.2},
                        },
                    ],
                },
            },
        ],
    })

    assert [item["tool_chain"] for item in normalized["trace"]["candidate_traces"]] == [
        ["timesfm2_forecast"],
        ["slice"],
    ]
    assert normalized["trace"]["winner_trace"]["tool_chain"] == ["timesfm2_forecast"]
    assert normalized["trace"]["winner_trace"]["metrics"] == {"mae": 0.1}


def test_load_checkpoint_rows_normalizes_legacy_schema(tmp_path):
    path = tmp_path / "train_demo__forecast.json"
    path.write_text(json.dumps([{
        "idx": 2,
        "dataset_name": "demo_ds",
        "task": "forecast",
        "tool_chain": ["timesfm2_forecast", "evaluate_batch_against_gt"],
        "status": "completed",
    }]))

    rows = _load_checkpoint_rows(path)

    assert rows == [{
        "idx": 2,
        "abs_idx": 2,
        "dataset_name": "demo_ds",
        "dataset": "demo_ds",
        "task": "forecast",
        "tool_chain": ["timesfm2_forecast"],
        "winner_tool_chain": ["timesfm2_forecast"],
        "status": "completed",
        "tools_run": 1,
    }]


def test_build_train_result_row_extracts_reasoning_preview_from_json_envelope():
    sample = MTBenchSample(
        file="00000.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    row = _build_train_result_row(
        ds_name="finance_aligned_short",
        sample_idx=7,
        sample=sample,
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["finance_aligned_short_forecast"],
        result={
            "status": "completed",
            "answer_type": "learning_summary",
            "prediction": [3.0],
            "trace_id": "trace-reasoning-json",
            "gate_accepted": True,
            "confidence": 0.9,
            "execution_log": [{"tool": "timesfm2_forecast"}],
            "tool_chain": ["timesfm2_forecast"],
            "total_ms": 123,
            "raw_answer": {"insight": "i", "recommendation": "r"},
            "reasoning": json.dumps(
                {
                    "thought": "Reason: Repair the final delivery while preserving the selected branch. Action: finish.",
                    "done": True,
                    "answer": {
                        "insight": "i",
                        "recommendation": "r",
                        "reasoning": "1. Compared the explored candidates.\n2. Preserved the winner-aligned summary.",
                    },
                }
            ),
            "evidence_summary": "summary",
        },
    )

    assert row["reasoning_preview"].startswith("1. Compared the explored candidates.")
    assert row["reasoning_preview"] != ""
    assert row["reasoning_preview"].lstrip().startswith("{") is False


def test_sanitize_learning_artifact_preview_removes_framework_wording():
    text = _sanitize_learning_artifact_preview(
        "Step 0: evaluate_batch_against_gt -> Reason: before finishing.\n"
        "Winner path stayed close to the recent regime.",
        limit=200,
    )

    assert "evaluate_batch_against_gt" not in text
    assert "before finishing" not in text
    assert "recent regime" in text


def test_sanitize_learning_artifact_preview_drops_jsonish_framework_preview():
    text = _sanitize_learning_artifact_preview(
        '{"insight":"GT-selected winner after evaluate_batch_against_gt before finishing.",'
        '"recommendation":"Use the winning path."}',
        limit=200,
    )

    assert text == ""


def test_sanitize_learning_artifact_preview_drops_control_flow_meta():
    text = _sanitize_learning_artifact_preview(
        "The current branch state confirms that this answer is available from a real tool result and may be returned unchanged as the final classification.",
        limit=200,
    )

    assert text == ""


def test_sanitize_learning_artifact_preview_keeps_clean_numbered_steps_from_single_line_reasoning():
    text = _sanitize_learning_artifact_preview(
        "1. Candidate path llm_reasoning_classification directly produced a task-valid label. "
        "2. Candidate path segment -> llm_reasoning_classification first created daily windows. "
        "3. The decisive evidence was the task's required daily-mean trend rule. "
        "4. The winner was selected as segment -> llm_reasoning_classification because it matched the required aggregation step.",
        limit=None,
    )

    assert "task-valid" not in text
    assert "2. Candidate path segment -> llm_reasoning_classification" in text
    assert "3. The decisive evidence" in text
    assert "4. The winner was selected as segment -> llm_reasoning_classification" in text


def test_sanitize_learning_artifact_preview_removes_task_valid_phrase_but_keeps_line():
    text = _sanitize_learning_artifact_preview(
        '2. llm_reasoning_classification produced the task-valid label "decreasing", which became the final winning answer.',
        limit=None,
    )

    assert "task-valid" not in text
    assert 'llm_reasoning_classification produced the label "decreasing"' in text


def test_extract_train_result_reasoning_preview_does_not_truncate_learning_summary_reasoning():
    long_reasoning = "1. " + ("Compared the explored candidates carefully. " * 40)
    preview = _extract_train_result_reasoning_preview(
        {"reasoning": ""},
        {"reasoning": long_reasoning},
    )

    assert preview.startswith("1. Compared the explored candidates carefully.")
    assert len(preview) >= len(long_reasoning.strip()) - 1
    assert "... [" not in preview


def test_extract_train_result_reasoning_preview_keeps_nested_json_reasoning_with_task_valid_phrase():
    result = {
        "reasoning": json.dumps(
            {
                "thought": (
                    "Reason: The winning path is already established and only the final delivery needs repair. "
                    "Action: Finish with the corrected learning-summary answer."
                ),
                "done": True,
                "answer": {
                    "reasoning": (
                        "1. segment was chosen because the prompt explicitly required computing daily mean temperatures "
                        "from 336 hourly readings. "
                        '2. llm_reasoning_classification was chosen because it produced the task-valid label "decreasing", '
                        "which became the final winning answer."
                    )
                },
            }
        )
    }

    preview = _extract_train_result_reasoning_preview(result, {})

    assert "task-valid" not in preview
    assert "1. segment was chosen" in preview
    assert '2. llm_reasoning_classification was chosen because it produced the label "decreasing"' in preview


def test_extract_train_result_reasoning_preview_can_use_execution_context_when_reasoning_is_empty():
    preview = _extract_train_result_reasoning_preview(
        {
            "reasoning": "",
            "execution_context": {
                "tool_chain": ["llm_reasoning_classification"],
                "public_reasoning": [
                    "The recent daily means are nearly flat, so the stable label is supported."
                ],
                "evidence_summary": "The recent daily means are nearly flat.",
            },
        },
        {},
    )

    assert "stable label is supported" in preview


def test_extract_train_result_reasoning_preview_prefers_winner_trace_reasoning_over_final_repair_text():
    result = {
        "raw_answer": {
            "winner_candidate": "segment",
            "winner_tool_chain": ["segment", "llm_reasoning_classification"],
            "winner_prediction": "decreasing",
        },
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "decreasing",
                            "reasoning": (
                                "The segmented path matched the daily aggregation required by the task."
                            ),
                            "execution_log": [
                                {
                                    "tool": "segment",
                                    "thought": "Reason: Convert the hourly sequence into daily windows before classifying. Action: segment.",
                                },
                                {
                                    "tool": "llm_reasoning_classification",
                                    "thought": "Reason: The daily means show a decreasing pattern after aggregation. Action: classify.",
                                    "observation": {"answer": "decreasing"},
                                },
                            ],
                        },
                    }
                ],
            }
        ],
        "reasoning": json.dumps(
            {
                "thought": "Reason: The corrected final answer should be retained in the branch. Action: finish.",
                "answer": {"reasoning": "1. Repaired summary.\n2. Winning answer retained in the branch."},
            }
        ),
    }

    preview = _extract_train_result_reasoning_preview(result, {})

    assert 'llm_reasoning_classification produced "decreasing"' in preview
    assert "retained in the branch" not in preview
    assert "repaired summary" not in preview.lower()


def test_build_train_result_evidence_preview_prefers_observation_over_reason():
    preview = _build_train_result_evidence_preview(
        {
            "execution_log": [
                {
                    "tool": "llm_reasoning_classification",
                    "thought": (
                        "Reason: The task explicitly benefits from comparing candidate branches. "
                        "Action: classify."
                    ),
                    "observation": {"answer": "stable"},
                }
            ]
        },
        ["llm_reasoning_classification"],
    )

    assert 'Observed: {"answer": "stable"}' in preview
    assert "Reason:" not in preview
    assert "candidate branches" not in preview


def test_build_train_result_row_preserves_explicit_failure_reason_fields():
    sample = MTBenchSample(
        file="00001.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    row = _build_train_result_row(
        ds_name="finance_aligned_short",
        sample_idx=4,
        sample=sample,
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["finance_aligned_short_forecast"],
        result={
            "status": "max_steps_exhausted",
            "answer_type": "",
            "prediction": None,
            "trace_id": "trace-456",
            "execution_log": [],
            "tool_chain": [],
            "total_ms": 321,
            "error": "executor exhausted max steps without producing a finalized answer",
            "error_type": "MaxStepsExhaustedError",
            "failure_reason": "max_steps_exhausted",
        },
    )

    assert row["status"] == "max_steps_exhausted"
    assert row["error_type"] == "MaxStepsExhaustedError"
    assert row["failure_reason"] == "max_steps_exhausted"


def test_extract_train_result_tool_chain_prefers_learning_summary_winner_chain():
    result = {
        "raw_answer": {
            "winner_tool_chain": [
                "hurst_exponent",
                "timesfm2_forecast",
                "evaluate_batch_against_gt",
            ]
        },
        "execution_log": [
            {"tool": "spawn_subagent"},
            {"tool": "evaluate_batch_against_gt"},
        ],
        "tool_chain": ["spawn_subagent", "evaluate_batch_against_gt"],
    }

    assert _extract_train_result_tool_chain(result) == [
        "hurst_exponent",
        "timesfm2_forecast",
    ]


def test_extract_train_result_tool_chain_respects_explicit_no_winner_state():
    result = {
        "raw_answer": {
            "winner_state": "none",
            "winner_tool_chain": ["llm_reasoning_classification"],
            "execution_context": {"tool_chain": ["llm_reasoning_classification"]},
        },
        "execution_log": [
            {"tool": "llm_reasoning_classification"},
            {"tool": "evaluate_against_gt"},
        ],
        "tool_chain": ["llm_reasoning_classification"],
    }

    assert _extract_train_result_tool_chain(result) == []


def test_extract_train_result_tool_chain_prefers_observed_path_when_explicit_winner_chain_is_wrong():
    result = {
        "raw_answer": {
            "winner_candidate": "top_level::llm_reasoning_classification",
            "winner_tool_chain": ["llm_reasoning_classification"],
            "winner_prediction": "-2% ~ +2%",
        },
        "execution_log": [
            {
                "tool": "llm_reasoning_classification",
                "observation": {"answer": "-2% ~ -4%"},
            },
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {"tool": "value_at", "observation": {"value": 38.98}},
                                {"tool": "value_at", "observation": {"value": 34.39}},
                                {"tool": "window_stats", "observation": {"mean": 34.8}},
                                {"tool": "llm_reasoning_classification", "observation": {"answer": "-2% ~ +2%"}},
                            ],
                            "answer": "-2% ~ +2%",
                        },
                    }
                ],
            },
        ],
    }

    assert _extract_train_result_tool_chain(result) == [
        "value_at",
        "value_at",
        "window_stats",
        "llm_reasoning_classification",
    ]


def test_extract_train_result_tool_chain_can_read_execution_context_winner_chain():
    result = {
        "raw_answer": {
            "execution_context": {
                "tool_chain": [
                    "hurst_exponent",
                    "timesfm2_forecast",
                    "evaluate_batch_against_gt",
                ]
            }
        },
        "execution_log": [
            {"tool": "spawn_subagent"},
            {"tool": "evaluate_batch_against_gt"},
        ],
    }

    assert _extract_train_result_tool_chain(result) == [
        "hurst_exponent",
        "timesfm2_forecast",
    ]


def test_build_train_result_row_uses_candidate_winner_chain_not_framework_chain():
    sample = MTBenchSample(
        file="00002.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    row = _build_train_result_row(
        ds_name="finance_aligned_short",
        sample_idx=5,
        sample=sample,
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["finance_aligned_short_forecast"],
        result={
            "status": "completed",
            "answer_type": "learning_summary",
            "prediction": [3.0],
            "trace_id": "trace-789",
            "gate_accepted": True,
            "confidence": 0.9,
            "execution_log": [
                {
                    "tool": "spawn_subagent",
                    "output": [
                        {
                            "success": True,
                            "result": {
                                "execution_log": [
                                    {"tool": "hurst_exponent", "observation": {"h": 0.73}},
                                    {"tool": "timesfm2_forecast", "observation": {"forecast": [3.0]}},
                                ],
                                "answer": [3.0],
                            },
                        }
                    ],
                },
                {
                    "tool": "evaluate_batch_against_gt",
                    "observation": {
                        "results": [
                            {
                                "name": "timesfm2_forecast",
                                "metrics": {"mae": 0.100},
                                "prediction": [3.0],
                            }
                        ]
                    },
                },
            ],
            "tool_chain": ["spawn_subagent", "evaluate_batch_against_gt"],
            "total_ms": 456,
            "raw_answer": {"insight": "i", "recommendation": "r"},
            "reasoning": "why",
            "evidence_summary": "summary",
        },
    )

    assert row["tool_chain"] == ["hurst_exponent", "timesfm2_forecast"]
    assert row["tools_run"] == 2


def test_build_train_result_evidence_preview_uses_winner_candidate_steps():
    result = {
        "raw_answer": {
            "winner_tool_chain": ["hurst_exponent", "timesfm2_forecast"],
        },
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {
                                    "tool": "hurst_exponent",
                                    "thought": "Estimate persistence before selecting a forecaster.",
                                    "observation": {"h": 0.73},
                                },
                                {
                                    "tool": "timesfm2_forecast",
                                    "thought": "Use the persistent regime to generate the forecast path.",
                                    "observation": {"point_forecast": [3.0]},
                                },
                            ],
                            "answer": [3.0],
                        },
                    }
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {
                            "name": "timesfm2_forecast",
                            "metrics": {"mae": 0.100},
                            "prediction": [3.0],
                        }
                    ]
                },
            },
        ],
        "evidence_summary": "Evidence Log (1 steps):\n  Successful steps: 1/1",
    }

    preview = _build_train_result_evidence_preview(
        result,
        ["hurst_exponent", "timesfm2_forecast"],
    )

    assert "Evidence Log (2 steps):" in preview
    assert "Step 0: hurst_exponent" in preview
    assert "Step 1: timesfm2_forecast" in preview
    assert "Successful steps: 2/2" in preview


def test_build_train_result_evidence_preview_prefers_exact_tool_chain_over_wrong_winner():
    result = {
        "raw_answer": {
            "winner_candidate": "top_level::llm_reasoning_classification",
            "winner_tool_chain": ["llm_reasoning_classification"],
        },
        "execution_log": [
            {
                "tool": "llm_reasoning_classification",
                "thought": "Top-level direct classifier.",
                "observation": {"answer": "-2% ~ -4%"},
            },
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {
                                    "tool": "value_at",
                                    "thought": "Read the first price.",
                                    "observation": {"value": 38.98},
                                },
                                {
                                    "tool": "value_at",
                                    "thought": "Read the last price.",
                                    "observation": {"value": 34.39},
                                },
                                {
                                    "tool": "window_stats",
                                    "thought": "Measure the recent regime.",
                                    "observation": {"mean": 34.8},
                                },
                                {
                                    "tool": "llm_reasoning_classification",
                                    "thought": "Classify with the measured move.",
                                    "observation": {"answer": "-2% ~ +2%"},
                                },
                            ],
                            "answer": "-2% ~ +2%",
                        },
                    }
                ],
            },
        ],
        "evidence_summary": "Evidence Log (1 steps):\n  Successful steps: 1/1",
    }

    preview = _build_train_result_evidence_preview(
        result,
        ["value_at", "value_at", "window_stats", "llm_reasoning_classification"],
    )

    assert "Evidence Log (4 steps):" in preview
    assert "Step 0: value_at" in preview
    assert "Step 2: window_stats" in preview
    assert "Step 3: llm_reasoning_classification" in preview
    assert "Successful steps: 4/4" in preview


def test_build_train_result_evidence_preview_disambiguates_same_chain_by_winner_prediction():
    result = {
        "raw_answer": {
            "winner_tool_chain": ["llm_reasoning_classification"],
            "winner_prediction": "increasing",
        },
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {
                                    "tool": "llm_reasoning_classification",
                                    "thought": "Classify the daily-mean slope directly.",
                                    "observation": {"answer": "stable"},
                                },
                            ],
                            "answer": "stable",
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {
                                    "tool": "llm_reasoning_classification",
                                    "thought": "Classify the daily-mean slope directly.",
                                    "observation": {"answer": "increasing"},
                                },
                            ],
                            "answer": "increasing",
                        },
                    },
                ],
            },
        ],
        "evidence_summary": "Evidence Log (1 steps):\n  Successful steps: 1/1",
    }

    preview = _build_train_result_evidence_preview(
        result,
        ["llm_reasoning_classification"],
    )

    assert 'Observed: {"answer": "increasing"}' in preview
    assert 'Observed: {"answer": "stable"}' not in preview


def test_build_train_result_evidence_preview_uses_observation_over_mismatched_action_text():
    result = {
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {
                                    "tool": "llm_reasoning_correlation",
                                    "thought": (
                                        "Reason: A task-valid label already exists. "
                                        "Action: finish with that exact label."
                                    ),
                                    "observation": {"answer": "No Correlation"},
                                },
                            ],
                            "answer": "No Correlation",
                        },
                    }
                ],
            },
        ],
        "evidence_summary": "Evidence Log (1 steps):\n  Successful steps: 1/1",
    }

    preview = _build_train_result_evidence_preview(
        result,
        ["llm_reasoning_correlation"],
    )

    assert "Step 0: llm_reasoning_correlation" in preview
    assert "Observed:" in preview
    assert "No Correlation" in preview
    assert "finish with that exact label" not in preview


def test_build_train_result_evidence_preview_does_not_truncate_long_step_details():
    long_answer = "x" * 1200
    result = {
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {
                                    "tool": "llm_reasoning_qa",
                                    "observation": {"answer": long_answer},
                                },
                            ],
                            "answer": long_answer,
                        },
                    }
                ],
            },
        ],
        "evidence_summary": "Evidence Log (1 steps):\n  Successful steps: 1/1",
    }

    preview = _build_train_result_evidence_preview(
        result,
        ["llm_reasoning_qa"],
    )

    assert long_answer in preview
    assert "... [" not in preview
    assert "Successful steps: 1/1" in preview


def test_build_train_result_evidence_preview_strips_control_flow_reason_from_observed_step():
    result = {
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "execution_log": [
                                {
                                    "tool": "llm_reasoning_forecast",
                                    "thought": (
                                        "Reason: A TimesFM forecast is already available, but the branch goal also asks "
                                        "to refine or interpret it with multimodal reasoning and the immediate-response "
                                        "rule requires a tool call now. Action: answer."
                                    ),
                                    "observation": {"answer": [1, 2, 3]},
                                },
                            ],
                            "answer": [1, 2, 3],
                        },
                    }
                ],
            },
        ],
        "evidence_summary": "Evidence Log (1 steps):\n  Successful steps: 1/1",
    }

    preview = _build_train_result_evidence_preview(
        result,
        ["llm_reasoning_forecast"],
    )

    assert "Observed:" in preview
    assert "branch goal" not in preview.lower()
    assert "requires a tool call now" not in preview.lower()


def test_build_train_result_evidence_preview_can_backfill_from_trace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    trace_dir = Path(".timeclaw/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace-backfill.jsonl"
    rows = [
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 1,
                    "step": 1,
                    "tool": "detect_trend",
                    "observation": {"preview": "{'direction': 'flat', 'slope': 0.005}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 1,
                    "step": 2,
                    "tool": "llm_reasoning_classification",
                    "observation": {"preview": "{'answer': 'stable'}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_complete",
                "summary": {
                    "sub_agent_idx": 1,
                    "tool_chain": {"preview": ["detect_trend", "llm_reasoning_classification"]},
                },
            },
        },
    ]
    trace_path.write_text("\n".join(json.dumps(item) for item in rows))

    preview = _build_train_result_evidence_preview(
        {
            "trace_id": "trace-backfill",
            "evidence_summary": "stale preview",
        },
        ["detect_trend", "llm_reasoning_classification"],
    )

    assert "Evidence Log (2 steps):" in preview
    assert "Step 0: detect_trend" in preview
    assert "Step 1: llm_reasoning_classification" in preview
    assert "Successful steps: 2/2" in preview


def test_build_train_result_evidence_preview_trace_backfill_handles_reused_subagent_idx(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    trace_dir = Path(".timeclaw/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace-reuse.jsonl"
    rows = [
        {
            "event": "agent_step",
            "payload": {"phase": "subagent_start", "summary": {"sub_agent_idx": 0}},
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 1,
                    "tool": "chronos2_forecast",
                    "observation": {"preview": "{'point_forecast': [1.0]}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_complete",
                "summary": {
                    "sub_agent_idx": 0,
                    "tool_chain": {"preview": ["chronos2_forecast", "llm_reasoning_indicator"]},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {"phase": "subagent_start", "summary": {"sub_agent_idx": 0}},
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 1,
                    "tool": "arima_forecast",
                    "observation": {"preview": "forecast=[42.67, 42.67]"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 2,
                    "tool": "llm_reasoning_indicator",
                    "observation": {"preview": "{'answer': {'upper_bb': [42.68, 42.67]}}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_complete",
                "summary": {
                    "sub_agent_idx": 0,
                    "tool_chain": {"preview": ["arima_forecast", "llm_reasoning_indicator"]},
                },
            },
        },
    ]
    trace_path.write_text("\n".join(json.dumps(item) for item in rows))

    preview = _build_train_result_evidence_preview(
        {
            "trace_id": "trace-reuse",
            "evidence_summary": "stale preview",
        },
        ["arima_forecast", "llm_reasoning_indicator"],
    )

    assert "Evidence Log (2 steps):" in preview
    assert "Step 0: arima_forecast" in preview
    assert "Step 1: llm_reasoning_indicator" in preview
    assert "chronos2_forecast" not in preview


def test_build_train_result_evidence_preview_trace_backfill_prefers_observed_chain_when_preview_is_truncated(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    trace_dir = Path(".timeclaw/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace-truncated-preview.jsonl"
    rows = [
        {
            "event": "agent_step",
            "payload": {"phase": "subagent_start", "summary": {"sub_agent_idx": 0}},
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 1,
                    "tool": "detect_changepoint",
                    "observation": {"preview": "{'changepoints': [370]}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 2,
                    "tool": "slice",
                    "observation": {"preview": "[12.69, 12.52, 12.255]"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 3,
                    "tool": "arima_forecast",
                    "observation": {"preview": "forecast=[12.05, 12.05]"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 4,
                    "tool": "window_stats",
                    "observation": {"preview": "{'mean': 12.05}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 5,
                    "tool": "llm_reasoning_indicator",
                    "observation": {"preview": "{'answer': {'upper_bb': [12.05, 12.05]}}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_complete",
                "summary": {
                    "sub_agent_idx": 0,
                    "tool_chain": {"preview": ["detect_changepoint", "slice", "arima_forecast", "window_stats"]},
                },
            },
        },
    ]
    trace_path.write_text("\n".join(json.dumps(item) for item in rows))

    preview = _build_train_result_evidence_preview(
        {
            "trace_id": "trace-truncated-preview",
            "evidence_summary": "stale preview",
        },
        ["detect_changepoint", "slice", "arima_forecast", "window_stats", "llm_reasoning_indicator"],
    )

    assert "Evidence Log (5 steps):" in preview
    assert "Step 4: llm_reasoning_indicator" in preview
    assert "Successful steps: 5/5" in preview


def test_build_train_result_evidence_preview_backfills_single_step_placeholder_from_trace(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    trace_dir = Path(".timeclaw/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace-single-step-backfill.jsonl"
    rows = [
        {
            "event": "agent_step",
            "payload": {"phase": "subagent_start", "summary": {"sub_agent_idx": 0}},
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_tool_result",
                "summary": {
                    "sub_agent_idx": 0,
                    "step": 1,
                    "tool": "llm_reasoning_indicator",
                    "observation": {"preview": "{'answer': {'macd': [0.1, 0.2]}}"},
                },
            },
        },
        {
            "event": "agent_step",
            "payload": {
                "phase": "subagent_complete",
                "summary": {
                    "sub_agent_idx": 0,
                    "tool_chain": {"preview": ["llm_reasoning_indicator"]},
                },
            },
        },
    ]
    trace_path.write_text("\n".join(json.dumps(item) for item in rows))

    preview = _build_train_result_evidence_preview(
        {
            "trace_id": "trace-single-step-backfill",
            "execution_log": [
                {
                    "tool": "spawn_subagent",
                    "output": [
                        {
                            "success": True,
                            "result": {
                                "execution_log": [{"tool": "llm_reasoning_indicator"}],
                                "answer": {"macd": [0.1, 0.2]},
                            },
                        }
                    ],
                }
            ],
            "evidence_summary": "Evidence Log (1 steps):\n  Successful steps: 1/1",
        },
        ["llm_reasoning_indicator"],
    )

    assert "Evidence Log (1 steps):" in preview
    assert "Step 0: llm_reasoning_indicator" in preview
    assert "Successful steps: 1/1" in preview
    assert "no detail recorded" not in preview


def test_build_train_result_row_sanitizes_raw_answer_preview():
    sample = MTBenchSample(
        file="00003.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    row = _build_train_result_row(
        ds_name="finance_aligned_short",
        sample_idx=6,
        sample=sample,
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["finance_aligned_short_forecast"],
        result={
            "status": "completed",
            "answer_type": "learning_summary",
            "prediction": [3.0],
            "trace_id": "trace-999",
            "gate_accepted": True,
            "confidence": 0.9,
            "tool_chain": ["timesfm2_forecast"],
            "total_ms": 456,
            "raw_answer": {
                "insight": "GT-selected winner after evaluate_batch_against_gt before finishing.",
                "recommendation": "Use the winning path.",
            },
            "reasoning": "why",
            "evidence_summary": "summary",
        },
    )

    assert row["raw_answer_preview"] == ""


def test_build_train_result_row_keeps_full_debug_payloads():
    sample = MTBenchSample(
        file="00004.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    result = {
        "status": "completed",
        "answer_type": "learning_summary",
        "prediction": [3.0],
        "trace_id": "trace-1000",
        "gate_accepted": True,
        "confidence": 0.9,
        "total_ms": 456,
        "raw_answer": {"insight": "i", "recommendation": "r"},
        "execution_context": {
            "tool_chain": ["timesfm2_forecast"],
            "public_reasoning": ["timesfm2_forecast produced a flat path"],
            "evidence_summary": "flat path around 3.0",
        },
        "control_context": {"completion_notes": "repair-only"},
        "execution_log": [{"tool": "timesfm2_forecast", "success": True}],
    }

    row = _build_train_result_row(
        ds_name="finance_aligned_short",
        sample_idx=7,
        sample=sample,
        task="forecast",
        pipeline_task="forecast",
        scope_keys=["finance_aligned_short_forecast"],
        result=result,
    )

    assert row["raw_answer"] == result["raw_answer"]
    assert row["execution_context"] == result["execution_context"]
    assert row["control_context"] == result["control_context"]
    assert row["execution_log"] == result["execution_log"]


@pytest.mark.asyncio
async def test_run_train_loop_no_merge_checkpoint_still_loads_split(monkeypatch):
    called = {"load_split": False}

    class DummyPipeline:
        def __init__(self, config, disable_memory=False):
            self.config = config
            self.disable_memory = disable_memory

        async def shutdown(self):
            return None

    class DummyConfig:
        def __init__(self):
            self.mcp = type("MCP", (), {"enabled": True})()

    def fake_load_split(ds_name, split, config=None):
        called["load_split"] = True
        return []

    monkeypatch.setattr(tl, "PipelineRunner", DummyPipeline)
    monkeypatch.setattr(tl, "_checkpoint_exists", lambda ds_name: True)
    monkeypatch.setattr(tl, "load_split", fake_load_split)
    monkeypatch.setattr(
        tl,
        "select_samples_with_original_indices",
        lambda samples, ds_name, idx_map: list(enumerate(samples)),
    )

    result = await tl.run_train_loop(
        config=DummyConfig(),
        dataset_name="finance_aligned_long",
        task_filter="indicator_upper_bb",
        max_samples=1,
        merge_checkpoint=False,
        concurrency=1,
    )

    assert called["load_split"] is True
    assert result == {}


@pytest.mark.asyncio
async def test_run_train_loop_manifest_uses_merged_task_rows(tmp_path, monkeypatch):
    real_pipeline_runner = tl.PipelineRunner

    monkeypatch.setattr(tl, "TRAIN_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(tl, "TRAIN_SUMMARY_PATH", tmp_path / "train_summary.json")
    monkeypatch.setattr(tl, "TRAIN_RUN_MANIFEST_PATH", tmp_path / "train_run_manifest.json")
    monkeypatch.setattr(
        tl,
        "expanded_benchmark_task_specs",
        lambda: [object()] * tl.BENCHMARK_EXPANDED_TASK_TARGET,
    )
    monkeypatch.setattr(tl, "datasets_with_benchmark_tasks", lambda: ["finance_aligned_short"])
    monkeypatch.setattr(tl, "_export_train_note_views", lambda **kwargs: {})

    sample = MTBenchSample(
        file="00000.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([3.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
    )

    existing_row = {
        "idx": 0,
        "abs_idx": 0,
        "dataset": "finance_aligned_short",
        "dataset_name": "finance_aligned_short",
        "sample_ref": "finance_aligned_short:old.parquet_row:0:forecast",
        "task": "forecast",
        "status": "completed",
        "answer_type": "learning_summary",
        "has_prediction": True,
        "gate_accepted": True,
        "winner_state": "selected",
        "winner_metrics": {"mae": 1.0},
    }

    def fake_load_train_checkpoint(ds_name, task=None):
        assert ds_name == "finance_aligned_short"
        if task in (None, "forecast"):
            return [dict(existing_row)]
        return []

    class DummyPipeline(real_pipeline_runner):
        def __init__(self, config, disable_memory=False):
            self.config = config
            self.disable_memory = disable_memory

        async def run(self, **kwargs):
            return {
                "status": "completed",
                "answer_type": "learning_summary",
                "prediction": [3.0],
                "trace_id": "trace-new",
                "gate_accepted": True,
                "confidence": 0.9,
                "total_ms": 123,
                "raw_answer": {
                    "winner_state": "selected",
                    "winner_candidate": "timesfm2_forecast",
                    "winner_prediction": [3.0],
                    "winner_tool_chain": ["timesfm2_forecast"],
                    "winner_metrics": {"mae": 0.0},
                    "insight": "timesfm won",
                    "recommendation": "prefer timesfm",
                },
                "execution_context": {
                    "tool_chain": ["timesfm2_forecast"],
                    "public_reasoning": ["timesfm2_forecast stayed on target."],
                    "evidence_summary": "flat path around 3.0",
                },
                "control_context": {},
                "execution_log": [
                    {
                        "tool": "timesfm2_forecast",
                        "success": True,
                        "output": {"forecast": [3.0]},
                    },
                ],
            }

        async def shutdown(self):
            return None

    class DummyConfig:
        def __init__(self):
            self.mcp = type("MCP", (), {"enabled": True})()

    monkeypatch.setattr(tl, "PipelineRunner", DummyPipeline)
    monkeypatch.setattr(tl, "_load_train_checkpoint", fake_load_train_checkpoint)
    monkeypatch.setattr(tl, "load_split", lambda ds_name, split, config=None: [sample])
    monkeypatch.setattr(
        tl,
        "select_samples_with_original_indices",
        lambda samples, ds_name, idx_map: [(1, sample)],
    )

    await tl.run_train_loop(
        config=DummyConfig(),
        dataset_name="finance_aligned_short",
        task_filter="forecast",
        max_samples=1,
        merge_checkpoint=True,
        concurrency=1,
    )

    manifest = json.loads((tmp_path / "train_run_manifest.json").read_text())
    task_info = manifest["tasks"]["finance_aligned_short::forecast"]
    assert task_info["rows"] == 2
    assert task_info["status_counts"] == {"completed": 2}


@pytest.mark.asyncio
async def test_process_one_task_sample_preserves_trace_fields_on_failure():
    class FailingPipeline:
        async def run(self, **kwargs):
            exc = MaxStepsExhaustedError(
                agent_kind="executor",
                max_steps=10,
                partial_result={
                    "answer": {
                        "insight": "timesfm2_forecast stayed close to the recent band.",
                        "recommendation": "When the window is smooth, keep the forecast path and verify the band.",
                    },
                    "answer_type": "learning_summary",
                    "reasoning": "1. timesfm2_forecast stayed near the recent band.",
                    "execution_context": {
                        "tool_chain": ["timesfm2_forecast", "llm_reasoning_indicator"],
                        "public_reasoning": ["timesfm2_forecast stayed close to the recent band."],
                        "evidence_summary": "forecast stayed near the recent band",
                    },
                    "control_context": {"completion_notes": "terminal packaging rejected"},
                    "execution_log": [
                        {"tool": "timesfm2_forecast", "success": True},
                        {"tool": "llm_reasoning_indicator", "success": True},
                    ],
                },
            )
            setattr(exc, "trace_id", "trace-fail-123")
            setattr(exc, "total_ms", 4321)
            setattr(exc, "error_type", "MaxStepsExhaustedError")
            setattr(exc, "failure_reason", "max_steps_exhausted")
            raise exc

    sample = MTBenchSample(
        file="00002.parquet_row",
        dataset="finance_aligned_long",
        domain="finance",
        task_types=["indicator"],
        input_window=np.asarray([1.0, 2.0, 3.0]),
        output_window=None,
        text="demo",
        timestamps=[0.0, 1.0, 2.0],
        start_year=2024,
        labels={"out_macd": [0.1, 0.2, 0.3]},
    )

    row = await _process_one_task_sample(
        ds_name="finance_aligned_long",
        sample_idx=2,
        sample=sample,
        task="indicator_macd",
        pipeline=FailingPipeline(),
        sem=asyncio.Semaphore(1),
    )

    assert row is not None
    assert row["status"] == "max_steps_exhausted"
    assert row["trace_id"] == "trace-fail-123"
    assert row["total_ms"] == 4321
    assert row["error_type"] == "MaxStepsExhaustedError"
    assert row["failure_reason"] == "max_steps_exhausted"
    assert row["answer_type"] == "learning_summary"
    assert row["raw_answer"]["insight"].startswith("timesfm2_forecast")
    assert [entry["tool"] for entry in row["execution_log"]] == [
        "timesfm2_forecast",
        "llm_reasoning_indicator",
    ]
