import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark import train_loop as tl
from timeclaw.benchmark.mtbench_support.sample_selection import (
    select_samples_with_original_indices,
)
from timeclaw.benchmark.train_loop import (
    _process_one_task_sample,
    _build_train_result_row,
    _build_train_result_evidence_preview,
    _extract_train_result_reasoning_preview,
    _extract_train_result_tool_chain,
    _load_checkpoint_rows,
    _normalize_train_result_row,
    _sanitize_learning_artifact_preview,
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
    assert row["learning_recommendation"] == "r"
    assert row["winner_tool_chain"] == ["timesfm2_forecast"]


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
                                        "Action: Call emit_final_answer to return it."
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
    assert "emit_final_answer" not in preview


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
