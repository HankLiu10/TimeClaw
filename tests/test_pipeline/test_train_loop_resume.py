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
    _extract_train_result_tool_chain,
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
    assert row["sample_ref"] == "finance_aligned_short:00000.parquet_row:3:forecast"
    assert row["trace_id"] == "trace-123"
    assert row["gate"] == "FAIL"
    assert row["gate_accepted"] is False
    assert row["scope_keys"] == ["finance_aligned_short_forecast"]
    assert row["learning_recommendation"] == "r"


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
            exc = MaxStepsExhaustedError(agent_kind="executor", max_steps=10)
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
