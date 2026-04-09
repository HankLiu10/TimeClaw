import asyncio

import numpy as np
import pytest

from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark import train_loop as tl
from timeclaw.benchmark.train_loop import (
    _process_one_task_sample,
    _build_train_result_row,
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
    monkeypatch.setattr(tl, "select_samples_by_index", lambda samples, ds_name, idx_map: samples)

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
