from __future__ import annotations

import numpy as np

from timeclaw.agents.executor import ExecutorAgent
from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark.mtbench_support.task_scoring import (
    align_numeric_prediction_to_ground_truth,
    backfill_scored_metrics,
    score_mtbench_prediction,
    validate_mtbench_sample_for_scoring,
)


def test_align_numeric_prediction_to_ground_truth_matches_mtbench_interp():
    aligned = align_numeric_prediction_to_ground_truth([0.0, 10.0], [0.0, 5.0, 10.0])
    assert aligned is not None
    assert aligned.tolist() == [0.0, 5.0, 10.0]


def test_score_finance_indicator_macd_uses_interp_alignment():
    metrics = score_mtbench_prediction(
        {"macd": [0.0, 10.0]},
        {"macd": [0.0, 5.0, 10.0]},
        "finance_aligned_long_indicator_macd",
    )

    assert metrics["macd_mae"] == 0.0
    assert metrics["macd_mse"] == 0.0


def test_validate_finance_trend_sample_skips_non_string_gt():
    sample = MTBenchSample(
        file="00001.parquet_row",
        dataset="finance_aligned_long",
        domain="finance",
        task_types=["trend"],
        input_window=np.asarray([1.0, 2.0, 3.0]),
        output_window=np.asarray([4.0, 5.0, 6.0]),
        text="demo",
        timestamps=[0.0, 1.0, 2.0],
        start_year=2024,
        labels={"trend_label": None},
    )

    valid, reason = validate_mtbench_sample_for_scoring(sample, "trend")

    assert valid is False
    assert reason == "invalid_ground_truth"


def test_score_weather_trend_parse_failure_is_excluded_style_error():
    metrics = score_mtbench_prediction(
        {"trend": "both increasing and decreasing"},
        "stable",
        "weather_aligned_short_trend",
    )

    assert metrics == {"parse_error": 1.0}


def test_backfill_scored_metrics_restores_finance_trend_3way_field():
    repaired = backfill_scored_metrics(
        "-2% ~ +2%",
        "-2% ~ +2%",
        "trend",
        {
            "accuracy": 1.0,
            "correct": True,
            "predicted_label": "-2% ~ +2%",
            "ground_truth_label": "-2% ~ +2%",
        },
    )

    assert repaired["accuracy_3way"] == 1.0


def test_backfill_scored_metrics_restores_correlation_brief_accuracy():
    repaired = backfill_scored_metrics(
        "No Correlation",
        "Moderate Negative Correlation",
        "correlation",
        {
            "accuracy": 0.0,
            "correct": False,
            "predicted_label": "No Correlation",
            "ground_truth_label": "Moderate Negative Correlation",
        },
    )

    assert repaired["exact_accuracy"] == 0.0
    assert repaired["brief_accuracy"] == 0.0


def test_executor_validation_accepts_weather_trend_alias_for_task_scope():
    valid, reason = ExecutorAgent._validate_final_answer(
        "classification",
        "up",
        domain="weather",
        scope_keys=["weather_aligned_long_trend_past"],
    )

    assert valid is True
    assert reason == ""


def test_executor_validation_rejects_finance_trend_three_way_alias_for_task_scope():
    valid, reason = ExecutorAgent._validate_final_answer(
        "classification",
        "down",
        domain="finance",
        scope_keys=["finance_aligned_long_trend"],
    )

    assert valid is False
    assert reason == "classification answer is outside the task label space"
