from __future__ import annotations

from timeclaw.benchmark.mtbench_support.official_metric_policy import (
    apply_official_metric_policy_to_row,
    summarize_official_inference_rows,
)


def test_finance_indicator_macd_row_above_threshold_is_filtered_from_summary():
    row = {
        "dataset": "finance_aligned_long",
        "task": "indicator_macd",
        "status": "completed",
        "metrics": {"macd_mse": 12.5, "macd_mae": 1.2},
    }

    annotated = apply_official_metric_policy_to_row(row)

    assert annotated["row_recorded"] is True
    assert annotated["summary_included"] is False
    assert annotated["summary_exclude_reason"] == "macd_mse>10"


def test_non_forecast_task_is_not_metric_filtered():
    row = {
        "dataset": "finance_qa_long",
        "task": "correlation",
        "status": "completed",
        "metrics": {"accuracy": 1.0},
    }

    annotated = apply_official_metric_policy_to_row(row)

    assert annotated["row_recorded"] is True
    assert annotated["summary_included"] is True
    assert annotated["summary_exclude_reason"] is None


def test_error_row_is_skipped_before_metric_filtering():
    row = {
        "dataset": "weather_aligned_short",
        "task": "forecast",
        "status": "error",
        "metrics": {"error": "parse_fail"},
    }

    annotated = apply_official_metric_policy_to_row(row)

    assert annotated["row_recorded"] is False
    assert annotated["summary_included"] is False
    assert annotated["summary_exclude_reason"] == "parse_fail"


def test_numeric_parse_error_metric_is_treated_as_parse_fail():
    row = {
        "dataset": "finance_qa_short",
        "task": "mcqa",
        "status": "ok",
        "metrics": {"parse_error": 1.0},
    }

    annotated = apply_official_metric_policy_to_row(row)

    assert annotated["row_recorded"] is False
    assert annotated["summary_included"] is False
    assert annotated["summary_exclude_reason"] == "parse_fail"


def test_summary_averages_only_officially_included_rows():
    rows = [
        {
            "dataset": "finance_aligned_long",
            "task": "indicator_macd",
            "status": "completed",
            "total_ms": 1000,
            "metrics": {"macd_mse": 2.0, "macd_mae": 1.0, "avg_mae": 1.0},
        },
        {
            "dataset": "finance_aligned_long",
            "task": "indicator_macd",
            "status": "completed",
            "total_ms": 1200,
            "metrics": {"macd_mse": 12.0, "macd_mae": 9.0, "avg_mae": 9.0},
        },
        {
            "dataset": "finance_aligned_long",
            "task": "indicator_macd",
            "status": "error",
            "total_ms": 1300,
            "metrics": {"error": "parse_fail"},
        },
    ]

    summary = summarize_official_inference_rows(rows)

    assert summary == [
        {
            "dataset": "finance_aligned_long",
            "task": "indicator_macd",
            "completed": 2,
            "total": 3,
            "errors": 1,
            "row_recorded_n": 2,
            "summary_included_n": 1,
            "threshold_filtered_n": 1,
            "invalid_skipped_n": 1,
            "avg_total_ms": 1166.667,
            "avg_macd_mse": 2.0,
            "avg_mse": 2.0,
            "avg_mae": 1.0,
        }
    ]


def test_weather_forecast_summary_drops_non_official_mape():
    rows = [
        {
            "dataset": "weather_aligned_short",
            "task": "forecast",
            "status": "completed",
            "metrics": {"mse": 4.0, "mae": 2.0, "rmse": 2.0, "mape": 99.0},
        }
    ]

    summary = summarize_official_inference_rows(rows)

    assert summary == [
        {
            "dataset": "weather_aligned_short",
            "task": "forecast",
            "completed": 1,
            "total": 1,
            "errors": 0,
            "row_recorded_n": 1,
            "summary_included_n": 1,
            "threshold_filtered_n": 0,
            "invalid_skipped_n": 0,
            "avg_mse": 4.0,
            "avg_mae": 2.0,
            "avg_rmse": 2.0,
        }
    ]


def test_finance_correlation_summary_projects_brief_accuracy_to_3way():
    rows = [
        {
            "dataset": "finance_qa_long",
            "task": "correlation",
            "status": "completed",
            "metrics": {
                "accuracy": 0.0,
                "exact_accuracy": 0.0,
                "brief_accuracy": 1.0,
            },
        }
    ]

    summary = summarize_official_inference_rows(rows)

    assert summary == [
        {
            "dataset": "finance_qa_long",
            "task": "correlation",
            "completed": 1,
            "total": 1,
            "errors": 0,
            "row_recorded_n": 1,
            "summary_included_n": 1,
            "threshold_filtered_n": 0,
            "invalid_skipped_n": 0,
            "avg_accuracy": 0.0,
            "avg_exact_accuracy": 0.0,
            "avg_accuracy_3way": 1.0,
        }
    ]


def test_finance_forecast_summary_keeps_official_mse_mae_rmse_mape():
    rows = [
        {
            "dataset": "finance_aligned_short",
            "task": "forecast",
            "status": "completed",
            "metrics": {
                "mse": 4.0,
                "mae": 2.0,
                "rmse": 2.0,
                "mape": 8.0,
            },
        }
    ]

    summary = summarize_official_inference_rows(rows)

    assert summary == [
        {
            "dataset": "finance_aligned_short",
            "task": "forecast",
            "completed": 1,
            "total": 1,
            "errors": 0,
            "row_recorded_n": 1,
            "summary_included_n": 1,
            "threshold_filtered_n": 0,
            "invalid_skipped_n": 0,
            "avg_mse": 4.0,
            "avg_mae": 2.0,
            "avg_rmse": 2.0,
            "avg_mape": 8.0,
        }
    ]


def test_finance_indicator_macd_summary_projects_generic_and_alias_metrics():
    rows = [
        {
            "dataset": "finance_aligned_short",
            "task": "indicator_macd",
            "status": "completed",
            "metrics": {
                "macd_mse": 1.5,
                "macd_mae": 1.0,
                "macd_rmse": 1.225,
                "macd_mape": 10.0,
            },
        }
    ]

    summary = summarize_official_inference_rows(rows)

    assert summary == [
        {
            "dataset": "finance_aligned_short",
            "task": "indicator_macd",
            "completed": 1,
            "total": 1,
            "errors": 0,
            "row_recorded_n": 1,
            "summary_included_n": 1,
            "threshold_filtered_n": 0,
            "invalid_skipped_n": 0,
            "avg_macd_mse": 1.5,
            "avg_mse": 1.5,
            "avg_mae": 1.0,
            "avg_rmse": 1.225,
            "avg_mape": 10.0,
        }
    ]


def test_legacy_official_fields_are_read_compatibly():
    row = {
        "dataset": "finance_aligned_long",
        "task": "indicator_macd",
        "official_recorded": True,
        "official_summary_included": False,
        "official_summary_skip_reason": "macd_mse>10",
        "metrics": {"macd_mse": 12.5},
    }

    annotated = apply_official_metric_policy_to_row(row)

    assert annotated["row_recorded"] is True
    assert annotated["summary_included"] is False
    assert annotated["summary_exclude_reason"] == "macd_mse>10"


def test_unfiltered_companion_retains_extreme_errors_and_failure_totals():
    rows=[{'dataset':'finance_aligned_short','task':'forecast','status':'completed','metrics':{'mse':2.,'mae':1.}},
          {'dataset':'finance_aligned_short','task':'forecast','status':'completed','metrics':{'mse':200.,'mae':10.}},
          {'dataset':'finance_aligned_short','task':'forecast','status':'failed','metrics':{}}]
    annotated=[apply_official_metric_policy_to_row(row) for row in rows]
    official=summarize_official_inference_rows(annotated)[0]
    all_valid=summarize_official_inference_rows(annotated,apply_thresholds=False)[0]
    assert official['total']==all_valid['total']==3
    assert official['summary_included_n']==1 and official['avg_mse']==2.
    assert all_valid['summary_included_n']==2 and all_valid['avg_mse']==101.
    assert official['errors']==all_valid['errors']==1
