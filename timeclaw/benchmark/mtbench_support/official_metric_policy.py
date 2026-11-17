from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

# Official MTBench threshold filters for cumulative forecast-style metrics.
#
# New definition:
#   These filters mirror the retired official-style evaluation path:
#   rows may still be recorded for auditing, but rows above the task-specific
#   threshold do not contribute to the averaged summary metrics.
_OFFICIAL_METRIC_FILTERS: dict[tuple[str, str], tuple[str, float]] = {
    ("weather", "forecast"): ("mse", 100.0),
    ("finance", "forecast"): ("mse", 100.0),
    ("finance", "indicator_macd"): ("macd_mse", 10.0),
    ("finance", "indicator_upper_bb"): ("upper_bb_mse", 100.0),
}

_ROW_RECORDED_KEY = "row_recorded"
_SUMMARY_INCLUDED_KEY = "summary_included"
_SUMMARY_EXCLUDE_REASON_KEY = "summary_exclude_reason"
_LEGACY_ROW_RECORDED_KEY = "official_recorded"
_LEGACY_SUMMARY_INCLUDED_KEY = "official_summary_included"
_LEGACY_SUMMARY_EXCLUDE_REASON_KEY = "official_summary_skip_reason"


def _domain_from_row(row: dict[str, Any]) -> str:
    domain = str(row.get("domain") or "").strip().lower()
    if domain:
        return domain
    dataset = str(row.get("dataset") or "").strip().lower()
    if dataset.startswith("weather_"):
        return "weather"
    if dataset.startswith("finance_"):
        return "finance"
    return ""


def _task_from_row(row: dict[str, Any]) -> str:
    return str(row.get("task") or "").strip().lower()


def _has_numeric_error_metric(metrics: dict[str, Any]) -> bool:
    for key in ("parse_error", "type_error"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)) and float(value) > 0:
            return True
    return False


def _is_completed_status(status: Any) -> bool:
    return str(status or "").strip().lower() == "completed"


def _official_row_status(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    if _row_summary_included(row) is False:
        if _row_recorded(row):
            return "filtered"
        if metrics.get("skipped") == "invalid_ground_truth" or _row_summary_exclude_reason(row) == "invalid_ground_truth":
            return "skipped"
        return "error"
    if isinstance(metrics, dict) and (metrics.get("error") or _has_numeric_error_metric(metrics)):
        return "error"
    return "completed"


def _numeric_metric_projection(row: dict[str, Any]) -> dict[str, float]:
    """Return the official benchmark-facing metrics that should be summarized.

    New definition:
        The persisted rows may contain auxiliary metrics such as RMSE or
        duplicated aliases. MTBench-style summaries should only aggregate the
        task/domain metrics that matter for official reporting.
    """
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    domain = _domain_from_row(row)
    task = _task_from_row(row)

    def pick(*pairs: tuple[str, str]) -> dict[str, float]:
        picked: dict[str, float] = {}
        for source_key, target_key in pairs:
            value = metrics.get(source_key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                picked[target_key] = float(value)
        return picked

    if domain == "weather" and task == "forecast":
        return pick(("mse", "avg_mse"), ("mae", "avg_mae"), ("rmse", "avg_rmse"))
    if domain == "finance" and task == "forecast":
        return pick(
            ("mse", "avg_mse"),
            ("mae", "avg_mae"),
            ("rmse", "avg_rmse"),
            ("mape", "avg_mape"),
        )
    if domain == "weather" and task in {"trend", "trend_past"}:
        return pick(("accuracy", "avg_accuracy"))
    if domain == "finance" and task == "trend":
        return pick(("accuracy", "avg_accuracy"), ("accuracy_3way", "avg_accuracy_3way"))
    if task == "mcqa":
        return pick(("accuracy", "avg_accuracy"))
    if domain == "finance" and task == "correlation":
        projected = pick(("accuracy", "avg_accuracy"), ("exact_accuracy", "avg_exact_accuracy"))
        brief = metrics.get("brief_accuracy")
        if isinstance(brief, (int, float)) and np.isfinite(float(brief)):
            projected["avg_accuracy_3way"] = float(brief)
        return projected
    if domain == "finance" and task == "indicator_macd":
        return pick(
            ("macd_mse", "avg_macd_mse"),
            ("macd_mse", "avg_mse"),
            ("macd_mae", "avg_mae"),
            ("macd_rmse", "avg_rmse"),
            ("macd_mape", "avg_mape"),
        )
    if domain == "finance" and task == "indicator_upper_bb":
        return pick(
            ("upper_bb_mse", "avg_upper_bb_mse"),
            ("upper_bb_mse", "avg_mse"),
            ("upper_bb_mae", "avg_mae"),
            ("upper_bb_rmse", "avg_rmse"),
            ("upper_bb_mape", "avg_mape"),
        )
    if domain == "weather" and task == "indicator":
        return pick(
            ("max_mse", "avg_max_mse"),
            ("min_mse", "avg_min_mse"),
            ("diff_mse", "avg_diff_mse"),
            ("max_mae", "avg_max_mae"),
            ("min_mae", "avg_min_mae"),
            ("diff_mae", "avg_diff_mae"),
        )

    picked: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            summary_key = key if key.startswith("avg_") else f"avg_{key}"
            picked[summary_key] = float(value)
    return picked


def timeclaw_result_fields(
    *,
    row_recorded: bool,
    summary_included: bool,
    summary_exclude_reason: str | None = None,
) -> dict[str, Any]:
    return {
        _ROW_RECORDED_KEY: row_recorded,
        _SUMMARY_INCLUDED_KEY: summary_included,
        _SUMMARY_EXCLUDE_REASON_KEY: summary_exclude_reason,
    }


def _row_recorded(row: dict[str, Any]) -> bool:
    if _ROW_RECORDED_KEY in row:
        return bool(row.get(_ROW_RECORDED_KEY))
    return bool(row.get(_LEGACY_ROW_RECORDED_KEY))


def _row_summary_included(row: dict[str, Any]) -> bool | None:
    if _SUMMARY_INCLUDED_KEY in row:
        return row.get(_SUMMARY_INCLUDED_KEY)
    return row.get(_LEGACY_SUMMARY_INCLUDED_KEY)


def _row_summary_exclude_reason(row: dict[str, Any]) -> str | None:
    if _SUMMARY_EXCLUDE_REASON_KEY in row:
        value = row.get(_SUMMARY_EXCLUDE_REASON_KEY)
        return str(value) if value is not None else None
    value = row.get(_LEGACY_SUMMARY_EXCLUDE_REASON_KEY)
    return str(value) if value is not None else None


def apply_official_metric_policy_to_row(row: dict[str, Any], *, apply_thresholds: bool = True) -> dict[str, Any]:
    """Annotate one inference row with MTBench-style summary inclusion fields.

    New definition:
        The current active inference path writes raw per-sample rows but does
        not itself decide whether extreme metric values should count toward the
        final benchmark-style averages. This helper re-applies the retired
        official MTBench policy onto the persisted row format so current
        inference summaries stay aligned with the official threshold filters.
    """
    normalized = dict(row)
    if not apply_thresholds:
        for key in (_ROW_RECORDED_KEY, _SUMMARY_INCLUDED_KEY, _SUMMARY_EXCLUDE_REASON_KEY,
                    _LEGACY_ROW_RECORDED_KEY, _LEGACY_SUMMARY_INCLUDED_KEY, _LEGACY_SUMMARY_EXCLUDE_REASON_KEY):
            normalized.pop(key, None)
    if (
        _ROW_RECORDED_KEY in normalized
        and _SUMMARY_INCLUDED_KEY in normalized
    ):
        return normalized
    if (
        _LEGACY_ROW_RECORDED_KEY in normalized
        and _LEGACY_SUMMARY_INCLUDED_KEY in normalized
    ):
        normalized.setdefault(_ROW_RECORDED_KEY, normalized.get(_LEGACY_ROW_RECORDED_KEY))
        normalized.setdefault(_SUMMARY_INCLUDED_KEY, normalized.get(_LEGACY_SUMMARY_INCLUDED_KEY))
        normalized.setdefault(_SUMMARY_EXCLUDE_REASON_KEY, normalized.get(_LEGACY_SUMMARY_EXCLUDE_REASON_KEY))
        return normalized

    metrics = normalized.get("metrics")
    if not isinstance(metrics, dict):
        normalized.update(
            timeclaw_result_fields(
                row_recorded=False,
                summary_included=False,
                summary_exclude_reason="missing_metrics",
            )
        )
        return normalized

    skipped_reason = metrics.get("skipped")
    if skipped_reason:
        normalized.update(
            timeclaw_result_fields(
                row_recorded=False,
                summary_included=False,
                summary_exclude_reason=str(skipped_reason),
            )
        )
        return normalized

    error_reason = metrics.get("error")
    if error_reason:
        normalized.update(
            timeclaw_result_fields(
                row_recorded=False,
                summary_included=False,
                summary_exclude_reason=str(error_reason),
            )
        )
        return normalized
    if _has_numeric_error_metric(metrics):
        normalized.update(
            timeclaw_result_fields(
                row_recorded=False,
                summary_included=False,
                summary_exclude_reason="parse_fail",
            )
        )
        return normalized

    status = str(normalized.get("status") or "").strip().lower()
    if status and not _is_completed_status(status):
        normalized.update(
            timeclaw_result_fields(
                row_recorded=False,
                summary_included=False,
                summary_exclude_reason=status,
            )
        )
        return normalized

    domain = _domain_from_row(normalized)
    task = str(normalized.get("task") or "").strip()
    metric_filter = _OFFICIAL_METRIC_FILTERS.get((domain, task)) if apply_thresholds else None
    if metric_filter is None:
        normalized.update(
            timeclaw_result_fields(row_recorded=True, summary_included=True)
        )
        return normalized

    metric_name, threshold = metric_filter
    value = metrics.get(metric_name)
    if isinstance(value, (int, float)) and np.isfinite(float(value)) and float(value) > threshold:
        normalized.update(
            timeclaw_result_fields(
                row_recorded=True,
                summary_included=False,
                summary_exclude_reason=f"{metric_name}>{threshold:g}",
            )
        )
        return normalized

    normalized.update(timeclaw_result_fields(row_recorded=True, summary_included=True))
    return normalized


def summarize_official_inference_rows(rows: list[dict[str, Any]], *, apply_thresholds: bool = True) -> list[dict[str, Any]]:
    """Summarize raw inference rows using MTBench-style inclusion rules."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalized = apply_official_metric_policy_to_row(row, apply_thresholds=apply_thresholds)
        dataset = str(normalized.get("dataset") or "").strip()
        task = str(normalized.get("task") or "").strip()
        grouped[(dataset, task)].append(normalized)

    summaries: list[dict[str, Any]] = []
    for (dataset, task), task_rows in sorted(grouped.items()):
        total = len(task_rows)
        completed = sum(_is_completed_status(r.get("status")) for r in task_rows)
        errors = sum(_official_row_status(r) == "error" for r in task_rows)
        row_recorded_n = sum(_row_recorded(r) for r in task_rows)
        summary_included_n = sum(bool(_row_summary_included(r)) for r in task_rows)
        threshold_filtered_n = sum(
            1
            for r in task_rows
            if _row_recorded(r) and _row_summary_included(r) is False
        )
        invalid_skipped_n = sum(
            1
            for r in task_rows
            if _row_recorded(r) is False and _row_summary_included(r) is False
        )

        metric_lists: dict[str, list[float]] = defaultdict(list)
        total_ms_values: list[float] = []
        for row in task_rows:
            total_ms = row.get("total_ms")
            if isinstance(total_ms, (int, float)) and np.isfinite(float(total_ms)):
                total_ms_values.append(float(total_ms))
            if _row_summary_included(row) is False:
                continue
            for key, value in _numeric_metric_projection(row).items():
                metric_lists[key].append(float(value))

        payload: dict[str, Any] = {
            "dataset": dataset,
            "task": task,
            "completed": completed,
            "total": total,
            "errors": errors,
            "row_recorded_n": row_recorded_n,
            "summary_included_n": summary_included_n,
            "threshold_filtered_n": threshold_filtered_n,
            "invalid_skipped_n": invalid_skipped_n,
        }
        if total_ms_values:
            payload["avg_total_ms"] = round(float(np.mean(total_ms_values)), 3)
        for key, values in sorted(metric_lists.items()):
            if values:
                payload[key] = round(float(np.mean(values)), 3)
        summaries.append(payload)
    return summaries
