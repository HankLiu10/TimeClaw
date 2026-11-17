from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.benchmark.ground_truth import normalize_numeric_ground_truth
from timeclaw.benchmark.metrics import mae, mse, rmse

# New definition:
#   This module is the shared MTBench-facing scorer. It does not change how the
#   agent reasons. It only standardizes two evaluation-only behaviors:
#   1. whether a sample/prediction is scoreable under MTBench-style rules
#   2. how numeric predictions are aligned to ground-truth length before
#      metrics are computed (NumPy linear interpolation, matching MTBench)

_FINANCE_TREND_3WAY = {
    "<-4%": "negative",
    "-2% ~ -4%": "negative",
    "-2% ~ +2%": "neutral",
    "+2% ~ +4%": "positive",
    ">+4%": "positive",
}

_CORRELATION_POSITIVE = {
    "Strong Positive Correlation",
    "Moderate Positive Correlation",
}
_CORRELATION_NEGATIVE = {
    "Strong Negative Correlation",
    "Moderate Negative Correlation",
}


def canonical_task_type(task_type: Any) -> str:
    text = str(task_type or "").strip().lower()
    if not text:
        return ""
    if text.endswith("_indicator_macd") or text == "indicator_macd":
        return "indicator_macd"
    if text.endswith("_indicator_upper_bb") or text == "indicator_upper_bb":
        return "indicator_upper_bb"
    if text.endswith("_forecast") or text == "forecast":
        return "forecast"
    if text.endswith("_trend_past") or text == "trend_past":
        return "trend_past"
    if text.endswith("_trend") or text == "trend":
        return "trend"
    if "_indicator_" in text or text.endswith("_indicator") or text == "indicator":
        return "indicator"
    if text.endswith("_mcqa") or text == "mcqa":
        return "mcqa"
    if text.endswith("_correlation") or text == "correlation":
        return "correlation"
    if text.endswith("_qa") or text == "qa":
        return "qa"
    if text.endswith("_classification") or text == "classification":
        return "classification"
    return text


def validate_mtbench_sample_for_scoring(sample: Any, task_type: Any) -> tuple[bool, str | None]:
    task = canonical_task_type(task_type)
    labels = getattr(sample, "labels", {}) if hasattr(sample, "labels") else {}
    domain = str(getattr(sample, "domain", "") or "").strip().lower()

    if task == "forecast":
        normalized = normalize_numeric_ground_truth(getattr(sample, "output_window", None))
        return normalized.valid, None if normalized.valid else "invalid_ground_truth"

    if task == "indicator_macd":
        normalized = normalize_numeric_ground_truth(labels.get("out_macd"))
        return normalized.valid, None if normalized.valid else "invalid_ground_truth"

    if task == "indicator_upper_bb":
        normalized = normalize_numeric_ground_truth(labels.get("out_upper_bb"))
        return normalized.valid, None if normalized.valid else "invalid_ground_truth"

    if task == "indicator":
        if domain == "weather":
            normalized = normalize_numeric_ground_truth(getattr(sample, "output_window", None))
            return normalized.valid, None if normalized.valid else "invalid_ground_truth"
        macd = normalize_numeric_ground_truth(labels.get("out_macd"))
        upper_bb = normalize_numeric_ground_truth(labels.get("out_upper_bb"))
        valid = macd.valid or upper_bb.valid
        return valid, None if valid else "invalid_ground_truth"

    if task in {"trend", "trend_past"} and domain == "finance":
        label = labels.get("trend_label")
        if isinstance(label, str) and _canonical_finance_trend_label(label) is not None:
            return True, None
        return False, "invalid_ground_truth"

    if task == "mcqa":
        answer = getattr(sample, "answer", None)
        if isinstance(answer, str) and prompts.extract_mcqa_letter(answer):
            return True, None
        return False, "invalid_ground_truth"

    if task == "correlation":
        label = labels.get("correlation")
        if isinstance(label, str) and label.strip():
            return True, None
        return False, "invalid_ground_truth"

    return True, None


def align_numeric_prediction_to_ground_truth(
    prediction: Any,
    ground_truth: Any,
) -> np.ndarray | None:
    pred_arr = _coerce_numeric_array(prediction)
    gt_arr = _coerce_numeric_ground_truth_array(ground_truth)
    if pred_arr is None or gt_arr is None or gt_arr.size == 0 or pred_arr.size == 0:
        return None
    if pred_arr.size == gt_arr.size:
        return pred_arr.astype(np.float64)
    if pred_arr.size == 1:
        return np.repeat(float(pred_arr[0]), gt_arr.size)
    return np.interp(
        np.linspace(0.0, 1.0, gt_arr.size),
        np.linspace(0.0, 1.0, pred_arr.size),
        pred_arr,
    ).astype(np.float64)


def score_mtbench_prediction(
    prediction: Any,
    ground_truth: Any,
    task_type: Any,
) -> dict[str, Any]:
    task = canonical_task_type(task_type)
    if task == "forecast":
        return _score_forecast(prediction, ground_truth)
    if task in {"indicator", "indicator_macd", "indicator_upper_bb"}:
        return _score_indicator(prediction, ground_truth, task)
    if task in {"trend", "trend_past"}:
        return _score_trend(prediction, ground_truth)
    if task == "mcqa":
        return _score_mcqa(prediction, ground_truth)
    if task == "correlation":
        return _score_correlation(prediction, ground_truth)
    return _score_generic_label(prediction, ground_truth)


def backfill_scored_metrics(
    prediction: Any,
    ground_truth: Any,
    task_type: Any,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fill missing scorer-produced fields from the same prediction/GT pair.

    New definition:
        Historical inference rows may contain only the primary metric
        (for example `accuracy`) while dropping derived MTBench fields such as
        finance trend `accuracy_3way` or correlation `brief_accuracy`.
        This helper re-scores the same sample and merges only the missing keys,
        without overriding already-persisted values.
    """
    current = dict(metrics) if isinstance(metrics, dict) else {}
    if not current:
        return current
    if current.get("skipped") or current.get("error"):
        return current
    if any(bool(current.get(key)) for key in ("parse_error", "type_error")):
        return current

    task = canonical_task_type(task_type)
    if task not in {"trend", "trend_past", "mcqa", "correlation"}:
        return current

    rescored = score_mtbench_prediction(prediction, ground_truth, task_type)
    if not isinstance(rescored, dict) or not rescored:
        return current
    if rescored.get("skipped") or rescored.get("error"):
        return current
    if any(bool(rescored.get(key)) for key in ("parse_error", "type_error")):
        return current

    missing_keys = [key for key in rescored if key not in current]
    if not missing_keys:
        return current

    repaired = dict(current)
    for key in missing_keys:
        repaired[key] = rescored[key]
    return repaired


def _score_forecast(prediction: Any, ground_truth: Any) -> dict[str, Any]:
    gt_arr = _coerce_numeric_ground_truth_array(ground_truth)
    if gt_arr is None:
        return {"skipped": "invalid_ground_truth"}
    aligned = align_numeric_prediction_to_ground_truth(
        _extract_forecast_like_payload(prediction),
        gt_arr,
    )
    if aligned is None:
        return {"type_error": 1.0}
    return _forecast_metric_payload(gt_arr, aligned)


def _score_indicator(prediction: Any, ground_truth: Any, task: str) -> dict[str, Any]:
    pred_payload = _unwrap_answer_dict(prediction)
    gt_payload = _unwrap_answer_dict(ground_truth)
    if not isinstance(pred_payload, dict) or not isinstance(gt_payload, dict):
        return {"type_error": 1.0}

    if task == "indicator_macd":
        target_keys = ["macd"]
    elif task == "indicator_upper_bb":
        target_keys = ["upper_bb"]
    else:
        target_keys = [key for key in ("macd", "upper_bb", "max", "min", "diff") if key in gt_payload]

    if not target_keys:
        return {"skipped": "invalid_ground_truth"}

    metrics: dict[str, Any] = {}
    valid_gt_keys = 0
    for key in target_keys:
        if key not in gt_payload:
            continue
        gt_value = gt_payload.get(key)
        pred_value = pred_payload.get(key)

        if key in {"max", "min", "diff"}:
            gt_scalar = _coerce_numeric_scalar(gt_value)
            if gt_scalar is None:
                continue
            valid_gt_keys += 1
            pred_scalar = _coerce_numeric_scalar(pred_value)
            if pred_scalar is None:
                metrics[f"{key}_error"] = 1.0
                continue
            metrics[f"{key}_mae"] = abs(pred_scalar - gt_scalar)
            metrics[f"{key}_mse"] = (pred_scalar - gt_scalar) ** 2
            continue

        gt_arr = _coerce_numeric_ground_truth_array(gt_value)
        if gt_arr is None:
            continue
        valid_gt_keys += 1
        aligned = align_numeric_prediction_to_ground_truth(pred_value, gt_arr)
        if aligned is None:
            metrics[f"{key}_error"] = 1.0
            continue
        metric_payload = _forecast_metric_payload(gt_arr, aligned)
        for metric_name, metric_value in metric_payload.items():
            metrics[f"{key}_{metric_name}"] = metric_value

    if valid_gt_keys == 0:
        return {"skipped": "invalid_ground_truth"}
    if not metrics:
        return {"parse_error": 1.0}

    mae_values = [
        float(value)
        for key, value in metrics.items()
        if key.endswith("_mae") and isinstance(value, (int, float))
    ]
    if mae_values:
        metrics["avg_mae"] = round(float(np.mean(mae_values)), 3)
    return metrics


def _score_trend(prediction: Any, ground_truth: Any) -> dict[str, Any]:
    gt_text = str(ground_truth or "").strip()
    if not gt_text:
        return {"skipped": "invalid_ground_truth"}

    gt_finance = _canonical_finance_trend_label(gt_text)
    if gt_finance is not None:
        pred_text = _extract_label_scalar(prediction)
        pred_label = _canonical_finance_trend_label(pred_text)
        if pred_label is None and isinstance(pred_text, str):
            parsed = prompts.parse_finance_trend_output(pred_text)
            pred_label = _canonical_finance_trend_label(parsed)
        if pred_label is None:
            return {"parse_error": 1.0}
        exact = float(pred_label == gt_finance)
        return {
            "accuracy": exact,
            "accuracy_3way": float(_FINANCE_TREND_3WAY[pred_label] == _FINANCE_TREND_3WAY[gt_finance]),
            "correct": bool(exact >= 1.0),
            "predicted_label": pred_label,
            "ground_truth_label": gt_finance,
        }

    gt_weather = _canonical_weather_trend_label(gt_text)
    if gt_weather is None:
        return {"skipped": "invalid_ground_truth"}
    pred_text = _extract_label_scalar(prediction)
    pred_label = _canonical_weather_trend_label(pred_text)
    if pred_label is None and isinstance(pred_text, str):
        pred_label = _canonical_weather_trend_label(prompts.parse_weather_trend_output(pred_text))
    if pred_label is None:
        return {"parse_error": 1.0}
    exact = float(pred_label == gt_weather)
    return {
        "accuracy": exact,
        "correct": bool(exact >= 1.0),
        "predicted_label": pred_label,
        "ground_truth_label": gt_weather,
    }


def _score_mcqa(prediction: Any, ground_truth: Any) -> dict[str, Any]:
    gt = prompts.extract_mcqa_letter(str(ground_truth or ""))
    if not gt:
        return {"skipped": "invalid_ground_truth"}
    pred = prompts.extract_mcqa_letter(str(_extract_label_scalar(prediction) or ""))
    if not pred:
        return {"parse_error": 1.0}
    exact = float(pred == gt)
    return {
        "accuracy": exact,
        "correct": bool(exact >= 1.0),
        "predicted_label": pred,
        "ground_truth_label": gt,
    }


def _score_correlation(prediction: Any, ground_truth: Any) -> dict[str, Any]:
    gt = _canonical_correlation_label(str(ground_truth or ""))
    if gt is None:
        return {"skipped": "invalid_ground_truth"}

    pred_raw = _extract_label_scalar(prediction)
    pred = _canonical_correlation_label(str(pred_raw or ""))
    if pred is None:
        pred = str(pred_raw or "").strip()

    exact = float(pred == gt)
    brief = float(_correlation_brief_bucket(pred) == _correlation_brief_bucket(gt))
    return {
        "accuracy": exact,
        "exact_accuracy": exact,
        "brief_accuracy": brief,
        "correct": bool(exact >= 1.0),
        "predicted_label": pred,
        "ground_truth_label": gt,
    }


def _score_generic_label(prediction: Any, ground_truth: Any) -> dict[str, Any]:
    pred = _canonicalize_label_text(_extract_label_scalar(prediction))
    gt = _canonicalize_label_text(_extract_label_scalar(ground_truth))
    if not gt:
        return {"skipped": "invalid_ground_truth"}
    exact = float(pred == gt)
    return {
        "accuracy": exact,
        "correct": bool(exact >= 1.0),
        "predicted_label": str(_extract_label_scalar(prediction)),
        "ground_truth_label": str(_extract_label_scalar(ground_truth)),
    }


def _forecast_metric_payload(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict[str, Any]:
    """Keep raw arithmetic until reporting; use pinned MTBench's nonzero MAPE.

    MAPE is undefined when every target is zero. Store null plus an explicit
    definition flag in that case, while retaining defined MAE/MSE/RMSE.
    Rounding before threshold admission would change which rows are included.
    """
    nonzero = gt_arr != 0
    percentage = float(np.mean(np.abs((gt_arr[nonzero] - pred_arr[nonzero]) / gt_arr[nonzero])) * 100) if nonzero.any() else None
    metrics = {"mae": mae(gt_arr, pred_arr), "rmse": rmse(gt_arr, pred_arr),
               "mape": percentage, "mse": mse(gt_arr, pred_arr)}
    if percentage is None:
        metrics["mape_defined"] = False
    return metrics


def _extract_forecast_like_payload(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("point_forecast", "forecast", "prediction", "values", "temperatures", "predictions", "data"):
            if key in value:
                return value[key]
        if "answer" in value:
            return _extract_forecast_like_payload(value["answer"])
    return value


def _unwrap_answer_dict(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    if isinstance(value, dict) and "answer" in value and isinstance(value["answer"], dict):
        return value["answer"]
    return value


def _coerce_numeric_ground_truth_array(value: Any) -> np.ndarray | None:
    normalized = normalize_numeric_ground_truth(value)
    if not normalized.valid or normalized.value is None:
        return None
    return np.asarray(normalized.value, dtype=np.float64).ravel()


def _coerce_numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    value = _extract_forecast_like_payload(value)
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (ValueError, TypeError):
        return None
    if arr.size == 0 or not bool(np.isfinite(arr).all()):
        return None
    return arr


def _coerce_numeric_scalar(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _extract_label_scalar(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        value = parsed
    if isinstance(value, dict):
        for key in ("trend", "answer", "correlation", "label", "prediction"):
            if key in value:
                return value[key]
    return value


def _canonicalize_label_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip()).lower()


def _canonical_weather_trend_label(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"increasing", "decreasing", "stable"}:
        return text
    return None


def _canonical_finance_trend_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    explicit = {
        "<-4%": "<-4%",
        "-2% ~ -4%": "-2% ~ -4%",
        "-2% ~ +2%": "-2% ~ +2%",
        "+2% ~ +4%": "+2% ~ +4%",
        ">+4%": ">+4%",
    }
    if text in explicit:
        return explicit[text]
    compact = text.replace(" ", "")
    compact_map = {
        "<-4%": "<-4%",
        "-2%~-4%": "-2% ~ -4%",
        "-2%~+2%": "-2% ~ +2%",
        "+2%~+4%": "+2% ~ +4%",
        ">+4%": ">+4%",
    }
    return compact_map.get(compact)


def _canonical_correlation_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    labels = [
        "Strong Positive Correlation",
        "Moderate Positive Correlation",
        "No Correlation",
        "Moderate Negative Correlation",
        "Strong Negative Correlation",
    ]
    lowered = text.lower()
    exact_map = {label.lower(): label for label in labels}
    if lowered in exact_map:
        return exact_map[lowered]
    hits = [label for label in labels if label.lower() in lowered]
    if len(hits) == 1:
        return hits[0]
    return None


def _correlation_brief_bucket(value: Any) -> str:
    label = _canonical_correlation_label(value)
    if label in _CORRELATION_POSITIVE:
        return "positive"
    if label in _CORRELATION_NEGATIVE:
        return "negative"
    if label == "No Correlation":
        return "none"
    return "other"
