"""Evaluate predictions against ground truth — training-mode only tool.

This tool computes task-specific metrics by comparing the agent's prediction
with the ground truth.  Ground truth is injected by the framework (via
``_ground_truth`` kwarg) and is NOT a user-facing parameter — the agent
only passes ``prediction`` and ``task_type``.

The return value includes both computed metrics AND the raw ground truth
so the agent can perform one-step qualitative analysis.  After that step,
the framework sanitises the conversation to remove the ground truth values.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from timeclaw.benchmark.metrics import mae, rmse, mape, mse
from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class EvaluateAgainstGTTool(ClawTool):
    """Compare a prediction against ground truth and return metrics.

    Available only during training mode.  The agent calls this tool with
    its prediction and task_type; the ground truth is supplied internally
    by the framework.
    """

    name: str = "evaluate_against_gt"
    category: ToolCategory = ToolCategory.VALIDATION
    description: str = (
        "Evaluate a prediction against the ground truth. "
        "Pass your prediction and the task_type. Returns evaluation metrics "
        "(MAE, RMSE, accuracy, etc.) and the ground truth values for analysis. "
        "Only available in learning mode."
    )
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "prediction": {
                "description": (
                    "Your prediction result — array for forecast, "
                    "dict for indicator, string/label for classification/qa/trend/correlation."
                ),
            },
            "task_type": {
                "type": "string",
                "enum": [
                    "forecast", "indicator", "trend",
                    "classification", "qa", "mcqa", "correlation",
                ],
                "description": "The type of the current task.",
            },
        },
        "required": ["prediction", "task_type"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        prediction = params.get("prediction")
        task_type = params.get("task_type", "")
        ground_truth = params.get("_ground_truth")

        if ground_truth is None:
            raise ValueError(
                "Ground truth not available. "
                "This tool is only usable during training/learning mode."
            )

        if prediction is None:
            raise ValueError("prediction is required.")

        metrics = self._compute_metrics(prediction, ground_truth, task_type)

        return {
            "metrics": metrics,
            "ground_truth": self._serialise(ground_truth),
        }

    # ── Metrics per task type ────────────────────────────────────────

    @staticmethod
    def _canonical_task_type(task_type: Any) -> str:
        """Map dataset-scoped task ids back to the evaluator's core task family.

        New definition:
            Training-time validation sometimes passes dataset-qualified task ids
            such as ``weather_aligned_short_forecast`` or
            ``finance_aligned_long_indicator_macd``.  Metric selection must be
            driven by the core task family, not the dataset-qualified wrapper,
            otherwise continuous-value tasks fall through into label-style
            exact-match scoring.
        """
        text = str(task_type or "").strip().lower()
        if not text:
            return ""
        if text in {
            "forecast", "indicator", "trend", "classification", "qa", "mcqa", "correlation",
        }:
            return text
        if text.endswith("_forecast"):
            return "forecast"
        if "_indicator_" in text or text.endswith("_indicator"):
            return "indicator"
        if text.endswith("_trend_past"):
            return "trend"
        if text.endswith("_trend") or text.endswith("_classification"):
            return "classification"
        if text.endswith("_mcqa"):
            return "mcqa"
        if text.endswith("_correlation"):
            return "correlation"
        if text.endswith("_qa"):
            return "qa"
        return text

    def _compute_metrics(
        self, prediction: Any, ground_truth: Any, task_type: str,
    ) -> dict[str, Any]:
        task_type = self._canonical_task_type(task_type)
        if task_type == "forecast":
            return self._forecast_metrics(prediction, ground_truth)
        if task_type == "indicator":
            return self._indicator_metrics(prediction, ground_truth)
        return self._label_metrics(prediction, ground_truth)

    @staticmethod
    def _forecast_metrics(pred: Any, gt: Any) -> dict[str, float]:
        if isinstance(pred, dict):
            for key in ("point_forecast", "forecast", "prediction", "values", "answer"):
                if key in pred and isinstance(pred[key], (list, np.ndarray)):
                    pred = pred[key]
                    break
        if isinstance(gt, dict):
            for key in ("point_forecast", "forecast", "values", "answer"):
                if key in gt and isinstance(gt[key], (list, np.ndarray)):
                    gt = gt[key]
                    break
        try:
            y_pred = np.asarray(pred, dtype=np.float64).ravel()
            y_true = np.asarray(gt, dtype=np.float64).ravel()
        except (ValueError, TypeError):
            return {"type_error": 1.0}
        n = min(len(y_true), len(y_pred))
        if n == 0:
            return {"type_error": 1.0}
        y_true, y_pred = y_true[:n], y_pred[:n]
        return {
            "mae": round(mae(y_true, y_pred), 3),
            "rmse": round(rmse(y_true, y_pred), 3),
            "mape": round(mape(y_true, y_pred), 3),
            "mse": round(mse(y_true, y_pred), 3),
        }

    @staticmethod
    def _indicator_metrics(pred: Any, gt: Any) -> dict[str, float]:
        if isinstance(pred, str):
            try:
                pred = json.loads(pred)
            except (json.JSONDecodeError, TypeError):
                return {"parse_error": 1.0}
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except (json.JSONDecodeError, TypeError):
                return {"parse_error": 1.0}
        if isinstance(pred, dict) and "answer" in pred and isinstance(pred["answer"], dict):
            pred = pred["answer"]
        if isinstance(gt, dict) and "answer" in gt and isinstance(gt["answer"], dict):
            gt = gt["answer"]
        if not isinstance(pred, dict) or not isinstance(gt, dict):
            return {"type_error": 1.0}

        metrics: dict[str, float] = {}
        for key in gt:
            if key in pred:
                pred_value = pred[key]
                gt_value = gt[key]
                try:
                    pred_arr = np.asarray(pred_value, dtype=np.float64).ravel()
                    gt_arr = np.asarray(gt_value, dtype=np.float64).ravel()
                except (ValueError, TypeError):
                    metrics[f"{key}_error"] = 1.0
                    continue

                if pred_arr.size == 0 or gt_arr.size == 0:
                    metrics[f"{key}_error"] = 1.0
                    continue

                if pred_arr.size == 1 and gt_arr.size == 1:
                    metrics[f"{key}_mae"] = round(abs(float(gt_arr[0]) - float(pred_arr[0])), 3)
                    continue

                n = min(len(gt_arr), len(pred_arr))
                gt_arr = gt_arr[:n]
                pred_arr = pred_arr[:n]
                mask = np.isfinite(gt_arr) & np.isfinite(pred_arr)
                gt_arr = gt_arr[mask]
                pred_arr = pred_arr[mask]
                if gt_arr.size == 0:
                    metrics[f"{key}_error"] = 1.0
                    continue

                metrics[f"{key}_mae"] = round(mae(gt_arr, pred_arr), 3)
                metrics[f"{key}_rmse"] = round(rmse(gt_arr, pred_arr), 3)
                metrics[f"{key}_mse"] = round(mse(gt_arr, pred_arr), 3)
                metrics[f"{key}_mape"] = round(mape(gt_arr, pred_arr), 3)
        if metrics:
            numeric = [v for k, v in metrics.items() if k.endswith("_mae") and isinstance(v, float)]
            if numeric:
                metrics["avg_mae"] = round(sum(numeric) / len(numeric), 3)
        return metrics

    @staticmethod
    def _label_metrics(pred: Any, gt: Any) -> dict[str, Any]:
        pred = EvaluateAgainstGTTool._normalise_label_prediction(pred)
        gt = EvaluateAgainstGTTool._normalise_label_prediction(gt)
        pred_str = str(pred).strip().lower()
        gt_str = str(gt).strip().lower()
        correct = pred_str == gt_str
        if (
            not correct
            and len(gt_str) == 1
            and gt_str.isalpha()
            and pred_str[:1].isalpha()
        ):
            correct = pred_str[:1] == gt_str
        return {
            "accuracy": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted_label": str(pred),
            "ground_truth_label": str(gt),
        }

    @staticmethod
    def _normalise_label_prediction(value: Any) -> Any:
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

    @staticmethod
    def _serialise(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj


tool = EvaluateAgainstGTTool()
