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

    def _compute_metrics(
        self, prediction: Any, ground_truth: Any, task_type: str,
    ) -> dict[str, Any]:
        if task_type == "forecast":
            return self._forecast_metrics(prediction, ground_truth)
        if task_type == "indicator":
            return self._indicator_metrics(prediction, ground_truth)
        return self._label_metrics(prediction, ground_truth)

    @staticmethod
    def _forecast_metrics(pred: Any, gt: Any) -> dict[str, float]:
        if isinstance(pred, dict):
            for key in ("point_forecast", "forecast", "prediction", "values"):
                if key in pred and isinstance(pred[key], (list, np.ndarray)):
                    pred = pred[key]
                    break
        if isinstance(gt, dict):
            for key in ("point_forecast", "forecast", "values"):
                if key in gt and isinstance(gt[key], (list, np.ndarray)):
                    gt = gt[key]
                    break
        y_pred = np.asarray(pred, dtype=np.float64).ravel()
        y_true = np.asarray(gt, dtype=np.float64).ravel()
        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]
        return {
            "mae": round(mae(y_true, y_pred), 6),
            "rmse": round(rmse(y_true, y_pred), 6),
            "mape": round(mape(y_true, y_pred), 4),
            "mse": round(mse(y_true, y_pred), 6),
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
        if not isinstance(pred, dict) or not isinstance(gt, dict):
            return {"type_error": 1.0}

        metrics: dict[str, float] = {}
        for key in gt:
            if key in pred:
                try:
                    metrics[f"{key}_mae"] = round(
                        abs(float(gt[key]) - float(pred[key])), 6
                    )
                except (ValueError, TypeError):
                    metrics[f"{key}_error"] = 1.0
        if metrics:
            numeric = [v for v in metrics.values() if isinstance(v, float)]
            if numeric:
                metrics["avg_mae"] = round(sum(numeric) / len(numeric), 6)
        return metrics

    @staticmethod
    def _label_metrics(pred: Any, gt: Any) -> dict[str, Any]:
        pred_str = str(pred).strip().lower()
        gt_str = str(gt).strip().lower()
        correct = pred_str == gt_str
        if not correct and len(pred_str) == 1 and len(gt_str) == 1:
            correct = pred_str == gt_str
        if not correct:
            correct = pred_str[:1] == gt_str[:1] and gt_str[:1].isalpha()
        return {
            "accuracy": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted_label": str(pred),
            "ground_truth_label": str(gt),
        }

    @staticmethod
    def _serialise(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj


tool = EvaluateAgainstGTTool()
