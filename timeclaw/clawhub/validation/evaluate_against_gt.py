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

from timeclaw.benchmark.mtbench_support.task_scoring import (
    align_numeric_prediction_to_ground_truth,
    canonical_task_type,
    score_mtbench_prediction,
)
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
        return canonical_task_type(task_type)

    def _compute_metrics(
        self, prediction: Any, ground_truth: Any, task_type: str,
    ) -> dict[str, Any]:
        return score_mtbench_prediction(prediction, ground_truth, task_type)

    @staticmethod
    def _forecast_metrics(pred: Any, gt: Any) -> dict[str, float]:
        return score_mtbench_prediction(pred, gt, "forecast")

    @staticmethod
    def _indicator_metrics(pred: Any, gt: Any) -> dict[str, float]:
        return score_mtbench_prediction(pred, gt, "indicator")

    @staticmethod
    def _label_metrics(pred: Any, gt: Any) -> dict[str, Any]:
        return score_mtbench_prediction(pred, gt, "classification")

    @staticmethod
    def _canonicalize_label_text(value: Any) -> str:
        return "".join(str(value).strip().lower().split())

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
    def _align_numeric_prediction_to_ground_truth(prediction: Any, ground_truth: Any) -> np.ndarray | None:
        return align_numeric_prediction_to_ground_truth(prediction, ground_truth)

    @staticmethod
    def _serialise(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj


tool = EvaluateAgainstGTTool()
