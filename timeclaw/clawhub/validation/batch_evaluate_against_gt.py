"""Batch evaluation against ground truth for training-mode exploration."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType
from timeclaw.clawhub.validation.evaluate_against_gt import EvaluateAgainstGTTool


class BatchEvaluateAgainstGTTool(ClawTool):
    name: str = "evaluate_batch_against_gt"
    category: ToolCategory = ToolCategory.VALIDATION
    description: str = (
        "Evaluate multiple candidate predictions against the same ground truth in one call. "
        "Use in learning mode after collecting several valid approaches."
    )
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "description": (
                    "List of candidate predictions to compare. Each item may be either "
                    "a raw prediction payload or an object like "
                    "{name: ..., prediction: ...}."
                ),
                "items": {
                    "description": (
                        "Raw prediction payload or wrapper object containing "
                        "`prediction` and optional `name`."
                    ),
                },
            },
            "task_type": {
                "type": "string",
                "enum": [
                    "forecast", "indicator", "trend",
                    "classification", "qa", "mcqa", "correlation",
                ],
            },
        },
        "required": ["predictions", "task_type"],
    }
    is_stub: bool = False

    @staticmethod
    def _normalise_prediction_item(item: Any, idx: int) -> tuple[str, Any] | None:
        """Accept either wrapped items or raw prediction payloads."""
        if isinstance(item, dict) and "prediction" in item:
            pred = item.get("prediction")
            if pred is None:
                return None
            name = str(item.get("name") or f"approach_{idx}")
            return name, pred

        if item is None:
            return None

        # Root-cause fix: agents often pass raw prediction payloads directly.
        # Treat the entire item as the prediction instead of silently dropping it.
        return f"approach_{idx}", item

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        task_type = params.get("task_type", "")
        predictions = params.get("predictions") or []
        ground_truth = params.get("_ground_truth")

        if ground_truth is None:
            raise ValueError("Ground truth not available. This tool is only usable during training mode.")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("predictions must be a non-empty list.")

        results = []
        for idx, item in enumerate(predictions):
            normalised = self._normalise_prediction_item(item, idx)
            if normalised is None:
                continue
            name, pred = normalised
            metrics = EvaluateAgainstGTTool()._compute_metrics(pred, ground_truth, task_type)
            results.append({
                "index": idx,
                "name": name,
                "metrics": metrics,
                "prediction": EvaluateAgainstGTTool._serialise(pred),
            })

        return {
            "results": results,
            "ground_truth": EvaluateAgainstGTTool._serialise(ground_truth),
        }

tool = BatchEvaluateAgainstGTTool()


class BatchEvaluateAgainstGTAliasTool(BatchEvaluateAgainstGTTool):
    name: str = "batch_evaluate_against_gt"


tool_alias = BatchEvaluateAgainstGTAliasTool()
