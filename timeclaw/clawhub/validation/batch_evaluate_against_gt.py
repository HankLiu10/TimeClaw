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
                "description": "List of candidate predictions to compare.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "prediction": {"description": "Candidate prediction payload."},
                    },
                    "required": ["prediction"],
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
            if not isinstance(item, dict):
                item = {"prediction": item}
            pred = item.get("prediction")
            if pred is None:
                continue
            metrics = EvaluateAgainstGTTool()._compute_metrics(pred, ground_truth, task_type)
            results.append({
                "index": idx,
                "name": item.get("name", f"approach_{idx}"),
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
