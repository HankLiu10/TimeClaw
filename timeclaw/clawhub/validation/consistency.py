"""Consistency check for time series."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class ConsistencyCheckTool(ClawTool):
    """Consistency check for time series or predictions."""

    name: str = "consistency_check"
    category: ToolCategory = ToolCategory.VALIDATION
    description: str = "Check consistency of predictions against historical time series data."
    io_type: ToolIOType = ToolIOType.XP_TO_ACCEPT
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "prediction": {
                "type": "array",
                "description": "Predicted values to check.",
            },
            "historical": {
                "type": "array",
                "description": "Historical time series values.",
            },
        },
        "required": ["prediction", "historical"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        prediction = np.asarray(params.get("prediction", []), dtype=float)
        historical = np.asarray(params.get("historical", ts_data), dtype=float)

        checks: list[dict[str, Any]] = []

        hist_std = float(np.std(historical)) if len(historical) > 1 else 0.0
        gap = abs(float(prediction[0]) - float(historical[-1]))
        threshold = 3.0 * hist_std if hist_std > 0 else float("inf")
        continuity_passed = gap <= threshold
        checks.append({
            "name": "continuity",
            "passed": continuity_passed,
            "details": {
                "gap": gap,
                "threshold": threshold,
                "historical_std": hist_std,
            },
        })

        recent_n = min(len(historical), max(2, len(historical) // 4))
        recent = historical[-recent_n:]
        hist_trend = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        pred_trend = float(np.polyfit(np.arange(len(prediction)), prediction, 1)[0])
        trend_passed = (hist_trend * pred_trend >= 0) or abs(hist_trend) < hist_std * 0.1
        checks.append({
            "name": "trend_consistency",
            "passed": trend_passed,
            "details": {
                "historical_trend_slope": hist_trend,
                "prediction_trend_slope": pred_trend,
            },
        })

        hist_var = float(np.var(historical)) if len(historical) > 1 else 0.0
        pred_var = float(np.var(prediction)) if len(prediction) > 1 else 0.0
        if hist_var > 0:
            variance_passed = pred_var <= 5.0 * hist_var
        else:
            variance_passed = pred_var == 0.0
        checks.append({
            "name": "variance_consistency",
            "passed": variance_passed,
            "details": {
                "historical_variance": hist_var,
                "prediction_variance": pred_var,
                "ratio": pred_var / hist_var if hist_var > 0 else float("inf") if pred_var > 0 else 0.0,
            },
        })

        n_passed = sum(1 for c in checks if c["passed"])
        return {
            "consistent": all(c["passed"] for c in checks),
            "checks": checks,
            "n_passed": n_passed,
            "n_total": len(checks),
        }


tool = ConsistencyCheckTool()
