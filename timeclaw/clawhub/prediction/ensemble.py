"""Ensemble forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class EnsembleForecastTool(ClawTool):
    """Ensemble multiple forecasts."""

    name: str = "ensemble_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = "Combine multiple forecast models into an ensemble."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "forecasts": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "List of forecast arrays from multiple models.",
            },
            "method": {
                "type": "string",
                "default": "mean",
                "enum": ["mean", "median", "weighted"],
                "description": "Aggregation method.",
            },
            "weights": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Weights for weighted ensemble (required when method='weighted').",
            },
        },
        "required": ["forecasts"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        forecasts = params.get("forecasts")
        if forecasts is None or len(forecasts) == 0:
            raise ValueError("forecasts parameter is required and must be non-empty")

        method: str = params.get("method", "mean")
        matrix = np.array(forecasts, dtype=float)

        if method == "mean":
            combined = np.mean(matrix, axis=0)
        elif method == "median":
            combined = np.median(matrix, axis=0)
        elif method == "weighted":
            weights = params.get("weights")
            if weights is None:
                raise ValueError("weights parameter is required for method='weighted'")
            w = np.array(weights, dtype=float)
            if len(w) != len(forecasts):
                raise ValueError(
                    f"weights length ({len(w)}) must match number of forecasts ({len(forecasts)})"
                )
            w = w / w.sum()
            combined = np.average(matrix, axis=0, weights=w)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'mean', 'median', or 'weighted'.")

        return {
            "forecast": combined.tolist(),
            "method": method,
            "n_models": len(forecasts),
        }


tool = EnsembleForecastTool()
