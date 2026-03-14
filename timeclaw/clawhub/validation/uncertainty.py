"""Uncertainty quantification for time series."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class UncertaintyQuantifyTool(ClawTool):
    """Uncertainty quantification for time series via bootstrap resampling."""

    name: str = "uncertainty_quantify"
    category: ToolCategory = ToolCategory.VALIDATION
    description: str = "Quantify prediction uncertainty using bootstrap resampling of historical residuals."
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "prediction": {
                "type": "array",
                "description": "Point prediction values.",
            },
            "historical": {
                "type": "array",
                "description": "Historical time series values.",
            },
            "n_bootstrap": {
                "type": "integer",
                "description": "Number of bootstrap samples.",
                "default": 1000,
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level for intervals.",
                "default": 0.95,
            },
        },
        "required": ["prediction", "historical"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        prediction = np.asarray(params.get("prediction", []), dtype=float)
        historical = np.asarray(params.get("historical", ts_data), dtype=float)
        n_bootstrap: int = int(params.get("n_bootstrap", 1000))
        confidence: float = float(params.get("confidence", 0.95))

        residuals = np.diff(historical)
        n_pred = len(prediction)
        rng = np.random.default_rng()

        bootstrap_paths = np.empty((n_bootstrap, n_pred), dtype=float)
        for i in range(n_bootstrap):
            sampled_residuals = rng.choice(residuals, size=n_pred, replace=True)
            bootstrap_paths[i] = prediction + np.cumsum(sampled_residuals) - np.cumsum(sampled_residuals).mean()

        alpha = (1.0 - confidence) / 2.0
        lower = np.percentile(bootstrap_paths, 100 * alpha, axis=0)
        upper = np.percentile(bootstrap_paths, 100 * (1.0 - alpha), axis=0)
        widths = upper - lower

        return {
            "lower_bound": lower.tolist(),
            "upper_bound": upper.tolist(),
            "confidence": confidence,
            "mean_width": float(np.mean(widths)),
        }


tool = UncertaintyQuantifyTool()
