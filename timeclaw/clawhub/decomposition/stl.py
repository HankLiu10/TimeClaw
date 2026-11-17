"""STL decomposition of time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class STLDecomposeTool(ClawTool):
    """STL decomposition using statsmodels."""

    name: str = "stl_decompose"
    category: ToolCategory = ToolCategory.DECOMPOSITION
    description: str = (
        "Decompose a time series into trend, seasonal, and residual components using STL "
        "(Seasonal-Trend decomposition using LOESS). Requires a known seasonal period; run "
        "detect_seasonality first if the period is unknown. Input: univariate or multivariate "
        "time series array/DataFrame. Output: dict with 'trend', 'seasonal', and 'residual' "
        "arrays (or a list of such dicts per channel for multivariate input). Use when you need "
        "to isolate a long-term trend from periodic fluctuations, remove seasonality before "
        "modeling, or examine residuals for anomalies. Key param: period (int, required) — the "
        "number of observations per seasonal cycle (e.g., 12 for monthly data with yearly "
        "seasonality)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "period": {
                "type": "integer",
                "description": "Period for seasonal component (e.g., 12 for monthly data with yearly seasonality)",
            },
        },
        "required": ["period"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        from statsmodels.tsa.seasonal import STL

        period = params.get("period")
        if period is None:
            raise ValueError("period is required")

        seasonal = self._ensure_odd(params.get("seasonal") or min(period - 1, 7))

        arr = self._to_array(ts_data)
        if arr.ndim > 1:
            results = []
            for i in range(arr.shape[1]):
                stl = STL(arr[:, i], period=period, seasonal=seasonal)
                res = stl.fit()
                results.append({
                    "trend": res.trend.tolist(),
                    "seasonal": res.seasonal.tolist(),
                    "residual": res.resid.tolist(),
                })
            return {"components": results}
        else:
            stl = STL(arr, period=period, seasonal=seasonal)
            res = stl.fit()
            return {
                "trend": res.trend.tolist(),
                "seasonal": res.seasonal.tolist(),
                "residual": res.resid.tolist(),
            }

    @staticmethod
    def _ensure_odd(val: int) -> int:
        """statsmodels STL requires seasonal to be an odd integer >= 3."""
        val = max(int(val), 3)
        return val if val % 2 == 1 else val + 1

    def _to_array(self, ts_data: Any) -> np.ndarray:
        if ts_data is None:
            raise ValueError("ts_data is required")
        if isinstance(ts_data, pd.DataFrame):
            return ts_data.values
        return np.asarray(ts_data, dtype=float)


tool = STLDecomposeTool()
