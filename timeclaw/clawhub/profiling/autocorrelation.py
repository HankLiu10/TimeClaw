"""Autocorrelation and partial autocorrelation analysis tool."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class AutocorrelationTool(ClawTool):
    name = "autocorrelation"
    category = ToolCategory.PROFILING
    description = (
        "Compute autocorrelation function (ACF) and/or partial autocorrelation function (PACF) "
        "at specified lags. Useful for identifying temporal dependencies and model order selection."
    )
    io_type = ToolIOType.X_TO_R
    parameters_schema = {
        "type": "object",
        "properties": {
            "max_lag": {"type": "integer", "description": "Maximum lag to compute", "default": 20},
            "mode": {
                "type": "string",
                "enum": ["acf", "pacf", "both"],
                "description": "Which function(s) to compute",
                "default": "both",
            },
        },
        "required": [],
    }

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        max_lag = params.get("max_lag", 20)
        mode = params.get("mode", "both")

        arr = np.asarray(ts_data, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        max_lag = min(max_lag, n - 1)

        result: dict[str, Any] = {}

        if mode in ("acf", "both"):
            mean = np.mean(arr)
            var = np.var(arr)
            acf_vals = []
            for lag in range(max_lag + 1):
                if var == 0:
                    acf_vals.append(1.0 if lag == 0 else 0.0)
                else:
                    c = np.mean((arr[: n - lag] - mean) * (arr[lag:] - mean))
                    acf_vals.append(c / var)
            result["acf"] = acf_vals
            result["acf_lags"] = list(range(max_lag + 1))

        if mode in ("pacf", "both"):
            try:
                from statsmodels.tsa.stattools import pacf as sm_pacf

                pacf_vals = sm_pacf(arr, nlags=max_lag).tolist()
            except ImportError:
                pacf_vals = [1.0] + [0.0] * max_lag
            result["pacf"] = pacf_vals
            result["pacf_lags"] = list(range(max_lag + 1))

        return result


tool = AutocorrelationTool()
