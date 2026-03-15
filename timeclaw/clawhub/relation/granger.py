"""Granger causality test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_2d(ts_data):
    if ts_data is None:
        raise ValueError("ts_data is required")
    if isinstance(ts_data, pd.DataFrame):
        return ts_data.values.astype(float)
    arr = np.asarray(ts_data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class GrangerCausalityTool(ClawTool):
    """Granger causality test between time series channels."""

    name: str = "granger_causality"
    category: ToolCategory = ToolCategory.RELATION
    description: str = (
        "Test Granger causality between two time series channels — whether past values of one "
        "channel significantly improve predictions of the other beyond its own past (directional "
        "predictive relationship). Input: multivariate time series with at least 2 channels "
        "(tests the first two columns). Output: dict with 'results' (list of per-lag dicts "
        "containing 'lag', 'f_statistic', and 'p_value'), 'significant_lags' (lags where "
        "p < 0.05), and 'max_lag'. Use when you need to establish whether one variable "
        "predictively leads another, e.g., determining if economic indicator X forecasts "
        "variable Y. Key param: max_lag (int, default 4) — maximum number of lags to test."
    )
    io_type: ToolIOType = ToolIOType.XK_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "max_lag": {"type": "integer", "default": 4},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        data = _to_2d(ts_data)
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 columns for Granger causality test")

        max_lag: int = int(params.get("max_lag", 4))
        pair = data[:, :2]

        try:
            raw = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
        except Exception as exc:
            raise ValueError(f"Granger test failed: {exc}") from exc

        results = []
        significant_lags = []
        for lag in range(1, max_lag + 1):
            test_result = raw[lag]
            f_stat = float(test_result[0]["ssr_ftest"][0])
            p_val = float(test_result[0]["ssr_ftest"][1])
            results.append({"lag": lag, "f_statistic": f_stat, "p_value": p_val})
            if p_val < 0.05:
                significant_lags.append(lag)

        return {
            "results": results,
            "significant_lags": significant_lags,
            "max_lag": max_lag,
        }


tool = GrangerCausalityTool()
