"""Cointegration test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

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


class CointegrationTestTool(ClawTool):
    """Cointegration test between time series channels."""

    name: str = "cointegration_test"
    category: ToolCategory = ToolCategory.RELATION
    description: str = "Test cointegration between time series channels (e.g., Engle-Granger)."
    io_type: ToolIOType = ToolIOType.XK_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        data = _to_2d(ts_data)
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 columns for cointegration test")

        t_stat, p_value, crit_values = coint(data[:, 0], data[:, 1])

        return {
            "cointegrated": bool(p_value < 0.05),
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "critical_values": {
                "1%": float(crit_values[0]),
                "5%": float(crit_values[1]),
                "10%": float(crit_values[2]),
            },
        }


tool = CointegrationTestTool()
