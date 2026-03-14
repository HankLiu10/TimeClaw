"""Detect trend direction using linear regression slope."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import linregress

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class DetectTrendTool(ClawTool):
    """Detect trend direction using linear regression slope."""

    name: str = "detect_trend"
    category: ToolCategory = ToolCategory.DETECTION
    description: str = "Detect trend direction using linear regression slope. Returns direction (up/down/flat), slope, and p-value."
    io_type: ToolIOType = ToolIOType.X_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = np.asarray(ts_data)
        if arr.size < 2:
            return {"direction": "flat", "slope": 0.0, "p_value": 1.0}
        x = np.arange(len(arr), dtype=float)
        result = linregress(x, arr)
        slope = float(result.slope)
        p_value = float(result.pvalue)
        if p_value > 0.05:
            direction = "flat"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"
        return {"direction": direction, "slope": slope, "p_value": p_value}


tool = DetectTrendTool()
