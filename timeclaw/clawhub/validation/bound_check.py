"""Bound check for time series values."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class BoundCheckTool(ClawTool):
    """Check if values are within specified bounds."""

    name: str = "bound_check"
    category: ToolCategory = ToolCategory.VALIDATION
    description: str = "Check if time series or prediction values are within lower/upper bounds."
    io_type: ToolIOType = ToolIOType.XP_TO_ACCEPT
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "lower": {
                "oneOf": [{"type": "number"}, {"type": "null"}],
                "description": "Lower bound (inclusive). None to skip.",
            },
            "upper": {
                "oneOf": [{"type": "number"}, {"type": "null"}],
                "description": "Upper bound (inclusive). None to skip.",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        lower = params.get("lower")
        upper = params.get("upper")

        arr = self._to_array(ts_data)
        violations = 0

        if lower is not None:
            violations += int(np.sum(arr < lower))
        if upper is not None:
            violations += int(np.sum(arr > upper))

        within_bounds = violations == 0
        return {
            "within_bounds": within_bounds,
            "violations": violations,
        }

    def _to_array(self, ts_data: Any) -> np.ndarray:
        if ts_data is None:
            raise ValueError("ts_data is required")
        if isinstance(ts_data, pd.DataFrame):
            return ts_data.values
        return np.asarray(ts_data, dtype=float)


tool = BoundCheckTool()
