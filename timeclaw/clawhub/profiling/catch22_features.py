"""Catch22 feature set extraction for time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pycatch22

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_array(ts_data):
    if ts_data is None:
        raise ValueError("ts_data is required")
    if isinstance(ts_data, pd.DataFrame):
        arr = ts_data.iloc[:, 0].values
    elif isinstance(ts_data, pd.Series):
        arr = ts_data.values
    else:
        arr = np.asarray(ts_data).ravel()
    return arr.astype(float)


class Catch22Tool(ClawTool):
    """Extract catch22 feature set from time series."""

    name: str = "catch22_features"
    category: ToolCategory = ToolCategory.PROFILING
    description: str = "Extract the catch22 feature set from a time series."
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        result = pycatch22.catch22_all(arr.tolist())
        features = dict(zip(result["names"], result["values"]))
        return {
            "features": features,
            "count": 22,
        }


tool = Catch22Tool()
