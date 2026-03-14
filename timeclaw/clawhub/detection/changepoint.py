"""Detect changepoints in time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import ruptures as rpt

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


class DetectChangepointTool(ClawTool):
    """Detect changepoints in time series."""

    name: str = "detect_changepoint"
    category: ToolCategory = ToolCategory.DETECTION
    description: str = "Detect changepoints in time series."
    io_type: ToolIOType = ToolIOType.X_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "penalty": {
                "type": "number",
                "description": "Penalty value for the PELT algorithm.",
                "default": 10,
            },
        },
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        penalty = float(params.get("penalty", 10))
        algo = rpt.Pelt(model="rbf").fit(arr.reshape(-1, 1))
        bkps = algo.predict(pen=penalty)
        changepoints = [int(b) for b in bkps[:-1]]
        return {
            "changepoints": changepoints,
            "n_changepoints": len(changepoints),
            "penalty": penalty,
        }


tool = DetectChangepointTool()
