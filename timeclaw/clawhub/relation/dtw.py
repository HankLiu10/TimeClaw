"""Dynamic time warping distance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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


class DTWDistanceTool(ClawTool):
    """Compute DTW distance between time series channels."""

    name: str = "dtw_distance"
    category: ToolCategory = ToolCategory.RELATION
    description: str = "Compute dynamic time warping (DTW) distance between channels."
    io_type: ToolIOType = ToolIOType.XK_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        data = _to_2d(ts_data)
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 columns for DTW distance")

        s = data[:, 0]
        t = data[:, 1]
        n, m = len(s), len(t)

        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = abs(s[i - 1] - t[j - 1])
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        distance = float(cost[n, m])

        i, j = n, m
        path_length = 1
        while i > 1 or j > 1:
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                candidates = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
                argmin = int(np.argmin(candidates))
                if argmin == 0:
                    i -= 1
                    j -= 1
                elif argmin == 1:
                    i -= 1
                else:
                    j -= 1
            path_length += 1

        return {
            "distance": distance,
            "normalized_distance": distance / path_length,
            "path_length": path_length,
        }


tool = DTWDistanceTool()
