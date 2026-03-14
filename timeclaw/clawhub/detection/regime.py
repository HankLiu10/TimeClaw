"""Detect regime switches in time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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


class DetectRegimeTool(ClawTool):
    """Detect regime switches in time series."""

    name: str = "detect_regime"
    category: ToolCategory = ToolCategory.DETECTION
    description: str = "Detect regime switches in time series."
    io_type: ToolIOType = ToolIOType.X_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "window": {
                "type": "integer",
                "description": "Rolling window size for computing statistics.",
                "default": 30,
            },
        },
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        window = int(params.get("window", 30))
        n = len(arr)

        if n < window * 2:
            return {
                "n_regimes": 1,
                "regime_boundaries": [0, n],
                "regime_stats": [{"mean": float(arr.mean()), "std": float(arr.std())}],
            }

        rolling_mean = np.array([arr[i:i + window].mean() for i in range(n - window + 1)])
        rolling_std = np.array([arr[i:i + window].std() for i in range(n - window + 1)])

        global_mean = rolling_mean.mean()
        global_std_of_mean = rolling_mean.std()

        if global_std_of_mean == 0:
            return {
                "n_regimes": 1,
                "regime_boundaries": [0, n],
                "regime_stats": [{"mean": float(arr.mean()), "std": float(arr.std())}],
            }

        mean_shift = np.abs(np.diff(rolling_mean))
        threshold = 2.0 * global_std_of_mean
        shift_indices = np.where(mean_shift > threshold)[0]

        boundaries = [0]
        min_gap = window // 2
        for idx in shift_indices:
            actual_idx = idx + window // 2
            if actual_idx - boundaries[-1] >= min_gap:
                boundaries.append(int(actual_idx))
        boundaries.append(n)

        regime_stats = []
        for i in range(len(boundaries) - 1):
            segment = arr[boundaries[i]:boundaries[i + 1]]
            regime_stats.append({
                "mean": float(segment.mean()),
                "std": float(segment.std()),
            })

        return {
            "n_regimes": len(boundaries) - 1,
            "regime_boundaries": boundaries,
            "regime_stats": regime_stats,
        }


tool = DetectRegimeTool()
