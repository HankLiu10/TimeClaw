"""Segment time series using a sliding window."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_array(ts_data: Any) -> np.ndarray:
    if ts_data is None:
        raise ValueError("ts_data is required")
    try:
        import pandas as pd
        if isinstance(ts_data, pd.DataFrame):
            return ts_data.values.astype(float)
        if isinstance(ts_data, pd.Series):
            return ts_data.values.astype(float)
    except ImportError:
        pass
    return np.asarray(ts_data, dtype=float)


class SegmentTool(ClawTool):
    """Segment time series into fixed-size windows with a sliding step."""

    name: str = "segment"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = (
        "Segment a time series into fixed-size windows using a sliding-window approach, "
        "producing overlapping or non-overlapping sub-arrays. "
        "Input: a 1-D numeric array (or DataFrame/Series flattened to 1-D). "
        "Output: dict with 'segments' (list of sub-arrays as lists), 'num_segments', "
        "'window_size', and 'step_size'. Use this for batch analysis, pattern mining, or "
        "preparing fixed-length input chunks for models that require uniform-length sequences. "
        "Parameters: window_size (int, default 100), step_size (int, default 50 — set equal "
        "to window_size for non-overlapping segments)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "window_size": {
                "type": "integer",
                "description": "Size of each segment window (default 100)",
            },
            "step_size": {
                "type": "integer",
                "description": "Step size between consecutive windows (default 50)",
            },
        },
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        if arr.ndim > 1:
            arr = arr.ravel()

        window_size = int(params.get("window_size", 100))
        step_size = int(params.get("step_size", 50))

        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")

        n = len(arr)
        segments = []
        start = 0
        while start + window_size <= n:
            segments.append(arr[start:start + window_size].tolist())
            start += step_size

        return {
            "segments": segments,
            "num_segments": len(segments),
            "window_size": window_size,
            "step_size": step_size,
        }


tool = SegmentTool()
