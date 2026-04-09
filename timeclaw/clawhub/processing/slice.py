"""Slice time series by start/end index or timestamp."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class SliceTool(ClawTool):
    """Slice time series by start/end index or timestamp."""

    name: str = "slice"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = (
        "Extract a contiguous sub-sequence from a time series by specifying start and end "
        "positions. Input: a numeric array, DataFrame, or Series. Output: the sliced array "
        "or DataFrame retaining the original data type. Use this when you need a specific "
        "time range or index range from a longer series — e.g., isolating a region of interest "
        "before analysis or trimming leading/trailing artifacts. CRITICAL: the returned slice "
        "becomes a new shorter series with its own current coordinate space for downstream tools. "
        "If you pass that sliced output into `value_at`, `window_stats`, or another index-based "
        "tool, indices are interpreted in the sliced series's CURRENT space, not the original "
        "full series. The state update may also show how that current space maps back to the "
        "original input via `coordinate_system.origin_space`, but that mapping is for reference "
        "only; downstream index arguments still use the current sliced space. For example, after "
        "slicing the last 78 points, use index=0 for the first point of the slice and index=-1 "
        "for the last point. "
        "Parameters (both required): start (int index or timestamp str), end (int index exclusive "
        "or timestamp str)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "start": {
                "oneOf": [
                    {"type": "integer", "description": "Start index (0-based)"},
                    {"type": "string", "description": "Start timestamp (for pandas)"},
                ],
                "description": "Start index or timestamp",
            },
            "end": {
                "oneOf": [
                    {"type": "integer", "description": "End index (exclusive)"},
                    {"type": "string", "description": "End timestamp (for pandas)"},
                ],
                "description": "End index or timestamp",
            },
        },
        "required": ["start", "end"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        start = params.get("start")
        end = params.get("end")

        if ts_data is None:
            raise ValueError("ts_data is required")

        if isinstance(ts_data, pd.DataFrame):
            if isinstance(start, str) or isinstance(end, str):
                return ts_data.loc[start:end]
            return ts_data.iloc[start:end]

        arr = np.asarray(ts_data)
        return arr[start:end]


tool = SliceTool()
