"""Retrieve a single value from a univariate time series by index."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _extract_univariate(ts_data: Any) -> tuple[np.ndarray, Any | None]:
    if ts_data is None:
        raise ValueError("ts_data is required")

    try:
        import pandas as pd

        if isinstance(ts_data, pd.DataFrame):
            if ts_data.shape[1] != 1:
                raise ValueError(
                    "value_at only supports univariate series; use select_channels first for multivariate data"
                )
            return ts_data.iloc[:, 0].to_numpy(dtype=float), ts_data.index
        if isinstance(ts_data, pd.Series):
            return ts_data.to_numpy(dtype=float), ts_data.index
    except ImportError:
        pass

    arr = np.asarray(ts_data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            "value_at only supports 1-D series; use select_channels first for multivariate data"
        )
    return arr, None


def _normalize_index(index: int, length: int) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved += length
    if resolved < 0 or resolved >= length:
        raise IndexError(f"index {index} is out of range for series length {length}")
    return resolved


class ValueAtTool(ClawTool):
    """Retrieve a single point value from a univariate time series."""

    name: str = "value_at"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = (
        "Retrieve the value at one specific position in a univariate time series. "
        "Input: a 1-D numeric array, pandas Series, or single-column DataFrame. "
        "Output: a dict with the requested value, the original index argument, the resolved "
        "0-based index, and the total series length. Use this when a question asks for the "
        "value at a specific point, the latest point (index=-1), the first forecast point, "
        "or an endpoint value before judging trend or comparing windows. CRITICAL: the index is "
        "always interpreted relative to the series currently passed into this tool. If the input "
        "comes from `slice`, `segment`, `resample`, or a forecast tool, use the CURRENT coordinate "
        "space shown in the latest state update. If that state update also includes "
        "`coordinate_system.origin_space`, treat it as provenance only, not as the index space for "
        "this tool call. Use index=0 for the first point of the current input and index=-1 for the "
        "last point. Parameter (required): index (integer, supports negative indexing like Python)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "0-based position within the current input series; negative values count from the end, e.g. -1 for the last point",
            }
        },
        "required": ["index"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr, ts_index = _extract_univariate(ts_data)
        length = len(arr)
        if length == 0:
            raise ValueError("ts_data is empty")

        requested_index = int(params["index"])
        resolved_index = _normalize_index(requested_index, length)

        result = {
            "index": requested_index,
            "resolved_index": resolved_index,
            "length": length,
            "value": float(arr[resolved_index]),
        }
        if ts_index is not None:
            result["timestamp"] = str(ts_index[resolved_index])
        return result


tool = ValueAtTool()
