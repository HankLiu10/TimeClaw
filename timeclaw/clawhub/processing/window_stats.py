"""Compute descriptive statistics over a contiguous time-series window."""

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
                    "window_stats only supports univariate series; use select_channels first for multivariate data"
                )
            return ts_data.iloc[:, 0].to_numpy(dtype=float), ts_data.index
        if isinstance(ts_data, pd.Series):
            return ts_data.to_numpy(dtype=float), ts_data.index
    except ImportError:
        pass

    arr = np.asarray(ts_data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            "window_stats only supports 1-D series; use select_channels first for multivariate data"
        )
    return arr, None


def _normalize_bound(bound: int, length: int) -> int:
    resolved = int(bound)
    if resolved < 0:
        resolved += length
    return min(max(resolved, 0), length)


class WindowStatsTool(ClawTool):
    """Compute statistics over a contiguous univariate time-series window."""

    name: str = "window_stats"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = (
        "Compute descriptive statistics for a contiguous window of a univariate time series. "
        "Input: a 1-D numeric array, pandas Series, or single-column DataFrame. "
        "Output: a dict containing the resolved half-open window [start, end), its length, "
        "endpoint values, min, max, mean, std, delta, and the absolute indices of the min/max "
        "points in the original series. Use this when you need the min/max over a recent range, "
        "the change across the last 24 points, or a quick summary of a forecast window before "
        "classification or answer selection. It always operates on the CURRENT input sequence "
        "it receives. If the input has already been sliced, resampled, normalized, or replaced "
        "by a forecast sequence, then start/end are relative to that current sequence rather "
        "than the original raw history. The state update may show `coordinate_system.origin_space` "
        "to explain provenance, but this tool's start/end arguments must still follow the current "
        "sequence's own coordinate space. "
        "Parameters (both required): start (integer start index), end (integer end index, exclusive). "
        "Negative indices follow Python slicing semantics. "
        "CRITICAL: after normalization, start must be strictly less than end; do not pass the same "
        "index for both bounds. Indices are always relative to the CURRENT sequence seen by this tool, "
        "not the original raw history length. For example, if you are summarizing a 72-step forecast, "
        "the first forecast day is start=0, end=24; the last 24 points are start=-24 and end=72. "
        "Another example: if a forecast artifact has current length 24 but provenance shows "
        "`origin_space=[144,168)`, you must still use start=0, end=24 here, not start=144, end=168."
    )
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "start": {
                "type": "integer",
                "description": "Inclusive start index of the window; negative values count from the end",
            },
            "end": {
                "type": "integer",
                "description": "Exclusive end index of the window; negative values count from the end",
            },
        },
        "required": ["start", "end"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr, ts_index = _extract_univariate(ts_data)
        length = len(arr)
        if length == 0:
            raise ValueError("ts_data is empty")

        requested_start = int(params["start"])
        requested_end = int(params["end"])
        resolved_start = _normalize_bound(requested_start, length)
        resolved_end = _normalize_bound(requested_end, length)

        if resolved_start >= resolved_end:
            example_msg = (
                f"Current sequence length is {length}. "
                "Use bounds relative to this current sequence. "
                "For example, the first 24 points are start=0, end=24; "
                "the last 24 points are start=-24, end=<current_length>."
            )
            raise ValueError(
                "window_stats requires start < end after normalization, got "
                f"{resolved_start} >= {resolved_end}. {example_msg}"
            )

        window = arr[resolved_start:resolved_end]
        local_argmin = int(np.argmin(window))
        local_argmax = int(np.argmax(window))
        absolute_argmin = resolved_start + local_argmin
        absolute_argmax = resolved_start + local_argmax

        result = {
            "start": requested_start,
            "end": requested_end,
            "resolved_start": resolved_start,
            "resolved_end": resolved_end,
            "length": int(len(window)),
            "start_value": float(window[0]),
            "end_value": float(window[-1]),
            "min": float(np.min(window)),
            "max": float(np.max(window)),
            "mean": float(np.mean(window)),
            "std": float(np.std(window)),
            "delta": float(window[-1] - window[0]),
            "argmin": absolute_argmin,
            "argmax": absolute_argmax,
        }
        if ts_index is not None:
            result["start_timestamp"] = str(ts_index[resolved_start])
            result["end_timestamp"] = str(ts_index[resolved_end - 1])
            result["argmin_timestamp"] = str(ts_index[absolute_argmin])
            result["argmax_timestamp"] = str(ts_index[absolute_argmax])
        return result


tool = WindowStatsTool()
