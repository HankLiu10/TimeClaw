"""Hurst exponent for long-range dependence in time series."""

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


class HurstExponentTool(ClawTool):
    """Compute Hurst exponent for long-range dependence."""

    name: str = "hurst_exponent"
    category: ToolCategory = ToolCategory.PROFILING
    description: str = "Compute Hurst exponent for long-range dependence in time series."
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        n = len(arr)
        if n < 20:
            raise ValueError("Time series too short for Hurst estimation (need >= 20)")

        max_k = n // 2
        sizes = []
        rs_values = []

        for exp in np.arange(1, int(np.log2(max_k)) + 1, 0.25):
            size = int(2 ** exp)
            if size < 10 or size > max_k:
                continue
            num_chunks = n // size
            if num_chunks < 1:
                continue

            rs_chunk = []
            for i in range(num_chunks):
                chunk = arr[i * size : (i + 1) * size]
                mean = np.mean(chunk)
                deviations = chunk - mean
                cumulative = np.cumsum(deviations)
                r = np.max(cumulative) - np.min(cumulative)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_chunk.append(r / s)

            if rs_chunk:
                sizes.append(size)
                rs_values.append(np.mean(rs_chunk))

        if len(sizes) < 2:
            raise ValueError("Not enough valid window sizes for regression")

        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)
        coeffs = np.polyfit(log_sizes, log_rs, 1)
        hurst = float(coeffs[0])

        if hurst < 0.45:
            interpretation = "mean-reverting"
        elif hurst > 0.55:
            interpretation = "trending"
        else:
            interpretation = "random walk"

        return {
            "hurst": hurst,
            "interpretation": interpretation,
        }


tool = HurstExponentTool()
