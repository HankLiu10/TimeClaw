"""Basic statistics for time series: mean, std, min, max, skewness, kurtosis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_array(ts_data: Any) -> np.ndarray:
    """Extract 1D array from ts_data (numpy, pandas, or list)."""
    if ts_data is None:
        raise ValueError("ts_data is required")
    if isinstance(ts_data, pd.DataFrame):
        arr = ts_data.iloc[:, 0].values
    elif isinstance(ts_data, pd.Series):
        arr = ts_data.values
    else:
        arr = np.asarray(ts_data).ravel()
    return arr.astype(float)


class BasicStatsTool(ClawTool):
    """Compute mean, std, min, max, skewness, kurtosis for a time series."""

    name: str = "basic_stats"
    category: ToolCategory = ToolCategory.PROFILING
    description: str = (
        "Compute basic statistics for a 1D numeric time series. "
        "Returns a dict with keys: mean, std, min, max, skewness, kurtosis. "
        "Use as a first step to understand data distribution, central tendency, and spread. "
        "Accepts any 1D numeric array (numpy, pandas Series/DataFrame, or list). "
        "No parameters required."
    )
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            raise ValueError("No valid (non-NaN) data points")
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "skewness": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr)),
        }


tool = BasicStatsTool()
