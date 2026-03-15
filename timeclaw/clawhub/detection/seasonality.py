"""Detect seasonality period in time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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


def _autocorrelation(arr: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(arr)
    x = arr - arr.mean()
    var = np.sum(x ** 2)
    if var == 0:
        return np.zeros(max_lag)
    acf = np.correlate(x, x, mode="full")
    acf = acf[n - 1: n - 1 + max_lag] / var
    return acf


class DetectSeasonalityTool(ClawTool):
    """Detect seasonality period in time series."""

    name: str = "detect_seasonality"
    category: ToolCategory = ToolCategory.DETECTION
    description: str = (
        "Detect the dominant seasonality period in a time series by analyzing autocorrelation function (ACF) peaks. "
        "Input: 1D numeric array, Series, or single-column DataFrame. "
        "Output: dict with 'seasonal' (bool), 'period' (int or null — the dominant cycle length in samples), "
        "'autocorrelation_peaks' (list of lag indices), and 'peak_strengths' (list of ACF values). "
        "Use as a prerequisite step before STL decomposition, seasonal ARIMA, or any method requiring a known period. "
        "No required parameters."
    )
    io_type: ToolIOType = ToolIOType.X_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        max_lag = min(len(arr) // 2, len(arr) - 1)
        if max_lag < 3:
            return {
                "seasonal": False,
                "period": None,
                "autocorrelation_peaks": [],
                "peak_strengths": [],
            }

        acf = _autocorrelation(arr, max_lag)
        peak_indices, properties = find_peaks(acf[1:], height=0.1, distance=2)
        peak_indices = peak_indices + 1

        if len(peak_indices) == 0:
            return {
                "seasonal": False,
                "period": None,
                "autocorrelation_peaks": [],
                "peak_strengths": [],
            }

        peak_strengths = [float(acf[p]) for p in peak_indices]
        best = int(np.argmax(peak_strengths))
        period = int(peak_indices[best])
        seasonal = peak_strengths[best] > 0.2

        return {
            "seasonal": seasonal,
            "period": period if seasonal else None,
            "autocorrelation_peaks": [int(p) for p in peak_indices],
            "peak_strengths": peak_strengths,
        }


tool = DetectSeasonalityTool()
