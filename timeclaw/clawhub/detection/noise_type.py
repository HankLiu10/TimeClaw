"""Classify noise type (white/red/pink) in time series."""

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


def _classify_slope(slope: float) -> str:
    if slope > -0.5:
        return "white"
    elif slope > -1.5:
        return "pink"
    else:
        return "brown"


class ClassifyNoiseTypeTool(ClawTool):
    """Classify noise type (white/red/pink) in time series."""

    name: str = "classify_noise_type"
    category: ToolCategory = ToolCategory.DETECTION
    description: str = (
        "Classify the noise type of a time series by analyzing its spectral slope in log-log space. "
        "White noise (~0 slope) indicates independent random values; pink noise (~-1 slope, 1/f) shows "
        "long-range correlations; brown/red noise (~-2 slope) is autocorrelated and slow-varying. "
        "Input: 1D numeric array, Series, or single-column DataFrame. "
        "Output: dict with 'noise_type' ('white', 'pink', or 'brown'), 'spectral_slope' (float), "
        "and 'r_squared' (goodness of fit). "
        "Use to understand the underlying stochastic process or to select appropriate models and filters. "
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
        n = len(arr)
        fft_vals = np.fft.rfft(arr - arr.mean())
        psd = np.abs(fft_vals) ** 2 / n
        freqs = np.fft.rfftfreq(n)

        mask = freqs > 0
        log_freq = np.log10(freqs[mask])
        log_psd = np.log10(psd[mask] + 1e-30)

        slope, intercept = np.polyfit(log_freq, log_psd, 1)
        predicted = slope * log_freq + intercept
        ss_res = np.sum((log_psd - predicted) ** 2)
        ss_tot = np.sum((log_psd - log_psd.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "noise_type": _classify_slope(slope),
            "spectral_slope": float(slope),
            "r_squared": float(r_squared),
        }


tool = ClassifyNoiseTypeTool()
