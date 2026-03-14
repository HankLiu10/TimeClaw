"""Power spectral density for time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch

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


class SpectralDensityTool(ClawTool):
    """Compute power spectral density."""

    name: str = "spectral_density"
    category: ToolCategory = ToolCategory.PROFILING
    description: str = "Compute power spectral density of a time series."
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "nperseg": {
                "type": "integer",
                "description": "Length of each segment for Welch's method.",
                "default": 256,
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        nperseg = int(params.get("nperseg", 256))
        nperseg = min(nperseg, len(arr))

        freqs, psd = welch(arr, nperseg=nperseg)

        dominant_idx = int(np.argmax(psd))
        dominant_freq = float(freqs[dominant_idx])
        total_power = float(np.trapezoid(psd, freqs))

        return {
            "frequencies": freqs.tolist(),
            "psd": psd.tolist(),
            "dominant_frequency": dominant_freq,
            "total_power": total_power,
        }


tool = SpectralDensityTool()
