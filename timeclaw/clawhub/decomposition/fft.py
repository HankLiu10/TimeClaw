"""FFT frequency spectrum analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import fft

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class FFTSpectrumTool(ClawTool):
    """FFT frequency spectrum analysis using scipy.fft."""

    name: str = "fft_spectrum"
    category: ToolCategory = ToolCategory.DECOMPOSITION
    description: str = "Compute FFT frequency spectrum (frequencies and magnitudes) of time series."
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = self._to_array(ts_data)
        if arr.ndim > 1:
            arr = arr[:, 0]  # Use first channel for spectrum
        n = len(arr)
        if n < 2:
            return {"frequencies": [], "magnitudes": []}

        # Compute FFT
        fft_vals = fft.fft(arr)
        fft_mag = np.abs(fft_vals[: n // 2 + 1])
        freqs = fft.fftfreq(n)[: n // 2 + 1]
        freqs = np.abs(freqs)

        return {
            "frequencies": freqs.tolist(),
            "magnitudes": fft_mag.tolist(),
        }

    def _to_array(self, ts_data: Any) -> np.ndarray:
        if ts_data is None:
            raise ValueError("ts_data is required")
        if isinstance(ts_data, pd.DataFrame):
            return ts_data.values
        return np.asarray(ts_data, dtype=float)


tool = FFTSpectrumTool()
