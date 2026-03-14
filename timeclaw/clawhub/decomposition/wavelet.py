"""Wavelet transform for time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pywt

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


class WaveletTransformTool(ClawTool):
    """Wavelet transform decomposition."""

    name: str = "wavelet_transform"
    category: ToolCategory = ToolCategory.DECOMPOSITION
    description: str = "Decompose time series using wavelet transform (e.g., pywt)."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "wavelet": {"type": "string", "default": "db4"},
            "level": {"type": "integer", "default": None},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        wavelet = params.get("wavelet", "db4")
        level = params.get("level", None)

        coeffs = pywt.wavedec(arr, wavelet, level=level)
        levels = len(coeffs) - 1

        return {
            "coefficients": [c.tolist() for c in coeffs],
            "wavelet": wavelet,
            "levels": levels,
            "lengths": [len(c) for c in coeffs],
        }


tool = WaveletTransformTool()
