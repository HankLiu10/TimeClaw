"""Resample time series to different length or frequency."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class ResampleTool(ClawTool):
    """Resample time series to a different length or frequency."""

    name: str = "resample"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = "Resample time series to a different frequency (e.g. '5min', '1H') or target length."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "freq": {
                "type": "string",
                "description": "Target frequency for datetime-indexed pandas data (e.g. '5min', '1H', '1D')",
            },
            "target_length": {
                "type": "integer",
                "description": "Target number of samples for numeric resampling via scipy",
            },
        },
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        if ts_data is None:
            raise ValueError("ts_data is required")

        freq = params.get("freq")
        target_length = params.get("target_length")

        if isinstance(ts_data, (pd.DataFrame, pd.Series)) and freq and isinstance(ts_data.index, pd.DatetimeIndex):
            resampled = ts_data.resample(freq).mean()
            return {
                "data": resampled.values.tolist(),
                "original_length": len(ts_data),
                "target_length": len(resampled),
            }

        if target_length is None:
            raise ValueError("target_length is required for non-datetime resampling")

        target_length = int(target_length)
        if target_length < 1:
            raise ValueError("target_length must be >= 1")

        if isinstance(ts_data, pd.DataFrame):
            arr = ts_data.values.astype(float)
        elif isinstance(ts_data, pd.Series):
            arr = ts_data.values.astype(float)
        else:
            arr = np.asarray(ts_data, dtype=float)

        original_length = arr.shape[0]

        if arr.ndim == 1:
            resampled = scipy_resample(arr, target_length)
        else:
            resampled = scipy_resample(arr, target_length, axis=0)

        return {
            "data": resampled.tolist(),
            "original_length": original_length,
            "target_length": target_length,
        }


tool = ResampleTool()
