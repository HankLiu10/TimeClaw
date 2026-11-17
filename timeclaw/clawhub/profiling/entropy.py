"""Entropy measures for time series: sample entropy, permutation entropy, spectral entropy."""

from __future__ import annotations

from typing import Any

import antropy
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


class EntropyTool(ClawTool):
    """Compute sample entropy, permutation entropy, and spectral entropy."""

    name: str = "entropy"
    category: ToolCategory = ToolCategory.PROFILING
    description: str = (
        "Measure the complexity and regularity of a time series via three entropy metrics. "
        "Low entropy indicates regular/predictable patterns; high entropy indicates chaotic/random behavior. "
        "Returns a dict with keys: sample_entropy, permutation_entropy, and spectral_entropy. "
        "Use to assess how predictable or complex a time series is before choosing forecasting methods. "
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

        samp_en = float(antropy.sample_entropy(arr))
        perm_en = float(antropy.perm_entropy(arr))
        spec_en = float(antropy.spectral_entropy(arr, sf=1.0))

        return {
            "sample_entropy": samp_en,
            "permutation_entropy": perm_en,
            "spectral_entropy": spec_en,
        }


tool = EntropyTool()
