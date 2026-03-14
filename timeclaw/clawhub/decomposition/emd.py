"""Empirical mode decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PyEMD import EMD

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


class EMDDecomposeTool(ClawTool):
    """Empirical mode decomposition."""

    name: str = "emd_decompose"
    category: ToolCategory = ToolCategory.DECOMPOSITION
    description: str = "Decompose time series into intrinsic mode functions (IMFs) using EMD."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)

        emd = EMD()
        imfs = emd.emd(arr)

        return {
            "imfs": [imf.tolist() for imf in imfs],
            "n_imfs": len(imfs),
            "residue": imfs[-1].tolist(),
        }


tool = EMDDecomposeTool()
