"""Singular spectrum analysis."""

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


class SSADecomposeTool(ClawTool):
    """Singular spectrum analysis decomposition."""

    name: str = "ssa_decompose"
    category: ToolCategory = ToolCategory.DECOMPOSITION
    description: str = "Decompose time series using singular spectrum analysis (SSA)."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "window_size": {"type": "integer", "default": None},
            "n_components": {"type": "integer", "default": 3},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        n = len(arr)
        window_size = params.get("window_size") or min(n // 2, 50)
        n_components = params.get("n_components", 3)

        if window_size < 2 or window_size >= n:
            raise ValueError(f"window_size must be in [2, {n - 1}], got {window_size}")

        k = n - window_size + 1
        trajectory = np.column_stack([arr[i : i + window_size] for i in range(k)])

        U, s, Vt = np.linalg.svd(trajectory, full_matrices=False)

        n_components = min(n_components, len(s))

        components = []
        for i in range(n_components):
            Xi = s[i] * np.outer(U[:, i], Vt[i, :])
            rec = np.zeros(n)
            counts = np.zeros(n)
            for j in range(k):
                rec[j : j + window_size] += Xi[:, j]
                counts[j : j + window_size] += 1.0
            rec /= counts
            components.append(rec.tolist())

        total_var = np.sum(s ** 2)
        explained = (s ** 2 / total_var).tolist() if total_var > 0 else [0.0] * len(s)

        return {
            "components": components,
            "singular_values": s.tolist(),
            "explained_variance_ratio": explained,
            "n_components": n_components,
        }


tool = SSADecomposeTool()
