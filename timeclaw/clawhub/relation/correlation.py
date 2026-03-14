"""Correlation matrix computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class CorrelationMatrixTool(ClawTool):
    """Compute correlation matrix between multiple time series channels."""

    name: str = "correlation_matrix"
    category: ToolCategory = ToolCategory.RELATION
    description: str = "Compute correlation matrix between channels (pearson, spearman, or kendall)."
    io_type: ToolIOType = ToolIOType.XK_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["pearson", "spearman", "kendall"],
                "description": "Correlation method: pearson, spearman, or kendall",
            },
        },
        "required": ["method"],
    }
    is_stub: bool = False

    _METHOD_ALIASES: dict[str, str] = {
        "pearson": "pearson", "spearman": "spearman", "kendall": "kendall",
        "pearson_r": "pearson", "spearman_rho": "spearman", "kendall_tau": "kendall",
        "linear": "pearson", "rank": "spearman",
    }

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        raw = params.get("method", "pearson")
        method = self._METHOD_ALIASES.get(str(raw).lower().strip(), "pearson")

        df = self._to_dataframe(ts_data)
        if df.shape[1] < 2:
            raise ValueError("ts_data must have at least 2 channels for correlation matrix")

        corr = df.corr(method=method)
        return {
            "matrix": corr.values.tolist(),
            "columns": list(corr.columns),
            "method": method,
        }

    def _to_dataframe(self, ts_data: Any) -> pd.DataFrame:
        if ts_data is None:
            raise ValueError("ts_data is required")
        if isinstance(ts_data, pd.DataFrame):
            return ts_data
        arr = np.asarray(ts_data)
        if arr.ndim == 1:
            raise ValueError("ts_data must have multiple channels (columns) for XK_TO_R")
        return pd.DataFrame(arr)


tool = CorrelationMatrixTool()
