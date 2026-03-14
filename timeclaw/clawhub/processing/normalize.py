"""Normalize time series (z-score or min-max)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
    ToolResult,
)


class NormalizeTool(ClawTool):
    """Normalize time series using z-score or min-max scaling."""

    name: str = "normalize"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = "Normalize time series using z-score or min-max scaling."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["zscore", "minmax"],
                "description": "Normalization method: 'zscore' or 'minmax'",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    _METHOD_ALIASES: dict[str, str] = {
        "zscore": "zscore", "z-score": "zscore", "z_score": "zscore",
        "standard": "zscore", "standardize": "zscore", "standardization": "zscore",
        "minmax": "minmax", "min-max": "minmax", "min_max": "minmax",
        "normalize": "zscore",
    }

    def _resolve_method(self, raw: str) -> str:
        return self._METHOD_ALIASES.get(raw.lower().strip(), "zscore")

    async def execute(
        self, ts_data: Any = None, *, text: Any = None, images: Any = None, **params: Any
    ) -> ToolResult:
        try:
            method = self._resolve_method(params.get("method", "zscore"))
            if ts_data is None:
                return ToolResult.fail("ts_data is required")

            if isinstance(ts_data, pd.DataFrame):
                arr = ts_data.values.astype(float)
            else:
                arr = np.asarray(ts_data, dtype=float)

            if arr.size == 0:
                return ToolResult.ok(arr)

            normalized, inv = self._normalize_with_inverse(arr, method)

            if isinstance(ts_data, pd.DataFrame):
                normalized = pd.DataFrame(normalized, index=ts_data.index, columns=ts_data.columns)

            return ToolResult(success=True, data=normalized, metadata={"inverse": inv, "method": method})
        except Exception as e:
            return ToolResult.fail(str(e))

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        raise NotImplementedError("Use execute() directly")

    def _normalize_with_inverse(self, arr: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
        arr = np.asarray(arr, dtype=float)
        if method == "zscore":
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
            if std == 0:
                std = 1.0
            return (arr - mean) / std, {"method": "zscore", "mean": mean, "std": std}
        if method == "minmax":
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            span = mx - mn
            if span == 0:
                span = 1.0
            return (arr - mn) / span, {"method": "minmax", "min": mn, "span": span}
        return self._normalize_with_inverse(arr, "zscore")

    @staticmethod
    def denormalize(values: Any, inv: dict) -> np.ndarray:
        """Reverse normalization using stored inverse params."""
        arr = np.asarray(values, dtype=float)
        if inv["method"] == "zscore":
            return arr * inv["std"] + inv["mean"]
        if inv["method"] == "minmax":
            return arr * inv["span"] + inv["min"]
        return arr


tool = NormalizeTool()
