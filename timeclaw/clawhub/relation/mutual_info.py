"""Mutual information between time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_2d(ts_data):
    if ts_data is None:
        raise ValueError("ts_data is required")
    if isinstance(ts_data, pd.DataFrame):
        return ts_data.values.astype(float)
    arr = np.asarray(ts_data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class MutualInformationTool(ClawTool):
    """Compute mutual information between time series channels."""

    name: str = "mutual_information"
    category: ToolCategory = ToolCategory.RELATION
    description: str = (
        "Compute the mutual information (MI) between two time series channels, measuring "
        "general (including nonlinear) statistical dependence. Unlike correlation, MI captures "
        "any kind of relationship, not just linear or monotonic. Input: multivariate time "
        "series with at least 2 channels (uses the first two columns). Output: dict with "
        "'mutual_information' (raw MI in nats) and 'normalized_mi' (scaled to [0, 1]). Use "
        "when you suspect nonlinear dependencies that Pearson/Spearman correlation would miss, "
        "or when you need a single dependency score without assuming a relationship form. Key "
        "param: bins (int, default 20) — number of histogram bins for density estimation."
    )
    io_type: ToolIOType = ToolIOType.XK_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "bins": {"type": "integer", "default": 20},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        data = _to_2d(ts_data)
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 columns for mutual information")

        bins: int = int(params.get("bins", 20))
        x = data[:, 0]
        y = data[:, 1]

        hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        pxy = hist_xy / hist_xy.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        mi = 0.0
        for i in range(pxy.shape[0]):
            for j in range(pxy.shape[1]):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

        hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
        denom = min(hx, hy) if min(hx, hy) > 0 else 1.0
        normalized_mi = float(mi / denom)

        return {
            "mutual_information": float(mi),
            "normalized_mi": normalized_mi,
        }


tool = MutualInformationTool()
