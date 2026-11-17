"""Fit statistical distributions to time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)

DISTRIBUTIONS = [
    ("norm", stats.norm),
    ("expon", stats.expon),
    ("lognorm", stats.lognorm),
    ("gamma", stats.gamma),
    ("uniform", stats.uniform),
]


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


class DistributionFitTool(ClawTool):
    """Fit statistical distributions to time series data."""

    name: str = "distribution_fit"
    category: ToolCategory = ToolCategory.PROFILING
    description: str = (
        "Test which probability distribution best fits the data values of a time series. "
        "Candidates: normal, exponential, lognormal, gamma, uniform. "
        "Returns a dict with keys: best_fit (distribution name), best_p_value (float), "
        "and fits (list of dicts each with name, params, ks_statistic, p_value). "
        "Use to characterize the statistical distribution of observations, e.g. for anomaly detection "
        "thresholds or simulation. No parameters required."
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

        fits = []
        for name, dist in DISTRIBUTIONS:
            try:
                fit_params = dist.fit(arr)
                ks_stat, p_value = stats.kstest(arr, name, args=fit_params)
                fits.append({
                    "name": name,
                    "params": [float(p) for p in fit_params],
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                })
            except Exception:
                continue

        if not fits:
            raise ValueError("No distributions could be fitted")

        best = max(fits, key=lambda f: f["p_value"])

        return {
            "best_fit": best["name"],
            "fits": fits,
            "best_p_value": best["p_value"],
        }


tool = DistributionFitTool()
