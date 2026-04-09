"""Stationarity testing tool (ADF and KPSS tests)."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class StationarityTool(ClawTool):
    name = "stationarity_test"
    category = ToolCategory.PROFILING
    description = (
        "Run stationarity tests on a time series. Supports Augmented Dickey-Fuller (ADF) "
        "and KPSS tests. ADF null hypothesis: series has a unit root (non-stationary). "
        "KPSS null hypothesis: series is stationary. "
        "Returns a dict with test statistics, p-values, critical values, and a stationarity "
        "conclusion ('stationary', 'non_stationary', 'trend_stationary', or 'difference_stationary'). "
        "Use to determine if differencing or detrending is needed before applying ARIMA or other "
        "stationary-assumption models. Key parameter: test ('adf', 'kpss', or 'both', default 'both')."
    )
    io_type = ToolIOType.X_TO_R
    parameters_schema = {
        "type": "object",
        "properties": {
            "test": {
                "type": "string",
                "enum": ["adf", "kpss", "both"],
                "description": "Which test(s) to run",
                "default": "both",
            },
        },
        "required": [],
    }

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        test = params.get("test", "both")
        arr = np.asarray(ts_data, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]

        if arr.size == 0:
            raise ValueError("stationarity_test requires at least one non-NaN value")

        if np.allclose(arr, arr[0]):
            constant_result = {
                "is_constant": True,
                "constant_value": float(arr[0]),
                "n_obs": int(arr.size),
                "conclusion": "constant_stationary",
            }
            if test in ("adf", "both"):
                constant_result["adf"] = {
                    "statistic": None,
                    "p_value": None,
                    "used_lag": 0,
                    "n_obs": int(arr.size),
                    "critical_values": {},
                    "is_stationary": True,
                    "note": "ADF is undefined for a constant series.",
                }
            if test in ("kpss", "both"):
                constant_result["kpss"] = {
                    "statistic": None,
                    "p_value": None,
                    "used_lag": 0,
                    "critical_values": {},
                    "is_stationary": True,
                    "note": "KPSS is undefined for a constant series.",
                }
            return constant_result

        result: dict[str, Any] = {}

        if test in ("adf", "both"):
            from statsmodels.tsa.stattools import adfuller

            adf_result = adfuller(arr, autolag="AIC")
            result["adf"] = {
                "statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "used_lag": int(adf_result[2]),
                "n_obs": int(adf_result[3]),
                "critical_values": {k: float(v) for k, v in adf_result[4].items()},
                "is_stationary": adf_result[1] < 0.05,
            }

        if test in ("kpss", "both"):
            from statsmodels.tsa.stattools import kpss

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(arr, regression="c", nlags="auto")
            result["kpss"] = {
                "statistic": float(kpss_result[0]),
                "p_value": float(kpss_result[1]),
                "used_lag": int(kpss_result[2]),
                "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
                "is_stationary": kpss_result[1] > 0.05,
            }

        if "adf" in result and "kpss" in result:
            result["conclusion"] = self._interpret(
                result["adf"]["is_stationary"], result["kpss"]["is_stationary"]
            )

        return result

    @staticmethod
    def _interpret(adf_stationary: bool, kpss_stationary: bool) -> str:
        if adf_stationary and kpss_stationary:
            return "stationary"
        elif not adf_stationary and not kpss_stationary:
            return "non_stationary"
        elif adf_stationary and not kpss_stationary:
            return "trend_stationary"
        else:
            return "difference_stationary"


tool = StationarityTool()
