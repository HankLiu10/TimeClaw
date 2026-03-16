"""TimesFM forecast tool — Google's decoder-only transformer foundation model."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType

_load_lock = threading.Lock()


def _statistical_fallback(ts_data: Any, horizon: int) -> dict[str, Any]:
    arr = np.asarray(ts_data, dtype=np.float32).ravel()
    if len(arr) < 2:
        forecast = np.full(horizon, arr[-1] if len(arr) > 0 else 0.0)
    else:
        trend = (arr[-1] - arr[-min(10, len(arr))]) / min(10, len(arr))
        forecast = arr[-1] + trend * np.arange(1, horizon + 1)
    return {"model": "timesfm_fallback", "horizon": horizon, "point_forecast": forecast.tolist()}


def _run_timesfm(ts_data: Any, horizon: int, num_samples: int = 20) -> dict[str, Any]:
    from timeclaw.sbm.registry import get_global_registry, auto_register

    registry = get_global_registry()
    if "timesfm" not in registry.list_available():
        try:
            from timeclaw.sbm.timesfm_adapter import TimesFMAdapter, TIMESFM_DEFAULT_CONFIG
            registry.register("timesfm", TimesFMAdapter, TIMESFM_DEFAULT_CONFIG)
        except ImportError:
            logger.warning("[timesfm_forecast] timesfm not installed")
            return _statistical_fallback(ts_data, horizon)

    try:
        adapter = registry.get("timesfm")
    except KeyError:
        logger.warning("[timesfm_forecast] adapter not available")
        return _statistical_fallback(ts_data, horizon)

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    if not adapter.is_loaded:
        with _load_lock:
            if not adapter.is_loaded:
                try:
                    adapter.load()
                    logger.info("[timesfm_forecast] model loaded")
                except Exception as e:
                    logger.error(f"[timesfm_forecast] load failed: {e}")
                    result = _statistical_fallback(ts_data, horizon)
                    result["model_error"] = f"load failed: {e}"
                    return result

    try:
        output = adapter.forecast(context, prediction_length=horizon, num_samples=num_samples)
        result: dict[str, Any] = {
            "model": "timesfm",
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist() if output.point_forecast is not None else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {str(q): v.tolist() for q, v in output.quantile_forecast.items()}
        return result
    except Exception as e:
        logger.error(f"[timesfm_forecast] inference failed: {e}")
        result = _statistical_fallback(ts_data, horizon)
        result["model_error"] = str(e)
        return result


_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {"type": "integer", "default": 96, "description": "Number of steps to forecast."},
    },
    "required": [],
}


class TimesFMForecastTool(ClawTool):
    name: str = "timesfm_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Google TimesFM, a decoder-only transformer foundation model "
        "pre-trained on 100 billion real-world time points. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: captures long-range dependencies and multi-scale patterns without fine-tuning; "
        "best for medium to long series (200+ points). Runs on GPU. "
        "Falls back to a linear-trend statistical forecast if the model server is unavailable. "
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _run_timesfm(ts_data, params.get("horizon", 96))


tool_timesfm = TimesFMForecastTool()
