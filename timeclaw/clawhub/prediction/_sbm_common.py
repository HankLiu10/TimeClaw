"""Shared logic for SBM foundation model forecast tools."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

_load_lock = threading.Lock()
_register_lock = threading.Lock()


def sbm_forecast(model_name: str, ts_data: Any, horizon: int, num_samples: int = 20) -> dict[str, Any]:
    """Call SBM registry for a specific model, with statistical fallback."""
    from timeclaw.sbm.registry import get_global_registry, auto_register

    registry = get_global_registry()
    with _register_lock:
        if not registry.list_available():
            auto_register()

    try:
        adapter = registry.get(model_name)
    except KeyError:
        logger.warning(f"[SBM] Model '{model_name}' not in registry, using fallback")
        return _statistical_fallback(ts_data, horizon, model_name)

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    if not adapter.is_loaded:
        with _load_lock:
            if not adapter.is_loaded:
                try:
                    adapter.load()
                    logger.info(f"[SBM] {model_name} loaded successfully")
                except Exception as e:
                    logger.error(f"[SBM] {model_name} load failed: {e}")
                    result = _statistical_fallback(ts_data, horizon, model_name)
                    result["sbm_error"] = f"load failed: {e}"
                    return result

    try:
        output = adapter.forecast(context, prediction_length=horizon, num_samples=num_samples)
        result: dict[str, Any] = {
            "model": model_name,
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist()
            if output.point_forecast is not None
            else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {
                str(q): v.tolist() for q, v in output.quantile_forecast.items()
            }
        return result
    except Exception as e:
        logger.error(f"[SBM] {model_name} forecast failed: {e}")
        result = _statistical_fallback(ts_data, horizon, model_name)
        result["sbm_error"] = str(e)
        return result


def _statistical_fallback(ts_data: Any, horizon: int, model_name: str) -> dict[str, Any]:
    arr = np.asarray(ts_data, dtype=np.float32).ravel()
    if len(arr) < 2:
        forecast = np.full(horizon, arr[-1] if len(arr) > 0 else 0.0)
    else:
        trend = (arr[-1] - arr[-min(10, len(arr))]) / min(10, len(arr))
        forecast = arr[-1] + trend * np.arange(1, horizon + 1)
    return {
        "model": f"{model_name}_fallback",
        "horizon": horizon,
        "point_forecast": forecast.tolist(),
    }


PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {
            "type": "integer",
            "default": 96,
            "description": "Number of steps to forecast.",
        },
    },
    "required": [],
}
