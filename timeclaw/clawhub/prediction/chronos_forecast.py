"""Chronos-2 forecast tool — Amazon Chronos-2 foundation model."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType

_load_lock = threading.Lock()


def _run_chronos(ts_data: Any, horizon: int, num_samples: int = 20) -> dict[str, Any]:
    from timeclaw.sbm.registry import configured_default_config, get_global_registry

    registry = get_global_registry()
    if "chronos2" not in registry.list_available():
        try:
            from timeclaw.sbm.chronos_adapter import ChronosAdapter, CHRONOS_DEFAULT_CONFIG
            registry.register(
                "chronos2",
                ChronosAdapter,
                configured_default_config("chronos2", CHRONOS_DEFAULT_CONFIG),
            )
        except ImportError:
            raise RuntimeError("chronos-forecasting>=2.2 not installed")

    try:
        adapter = registry.get("chronos2")
    except KeyError:
        raise RuntimeError("chronos2 adapter not available")

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    if not adapter.is_loaded:
        with _load_lock:
            if not adapter.is_loaded:
                try:
                    adapter.load()
                    logger.info("[chronos2_forecast] model loaded")
                except Exception as e:
                    logger.error(f"[chronos2_forecast] load failed: {e}")
                    raise RuntimeError(f"chronos load failed: {e}") from e

    try:
        output = adapter.forecast(context, prediction_length=horizon, num_samples=num_samples)
        result: dict[str, Any] = {
            "model": str(output.metadata.get("model", "chronos-2")),
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist() if output.point_forecast is not None else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {str(q): v.tolist() for q, v in output.quantile_forecast.items()}
        return result
    except Exception as e:
        logger.error(f"[chronos2_forecast] inference failed: {e}")
        raise RuntimeError(f"chronos inference failed: {e}") from e


_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {"type": "integer", "default": 96, "description": "Number of steps to forecast."},
    },
    "required": [],
}


class ChronosForecastTool(ClawTool):
    name: str = "chronos2_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Amazon Chronos-2. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: strong zero-shot forecasting and direct quantile prediction; "
        "handles short to medium series well (50-500 points). Runs on GPU. "
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _run_chronos(ts_data, params.get("horizon", 96))


tool_chronos = ChronosForecastTool()
