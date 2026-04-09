"""Moirai 1.1 R-base forecast tool — Salesforce universal forecasting model."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType

_load_lock = threading.Lock()


def _run_moirai(ts_data: Any, horizon: int, num_samples: int = 20) -> dict[str, Any]:
    from timeclaw.sbm.registry import configured_default_config, get_global_registry
    from timeclaw.sbm.moirai_adapter import (
        MoiraiAdapter,
        MOIRAI_DEFAULT_CONFIG,
        _ensure_moirai_imports,
    )

    registry = get_global_registry()
    if "moirai1_1_r_base" not in registry.list_available():
        try:
            _ensure_moirai_imports()
            registry.register(
                "moirai1_1_r_base",
                MoiraiAdapter,
                configured_default_config("moirai1_1_r_base", MOIRAI_DEFAULT_CONFIG),
            )
        except ImportError as e:
            raise RuntimeError(f"moirai import unavailable: {e}") from e

    try:
        adapter = registry.get("moirai1_1_r_base")
    except KeyError:
        raise RuntimeError("moirai1_1_r_base adapter not available")

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    if not adapter.is_loaded:
        with _load_lock:
            if not adapter.is_loaded:
                try:
                    adapter.load()
                    logger.info("[moirai1_1_r_base_forecast] model loaded")
                except Exception as e:
                    logger.warning(f"[moirai1_1_r_base_forecast] first load failed: {e}")
                    # Some runs showed transient false-negative import errors
                    # under concurrent startup. Re-check imports and retry once.
                    try:
                        _ensure_moirai_imports()
                        adapter.load()
                        logger.info("[moirai1_1_r_base_forecast] model loaded on retry")
                    except Exception as retry_e:
                        logger.error(f"[moirai1_1_r_base_forecast] load failed: {retry_e}")
                        raise RuntimeError(f"moirai load failed: {retry_e}") from retry_e

    try:
        output = adapter.forecast(context, prediction_length=horizon, num_samples=num_samples)
        result: dict[str, Any] = {
            "model": "Salesforce/moirai-1.1-R-base",
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist() if output.point_forecast is not None else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {str(q): v.tolist() for q, v in output.quantile_forecast.items()}
        return result
    except Exception as e:
        logger.error(f"[moirai1_1_r_base_forecast] inference failed: {e}")
        raise RuntimeError(f"moirai inference failed: {e}") from e


_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {"type": "integer", "default": 96, "description": "Number of steps to forecast."},
    },
    "required": [],
}


class MoiraiForecastTool(ClawTool):
    name: str = "moirai1_1_r_base_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Salesforce Moirai 1.1 R-base, "
        "a masked encoder-based universal forecasting model "
        "trained on 27 billion observations across 9 domains with any-variate attention. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: strong at multi-step forecasting, adapts to varying frequencies without fine-tuning; "
        "best for medium to long series (100+ points). Runs on GPU. "
        "Fails explicitly if Moirai 1.1 R-base is unavailable. "
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _run_moirai(ts_data, params.get("horizon", 96))


tool_moirai = MoiraiForecastTool()
