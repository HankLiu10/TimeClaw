"""SBM Model Server: serves all time-series foundation models via HTTP.

Loads Chronos-2, Moirai 1.1 R-base, TimesFM 2.0, Sundial Base 128M, and optionally Aurora onto GPU at startup.
Exposes /forecast endpoint for high-throughput concurrent access.

Usage:
    python -m timeclaw.sbm.server                  # default port 8100
    python -m timeclaw.sbm.server --port 8100 --gpu 0
    python -m timeclaw.sbm.server --models chronos2,sundial_base_128m
"""

from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from timeclaw.sbm.base import SBMConfig
from timeclaw.sbm.registry import SBMRegistry

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "chronos2": {
        "cls": "timeclaw.sbm.chronos_adapter.ChronosAdapter",
        "default_cfg": "timeclaw.sbm.chronos_adapter.CHRONOS_DEFAULT_CONFIG",
    },
    "moirai1_1_r_base": {
        "cls": "timeclaw.sbm.moirai_adapter.MoiraiAdapter",
        "default_cfg": "timeclaw.sbm.moirai_adapter.MOIRAI_DEFAULT_CONFIG",
    },
    "timesfm2": {
        "cls": "timeclaw.sbm.timesfm_adapter.TimesFMAdapter",
        "default_cfg": "timeclaw.sbm.timesfm_adapter.TIMESFM_DEFAULT_CONFIG",
    },
    "sundial_base_128m": {
        "cls": "timeclaw.sbm.timer_adapter.TimerAdapter",
        "default_cfg": "timeclaw.sbm.timer_adapter.TIMER_DEFAULT_CONFIG",
    },
    "aurora": {
        "cls": "timeclaw.sbm.aurora_adapter.AuroraAdapter",
        "default_cfg": "timeclaw.sbm.aurora_adapter.AURORA_DEFAULT_CONFIG",
    },
}

registry = SBMRegistry()


def _import(dotted_path: str) -> Any:
    module_path, attr = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _parse_device_map(raw: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in str(raw or "").split(","):
        piece = chunk.strip()
        if not piece or "=" not in piece:
            continue
        name, device = piece.split("=", 1)
        name = name.strip()
        device = device.strip()
        if name and device:
            mapping[name] = device
    return mapping


def _device_for_model(
    name: str,
    *,
    default_device: str,
    device_map: dict[str, str] | None,
) -> str:
    return str((device_map or {}).get(name) or default_device)


def load_models(
    model_names: list[str],
    *,
    default_device: str = "cuda",
    device_map: dict[str, str] | None = None,
) -> None:
    for name in model_names:
        if name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {name}, skipping")
            continue
        info = MODEL_CONFIGS[name]
        try:
            adapter_cls = _import(info["cls"])
            default_cfg: SBMConfig = _import(info["default_cfg"])
            device = _device_for_model(
                name,
                default_device=default_device,
                device_map=device_map,
            )
            cfg = default_cfg.model_copy(update={"device": device})
            registry.register(name, adapter_cls, cfg)
            logger.info(f"[SBM-Server] Loading {name} onto {device}...")
            t0 = time.time()
            adapter = registry.get(name)
            adapter.load()
            elapsed = time.time() - t0
            logger.info(f"[SBM-Server] {name} loaded in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"[SBM-Server] Failed to load {name}: {e}")


class ForecastRequest(BaseModel):
    model: str = "auto"
    context: list[float]
    prediction_length: int = 96
    num_samples: int = 20


class ForecastResponse(BaseModel):
    model: str
    point_forecast: list[float]
    quantiles: dict[str, list[float]] | None = None
    elapsed_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    models_available: list[str]
    model_devices: dict[str, str] = Field(default_factory=dict)


_thread_pool: ThreadPoolExecutor | None = None


def create_app(
    model_names: list[str],
    *,
    default_device: str = "cuda",
    device_map: dict[str, str] | None = None,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _thread_pool
        _thread_pool = ThreadPoolExecutor(max_workers=len(model_names))
        load_models(
            model_names,
            default_device=default_device,
            device_map=device_map,
        )
        logger.info(
            f"[SBM-Server] Ready: {registry.list_loaded()} "
            f"(default_device={default_device}, device_map={device_map or {}})"
        )
        yield
        registry.unload_all()
        if _thread_pool:
            _thread_pool.shutdown(wait=False)
        logger.info("[SBM-Server] All models unloaded")

    app = FastAPI(
        title="TimeClaw SBM Model Server",
        description="Serves time-series foundation models (Chronos-2, Moirai 1.1 R-base, TimesFM 2.0, Sundial Base 128M, Aurora)",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        model_devices = {
            name: _device_for_model(
                name,
                default_device=default_device,
                device_map=device_map,
            )
            for name in registry.list_loaded()
        }
        return HealthResponse(
            status="ok",
            models_loaded=registry.list_loaded(),
            models_available=registry.list_available(),
            model_devices=model_devices,
        )

    @app.post("/forecast", response_model=ForecastResponse)
    async def forecast(req: ForecastRequest):
        model_name = req.model
        if model_name == "auto":
            loaded = registry.list_loaded()
            if not loaded:
                raise HTTPException(500, "No models loaded")
            for pref in ["chronos2", "moirai1_1_r_base", "timesfm2", "sundial_base_128m"]:
                if pref in loaded:
                    model_name = pref
                    break
            else:
                model_name = loaded[0]

        try:
            adapter = registry.get(model_name)
        except KeyError:
            raise HTTPException(404, f"Model '{model_name}' not available. Loaded: {registry.list_loaded()}")

        if not adapter.is_loaded:
            try:
                adapter.load()
            except Exception as e:
                raise HTTPException(500, f"Failed to load {model_name}: {e}")

        context = np.array(req.context, dtype=np.float32)
        t0 = time.time()

        def _do_forecast():
            return adapter.forecast(
                context,
                prediction_length=req.prediction_length,
                num_samples=req.num_samples,
            )

        try:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(_thread_pool, _do_forecast)
        except Exception as e:
            raise HTTPException(500, f"Forecast failed: {e}")

        elapsed_ms = int((time.time() - t0) * 1000)

        point = output.point_forecast
        if point is not None:
            point = np.asarray(point).ravel().tolist()
        else:
            point = []

        quantiles = None
        if output.quantile_forecast:
            quantiles = {
                str(q): np.asarray(v).ravel().tolist()
                for q, v in output.quantile_forecast.items()
            }

        return ForecastResponse(
            model=model_name,
            point_forecast=point,
            quantiles=quantiles,
            elapsed_ms=elapsed_ms,
            metadata=output.metadata,
        )

    @app.get("/models")
    async def list_models():
        return {"profiles": registry.get_profiles()}

    return app


def main():
    parser = argparse.ArgumentParser(description="TimeClaw SBM Model Server")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Default device for models without an explicit mapping, e.g. cuda:2.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="",
        help="Comma-separated per-model devices, e.g. chronos2=cuda:2,moirai1_1_r_base=cuda:3",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Legacy single-GPU shortcut. Ignored when --device-map is set.",
    )
    parser.add_argument("--models", type=str, default="chronos2,moirai1_1_r_base,timesfm2,sundial_base_128m",
                        help="Comma-separated model names to load")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    device_map = _parse_device_map(args.device_map)
    default_device = args.device
    if args.gpu is not None and not device_map:
        default_device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    app = create_app(
        model_names,
        default_device=default_device,
        device_map=device_map,
    )

    import uvicorn
    logger.info(
        "[SBM-Server] Starting on "
        f"{args.host}:{args.port}, models={model_names}, "
        f"default_device={default_device}, device_map={device_map}"
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
