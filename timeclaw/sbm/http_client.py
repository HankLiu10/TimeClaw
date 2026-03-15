"""HTTP client adapter for remote SBM model server.

Drop-in replacement for local SBM adapters.  When the SBM server is
running, all forecast calls become async HTTP requests — perfect for
high-concurrency pipelines where multiple samples call SBM in parallel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

DEFAULT_SERVER_URL = "http://localhost:8100"


class SBMHttpClient(SBMBase):
    """Adapter that delegates forecast calls to a remote SBM server."""

    name: str = "sbm_http"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = False  # GPU is on the server side

    def __init__(
        self,
        config: SBMConfig | None = None,
        server_url: str = DEFAULT_SERVER_URL,
        target_model: str = "auto",
    ):
        cfg = config or SBMConfig(model_name="sbm_http")
        super().__init__(cfg)
        self.server_url = server_url.rstrip("/")
        self.target_model = target_model
        self._model = True  # always "loaded"

    def load(self) -> None:
        self._model = True

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        """Synchronous forecast via HTTP (used by existing sync call sites)."""
        import requests

        pred_len = prediction_length or self.config.prediction_length
        num_samples = kwargs.get("num_samples", self.config.num_samples)

        payload = {
            "model": self.target_model,
            "context": np.asarray(context, dtype=np.float64).ravel().tolist(),
            "prediction_length": pred_len,
            "num_samples": num_samples,
        }

        try:
            resp = requests.post(
                f"{self.server_url}/forecast",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[SBMHttpClient] Request failed: {e}")
            raise RuntimeError(f"SBM server request failed: {e}") from e

        point = np.array(data["point_forecast"], dtype=np.float64)

        quantiles = None
        if data.get("quantiles"):
            quantiles = {
                float(q): np.array(v, dtype=np.float64)
                for q, v in data["quantiles"].items()
            }

        return SBMOutput(
            point_forecast=point,
            quantile_forecast=quantiles,
            metadata={
                "model": data.get("model", self.target_model),
                "elapsed_ms": data.get("elapsed_ms", 0),
                "server": self.server_url,
                **data.get("metadata", {}),
            },
        )

    async def async_forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        """Async forecast via HTTP — ideal for concurrent pipelines."""
        import aiohttp

        pred_len = prediction_length or self.config.prediction_length
        num_samples = kwargs.get("num_samples", self.config.num_samples)

        payload = {
            "model": self.target_model,
            "context": np.asarray(context, dtype=np.float64).ravel().tolist(),
            "prediction_length": pred_len,
            "num_samples": num_samples,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/forecast",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            logger.error(f"[SBMHttpClient] Async request failed: {e}")
            raise RuntimeError(f"SBM server request failed: {e}") from e

        point = np.array(data["point_forecast"], dtype=np.float64)

        quantiles = None
        if data.get("quantiles"):
            quantiles = {
                float(q): np.array(v, dtype=np.float64)
                for q, v in data["quantiles"].items()
            }

        return SBMOutput(
            point_forecast=point,
            quantile_forecast=quantiles,
            metadata={
                "model": data.get("model", self.target_model),
                "elapsed_ms": data.get("elapsed_ms", 0),
                "server": self.server_url,
                **data.get("metadata", {}),
            },
        )


def is_server_available(server_url: str = DEFAULT_SERVER_URL) -> bool:
    """Check if the SBM server is reachable."""
    import requests
    try:
        resp = requests.get(f"{server_url}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_server_models(server_url: str = DEFAULT_SERVER_URL) -> list[str]:
    """Get list of loaded models from the server."""
    import requests
    try:
        resp = requests.get(f"{server_url}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("models_loaded", [])
    except Exception:
        pass
    return []
