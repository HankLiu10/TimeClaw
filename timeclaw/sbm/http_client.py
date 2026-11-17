"""HTTP client adapter for remote SBM model server.

Drop-in replacement for local SBM adapters.  When the SBM server is
running, all forecast calls become async HTTP requests — perfect for
high-concurrency pipelines where multiple samples call SBM in parallel.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any

import aiohttp
import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

DEFAULT_SERVER_URL = "http://localhost:8100"


def get_server_health(server_url: str = DEFAULT_SERVER_URL) -> dict[str, Any]:
    """Return raw health payload from an SBM server."""
    import requests

    resp = requests.get(f"{server_url}/health", timeout=3)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


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
        request_timeout_s: float | None = None,
        first_byte_timeout_s: float | None = None,
    ):
        cfg = config or SBMConfig(model_name="sbm_http")
        super().__init__(cfg)
        self.server_url = server_url.rstrip("/")
        self.target_model = target_model
        self.request_timeout_s = (
            float(request_timeout_s) if request_timeout_s is not None else None
        )
        self.first_byte_timeout_s = (
            float(first_byte_timeout_s)
            if first_byte_timeout_s is not None
            else None
        )
        self._model = True  # always "loaded"

    def _requests_timeout(self) -> tuple[float, float] | None:
        read_timeout = self.first_byte_timeout_s or self.request_timeout_s
        if read_timeout is None:
            return None
        return (read_timeout, read_timeout)

    def _aiohttp_timeout(self) -> "aiohttp.ClientTimeout":
        connect_timeout = (
            min(self.request_timeout_s, 30.0)
            if self.request_timeout_s is not None
            else 30.0
        )
        return aiohttp.ClientTimeout(
            total=self.request_timeout_s,
            connect=connect_timeout,
            sock_connect=connect_timeout,
            sock_read=self.first_byte_timeout_s or self.request_timeout_s,
        )

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
                timeout=self._requests_timeout(),
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
                    timeout=self._aiohttp_timeout(),
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
    try:
        get_server_health(server_url)
        return True
    except Exception:
        return False


def get_server_models(server_url: str = DEFAULT_SERVER_URL) -> list[str]:
    """Get list of loaded models from the server."""
    try:
        return list(get_server_health(server_url).get("models_loaded", []) or [])
    except Exception:
        pass
        return []


class RoundRobinSBMHttpClient(SBMBase):
    """HTTP client that exposes many SBM replicas through one shared queue.

    New definition:
    - Multiple replica URLs of the same logical SBM model no longer behave as
      independent per-replica queues selected by naive round-robin.
    - Instead, this client maintains one global FIFO dispatch queue per model:
      a request waits locally until any replica becomes idle, then it is bound
      to that free replica and sent once.
    - Therefore queued work is shared across all replicas of the same model,
      matching a "single queue, many workers" scheduler.
    """

    name: str = "sbm_http_rr"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = False

    def __init__(
        self,
        config: SBMConfig | None = None,
        server_urls: list[str] | tuple[str, ...] | None = None,
        target_model: str = "auto",
        request_timeout_s: float | None = None,
        first_byte_timeout_s: float | None = None,
    ):
        cfg = config or SBMConfig(model_name="sbm_http_rr")
        super().__init__(cfg)
        urls = [str(url).rstrip("/") for url in (server_urls or []) if str(url).strip()]
        if not urls:
            raise ValueError("RoundRobinSBMHttpClient requires at least one server URL")
        self.server_urls = urls
        self.target_model = target_model
        self.request_timeout_s = (
            float(request_timeout_s) if request_timeout_s is not None else None
        )
        self.first_byte_timeout_s = (
            float(first_byte_timeout_s)
            if first_byte_timeout_s is not None
            else None
        )
        self._model = True
        self._dispatch_cond = threading.Condition()
        self._available_server_urls = deque(self.server_urls)
        self._next_ticket = 0
        self._dispatch_ticket = 0

    def load(self) -> None:
        self._model = True

    def _build_client(self, server_url: str) -> SBMHttpClient:
        return SBMHttpClient(
            config=self.config,
            server_url=server_url,
            target_model=self.target_model,
            request_timeout_s=self.request_timeout_s,
            first_byte_timeout_s=self.first_byte_timeout_s,
        )

    def _acquire_server_url_blocking(self) -> str:
        with self._dispatch_cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            while (
                ticket != self._dispatch_ticket
                or not self._available_server_urls
            ):
                self._dispatch_cond.wait()
            server_url = self._available_server_urls.popleft()
            self._dispatch_ticket += 1
            self._dispatch_cond.notify_all()
            return server_url

    def _release_server_url(self, server_url: str) -> None:
        with self._dispatch_cond:
            self._available_server_urls.append(str(server_url).rstrip("/"))
            self._dispatch_cond.notify_all()

    async def _run_async_forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        server_url = await asyncio.to_thread(self._acquire_server_url_blocking)
        client = self._build_client(server_url)
        try:
            return await client.async_forecast(
                context,
                prediction_length=prediction_length,
                **kwargs,
            )
        finally:
            self._release_server_url(server_url)

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "RoundRobinSBMHttpClient.forecast() cannot be called inside a running event loop; "
                "use async_forecast() instead."
            )
        server_url = self._acquire_server_url_blocking()
        client = self._build_client(server_url)
        try:
            return client.forecast(
                context,
                prediction_length=prediction_length,
                **kwargs,
            )
        finally:
            self._release_server_url(server_url)

    async def async_forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        return await self._run_async_forecast(
            context,
            prediction_length=prediction_length,
            **kwargs,
        )
