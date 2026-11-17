from __future__ import annotations

import threading
import time

from timeclaw.core.config import LLMConfig, SBMRuntimeConfig, TimeClawConfig
from timeclaw.sbm.http_client import RoundRobinSBMHttpClient, SBMHttpClient
from timeclaw.sbm.registry import get_global_registry, register_remote_server_client_map


def test_http_client_allows_infinite_queue_wait():
    client = SBMHttpClient(
        server_url="http://127.0.0.1:8100",
        target_model="timesfm2",
        request_timeout_s=None,
        first_byte_timeout_s=None,
    )

    assert client._requests_timeout() is None

    timeout = client._aiohttp_timeout()
    assert timeout.total is None
    assert timeout.sock_read is None
    assert timeout.connect == 30.0
    assert timeout.sock_connect == 30.0


def test_remote_client_registration_uses_sbm_timeouts(monkeypatch):
    cfg = TimeClawConfig(
        llm=LLMConfig(timeout=123.0, first_token_timeout=45.0),
        sbm=SBMRuntimeConfig(
            request_timeout_s=None,
            first_byte_timeout_s=None,
        ),
    )
    monkeypatch.setattr(TimeClawConfig, "load", staticmethod(lambda: cfg))
    monkeypatch.setattr("timeclaw.sbm.http_client.is_server_available", lambda _url: True)

    registry = get_global_registry()
    model_name = "timesfm2"
    if model_name in registry._instances:
        registry.unload(model_name)

    try:
        register_remote_server_client_map({model_name: "http://127.0.0.1:8111"})
        client = registry._instances[model_name]
        assert client.request_timeout_s is None
        assert client.first_byte_timeout_s is None
    finally:
        if model_name in registry._instances:
            registry.unload(model_name)


def test_round_robin_client_waits_for_any_free_replica():
    client = RoundRobinSBMHttpClient(
        server_urls=["http://127.0.0.1:8112", "http://127.0.0.1:8113"],
        target_model="timesfm2",
    )
    first = client._acquire_server_url_blocking()
    second = client._acquire_server_url_blocking()
    assert {first, second} == {
        "http://127.0.0.1:8112",
        "http://127.0.0.1:8113",
    }

    acquired: dict[str, str] = {}

    def _worker() -> None:
        acquired["url"] = client._acquire_server_url_blocking()

    thread = threading.Thread(target=_worker)
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()

    client._release_server_url(second)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert acquired["url"] == second
    client._release_server_url(first)
    client._release_server_url(acquired["url"])


def test_round_robin_client_global_queue_is_fifo():
    client = RoundRobinSBMHttpClient(
        server_urls=["http://127.0.0.1:8112"],
        target_model="timesfm2",
    )
    initial = client._acquire_server_url_blocking()
    acquire_order: list[str] = []

    def _worker(name: str) -> None:
        url = client._acquire_server_url_blocking()
        acquire_order.append(name)
        client._release_server_url(url)

    first = threading.Thread(target=_worker, args=("first",))
    second = threading.Thread(target=_worker, args=("second",))
    first.start()
    second.start()
    time.sleep(0.05)

    client._release_server_url(initial)
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert acquire_order == ["first", "second"]
