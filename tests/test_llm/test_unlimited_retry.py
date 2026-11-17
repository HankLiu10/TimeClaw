"""Retry failures locally; these tests never call the paid provider."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import timeclaw.llm.base as base_module
from timeclaw.llm.base import LLMBackend, LLMRequestGate
from timeclaw.core.accounting import RequestLedger
from tests.test_llm.test_base import (
    _DummyHTTPAsyncClient, _DummyHTTPStreamResponse, _DummyWebSocket,
)
from tests.test_llm.test_native_chat_transport import backend, chat_server, completion_body


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body", [
    (429, '{"error":{"code":"1302","message":"账户限流"}}'),
    (503, '{"error":{"message":"temporarily unavailable"}}'),
    (500, '{"error":{"message":"internal error"}}'),
    (200, '{"error":{"code":"1302","message":"账户限流"}}\n'),
    (200, 'data: {"error":{"message":"The service is temporarily unavailable. Please retry later."}}\n\n'),
    (200, 'data: {"error":{"code":"upstream_http2_stream_error","message":"Upstream HTTP/2 stream failed"}}\n\n'),
    (200, 'data: {"error":{"code":"upstream_stream_read_error","message":"Upstream response stream was interrupted"}}\n\n'),
])
async def test_unlimited_retries_exceed_ten_and_reset_request_deadline(
    chat_server, tmp_path, monkeypatch, status, body,
):
    chat_server["responses"] = [(status, body)] * 12 + [(200, completion_body())]
    sleeps = []
    clock = [0.0]
    real_sleep = asyncio.sleep

    async def sleep(seconds):
        sleeps.append(seconds)
        # Exceed this backend's 600-second request timeout between attempts.
        # Retries must start a fresh timeout rather than exhaust a total deadline.
        clock[0] += 601
        await real_sleep(0)

    monkeypatch.setattr(base_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(base_module.asyncio, "sleep", sleep)
    llm = backend(chat_server, network_retry_attempts=None)
    ledger = RequestLedger("unlimited", tmp_path)
    try:
        with ledger.activate():
            response = await asyncio.wait_for(
                llm.complete([{"role": "user", "content": "test"}]), timeout=10,
            )
    finally:
        await llm.shutdown()
    assert response.text == '{"ok":true}'
    assert len(chat_server["requests"]) == 13
    assert sleeps == [30.0] * 12
    summary = ledger.summary()
    assert summary["physical_attempts"] == 13
    assert summary["logical_calls"] == 1
    assert summary["status"] == "completed"
    assert summary["usage_covers_all_attempts"] is False
    assert summary["billed_usd"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_permanent_http_errors_do_not_retry_even_with_transient_words(
    chat_server, monkeypatch, status,
):
    chat_server["responses"] = [(status, json.dumps({
        "error": {"message": "invalid request/key, not a connection or rate limit failure"},
    }))]
    sleep = AsyncMock()
    monkeypatch.setattr(base_module.asyncio, "sleep", sleep)
    llm = backend(chat_server)
    try:
        with pytest.raises(Exception):
            await llm.complete([{"role": "user", "content": "test"}])
    finally:
        await llm.shutdown()
    assert len(chat_server["requests"]) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelling_retry_sleep_stops_retries_and_releases_request_gate(
    chat_server, tmp_path, monkeypatch,
):
    chat_server["responses"] = [
        (429, '{"error":{"code":"1302","message":"rate limit"}}'),
        (200, completion_body()),
    ]
    waiting = asyncio.Event()

    async def sleep(seconds):
        assert seconds == 30
        waiting.set()
        await asyncio.Future()

    monkeypatch.setattr(base_module.asyncio, "sleep", sleep)
    llm = backend(chat_server, request_gate=LLMRequestGate(1))
    ledger = RequestLedger("cancelled", tmp_path)

    async def run():
        with ledger.activate():
            return await llm.complete([{"role": "user", "content": "test"}])

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(chat_server["requests"]) == 1
        assert ledger.summary()["status"] == "cancelled"
        # A subsequent request on the same one-slot gate must still complete.
        response = await asyncio.wait_for(
            llm.complete([{"role": "user", "content": "next"}]), timeout=5,
        )
        assert response.text == '{"ok":true}'
        assert len(chat_server["requests"]) == 2
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await llm.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("websocket", [False, True])
async def test_responses_transports_also_retry_beyond_ten(monkeypatch, websocket):
    done = {"usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}}
    if websocket:
        response = _DummyWebSocket([
            json.dumps({"type": "response.output_text.delta", "delta": "pong"}),
            json.dumps({"type": "response.completed", "response": done}),
        ])
        target = "websockets.connect"
    else:
        response = _DummyHTTPAsyncClient(_DummyHTTPStreamResponse([
            "event: response.output_text.delta", 'data: {"delta":"pong"}', "",
            "event: response.completed", "data: " + json.dumps({"response": done}), "",
        ]))
        target = "httpx.AsyncClient"
    connect = Mock(side_effect=[RuntimeError("HTTP 429 rate limit")] * 12 + [response])
    sleep = AsyncMock()
    monkeypatch.setattr(target, connect)
    monkeypatch.setattr(base_module.asyncio, "sleep", sleep)
    llm = LLMBackend(
        default_model="openai/test", api_base="http://proxy.example/v1", api_key="local-test-only",
        max_retries=1, network_retry_attempts=None,
        supports_websockets=websocket, responses_websockets_v2=websocket,
    )
    result = await llm.complete([{"role": "user", "content": "test"}])
    assert result.text == "pong"
    assert connect.call_count == 13
    assert [call.args[0] for call in sleep.await_args_list] == [30.0] * 12
