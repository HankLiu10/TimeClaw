"""Wire-level tests use a local HTTP server, never an external paid endpoint."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from timeclaw.core.config import LLMConfig
from timeclaw.llm.base import LLMBackend, ProviderAccountQuotaError
from timeclaw.llm.router import ModelRouter
from timeclaw.core.accounting import RequestLedger


def completion_body(*, finish="stop", usage=True, text='{"ok":true}'):
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "local reasoning"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
    ]
    if finish is not None:
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage:
        chunks.append({"choices": [], "usage": {
            "prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": 9},
        }})
    return "".join("data: " + json.dumps(dict(id="local", object="chat.completion.chunk", created=1,
                                             model="glm-5.3-flash", **chunk)) + "\n\n"
                   for chunk in chunks) + "data: [DONE]\n\n"


@pytest.fixture
def chat_server():
    state = {"requests": [], "responses": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "body": request})
            status, body = state["responses"].pop(0) if state["responses"] else (200, completion_body())
            body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if status == 200 else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/api/paas/v4"
    yield state
    server.shutdown()
    thread.join()
    server.server_close()


def backend(server, **kwargs):
    return LLMBackend(default_model="openai/glm-5.3-flash", api_base=server["url"],
                      api_key="local-test-only", api_transport="chat_completions",
                      max_retries=1, timeout=600, **kwargs)


@pytest.mark.asyncio
async def test_text_preparation_inherits_configured_effort_on_actual_wire(chat_server, tmp_path, monkeypatch):
    import numpy as np
    from timeclaw.benchmark.mtbench_adapter import MTBenchSample
    from timeclaw.text import task_pruning
    raw = 'Hourly observations only; no future measurements.'
    chat_server['responses'] = [(200, completion_body(
        text=f'CLEANED_TEXT:\n{raw}\n\nREMOVED_SUMMARY:\n- none'))]
    monkeypatch.setattr(task_pruning, 'RAW_TEXT_DIR', tmp_path/'raw')
    monkeypatch.setattr(task_pruning, 'CLEANED_TEXT_DIR', tmp_path/'cleaned')
    llm = LLMBackend(default_model='openai/gpt-6.1-sol', api_base=chat_server['url'],
                     api_key='fixture-key', api_transport='chat_completions',
                     reasoning_effort='low', network_retry_attempts=0, send_temperature=False)
    sample = MTBenchSample(file='case.json', dataset='weather_aligned_short', domain='weather',
                           task_types=['trend_past'], input_window=np.arange(168.),
                           output_window=None, text=raw, timestamps=list(np.arange(168.)*3600.),
                           start_year=2024, forecast_horizon=24)
    try:
        view = await task_pruning.prepare_task_text(sample, 'trend_past', llm=llm)
    finally:
        await llm.shutdown()
    assert len(chat_server['requests']) == 1
    assert chat_server['requests'][0]['body']['reasoning_effort'] == 'low'
    assert 'temperature' not in chat_server['requests'][0]['body']
    assert view['cleaned_text'] == raw


@pytest.mark.asyncio
async def test_native_chat_wire_payload_usage_and_single_accounted_attempt(chat_server, tmp_path):
    llm = backend(chat_server, reasoning_effort="medium", max_tokens=9,
                  disable_response_storage=True, supports_websockets=True,
                  responses_websockets_v2=True)
    ledger = RequestLedger("chat", tmp_path)
    try:
        with ledger.activate():
            response = await llm.complete(
                [{"role": "user", "content": "Return JSON"}], response_format="json",
                reasoning_effort="none", max_output_tokens=3, max_completion_tokens=2,
            )
    finally:
        await llm.shutdown()
    request, = chat_server["requests"]
    assert request["path"] == "/api/paas/v4/chat/completions"
    body = request["body"]
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Return JSON"}]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "max"
    assert body["thinking"] == {"type": "enabled"}
    assert not ({"max_tokens", "max_output_tokens", "max_completion_tokens", "store", "reasoning"} & body.keys())
    assert response.parsed == {"ok": True}
    assert response.raw.choices[0].message.reasoning_content == "local reasoning"
    assert response.usage == {"prompt_tokens": 11, "completion_tokens": 13,
                              "total_tokens": 24, "cached_tokens": 7, "reasoning_tokens": 9}
    summary = ledger.summary()
    assert summary["logical_calls"] == summary["physical_attempts"] == 1
    assert summary["usage_covers_all_attempts"] is True
    assert summary["observed_completion_tokens"] == 13
    assert summary["billed_usd"] is None


@pytest.mark.asyncio
async def test_ten_reconnects_are_explicit_and_every_attempt_is_accounted(chat_server, tmp_path, monkeypatch):
    chat_server["responses"] = [(503, '{"error":{"message":"temporarily unavailable"}}')] * 11
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("timeclaw.llm.base.asyncio.sleep", sleep)
    llm = backend(chat_server, network_retry_attempts=10)
    ledger = RequestLedger("retry", tmp_path)
    try:
        with pytest.raises(Exception):
            with ledger.activate():
                await llm.complete([{"role": "user", "content": "test"}])
    finally:
        await llm.shutdown()
    assert len(chat_server["requests"]) == 11
    assert sleeps == [30.0] * 10
    summary = ledger.summary()
    assert summary["physical_attempts"] == 11
    assert summary["logical_calls"] == 1
    assert summary["usage_covers_all_attempts"] is False
    assert summary["status"] == "failed"


@pytest.mark.asyncio
async def test_account_1113_stops_once_without_reconnect_or_invented_usage(chat_server, tmp_path):
    chat_server["responses"] = [(429, '{"error":{"code":"1113","message":"no balance"}}')]
    llm = backend(chat_server)
    ledger = RequestLedger("account", tmp_path)
    try:
        with pytest.raises(ProviderAccountQuotaError, match="1113"):
            with ledger.activate():
                await llm.complete([{"role": "user", "content": "test"}])
    finally:
        await llm.shutdown()
    assert len(chat_server["requests"]) == 1
    summary = ledger.summary()
    assert summary["physical_attempts"] == 1
    assert summary["usage_covers_all_attempts"] is False
    assert summary["billed_usd"] is None
    end, = [event for event in ledger.events if event["kind"] == "attempt_end"]
    assert end["usage"] == {}
    assert end["error_type"] == "ProviderAccountQuotaError"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,exception", [(None, RuntimeError), ("length", ValueError)])
async def test_incomplete_output_is_rejected_with_observed_usage_preserved(chat_server, tmp_path, finish, exception):
    chat_server["responses"] = [(200, completion_body(finish=finish))]
    llm = backend(chat_server, network_retry_attempts=0)
    ledger = RequestLedger("incomplete", tmp_path)
    try:
        with pytest.raises(exception):
            with ledger.activate():
                await llm.complete([{"role": "user", "content": "test"}])
    finally:
        await llm.shutdown()
    assert len(chat_server["requests"]) == 1
    summary = ledger.summary()
    assert summary["status"] == "failed"
    assert summary["observed_completion_tokens"] == 13


@pytest.mark.asyncio
async def test_missing_usage_remains_unknown(chat_server, tmp_path):
    chat_server["responses"] = [(200, completion_body(usage=False))]
    llm = backend(chat_server)
    ledger = RequestLedger("missing-usage", tmp_path)
    try:
        with ledger.activate():
            result = await llm.complete([{"role": "user", "content": "test"}])
    finally:
        await llm.shutdown()
    assert result.usage == {}
    assert ledger.summary()["usage_covers_all_attempts"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status,prefix,ending", [
    (200, "", "\n"), (200, "", ""), (200, "data: ", "\n\n"), (400, "", ""),
])
async def test_bare_json_content_rejection_at_stream_tail_is_not_retried(
    chat_server, tmp_path, monkeypatch, status, prefix, ending,
):
    # Reproduces the observed HTTP 200 SSE stream ending in unframed JSON.
    partial = completion_body(finish=None, usage=False).replace("data: [DONE]\n\n", "")
    rejection = json.dumps({"error": {"code": "1301", "message": "内容审核拒绝; connection closed"}}, ensure_ascii=False)
    body = (partial if status == 200 else "") + prefix + rejection + ending
    chat_server["responses"] = [(status, body)] * 11
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("timeclaw.llm.base.asyncio.sleep", sleep)
    llm = backend(chat_server)
    ledger = RequestLedger("content-rejection", tmp_path)
    try:
        with pytest.raises(RuntimeError, match="1301"):
            with ledger.activate():
                await llm.complete([{"role": "user", "content": "test"}])
    finally:
        await llm.shutdown()
    assert len(chat_server["requests"]) == 1
    assert sleeps == []
    end, = [e for e in ledger.events if e["kind"] == "attempt_end"]
    assert end["error_type"] == "ProviderContentPolicyError"
    assert ledger.summary()["usage_covers_all_attempts"] is False


def test_default_config_and_router_select_gpt6_chat_without_credentials():
    config = LLMConfig()
    assert set(config.models.model_dump().values()) == {"openai/gpt-6-sol"}
    assert config.api_base is None
    assert config.api_key is config.azure_api_key is None
    assert config.max_tokens is None
    assert config.timeout == 3600
    assert config.network_retry_attempts is None
    assert config.reasoning_effort == "medium"
    assert config.allow_fallback is False
    configured_backend = ModelRouter(config).get_backend("manager")
    assert configured_backend.api_transport == "chat_completions"
    assert configured_backend.network_retry_attempts is None
    assert LLMBackend().api_transport == "responses"
