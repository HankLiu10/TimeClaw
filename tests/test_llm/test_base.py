from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from timeclaw.llm.base import (
    LLMBackend,
    _exception_diagnostic,
    _is_unbounded_network_retry,
    _is_unbounded_supplier_backpressure,
    _reasoning_effort_for_model,
)
from timeclaw.core.retry_control import RetryLater, deferred_retry_scope


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.TimeoutError(),
        httpx.ReadTimeout(""),
        httpx.ConnectError(""),
    ],
)
def test_blank_transport_exceptions_use_unbounded_network_retry(exc):
    assert str(exc) == ""
    assert _is_unbounded_network_retry(exc) is True
    assert _exception_diagnostic(exc).endswith(type(exc).__name__)


def test_grok_translates_unsupported_none_reasoning_effort_to_high():
    assert _reasoning_effort_for_model("openai/grok-4.5-latest", "none") == "high"
    assert _reasoning_effort_for_model("openai/grok-4.5-latest", "medium") == "medium"
    assert _reasoning_effort_for_model("openai/gpt-5.4", "none") == "none"


def test_upstream_rate_limit_is_unbounded_supplier_backpressure():
    exc = RuntimeError(
        'HTTP 429: {"error":{"message":"Upstream rate limit exceeded, please retry later"}}'
    )
    assert _is_unbounded_supplier_backpressure(exc) is True


def test_provider_processing_failure_uses_network_retry_policy():
    exc = RuntimeError("The request could not be processed. Please check the request parameters.")
    assert _is_unbounded_network_retry(exc) is True


class _DummyLiteLLMStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _DummyHTTPStreamResponse:
    def __init__(self, lines: list[object], status_code: int = 200):
        self._lines = list(lines)
        self.status_code = status_code
        self.reason_phrase = "OK"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield str(line)


class _DummyHTTPAsyncClient:
    def __init__(self, response: _DummyHTTPStreamResponse, *args, **kwargs):
        self._response = response
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        self.stream_args = args
        self.stream_kwargs = kwargs
        return self._response


class _DummyWebSocket:
    def __init__(self, incoming_messages: list[str]):
        self._incoming_messages = list(incoming_messages)
        self.sent_messages: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, payload: str):
        self.sent_messages.append(payload)

    async def recv(self):
        if not self._incoming_messages:
            raise AssertionError("recv called after stream exhausted")
        return self._incoming_messages.pop(0)


def test_parse_response_accepts_dict_responses_payload():
    payload = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": '{"ok": true, "msg": "pong"}'},
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }

    resp = LLMBackend._parse_response(payload, "openai/gpt-5.4")

    assert resp.text == '{"ok": true, "msg": "pong"}'
    assert resp.parsed == {"ok": True, "msg": "pong"}
    assert resp.usage["total_tokens"] == 8


def test_parse_response_accepts_nested_output_text_value():
    payload = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": {"value": '{"ok": true, "msg": "pong"}'}},
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }

    resp = LLMBackend._parse_response(payload, "openai/gpt-5.4")

    assert resp.text == '{"ok": true, "msg": "pong"}'
    assert resp.parsed == {"ok": True, "msg": "pong"}
    assert resp.usage["total_tokens"] == 8


def test_responses_ws_endpoint_rewrites_http_base():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
    )
    assert backend._responses_ws_endpoint() == "wss://proxy.example/v1/responses"


@pytest.mark.asyncio
async def test_complete_propagates_retry_later_without_wrapping():
    backend = LLMBackend(default_model="openai/gpt-5.4")

    async def raise_retry_later(*args, **kwargs):
        raise RetryLater(delay_s=0.01, reason="busy", layer="llm")

    backend._call_with_retry = raise_retry_later  # type: ignore[method-assign]
    with pytest.raises(RetryLater):
        await backend.complete([{"role": "user", "content": "hello"}], model="ollama/demo")


@pytest.mark.asyncio
async def test_complete_propagates_provider_failure_instead_of_returning_empty_response():
    backend = LLMBackend(default_model="ollama/demo")

    async def fail_provider(*args, **kwargs):
        raise RuntimeError("provider could not process request")

    backend._call_with_retry = fail_provider  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="provider could not process request"):
        await backend.complete(
            [{"role": "user", "content": "hello"}],
            model="ollama/demo",
        )


@pytest.mark.asyncio
async def test_call_native_openai_uses_websocket_responses_transport():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        disable_response_storage=True,
        api_transport="responses",
        supports_websockets=True,
        responses_websockets_v2=True,
    )
    dummy_ws = _DummyWebSocket(
        [
            json.dumps({"type": "response.output_text.delta", "delta": "po"}),
            json.dumps({"type": "response.output_text.delta", "delta": "ng"}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
                    },
                }
            ),
        ]
    )

    with patch("websockets.connect", return_value=dummy_ws) as mock_connect:
        resp = await backend._call_native_openai(
            call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
            model_label="openai/gpt-5.4",
            response_format=None,
            messages=[{"role": "user", "content": "Reply with exactly pong"}],
            reasoning_effort="none",
        )

    assert resp is not None
    assert resp.text == "pong"
    assert resp.usage["total_tokens"] == 6
    _, kwargs = mock_connect.call_args
    assert kwargs["additional_headers"]["Authorization"] == "Bearer sk-test"
    assert kwargs["additional_headers"]["OpenAI-Beta"] == "responses=v1"
    assert kwargs["additional_headers"]["session_id"].startswith("timeclaw-session-")
    sent_payload = json.loads(dummy_ws.sent_messages[0])
    assert sent_payload == {
        "type": "response.create",
        "model": "gpt-5.4",
        "stream": True,
        "store": False,
        "instructions": "",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with exactly pong"}],
            }
        ],
        "reasoning": {"effort": "none"},
    }


@pytest.mark.asyncio
async def test_call_native_openai_retries_on_websocket_busy_close(monkeypatch):
    sleeps = []
    async def simulated_sleep(seconds):
        sleeps.append(seconds)
    monkeypatch.setattr("timeclaw.llm.base.asyncio.sleep", simulated_sleep)
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        max_retries=2,
        api_transport="responses",
        supports_websockets=True,
        responses_websockets_v2=True,
    )
    dummy_ws = _DummyWebSocket(
        [
            json.dumps({"type": "response.output_text.delta", "delta": "pong"}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
                    },
                }
            ),
        ]
    )

    with patch(
        "websockets.connect",
        side_effect=[RuntimeError("code=1013 account is busy, please retry later"), dummy_ws],
    ) as mock_connect:
        resp = await backend._call_native_openai(
            call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
            model_label="openai/gpt-5.4",
            response_format=None,
            messages=[{"role": "user", "content": "Reply with exactly pong"}],
            reasoning_effort="none",
        )

    assert resp is not None
    assert resp.text == "pong"
    assert mock_connect.call_count == 2
    assert sleeps == [30.0]



@pytest.mark.asyncio
async def test_call_native_openai_retries_on_response_incomplete():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        max_retries=2,
        api_transport="responses",
        supports_websockets=True,
        responses_websockets_v2=True,
    )
    incomplete_ws = _DummyWebSocket(
        [
            json.dumps(
                {
                    "type": "response.incomplete",
                    "response": {
                        "status": "incomplete",
                    },
                }
            ),
        ]
    )
    good_ws = _DummyWebSocket(
        [
            json.dumps({"type": "response.output_text.delta", "delta": "pong"}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
                    },
                }
            ),
        ]
    )

    with patch("websockets.connect", side_effect=[incomplete_ws, good_ws]) as mock_connect:
        resp = await backend._call_native_openai(
            call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
            model_label="openai/gpt-5.4",
            response_format=None,
            messages=[{"role": "user", "content": "Reply with exactly pong"}],
            reasoning_effort="none",
        )

    assert resp is not None
    assert resp.text == "pong"
    assert mock_connect.call_count == 2


@pytest.mark.asyncio
async def test_call_native_openai_keeps_supplier_retry_local_inside_deferred_scope(monkeypatch):
    sleeps = []
    async def simulated_sleep(seconds):
        sleeps.append(seconds)
    monkeypatch.setattr("timeclaw.llm.base.asyncio.sleep", simulated_sleep)
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        max_retries=2,
        api_transport="responses",
        supports_websockets=True,
        responses_websockets_v2=True,
    )
    dummy_ws = _DummyWebSocket(
        [
            json.dumps({"type": "response.output_text.delta", "delta": "pong"}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
                    },
                }
            ),
        ]
    )

    with patch(
        "websockets.connect",
        side_effect=[RuntimeError("code=1013 account is busy, please retry later"), dummy_ws],
    ) as mock_connect:
        with deferred_retry_scope("train:test:0:forecast"):
            resp = await backend._call_native_openai(
                call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
                model_label="openai/gpt-5.4",
                response_format=None,
                messages=[{"role": "user", "content": "Reply with exactly pong"}],
                reasoning_effort="none",
            )

    assert resp is not None
    assert resp.text == "pong"
    assert mock_connect.call_count == 2
    assert sleeps == [30.0]



@pytest.mark.asyncio
async def test_call_native_openai_keeps_ws_1011_proxy_retry_local_inside_deferred_scope(monkeypatch):
    sleeps = []
    async def simulated_sleep(seconds):
        sleeps.append(seconds)
    monkeypatch.setattr("timeclaw.llm.base.asyncio.sleep", simulated_sleep)
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        max_retries=2,
        api_transport="responses",
        supports_websockets=True,
        responses_websockets_v2=True,
    )

    with patch(
        "websockets.connect",
        side_effect=[
            RuntimeError("websocket closed before any event: code=1011 reason=upstream websocket proxy failed"),
            _DummyWebSocket(
                [
                    json.dumps({"type": "response.output_text.delta", "delta": "pong"}),
                    json.dumps(
                        {
                            "type": "response.completed",
                            "response": {
                                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
                            },
                        }
                    ),
                ]
            ),
        ],
    ) as mock_connect:
        with deferred_retry_scope("train:test:0:forecast"):
            resp = await backend._call_native_openai(
                call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
                model_label="openai/gpt-5.4",
                response_format=None,
                messages=[{"role": "user", "content": "Reply with exactly pong"}],
                reasoning_effort="none",
            )

    assert resp is not None
    assert resp.text == "pong"
    assert mock_connect.call_count == 2
    assert sleeps == [30.0]



@pytest.mark.asyncio
async def test_call_native_openai_keeps_empty_ws_retry_local_inside_deferred_scope():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        max_retries=2,
        api_transport="responses",
        supports_websockets=True,
        responses_websockets_v2=True,
    )
    empty_ws = _DummyWebSocket(
        [
            json.dumps({"type": "response.completed", "response": {"usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1}}}),
        ]
    )
    good_ws = _DummyWebSocket(
        [
            json.dumps({"type": "response.output_text.delta", "delta": "pong"}),
            json.dumps({"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}}}),
        ]
    )

    with patch("websockets.connect", side_effect=[empty_ws, good_ws]) as mock_connect:
        with deferred_retry_scope("train:test:0:forecast"):
            resp = await backend._call_native_openai(
                call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
                model_label="openai/gpt-5.4",
                response_format=None,
                messages=[{"role": "user", "content": "Reply with exactly pong"}],
                reasoning_effort="none",
            )

    assert resp is not None
    assert resp.text == "pong"
    assert mock_connect.call_count == 2


def test_parse_responses_ws_events_accepts_content_part_and_output_text_done():
    events = [
        {
            "type": "response.content_part.added",
            "part": {"type": "output_text", "text": '{"answer": '},
        },
        {
            "type": "response.output_text.done",
            "text": '"pong"}',
        },
        {
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            },
        },
    ]

    resp = LLMBackend._parse_responses_ws_events(events, "openai/gpt-5.4")

    assert resp.text is not None
    assert resp.parsed == {"answer": "pong"}
    assert resp.usage["total_tokens"] == 10


def test_parse_responses_ws_events_preserves_chunk_whitespace():
    events = [
        {"type": "response.output_text.delta", "delta": "Predict"},
        {"type": "response.output_text.delta", "delta": " the"},
        {"type": "response.output_text.delta", "delta": " future"},
        {"type": "response.output_text.delta", "delta": " bucket"},
        {
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
            },
        },
    ]

    resp = LLMBackend._parse_responses_ws_events(events, "openai/gpt-5.4")

    assert resp.text == "Predict the future bucket"


def test_shadow_extract_text_from_ws_events_recovers_completed_response_text():
    events = [
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": {"value": '{"ok": true, "msg": "pong"}'},
                            }
                        ],
                    }
                ]
            },
        }
    ]

    text = LLMBackend._shadow_extract_text_from_ws_events(events)

    assert text == '{"ok": true, "msg": "pong"}'


def test_parse_responses_sse_accepts_output_item_added_and_nested_output_text():
    raw = (
        "event: response.output_item.added\n"
        'data: {"item":{"type":"message","content":[{"type":"output_text","text":{"value":"{\\"ok\\": true}"}}]}}\n\n'
        "event: response.completed\n"
        'data: {"response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n'
    )

    resp = LLMBackend._parse_responses_sse_string(raw, "openai/gpt-5.4")

    assert resp.text == '{"ok": true}'
    assert resp.parsed == {"ok": True}
    assert resp.usage["total_tokens"] == 8


def test_parse_responses_sse_preserves_chunk_whitespace():
    raw = (
        "event: response.output_text.delta\n"
        'data: {"delta":"Treat"}\n\n'
        "event: response.output_text.delta\n"
        'data: {"delta":" hedged"}\n\n'
        "event: response.output_text.delta\n"
        'data: {"delta":" articles"}\n\n'
        "event: response.completed\n"
        'data: {"response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n'
    )

    resp = LLMBackend._parse_responses_sse_string(raw, "openai/gpt-5.4")

    assert resp.text == "Treat hedged articles"


@pytest.mark.asyncio
async def test_call_native_openai_http_stops_after_response_completed():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="http://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        api_transport="responses",
        supports_websockets=False,
        responses_websockets_v2=False,
    )
    response = _DummyHTTPStreamResponse(
        [
            "event: response.output_text.delta",
            'data: {"delta":"pong"}',
            "",
            "event: response.completed",
            'data: {"response":{"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}',
            "",
            AssertionError("HTTP SSE reader should stop after response.completed"),
        ]
    )

    with patch("httpx.AsyncClient", side_effect=lambda *args, **kwargs: _DummyHTTPAsyncClient(response, *args, **kwargs)):
        resp = await backend._call_native_openai(
            call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
            model_label="openai/gpt-5.4",
            response_format=None,
            messages=[{"role": "user", "content": "Reply with exactly pong"}],
            reasoning_effort="none",
        )

    assert resp is not None
    assert resp.text == "pong"
    assert resp.usage["total_tokens"] == 6


@pytest.mark.asyncio
async def test_call_native_openai_http_retries_stream_without_terminal_event():
    backend = LLMBackend(
        default_model="openai/grok-4.5-latest",
        api_base="http://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
        max_retries=1,
        api_transport="responses",
        supports_websockets=False,
        responses_websockets_v2=False,
    )
    responses = iter([
        _DummyHTTPStreamResponse([
            "event: response.output_text.delta",
            'data: {"delta":"partial"}',
            "",
        ]),
        _DummyHTTPStreamResponse([
            "event: response.output_text.delta",
            'data: {"delta":"pong"}',
            "",
            "event: response.completed",
            'data: {"response":{"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}',
            "",
        ]),
    ])
    clients: list[_DummyHTTPAsyncClient] = []

    def client_factory(*args, **kwargs):
        client = _DummyHTTPAsyncClient(next(responses), *args, **kwargs)
        clients.append(client)
        return client

    with (
        patch("httpx.AsyncClient", side_effect=client_factory),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        resp = await backend._call_native_openai(
            call_kwargs={"model": "openai/grok-4.5-latest", "max_tokens": 32},
            model_label="openai/grok-4.5-latest",
            response_format=None,
            messages=[{"role": "user", "content": "Reply with exactly pong"}],
            reasoning_effort="high",
        )

    assert len(clients) == 2
    assert resp is not None
    assert resp.text == "pong"


@pytest.mark.asyncio
async def test_call_with_retry_collects_streamed_litellm_chunks():
    backend = LLMBackend(
        default_model="azure/gpt-5.4",
        timeout=30,
    )
    stream = _DummyLiteLLMStream([
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content='{"answer":'))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=' "pong"}'))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        ),
    ])

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = stream
        resp = await backend._call_with_retry(
            call_kwargs={"model": "azure/gpt-5.4", "messages": [{"role": "user", "content": "Reply with JSON"}]},
            model_label="azure/gpt-5.4",
            response_format=None,
            messages=[{"role": "user", "content": "Reply with JSON"}],
            reasoning_effort="none",
        )

    assert resp is not None
    assert resp.text == '{"answer": "pong"}'
    assert resp.parsed == {"answer": "pong"}
    assert resp.usage["total_tokens"] == 10
    assert mock_acompletion.await_count == 1
    _, kwargs = mock_acompletion.await_args
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


def test_native_openai_transport_is_websocket_only():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=300,
        first_token_timeout=60,
    )

    assert not hasattr(backend, "_get_openai_client")
    assert backend._responses_ws_endpoint() == "wss://proxy.example/v1/responses"
