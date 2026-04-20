from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from timeclaw.llm.base import LLMBackend


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _DummyStreamResponse:
    def __init__(self, status_code: int, chunks: list[str]):
        self.status_code = status_code
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk


class _DummyClient:
    def __init__(self, response: _DummyStreamResponse):
        self.response = response
        self.calls: list[dict] = []
        self.closed = False

    def stream(self, method: str, url: str, json: dict, headers: dict | None = None):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return self.response

    async def close(self):
        self.closed = True


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


@pytest.mark.asyncio
async def test_call_native_openai_posts_directly_to_proxy_responses_endpoint():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=30,
    )
    dummy = _DummyClient(
        _DummyStreamResponse(
            200,
            [
                'event: response.output_item.done\n'
                'data: {"item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"pong"}]}}\n\n',
                'event: response.completed\n'
                'data: {"response":{"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}\n\n',
            ],
        )
    )
    backend._oai_client = dummy

    resp = await backend._call_native_openai(
        call_kwargs={"model": "openai/gpt-5.4", "max_tokens": 32},
        model_label="openai/gpt-5.4",
        response_format=None,
        messages=[{"role": "user", "content": "Reply with exactly pong"}],
        reasoning_effort="none",
    )

    assert resp is not None
    assert resp.text == "pong"
    assert dummy.calls == [
        {
            "method": "POST",
            "url": "https://proxy.example/v1/responses",
            "json": {
                "model": "gpt-5.4",
                "max_output_tokens": 32,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Reply with exactly pong"}],
                    }
                ],
                "reasoning": {"effort": "none"},
                "stream": True,
            },
            "headers": {"Accept": "text/event-stream"},
        }
    ]


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


def test_native_openai_client_uses_separate_first_token_timeout():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        api_base="https://proxy.example/v1",
        api_key="sk-test",
        timeout=300,
        first_token_timeout=60,
    )

    client = backend._get_openai_client()

    assert isinstance(client.timeout, type(__import__("httpx").Timeout(1.0)))
    assert client.timeout.connect == 300
    assert client.timeout.read == 60
    assert client.timeout.write == 300
    assert client.timeout.pool == 300
