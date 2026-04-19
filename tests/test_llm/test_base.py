from __future__ import annotations

import pytest

from timeclaw.llm.base import LLMBackend


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _DummyClient:
    def __init__(self, response: _DummyResponse):
        self.response = response
        self.calls: list[dict] = []
        self.closed = False

    async def post(self, url: str, json: dict):
        self.calls.append({"url": url, "json": json})
        return self.response

    async def close(self):
        self.closed = True


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
        _DummyResponse(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "pong"}],
                    }
                ]
            },
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
            },
        }
    ]
