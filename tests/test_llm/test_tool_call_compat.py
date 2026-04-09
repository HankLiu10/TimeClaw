from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from timeclaw.llm.base import LLMBackend


def _tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "emit_tool_call",
                "description": "Emit one tool call.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "emit_final_answer",
                "description": "Emit one final answer.",
                "parameters": {"type": "object"},
            },
        },
    ]


def test_recover_textual_tool_call_accepts_proxy_style_function_prefix():
    recovered = LLMBackend._recover_textual_tool_call(
        'emit_tool_call{"thought":"t","tool":"noop","args":{"x":1}}',
        allowed_tool_names=["emit_tool_call", "emit_final_answer"],
    )

    assert recovered == [{
        "name": "emit_tool_call",
        "arguments": {"thought": "t", "tool": "noop", "args": {"x": 1}},
    }]


def test_recover_textual_tool_call_rejects_trailing_second_object():
    recovered = LLMBackend._recover_textual_tool_call(
        'emit_tool_call{"thought":"t","tool":"noop","args":{"x":1}}{"done":true}',
        allowed_tool_names=["emit_tool_call", "emit_final_answer"],
    )

    assert recovered == []


@pytest.mark.asyncio
async def test_complete_with_tools_recovers_proxy_degraded_tool_call():
    backend = LLMBackend(
        default_model="mock-model",
        fallback_model=None,
        api_base=None,
        api_key=None,
    )

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = (
        'emit_tool_call{"thought":"t","tool":"noop","args":{"x":1}}'
    )
    response.choices[0].message.tool_calls = None
    response.usage = None

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = response
        parsed = await backend.complete_with_tools(
            [{"role": "user", "content": "test"}],
            tools=_tool_specs(),
            model="mock-model",
        )

    assert parsed.tool_calls == [{
        "name": "emit_tool_call",
        "arguments": {"thought": "t", "tool": "noop", "args": {"x": 1}},
    }]
    assert parsed.parsed == {"thought": "t", "tool": "noop", "args": {"x": 1}}


@pytest.mark.asyncio
async def test_complete_with_tools_native_openai_forwards_tool_choice_and_parallel_flag():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        fallback_model=None,
        api_base="https://example.invalid/v1",
        api_key="test-key",
    )

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = ""
    response.choices[0].message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="emit_tool_call",
                arguments='{"thought":"t","tool":"noop","args":{"x":1}}',
            )
        )
    ]
    response.usage = None

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=response))
            )

    fake_client = _FakeClient()
    backend._oai_client = fake_client

    parsed = await backend.complete_with_tools(
        [{"role": "user", "content": "test"}],
        tools=_tool_specs(),
        model="openai/gpt-5.4",
        tool_choice="required",
        parallel_tool_calls=False,
    )

    create_call = fake_client.chat.completions.create
    kwargs = create_call.await_args.kwargs
    assert kwargs["tool_choice"] == "required"
    assert kwargs["parallel_tool_calls"] is False
    assert parsed.tool_calls == [{
        "name": "emit_tool_call",
        "arguments": {"thought": "t", "tool": "noop", "args": {"x": 1}},
    }]
