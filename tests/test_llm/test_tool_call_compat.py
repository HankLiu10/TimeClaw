from __future__ import annotations

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
