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


def test_messages_to_responses_request_preserves_assistant_history():
    instructions, items = LLMBackend._messages_to_responses_request([
        {"role": "system", "content": "Follow the contract."},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": '{"thought":"done"}'},
    ])

    assert instructions == "Follow the contract."
    assert items[0]["role"] == "user"
    assert items[0]["content"] == [{"type": "input_text", "text": "hello"}]
    assert items[1]["role"] == "assistant"
    assert "id" not in items[1]
    assert "status" not in items[1]
    assert "phase" not in items[1]
    assert items[1]["content"] == [{
        "type": "output_text",
        "text": '{"thought":"done"}',
        "annotations": [],
    }]


def test_parse_responses_sse_string_parses_output_text():
    raw = """event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":"{\\"ok\\":true}"}],"phase":"final_answer","role":"assistant"},"output_index":0,"sequence_number":1}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":2},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":15}}}
"""
    parsed = LLMBackend._parse_response(raw, "openai/gpt-5.4")

    assert parsed.text == '{"ok":true}'
    assert parsed.parsed == {"ok": True}
    assert parsed.tool_calls == []
    assert parsed.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cached_tokens": 2,
        "reasoning_tokens": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }


def test_parse_responses_sse_string_parses_function_calls():
    raw = """event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\\"x\\":1}","call_id":"call_1","name":"emit_tool_call"},"output_index":0,"sequence_number":1}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}
"""
    parsed = LLMBackend._parse_response(raw, "openai/gpt-5.4")

    assert parsed.text is None
    assert parsed.tool_calls == [{
        "name": "emit_tool_call",
        "arguments": {"x": 1},
    }]


@pytest.mark.asyncio
async def test_complete_with_tools_native_openai_responses_path_converts_tools_and_parses_calls():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        fallback_model=None,
        api_base="https://example.com/v1",
        api_key="sk-test",
    )

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="emit_tool_call",
                arguments='{"thought":"t","tool":"noop","args":{"x":1}}',
            )
        ],
        output_text="",
        usage=None,
    )
    mock_client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=response))
    )

    with patch.object(backend, "_get_openai_client", return_value=mock_client):
        parsed = await backend.complete_with_tools(
            [{"role": "user", "content": "test"}],
            tools=_tool_specs(),
            model="openai/gpt-5.4",
            tool_choice="required",
        )

    sent_kwargs = mock_client.responses.create.await_args.kwargs
    assert sent_kwargs["input"][0]["role"] == "user"
    assert "temperature" not in sent_kwargs
    assert sent_kwargs["tools"][0] == {
        "type": "function",
        "name": "emit_tool_call",
        "description": "Emit one tool call.",
        "parameters": {"type": "object"},
        "strict": False,
    }
    assert parsed.tool_calls == [{
        "name": "emit_tool_call",
        "arguments": {"thought": "t", "tool": "noop", "args": {"x": 1}},
    }]
    assert parsed.parsed is None


@pytest.mark.asyncio
async def test_complete_native_openai_responses_path_parses_json_output():
    backend = LLMBackend(
        default_model="openai/gpt-5.4",
        fallback_model=None,
        api_base="https://example.com/v1",
        api_key="sk-test",
    )

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"label":"up"}')
                ],
            )
        ],
        output_text='{"label":"up"}',
        usage=None,
    )
    mock_client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=response))
    )

    with patch.object(backend, "_get_openai_client", return_value=mock_client):
        parsed = await backend.complete(
            [{"role": "user", "content": "classify"}],
            model="openai/gpt-5.4",
            response_format="json",
        )

    sent_kwargs = mock_client.responses.create.await_args.kwargs
    assert sent_kwargs.get("instructions") in {None, ""}
    assert "temperature" not in sent_kwargs
    assert sent_kwargs["text"] == {"format": {"type": "json_object"}}
    assert "json" in sent_kwargs["input"][0]["content"][0]["text"].lower()
    assert parsed.text == '{"label":"up"}'
    assert parsed.parsed == {"label": "up"}
