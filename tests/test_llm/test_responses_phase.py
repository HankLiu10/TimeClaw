"""Intermediate assistant updates must not become a TimeClaw tool decision."""
import json
from types import SimpleNamespace

import pytest

from timeclaw.llm.base import LLMBackend


def message(phase, text):
    return {"id": phase, "type": "message", "role": "assistant", "phase": phase,
            "content": [{"type": "output_text", "text": text}]}


@pytest.mark.parametrize("transport", ["dict", "sdk", "sse", "ws"])
@pytest.mark.parametrize("with_final", [True, False])
def test_commentary_is_not_a_decision(transport, with_final):
    intermediate = '{"done":false,"tool":"wrong_intermediate_tool","args":{}}'
    final = '{"done":false,"tool":"spawn_subagent","args":{}}'
    output = [message("commentary", intermediate)]
    if with_final:
        output.append(message("final_answer", final))
    response = {"status": "completed", "output": output,
                "output_text": intermediate + (final if with_final else ""),
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}
    events = []
    for i, item in enumerate(output):
        events += [
            {"type": "response.output_item.added", "output_index": i,
             "item": {**item, "content": []}},
            {"type": "response.output_text.delta", "output_index": i,
             "item_id": item["id"], "delta": item["content"][0]["text"]},
            {"type": "response.output_item.done", "output_index": i, "item": item},
        ]
    events.append({"type": "response.completed", "response": response})
    if transport == "dict":
        result = LLMBackend._parse_response(response, "gpt-6-sol")
    elif transport == "sdk":
        result = LLMBackend._parse_response(SimpleNamespace(**response), "gpt-6-sol")
    elif transport == "ws":
        result = LLMBackend._parse_responses_ws_events(events, "gpt-6-sol")
    else:
        raw = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events)
        result = LLMBackend._parse_responses_sse_string(raw, "gpt-6-sol")
    assert result.text == (final if with_final else None)
    assert result.parsed == (json.loads(final) if with_final else None)
    assert result.usage["total_tokens"] == 30


def test_assistant_history_preserves_phase_without_adding_it_to_user():
    _, items = LLMBackend._messages_to_responses_request([
        {"role": "assistant", "phase": "commentary", "content": "Working."},
        {"role": "assistant", "phase": "final_answer", "content": '{"done":false}'},
        {"role": "user", "content": "Tool observation"},
    ])
    assert items[0]["phase"] == "commentary"
    assert items[1]["phase"] == "final_answer"
    assert "phase" not in items[2]
