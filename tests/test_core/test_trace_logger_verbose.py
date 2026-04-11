from timeclaw.core import trace_logger


def test_verbose_llm_call_uses_plain_text_when_parsed_is_none(monkeypatch):
    messages: list[str] = []

    def fake_info(message: str) -> None:
        messages.append(message)

    trace_logger.configure(verbose=True, full_dump=False)
    monkeypatch.setattr(trace_logger.logger, "info", fake_info)

    trace_logger.llm_call(
        "unit-test",
        [{"role": "user", "content": "Say hello"}],
        type(
            "Resp",
            (),
            {
                "text": "Predicted Prices: 1.0, 2.0, 3.0",
                "parsed": None,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )(),
    )

    assert messages
    assert "Predicted Prices: 1.0, 2.0, 3.0" in messages[0]
    assert "None" not in messages[0]
