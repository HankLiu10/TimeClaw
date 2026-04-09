import random

from timeclaw.agents.tool_dropout import apply_dropout, keep_probability


def test_keep_probability_matches_final_formula():
    counts = {"chronos2_forecast": 10, "prophet_forecast": 2, "ets_forecast": 2}
    p = keep_probability(
        "chronos2_forecast",
        counts,
        ["chronos2_forecast", "prophet_forecast", "ets_forecast"],
        alpha=1.0,
    )
    assert p == 3 / 11


def test_keep_probability_returns_one_for_never_drop_and_coldest_tool():
    counts = {"chronos2_forecast": 10, "prophet_forecast": 0, "llm_reasoning_forecast": 7}
    assert keep_probability(
        "prophet_forecast",
        counts,
        ["chronos2_forecast", "prophet_forecast", "llm_reasoning_forecast"],
        alpha=1.0,
    ) == 1.0
    assert keep_probability(
        "llm_reasoning_forecast",
        counts,
        ["chronos2_forecast", "prophet_forecast", "llm_reasoning_forecast"],
        alpha=1.0,
        never_drop={"llm_reasoning_forecast"},
    ) == 1.0


def test_apply_dropout_keeps_never_drop_and_can_hide_hot_tool():
    counts = {
        "chronos2_forecast": 10,
        "prophet_forecast": 0,
        "llm_reasoning_forecast": 8,
    }
    visible, keep_probs = apply_dropout(
        ["chronos2_forecast", "prophet_forecast", "llm_reasoning_forecast"],
        counts,
        never_drop={"llm_reasoning_forecast"},
        rng=random.Random(0),
    )

    assert "llm_reasoning_forecast" in visible
    assert keep_probs["prophet_forecast"] == 1.0
    assert keep_probs["chronos2_forecast"] < 1.0

