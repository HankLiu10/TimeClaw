from types import SimpleNamespace

import numpy as np

from timeclaw.benchmark.mtbench_prompts import build_prompt


def test_finance_correlation_prompt_explains_task_semantics():
    sample = SimpleNamespace(
        dataset="finance_qa_short",
        domain="finance",
        metadata={
            "sticker": "RHI",
            "published_utc": "2024-01-01 12:00:00",
        },
        timestamps=[1704067200.0, 1704672000.0],
        input_window=np.array([100.0, 101.5, 103.0], dtype=float),
        text="Analysts upgraded the stock and highlighted strong near-term demand.",
        labels={},
    )

    system_prompt, user_prompt = build_prompt(sample, "correlation")
    prompt = f"{system_prompt}\n{user_prompt}"

    assert "directional alignment" in prompt
    assert "not a Pearson-style statistical correlation" in prompt
    assert "not a proof of causality" in prompt
    assert "Use No Correlation when the article signal is weak" in prompt
