from types import SimpleNamespace

import numpy as np

from timeclaw.benchmark.mtbench_prompts import build_prompt


def test_finance_correlation_prompt_preserves_official_labels_and_news():
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

    for label in ("Strong Positive Correlation", "Moderate Positive Correlation", "No Correlation",
                  "Moderate Negative Correlation", "Strong Negative Correlation"):
        assert label in prompt
    assert sample.text in prompt
    assert "RHI" in prompt


def test_mcqa_parser_does_not_confuse_word_prefixes_with_answer_letters():
    from timeclaw.benchmark.mtbench_prompts import extract_mcqa_letter
    assert extract_mcqa_letter('Answer: B')=='B'
    assert extract_mcqa_letter('Choose D')=='D'
    assert extract_mcqa_letter('B')=='B'
    assert extract_mcqa_letter('Clearly there is no option')==''
    assert extract_mcqa_letter('A or B')==''
