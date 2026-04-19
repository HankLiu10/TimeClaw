from __future__ import annotations

import re
from typing import Any


def infer_indicator_key(question: str, context: str) -> str | None:
    text = f"{question}\n{context}".lower()
    if "upper_bb" in text or "upper bollinger band" in text:
        return "upper_bb"
    if re.search(r"\bmacd\b", text) or "moving average convergence divergence" in text:
        return "macd"
    return None


def canonical_indicator_key(key: str) -> str:
    normalized = str(key).strip().lower()
    alias_map = {
        "upper_bollinger_band": "upper_bb",
        "upper_bollinger_bands": "upper_bb",
        "bollinger_upper": "upper_bb",
        "bb_upper": "upper_bb",
        "macd_values": "macd",
        "macd_forecast": "macd",
        "requested_indicator_key": "indicator_name",
    }
    return alias_map.get(normalized, normalized)


def canonicalize_indicator_answer(
    answer: Any,
    *,
    indicator_output_keys: list[str] | None = None,
    question: str = "",
    context: str = "",
) -> Any:
    candidate = answer
    expected_keys = [
        str(key).strip()
        for key in (indicator_output_keys or [])
        if str(key).strip()
    ]

    if isinstance(candidate, dict) and isinstance(candidate.get("answer"), dict):
        normalized_inner = canonicalize_indicator_answer(
            candidate["answer"],
            indicator_output_keys=indicator_output_keys,
            question=question,
            context=context,
        )
        if normalized_inner is candidate["answer"]:
            return candidate
        wrapped = dict(candidate)
        wrapped["answer"] = normalized_inner
        return wrapped
    if not isinstance(candidate, dict):
        if len(expected_keys) == 1 and candidate is not None:
            return {expected_keys[0]: candidate}
        return answer
    if not candidate:
        return answer
    inferred_key = infer_indicator_key(question, context)
    actual_items = list(candidate.items())
    canonical_actual = [
        (canonical_indicator_key(str(key)), value) for key, value in actual_items
    ]

    if len(canonical_actual) == 1:
        canonical_key, value = canonical_actual[0]
        if expected_keys and (canonical_key == "indicator_name" or canonical_key == expected_keys[0]):
            return {expected_keys[0]: value}
        if canonical_key == "indicator_name" and inferred_key:
            return {inferred_key: value}

    normalized: dict[str, Any] = {}
    for canonical_key, value in canonical_actual:
        if canonical_key == "indicator_name" and inferred_key:
            canonical_key = inferred_key
        normalized[canonical_key] = value
    return normalized
