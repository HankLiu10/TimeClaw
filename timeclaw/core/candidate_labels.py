from __future__ import annotations

import re
from typing import Any

_BRANCH_LABEL_RE = re.compile(
    r"^(?:(?:slot|candidate|approach)_\d+|top_level::\d+)\s*::\s*",
    re.IGNORECASE,
)
_BRANCH_SCAFFOLD_RE = re.compile(
    r"(?:(?:slot|candidate|approach)_\d+|top_level::\d+)\s*::\s*",
    re.IGNORECASE,
)
_GENERIC_LABEL_RE = re.compile(
    r"(?:slot|candidate|approach|top_level)(?:_\d+|::\d+)?",
    re.IGNORECASE,
)


def normalize_tool_chain(tool_chain: Any) -> list[str]:
    return [
        str(tool).strip()
        for tool in (tool_chain or [])
        if str(tool).strip()
    ]


def strip_branch_prefix(label: Any) -> str:
    text = str(label or "").strip()
    while text:
        updated = _BRANCH_LABEL_RE.sub("", text, count=1).strip()
        if updated == text:
            break
        text = updated
    return text


def canonicalize_candidate_mentions(text: Any) -> str:
    return _BRANCH_SCAFFOLD_RE.sub("", str(text or ""))


def is_generic_candidate_label(label: Any) -> bool:
    text = str(label or "").strip()
    if not text:
        return True
    return bool(_GENERIC_LABEL_RE.fullmatch(text))


def canonical_candidate_label(
    *,
    candidate_key: Any = None,
    name: Any = None,
    tool_chain: Any = None,
    fallback: str = "",
) -> str:
    cleaned_chain = normalize_tool_chain(tool_chain)
    chain_text = " -> ".join(cleaned_chain) if cleaned_chain else ""

    cleaned_labels = [
        strip_branch_prefix(raw)
        for raw in (candidate_key, name)
    ]
    for text in cleaned_labels:
        if (
            text
            and not is_generic_candidate_label(text)
            and ("->" in text or "→" in text)
        ):
            return text

    if chain_text:
        return chain_text

    for text in cleaned_labels:
        if text and not is_generic_candidate_label(text):
            return text

    for raw in (candidate_key, name):
        text = str(raw or "").strip()
        if text:
            return text

    return fallback
