from __future__ import annotations

import re


def normalize_learning_clause(text: str) -> str:
    """Normalize agent self-talk into reusable evidence phrasing.

    New definition:
        ``normalize_learning_clause`` is a shared lightweight text normalizer
        for learning artifacts. It keeps task evidence, but removes
        orchestration-style prefaces and rewrites first-person tool narration
        into neutral evidence language.
    """

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"(?i)^i used ([a-z0-9_]+) to ", r"\1 was used to ", cleaned)
    cleaned = re.sub(r"(?i)^i used ([a-z0-9_]+) with ", r"\1 was used with ", cleaned)
    cleaned = re.sub(r"(?i)^i used ([a-z0-9_]+) on ", r"\1 was used on ", cleaned)
    cleaned = re.sub(r"(?i)^i used ([a-z0-9_]+)\b", r"\1 was used", cleaned)
    cleaned = re.sub(r"(?i)^i (?:first |then )?ran ([a-z0-9_]+)\b", r"\1 was run", cleaned)
    cleaned = re.sub(r"(?i)^i (?:first )?called ([a-z0-9_]+)\b", r"\1 was called", cleaned)
    cleaned = re.sub(r"(?i)^i chose ([a-z0-9_]+)\b", r"\1 was chosen", cleaned)
    cleaned = re.sub(r"(?i)^used ([a-z0-9_]+)\b", r"\1 was used", cleaned)

    framework_preface_patterns = (
        r"(?i)^the\s+required\s+(?:hinted\s+)?tool\s+was\s+used:\s*",
        r"(?i)^(?:we|i)\s+(?:now\s+)?(?:have|had|already\s+have|already\s+had|"
        r"have\s+already\s+satisfied|already\s+satisfied|satisfied)\s+the\s+"
        r"(?:required|mandated|prioritized?)\s+[^.]{0,120}?\b"
        r"(?:evidence|diagnostic|call|tool(?:\s+call)?|step|path|requirement|constraint)\b"
        r"(?:\s*,\s*(?:and|but|plus)\s+|\s+and\s+now\s+|\s+and\s+|\s+but\s+|\s+so\s+)?",
        r"(?i)^after\s+satisfying\s+the\s+(?:required|mandated|prioritized?)\s+[^,.;]{0,120},\s*",
    )
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        for pattern in framework_preface_patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()

    cleaned = re.sub(
        r"(?i)\s+as\s+required\s+by\s+the\s+(?:assigned\s+exploration\s+path|assigned\s+path|branch|path)\b",
        "",
        cleaned,
    ).strip()

    return cleaned
