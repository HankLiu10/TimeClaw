from __future__ import annotations

import re


_BROKEN_PLUS_MINUS_RE = re.compile(r"\x00[bB]1")
_DISALLOWED_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FRAMEWORK_TOOL_MARKERS = (
    "spawn_subagent",
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
)
_META_EXECUTION_PATTERNS = (
    r"(?i)\b(?:assigned(?: exploration)? path|assigned branch|execution context|ground truth|training mode|learning mode|learning-mode)\b.{0,120}\b(?:require|requires|required|prioritiz|mandat|must|before finishing|encourag)\w*",
    r"(?i)\b(?:latest|current|developer|immediate-response)\s+instruction\b.{0,120}\brequire\w*\b.{0,80}\btool call\b",
    r"(?i)\b(?:must call|must use|need(?:ed)? to call|tool call is required|tool must be called)\b.{0,120}\b(?:tool|turn|now|finish)\b",
    r"(?i)\b(?:before finishing|before finalizing|finish-only turn|evaluation-only turn|completion contract)\b",
    r"(?i)\b(?:branch instructions?|branch-valid|failed consistency check|reserved for finishing)\b",
    r"(?i)\b(?:branch comparison|independent branches?|multiple branches?|separately reason)\b",
    r"(?i)\b(?:task\b.{0,80}\brequires?|explicitly prioritiz(?:e|es|ed|ing)|selected forecast)\b",
    r"(?i)\b(?:most|next|additional|final|remaining)\b.{0,120}\b(?:step|evidence|diagnostic|validation|check)\b",
    r"(?i)\bmost\s+(?:relevant|useful|efficient|direct|informative)\b.{0,120}\b(?:tool|path|candidate|step|evidence|validation)\b",
    r"(?i)\bsupport(?:s|ing)? the existing (?:classification|forecast|answer)\b",
    r"(?i)\b(?:i|we)\s+still need\b.{0,120}\b(?:evidence|tool|step|diagnostic|validation|check)\b",
    r"(?i)\b(?:current branch state confirms|available from a real tool result)\b",
    r"(?i)\b(?:may be returned unchanged|returned unchanged as the final)\b",
)
_META_OUTPUT_PATTERNS = (
    r"(?i)\b(?:re-?wrap(?:ping|ped)?|repair(?:ed|ing)?|corrected)\b.{0,80}\b(?:final|json|answer|payload)\b",
    r"(?i)\b(?:saved|existing|unchanged|exact)\b.{0,50}\b(?:answer|label|payload|json)\b",
    r"(?i)\b(?:return(?:ed|ing)?|reuse(?:d|ing)?|preserv(?:e|ing)?)\b.{0,80}\b(?:answer|label|payload|json)\b",
    r"(?i)\btask-valid\b.{0,100}\b(?:already exists|already available|remains available|is available|returned unchanged)\b",
)
_EVIDENCE_TEXT_REPLACEMENTS = (
    (r"(?i)\btask-valid\s+", ""),
    (r"(?i)\bexact task-valid format\b", "exact format"),
    (r"(?i)\bcurrent branch state confirms(?: that)?\b", ""),
    (r"(?i)\bavailable from a real tool result\b", ""),
    (r"(?i)\bmay be returned unchanged\b", ""),
    (r"(?i)\breturned unchanged as the final\b", ""),
    (r"(?i)\breturn(?:ed|ing)? unchanged\b", ""),
    (r"(?i)\bused unchanged\b", ""),
    (r"(?i)\bprovided reuse answer\b", ""),
    (r"(?i)\breuse answer\b", ""),
    (r"(?i)\baccepted (?:result|answer|forecast|payload)\b", "result"),
    (r"(?i)\bwinning (?:branch|candidate|path)\b", "path"),
    (r"(?i)\bbest candidate among explored branches\b", "path"),
    (r"(?i)\bexplored branches\b", "tested paths"),
    (r"(?i)\brepair-only\b", ""),
    (r"(?i)\bfinish-only\b", ""),
    (r"(?i)\bfinal JSON delivery\b", ""),
    (r"(?i)\bfinal answer\b", ""),
    (r"(?i)\bcompletion mechanics\b", ""),
    (r"(?i)\bpayload wrapping\b", ""),
    (r"(?i)\bcontroller-side\b", ""),
    (r"(?i)\bcontroller\b", ""),
    (r"(?i)\bcorrected final answer\b", ""),
    (r"(?i)\brepaired summary\b", ""),
    (r"(?i)\bretained in the branch\b", ""),
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|;\s+")


def sanitize_learning_text(text: str) -> str:
    """Remove storage-breaking control bytes from learning artifacts.

    New definition:
        ``sanitize_learning_text`` is the shared storage-facing text cleaner for
        learning artifacts. It preserves normal Unicode content, repairs the
        common broken ``±`` sequence that occasionally arrives as ``\\x00b1``,
        and strips other disallowed control bytes so notes/results stay valid
        text files instead of silently becoming binary-ish artifacts.
    """

    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not cleaned:
        return ""
    cleaned = _BROKEN_PLUS_MINUS_RE.sub("±", cleaned)
    cleaned = _DISALLOWED_CONTROL_RE.sub("", cleaned)
    return cleaned


def normalize_learning_clause(text: str) -> str:
    """Normalize agent self-talk into reusable evidence phrasing.

    New definition:
        ``normalize_learning_clause`` is a shared lightweight text normalizer
        for learning artifacts. It keeps task evidence, but removes
        orchestration-style prefaces and rewrites first-person tool narration
        into neutral evidence language.
    """

    cleaned = sanitize_learning_text(text).strip()
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

    return sanitize_learning_text(cleaned).strip()


def has_meta_learning_content(text: str) -> bool:
    """Detect control-flow / packaging language in learning artifacts.

    New definition:
        ``meta learning content`` means wording about branch control,
        tool-mandate mechanics, packaging/repair, or "reuse the saved answer"
        behavior, rather than evidence from the executed tool path itself.
    """

    cleaned = sanitize_learning_text(text).strip()
    if not cleaned:
        return False
    lowered = f" {cleaned.lower()} "
    if any(marker in lowered for marker in _FRAMEWORK_TOOL_MARKERS):
        return True
    return any(
        re.search(pattern, cleaned)
        for pattern in (*_META_EXECUTION_PATTERNS, *_META_OUTPUT_PATTERNS)
    )


def rewrite_learning_evidence_text(text: str) -> str:
    """Rewrite lightweight meta phrases without discarding the evidence line."""

    cleaned = normalize_learning_clause(text)
    if not cleaned:
        return ""
    for pattern, repl in _EVIDENCE_TEXT_REPLACEMENTS:
        cleaned = re.sub(pattern, repl, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" .,:;|-")


def sanitize_learning_evidence_text(
    text: str,
    *,
    drop_imperatives: bool = False,
) -> str:
    """Keep only evidence-facing language from a learning artifact snippet."""

    cleaned = normalize_learning_clause(text)
    if not cleaned:
        return ""

    kept_units: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(cleaned):
        raw_unit = sanitize_learning_text(part).strip()
        unit = rewrite_learning_evidence_text(raw_unit)
        if not unit:
            continue
        if has_meta_learning_content(raw_unit) or has_meta_learning_content(unit):
            continue
        if drop_imperatives and re.match(r"(?i)^(use|return|wrap|resend|copy|finish)\b", unit):
            continue
        kept_units.append(unit)
    if not kept_units:
        return ""
    return ". ".join(kept_units).strip().rstrip(".")
