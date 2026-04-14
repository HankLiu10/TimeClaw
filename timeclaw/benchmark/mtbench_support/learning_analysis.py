from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from timeclaw.core.config import TimeClawConfig
from timeclaw.core.learning_text import normalize_learning_clause
from timeclaw.llm.router import ModelRouter

_analysis_llm = None

_SYSTEM_PROMPT = """You are writing a single-sample learning analysis for a time-series agent.

Your output will be stored in notes and later distilled into reusable memory.
Do not write a generic summary. Write compact, evidence-grounded learning material.

Extract exactly five fields:
0. text signals: 3-6 short original-text snippets from the task's text/context that are most useful for solving this sample
1. sample conditions that matter for tool choice
2. concrete evidence for why the accepted tool/path matched this sample better than the other tested paths
3. failure signals from weaker tested paths
4. a reusable decision rule for future similar cases

Rules:
- Be specific and operational.
- Use evaluation metrics when available.
- Refer to actual tools and tool paths when they matter, but phrase them as
  evidence patterns rather than candidate bookkeeping.
- Explain method-data fit before metric outcomes: first state what data/task
  characteristics made the accepted tool/path suitable, then use metrics only
  as supporting evidence that this fit held on this sample.
- Do not treat better metrics as the cause of success. Metrics are validation
  evidence, not the underlying reason the path matched the sample.
- Avoid bullets like "X won because it had lower error" or "X was better
  because MAE was lower" unless the same bullet already states the structural
  reason those metrics improved.
- For text_signals, use the task description only to understand what the task needs.
- For text_signals, extract snippets ONLY from the dedicated Text/Context Source block provided below.
- For text_signals, NEVER copy from scope, task type, task instructions, task-description scaffolding, time-series data section, fingerprint, candidate traces, metrics, or analysis text.
- For text_signals, copy short original snippets or sentences from the provided text/context when possible.
- For text_signals, do not invent facts, do not summarize the whole passage, and do not include more than 6 items.
- Keep the four sections cleanly separated by purpose:
  - sample_conditions: only input/task conditions that matter for choice,
    such as trend strength, seasonality, volatility, lag structure, horizon,
    missingness, or alignment between text context and the series
  - winner_evidence: only relative, evidence-backed reasons the winner was stronger
  - failure_signals: only concrete weaknesses of weaker candidates
  - reusable_rule: only one future-facing decision rule
- Frame conclusions relatively to the evaluated candidates in this sample, not as universal facts.
- Do not use absolute claims such as "best", "optimal", "always", "guaranteed", or similar strongest-form wording.
- Do not mention orchestration/framework actions such as spawn_subagent or evaluate_batch_against_gt.
- Do not mention branch state, support notes, validator/gate status, fallback handling,
  "next step", "need_evidence", "can be returned unchanged", or any other control-flow language.
- Do not restate the full task prompt.
- Do not include ground-truth values or label distributions.
- Do not write vague claims like "performed well" or "good result".
- In winner_evidence and failure_signals, name the relevant compared tool/path whenever possible.
- Do not use labels like "candidate A", "candidate B", "winner", or "runner-up" in the returned bullets.
- In reusable_rule, write exactly one bullet that starts with "When".
- Keep each bullet short and factual.
- If no clear failure signal exists, return an empty list for failure_signals.
- If `Winner correctness` is `incorrect`, treat the selected winner as the
  strongest failed challenger, not a success pattern:
  - winner_evidence should explain why it still surfaced as the strongest challenger
  - failure_signals should include why it was still wrong or fragile
  - reusable_rule must be a cautionary rule that starts with "When" and warns what
    to avoid, verify, or compare before trusting a similar path

Return ONLY JSON with keys:
text_signals, sample_conditions, winner_evidence, failure_signals, reusable_rule"""


def _get_analysis_llm():
    global _analysis_llm
    if _analysis_llm is None:
        cfg = TimeClawConfig.load()
        router = ModelRouter(cfg.llm)
        try:
            _analysis_llm = router.get_backend("reasoner")
        except Exception:
            _analysis_llm = router.get_backend("executor")
    return _analysis_llm


def _short_text(value: Any, limit: int = 400) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _strip_ground_truth_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in {"ground_truth", "ground_truth_label"}:
                continue
            cleaned[key_str] = _strip_ground_truth_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_ground_truth_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_ground_truth_fields(item) for item in value]
    return value


def _remove_ground_truth_lines(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned_lines = [
        line for line in text.splitlines()
        if "ground truth" not in line.lower() and "ground_truth" not in line.lower()
    ]
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _candidate_prompt_block(candidate_traces: list[dict[str, Any]]) -> str:
    def _display_candidate_name(trace: dict[str, Any], index: int) -> str:
        name = str(trace.get("name") or f"candidate_{index}").strip()
        tool_chain = [t for t in trace.get("tool_chain", []) if t]
        if name and not re.fullmatch(r"(slot|candidate|approach)_\d+", name):
            return name
        if tool_chain:
            return " -> ".join(tool_chain[:5])
        return name

    lines: list[str] = []
    for idx, trace in enumerate(candidate_traces[:4], 1):
        if not isinstance(trace, dict):
            continue
        name = _display_candidate_name(trace, idx)
        tool_chain = [t for t in trace.get("tool_chain", []) if t]
        metrics = _strip_ground_truth_fields(trace.get("metrics") or {})
        lines.append(f"- Tested path: {name}")
        if tool_chain:
            lines.append(f"  Tool chain: {' -> '.join(tool_chain[:5])}")
        if metrics:
            lines.append(f"  Metrics: {_short_text(metrics, 300)}")
        public_snippets = trace.get("public_reasoning_snippets")
        if isinstance(public_snippets, list):
            cleaned_public = [
                str(item).strip()
                for item in public_snippets
                if str(item).strip()
            ]
            if cleaned_public:
                lines.append(
                    "  Execution evidence: "
                    + " | ".join(_short_text(item, 240) for item in cleaned_public[:3])
                )
        steps = trace.get("steps")
        if isinstance(steps, list) and steps:
            lines.append("  Steps:")
            for step in steps[:3]:
                if not isinstance(step, dict):
                    continue
                tool = str(step.get('tool') or '').strip()
                observation = _short_text(step.get("observation"), 240)
                lines.append(f"    - tool={tool}; observation={observation}")
    return "\n".join(lines) if lines else "(no candidate traces)"


def _candidate_alias_map(candidate_traces: list[dict[str, Any]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for idx, trace in enumerate(candidate_traces):
        if not isinstance(trace, dict):
            continue
        display = ""
        name = str(trace.get("name") or "").strip()
        tool_chain = [str(t).strip() for t in trace.get("tool_chain", []) if str(t).strip()]
        if name and not re.fullmatch(r"(slot|candidate|approach)_\d+", name):
            display = name
        elif tool_chain:
            display = " -> ".join(tool_chain[:5])
        elif name:
            display = name
        else:
            display = f"candidate_{idx}"
        for raw in (trace.get("name"), trace.get("candidate_key")):
            alias = str(raw or "").strip()
            if not alias:
                continue
            aliases[alias.lower()] = display
    return aliases


def _sanitize_analysis_item(text: Any, alias_map: dict[str, str]) -> str:
    cleaned = _short_text(text, 500).strip()
    if not cleaned:
        return ""
    if re.search(
        r"(?i)\b(spawn_subagent|evaluate_against_gt|evaluate_batch_against_gt|batch_evaluate_against_gt)\b",
        cleaned,
    ):
        return ""
    for alias, display in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        cleaned = re.sub(rf"(?i)\b{re.escape(alias)}\b", display, cleaned)
    for display in sorted(set(alias_map.values()), key=len, reverse=True):
        cleaned = re.sub(
            rf"(?i)\b{re.escape(display)}\b\s+using\s+\b{re.escape(display)}\b",
            display,
            cleaned,
        )
    cleaned = normalize_learning_clause(cleaned)
    if re.search(
        r"(?i)\b(execution context|assigned(?: exploration)? path|assigned branch|branch(?: instructions?)?|training mode|learning mode|learning-mode|ground truth)\b",
        cleaned,
    ):
        return ""
    if re.search(r"(?i)\bbefore finishing\b", cleaned):
        return ""
    if re.search(
        r"(?i)\b(gt-selected|task-valid(?: candidate)?|branch comparison|most efficient (?:next step|way)|most direct next step|available from a real tool result|may be returned unchanged|current branch state confirms)\b",
        cleaned,
    ):
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;|-")
    return cleaned


def _extract_text_context(task_description: str) -> str:
    if not isinstance(task_description, str) or not task_description.strip():
        return ""
    text = task_description
    if "\n\nUser prompt:\n" in text:
        text = text.split("\n\nUser prompt:\n", 1)[1]
    for label in ("Weather report:", "News:", "Text:", "Context:"):
        idx = text.find(label)
        if idx < 0:
            continue
        snippet = text[idx + len(label):].strip()
        for stop in (
            "\n\nYour task is",
            "\nYour task is",
            "\n\nReturn ONLY",
            "\nReturn ONLY",
            "\n\nQuestion:",
            "\nQuestion:",
        ):
            stop_idx = snippet.find(stop)
            if stop_idx >= 0:
                snippet = snippet[:stop_idx].strip()
                break
        return " ".join(snippet.replace("||", " ").split())
    return ""


def _sanitize_text_signals(
    text_signals: list[str],
    text_context: str,
) -> list[str]:
    if not text_signals:
        return []
    context_lower = text_context.lower()
    cleaned: list[str] = []
    seen: set[str] = set()
    banned_fragments = {
        "scope:",
        "task type:",
        "task description:",
        "sample fingerprint:",
        "weather_aligned_",
        "finance_aligned_",
        "qa_short",
        "qa_long",
        "horizon=",
    }
    for item in text_signals:
        text = str(item).strip().strip('"')
        if not text:
            continue
        lower = text.lower()
        if any(fragment in lower for fragment in banned_fragments):
            continue
        if context_lower and lower not in context_lower:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        cleaned.append(text)
    return cleaned[:6]


def render_learning_analysis_text(analysis: dict[str, Any]) -> str:
    sections = [
        ("Text Signals", analysis.get("text_signals")),
        ("Sample Conditions", analysis.get("sample_conditions")),
        ("Winner Evidence", analysis.get("winner_evidence")),
        ("Failure Signals", analysis.get("failure_signals")),
        ("Reusable Rule", analysis.get("reusable_rule")),
    ]
    lines: list[str] = []
    for title, items in sections:
        lines.append(f"{title}:")
        if isinstance(items, list) and items:
            for item in items:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
        else:
            lines.append("- (none)")
        lines.append("")
    return "\n".join(lines).strip()


def _metric_summary_text(metrics: dict[str, Any]) -> str:
    if not isinstance(metrics, dict):
        return ""
    ordered_keys = (
        "accuracy",
        "correct",
        "mae",
        "avg_mae",
        "rmse",
        "mse",
        "max_mae",
        "mape",
    )
    parts: list[str] = []
    for key in ordered_keys:
        value = metrics.get(key)
        if isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={float(value):.3f}")
    return ", ".join(parts[:4])


def _tool_chain_text(trace: dict[str, Any] | None) -> str:
    if not isinstance(trace, dict):
        return ""
    cleaned = [str(tool).strip() for tool in (trace.get("tool_chain") or []) if str(tool).strip()]
    if not cleaned:
        return ""
    return " -> ".join(cleaned[:5])


def _winner_name_text(trace: dict[str, Any] | None) -> str:
    if not isinstance(trace, dict):
        return "candidate"
    name = str(trace.get("name") or trace.get("candidate_key") or "").strip()
    chain = _tool_chain_text(trace)
    if name and not re.fullmatch(r"(slot|candidate|approach|top_level)_\d+", name.lower()):
        return name
    if chain:
        return chain
    return name or "candidate"


def _scalar_prediction_text(prediction: Any) -> str:
    if prediction is None or isinstance(prediction, (list, tuple, dict)):
        return ""
    return _short_text(prediction, 80).strip().strip('"')


def _join_analysis_items(items: list[str] | Any, limit: int = 2) -> str:
    if not isinstance(items, list):
        return ""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip().strip("- ").rstrip(".")
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return "; ".join(cleaned)


def render_learning_summary_from_analysis(
    analysis: dict[str, Any] | None,
    *,
    winner_trace: dict[str, Any] | None,
    winner_correct: bool | None,
) -> tuple[str, str]:
    """Render final learning text from structured analysis only.

    New definition:
        Final `insight` / `recommendation` should be written in terms of data
        characteristics, tool evidence, and candidate comparison, not executor
        control flow.
    """
    if not isinstance(analysis, dict):
        return "", ""

    sample_conditions = _join_analysis_items(analysis.get("sample_conditions"), limit=2)
    winner_evidence = _join_analysis_items(analysis.get("winner_evidence"), limit=2)
    failure_signals = _join_analysis_items(analysis.get("failure_signals"), limit=2)
    reusable_rule = _join_analysis_items(analysis.get("reusable_rule"), limit=1)

    if winner_correct is False:
        parts: list[str] = []
        if sample_conditions:
            parts.append(f"Relevant data characteristics: {sample_conditions}")
        if winner_evidence:
            parts.append(f"Why this path still looked plausible: {winner_evidence}")
        if failure_signals:
            parts.append(f"Observed failure signals: {failure_signals}")
    else:
        parts = []
        if sample_conditions:
            parts.append(f"Relevant data characteristics: {sample_conditions}")
        if winner_evidence:
            parts.append(f"Why this tool/path fit the data: {winner_evidence}")
        if failure_signals:
            parts.append(f"Observed failure modes in other tested paths: {failure_signals}")

    insight = ". ".join(part.strip().rstrip(".") for part in parts if part).strip()
    if insight and not insight.endswith("."):
        insight += "."

    recommendation = reusable_rule.strip()
    if recommendation and not recommendation.endswith("."):
        recommendation += "."
    return insight, recommendation


def _normalize_analysis_payload(
    payload: dict[str, Any] | None,
    candidate_traces: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]] | None:
    if not isinstance(payload, dict):
        return None
    alias_map = _candidate_alias_map(candidate_traces or [])
    normalized: dict[str, list[str]] = {}
    for key in ("text_signals", "sample_conditions", "winner_evidence", "failure_signals", "reusable_rule"):
        value = payload.get(key)
        if value is None:
            normalized[key] = []
        elif isinstance(value, list):
            items = [
                _sanitize_analysis_item(item, alias_map) if key != "text_signals" else str(item).strip()
                for item in value
            ]
            normalized[key] = [item for item in items if item]
        else:
            text = (
                _sanitize_analysis_item(value, alias_map)
                if key != "text_signals"
                else str(value).strip()
            )
            normalized[key] = [text] if text else []
    if not normalized["sample_conditions"] or not normalized["winner_evidence"] or not normalized["reusable_rule"]:
        return None
    return normalized


async def generate_learning_analysis(
    *,
    scope_key: str = "",
    scope_keys: list[str] | None = None,
    task_type: str,
    domain: str,
    task_description: str,
    sample_fingerprint: str,
    candidate_traces: list[dict[str, Any]],
    best_metrics: dict[str, Any],
    selected_winner_trace: dict[str, Any] | None = None,
    winner_correct: bool | None = None,
    fallback_summary: str = "",
    llm: Any | None = None,
) -> tuple[dict[str, list[str]] | None, str]:
    llm = llm or _get_analysis_llm()
    text_context = _extract_text_context(task_description)
    best_metrics = _strip_ground_truth_fields(best_metrics)
    fallback_summary = _remove_ground_truth_lines(fallback_summary)
    normalized_scope_keys: list[str] = []
    for raw_scope in scope_keys or ([scope_key] if scope_key else []):
        scope = str(raw_scope).strip()
        if scope and scope not in normalized_scope_keys:
            normalized_scope_keys.append(scope)
    user_prompt = (
        f"Scope targets: {_short_text(normalized_scope_keys or ['(unknown)'], 2000)}\n"
        f"Task type: {task_type}\n"
        f"Domain: {domain}\n\n"
        f"Task description:\n{_short_text(task_description, 12000)}\n\n"
        f"Text/Context Source for text_signals ONLY:\n{_short_text(text_context or '(none)', 12000)}\n\n"
        f"Sample fingerprint:\n{_short_text(sample_fingerprint, 2000)}\n\n"
        f"Candidates:\n{_candidate_prompt_block(candidate_traces)}\n\n"
        f"Final accepted winner:\n{_candidate_prompt_block([selected_winner_trace] if selected_winner_trace else [])}\n\n"
        f"Best evaluated metrics (may belong to a non-winning compared candidate):\n{_short_text(best_metrics, 800)}\n\n"
        f"Winner correctness:\n{winner_correct if winner_correct is not None else 'unknown'}\n\n"
        f"Fallback summary:\n{_short_text(fallback_summary, 3000)}"
    )
    try:
        resp = await llm.complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format="json",
        )
        payload = _normalize_analysis_payload(resp.parsed if resp else None, candidate_traces)
        if payload:
            payload["text_signals"] = _sanitize_text_signals(payload.get("text_signals", []), text_context)
            return payload, render_learning_analysis_text(payload)
    except Exception as exc:
        logger.debug(f"[LearningAnalysis] generation failed: {exc}")

    fallback_text = fallback_summary.strip() or "(analysis unavailable)"
    return None, fallback_text
