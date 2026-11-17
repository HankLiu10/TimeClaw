from __future__ import annotations

import asyncio
import ast
import json
import re
from dataclasses import replace
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from timeclaw.benchmark.mtbench_adapter import MTBenchSample


TEXT_ARTIFACT_ROOT = Path(".timeclaw/text_artifacts")
RAW_TEXT_DIR = TEXT_ARTIFACT_ROOT / "raw"
CLEANED_TEXT_DIR = TEXT_ARTIFACT_ROOT / "cleaned"
_TASK_TEXT_VIEW_KEY = "task_text_views"
_TASK_PRUNING_CACHE_VERSION = "v2"
_TEXT_PRUNING_MAX_RETRY_ATTEMPTS = 4
_TEXT_PRUNING_DEFERRED_RETRY_STATE: dict[str, dict[str, int]] = {}
_TEXT_PRUNING_LLM_MAX_RETRIES = 3

_PRUNING_SYSTEM_PROMPT = """You clean task context text for a downstream time-series agent.

Goal:
- Remove only text that is almost certainly irrelevant boilerplate.
- Keep the original wording and order whenever possible.
- Keep all potentially task-relevant facts, entities, numbers, times, and directional statements.
- For long sources, keep only the passages that are actually relevant to the current task prompt instead of echoing the full source.

Rules:
- Use the full task prompt only to judge relevance.
- Do not answer the task.
- Do not summarize or paraphrase the kept passages into fewer facts.
- Do not infer labels, trends, tool choices, or conclusions.
- `cleaned_text` must not be empty when the original text is non-empty.
- If you are unsure whether a short local span matters, keep it.
- If a long section does not bear on the asked entity, time period, comparison target, answer options, or evidence needed for the current task, delete it.
- For transcripts, live blogs, and long articles, it is correct to keep only the turns or paragraphs that carry task-relevant evidence.
- If deleting irrelevant text would make the output empty, return the minimally cleaned original text instead.
- Delete anything that is clearly unrelated to the current task or source evidence.
- Delete obvious duplicates and near-duplicates; keep the most informative original wording once.
- Delete boilerplate such as cookie banners, login prompts, navigation, legal disclaimers, repeated site chrome, ads, publisher chrome, image captions, ticker quote widgets, and account prompts when they do not add task-relevant evidence.
- Delete publisher/editorial wrapper text that promotes the publisher, explains its products, subscription tiers, rankings, screens, or generic investing guidance unless that text is itself the evidence source the task is asking about.
- Delete isolated bylines, date stamps, image markers, empty fragments, and ticker-header snippets when they do not carry evidence.
- If the source is wrapped in serialization such as `Title:` / `Content:` or list-like quoted items, keep the useful text content but remove the wrapper syntax, quote syntax, and empty items.
- If a sentence might matter, keep it.

`removed_summary` requirements:
- Write short, concrete, human-readable descriptions of what was removed.
- Describe the omitted content itself, not your pruning process.
- Good examples: "publisher byline and publish date", "duplicate flood-damage paragraph", "image caption and stock-chart marker".
- Bad examples: "removed irrelevant boilerplate", "removed obvious text corruption/wrapper artifacts while preserving content".

Return ONLY JSON:
{"cleaned_text": "<cleaned original text>", "removed_summary": ["concrete removed-content description 1", "concrete removed-content description 2"]}"""

_PRUNING_PLAIN_RESPONSE_PROMPT = """You clean task context text for a downstream time-series agent.

Goal:
- Remove only text that is almost certainly irrelevant boilerplate.
- Keep the original wording and order whenever possible.
- Keep all potentially task-relevant facts, entities, numbers, times, and directional statements.
- For long sources, keep only the passages that are actually relevant to the current task prompt instead of echoing the full source.

Rules:
- Use the full task prompt only to judge relevance.
- Do not answer the task.
- Do not summarize or paraphrase the kept passages into fewer facts.
- Do not infer labels, trends, tool choices, or conclusions.
- The cleaned text must not be empty when the original text is non-empty.
- If you are unsure whether a short local span matters, keep it.
- If a long section does not bear on the asked entity, time period, comparison target, answer options, or evidence needed for the current task, delete it.
- For transcripts, live blogs, and long articles, it is correct to keep only the turns or paragraphs that carry task-relevant evidence.
- If deleting irrelevant text would make the output empty, return the minimally cleaned original text instead.
- Delete anything that is clearly unrelated to the current task or source evidence.
- Delete obvious duplicates and near-duplicates; keep the most informative original wording once.
- Delete boilerplate such as cookie banners, login prompts, navigation, legal disclaimers, repeated site chrome, ads, publisher chrome, image captions, ticker quote widgets, and account prompts when they do not add task-relevant evidence.
- Delete publisher/editorial wrapper text that promotes the publisher, explains its products, subscription tiers, rankings, screens, or generic investing guidance unless that text is itself the evidence source the task is asking about.
- Delete isolated bylines, date stamps, image markers, empty fragments, and ticker-header snippets when they do not carry evidence.
- If the source is wrapped in serialization such as `Title:` / `Content:` or list-like quoted items, keep the useful text content but remove the wrapper syntax, quote syntax, and empty items.
- If a sentence might matter, keep it.

Output format:
- Return plain text only.
- First line must be exactly `CLEANED_TEXT:`
- Then output the cleaned text.
- Then output a blank line and a line exactly `REMOVED_SUMMARY:`
- Then output 1 to 8 bullet lines starting with `- `
- Each bullet must be a short, concrete, human-readable description of what was removed.
- Describe the omitted content itself, not your pruning process.
- If nothing was removed, output exactly `- none`
- Do not output JSON, markdown code fences, or any extra headings."""


def _normalize_text(text: Any) -> str:
    return str(text or "").replace("\xa0", " ").strip()


def _task_view_store(sample: "MTBenchSample") -> dict[str, Any]:
    metadata = sample.metadata
    store = metadata.get(_TASK_TEXT_VIEW_KEY)
    if not isinstance(store, dict):
        store = {}
        metadata[_TASK_TEXT_VIEW_KEY] = store
    return store


def task_text_view(sample: "MTBenchSample", task: str) -> dict[str, Any]:
    store = sample.metadata.get(_TASK_TEXT_VIEW_KEY)
    if not isinstance(store, dict):
        return {}
    view = store.get(str(task).strip())
    return view if isinstance(view, dict) else {}


def task_text(sample: "MTBenchSample", task: str) -> str:
    view = task_text_view(sample, task)
    cleaned = _normalize_text(view.get("cleaned_text"))
    return cleaned if cleaned else _normalize_text(sample.text)


def _coerce_metadata_timestamp_to_epoch(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value <= 0:
            return None
        # Dataset JSON sometimes stores milliseconds.
        if value > 1e11:
            value /= 1000.0
        return value
    text = str(raw).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _task_text_timestamp(sample: "MTBenchSample") -> tuple[int | str, str | None]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw_value = metadata.get("published_utc")
    if raw_value in (None, ""):
        raw_value = metadata.get("timestamp_ms")
    raw_text = str(raw_value).strip() if raw_value not in (None, "") else None
    epoch = _coerce_metadata_timestamp_to_epoch(raw_value)
    timestamps = [
        float(ts)
        for ts in (sample.timestamps or [])
        if isinstance(ts, (int, float)) and float(ts) > 0
    ]
    if epoch is None or not timestamps:
        return "", raw_text
    nearest_idx = min(
        range(len(timestamps)),
        key=lambda idx: abs(timestamps[idx] - epoch),
    )
    return int(nearest_idx), raw_text


def task_text_data(sample: "MTBenchSample", task: str) -> list[dict[str, Any]] | None:
    text_value = task_text(sample, task)
    if not text_value:
        return None
    view = task_text_view(sample, task)
    timestamp, timestamp_raw = _task_text_timestamp(sample)
    payload: dict[str, Any] = {
        "content": text_value,
        "source": sample.domain,
        "timestamp": timestamp,
    }
    if timestamp_raw:
        payload["timestamp_raw"] = timestamp_raw
    raw_text_ref = str(view.get("raw_text_ref") or "").strip()
    cleaned_text_ref = str(view.get("cleaned_text_ref") or "").strip()
    removed_summary = view.get("removed_summary")
    if raw_text_ref:
        payload["raw_text_ref"] = raw_text_ref
    if cleaned_text_ref:
        payload["cleaned_text_ref"] = cleaned_text_ref
    if isinstance(removed_summary, list):
        cleaned_removed = [str(item).strip() for item in removed_summary if str(item).strip()]
        if cleaned_removed:
            payload["removed_summary"] = cleaned_removed
    return [payload]


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_text_ref(sample: "MTBenchSample") -> str:
    sample_key = f"{sample.dataset}:{sample.file}:{sha1(_normalize_text(sample.text).encode('utf-8')).hexdigest()[:12]}"
    return f"raw_text:{sample_key}"


def _cleaned_text_ref(raw_text_ref: str, task: str, task_prompt: str) -> str:
    digest = sha1(
        f"{_TASK_PRUNING_CACHE_VERSION}|{raw_text_ref}|{task}|{task_prompt}".encode("utf-8")
    ).hexdigest()[:12]
    return f"cleaned_text:{task}:{digest}"


def _artifact_file(root: Path, ref: str) -> Path:
    safe_name = ref.replace(":", "__").replace("/", "_")
    return root / f"{safe_name}.json"


def _task_prompt_text(sample: "MTBenchSample", task: str) -> str:
    from timeclaw.benchmark import mtbench_prompts as prompts

    # The source text is supplied separately to the pruning LLM below.
    # Re-inlining it inside the task prompt doubles the largest article/transcript
    # payloads and can push long finance QA samples into unstable provider behavior.
    raw_sample = replace(
        sample,
        text="[SOURCE_TEXT_PROVIDED_SEPARATELY_FOR_TASK_PRUNING]",
        metadata={k: v for k, v in sample.metadata.items() if k != _TASK_TEXT_VIEW_KEY},
    )
    sys_prompt, user_prompt = prompts.build_prompt(raw_sample, task)
    if sys_prompt.strip():
        return f"System prompt:\n{sys_prompt}\n\nUser prompt:\n{user_prompt}"
    return f"User prompt:\n{user_prompt}"


def _cleanup_whitespace(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_pruning_transport_text(text: str) -> str:
    """Normalize typography that is semantically equivalent but transport-fragile.

    New definition:
    - This normalization is only for the text sent to the pruning LLM.
    - It must not change the persisted raw-text artifact.
    - It should preserve meaning while removing provider-fragile typography.
    - Currency symbols may be rewritten into equivalent textual amounts when
      the provider repeatedly treats the symbol form as transport-fragile.
    """

    normalized = (
        str(text or "")
        .replace("\xa0", " ")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
    )
    normalized = re.sub(r"\$(\d+(?:\.\d+)?)", r"USD \1", normalized)
    return normalized


def _strip_pruning_transport_wrappers(text: str) -> str:
    """Remove provider-fragile publisher wrappers from pruning transport text.

    New definition:
    - This only affects the transient text sent to the pruning LLM.
    - It removes obvious publisher/distribution wrappers that are not part of
      the article body itself, such as newswire headers and source badges.
    """

    cleaned = str(text or "")
    cleaned = re.sub(r"\bPublished on:\s*[^\n]*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\|\s*Source:\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(GLOBE NEWSWIRE\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_TRANSPORT_GLOSSARY_LINE_PATTERNS = (
    re.compile(r"\breflect the total number of shares sold short\b", flags=re.IGNORECASE),
    re.compile(
        r"\bis generally understood to mean the sale of a security that the seller does not own\b",
        flags=re.IGNORECASE,
    ),
)


def _line_has_long_quoted_span(line: str, *, min_chars: int = 120) -> bool:
    quote_char = '"'
    parts = str(line or "").split(quote_char)
    for idx in range(1, len(parts), 2):
        if len(parts[idx]) >= min_chars:
            return True
    return False


def _strip_pruning_transport_fragile_lines(text: str) -> str:
    """Drop transport-fragile lines that repeatedly trigger provider incompletes.

    New definition:
    - This runs only on pruning transport text.
    - It removes two empirically bad line types that stayed non-essential for
      the downstream pruning decision in our failed samples:
      1. glossary/methodology boilerplate lines about short-interest mechanics
      2. single lines that contain exceptionally long direct-quote spans
    - Raw text artifacts remain unchanged and can still be inspected later via
      `raw_text_ref`.
    """

    kept_lines: list[str] = []
    for line in str(text or "").splitlines():
        clean_line = _cleanup_whitespace(line)
        if clean_line:
            if any(pattern.search(clean_line) for pattern in _TRANSPORT_GLOSSARY_LINE_PATTERNS):
                continue
            if _line_has_long_quoted_span(clean_line):
                continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _dedupe_adjacent_blocks(blocks: list[str]) -> list[str]:
    deduped: list[str] = []
    prev_norm = ""
    for block in blocks:
        clean_block = _cleanup_whitespace(block)
        if not clean_block:
            continue
        norm = re.sub(r"\s+", " ", clean_block).strip().lower()
        if norm and norm == prev_norm:
            continue
        deduped.append(clean_block)
        prev_norm = norm
    return deduped


def _dedupe_adjacent_lines_preserve_blank_lines(text: str) -> str:
    """Dedupe repeated neighboring lines without collapsing paragraph breaks.

    New definition:
    - This runs only on pruning transport text.
    - It preserves blank-line separators so long wrapped sources do not get
      flattened into one giant paragraph before reaching the pruning LLM.
    - Only adjacent non-empty duplicate lines are removed.
    """

    out_lines: list[str] = []
    prev_norm = ""
    last_was_blank = False
    for raw_line in str(text or "").splitlines():
        clean_line = _cleanup_whitespace(raw_line)
        if not clean_line:
            if out_lines and not last_was_blank:
                out_lines.append("")
            last_was_blank = True
            prev_norm = ""
            continue
        norm = re.sub(r"\s+", " ", clean_line).strip().lower()
        if norm and norm == prev_norm and not last_was_blank:
            continue
        out_lines.append(clean_line)
        prev_norm = norm
        last_was_blank = False
    return "\n".join(out_lines).strip()


def _unwrap_title_content_wrapper(text: str) -> str:
    match = re.fullmatch(r"\s*Title:\s*(.*?)\s+Content:\s*(\[.*\])\s*", text, flags=re.DOTALL)
    if not match:
        return text
    title = _cleanup_whitespace(match.group(1))
    content_literal = match.group(2)
    try:
        content_items = ast.literal_eval(content_literal)
    except Exception:
        return text
    if not isinstance(content_items, (list, tuple)):
        return text

    blocks: list[str] = []
    if title:
        blocks.append(title)
    for item in content_items:
        clean_item = _cleanup_whitespace(str(item or ""))
        if clean_item:
            blocks.append(clean_item)
    deduped = _dedupe_adjacent_blocks(blocks)
    return "\n\n".join(deduped) if deduped else text


def _prepare_pruning_source_text(raw_text: str) -> str:
    prepared = _cleanup_whitespace(raw_text)
    if not prepared:
        return ""

    prepared = _unwrap_title_content_wrapper(prepared)
    prepared = _normalize_pruning_transport_text(prepared)
    prepared = _strip_pruning_transport_wrappers(prepared)
    prepared = _strip_pruning_transport_fragile_lines(prepared)
    prepared = re.sub(r"\n\s*Article Content:\s*\n+", "\n\n", prepared, flags=re.IGNORECASE)

    prepared = _dedupe_adjacent_lines_preserve_blank_lines(prepared)

    paragraph_blocks = _dedupe_adjacent_blocks(re.split(r"\n\s*\n", prepared))
    if paragraph_blocks:
        prepared = "\n\n".join(paragraph_blocks)

    return _cleanup_whitespace(prepared)


def _normalize_removed_summary(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _cleanup_whitespace(str(item or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out[:8]


def _is_incomplete_error_message(message: str | None) -> bool:
    return "incomplete" in str(message or "").strip().lower()


def _build_original_text_fallback_view(
    *,
    sample: "MTBenchSample",
    task_name: str,
    raw_text_ref: str,
    cleaned_text_ref: str,
    raw_text: str,
    pruning_source_text: str,
    cache_path: Path,
) -> dict[str, Any]:
    """Persist a pruning fallback that keeps the original text unchanged.

    New definition:
    - This fallback is only for terminal pruning incompletes or parser-empty
      pruning responses after retry exhaustion.
    - It does not invent removals; `removed_summary` stays empty.
    - It preserves the existing artifact shape so downstream consumers do not
      need any schema changes.
    """

    cleaned_text = _cleanup_whitespace(raw_text)
    _json_write(cache_path, {
        "ref": cleaned_text_ref,
        "task": task_name,
        "raw_text_ref": raw_text_ref,
        "cleaned_text": cleaned_text,
        "removed_summary": [],
        "raw_chars": len(raw_text),
        "pruning_source_chars": len(pruning_source_text),
        "cleaned_chars": len(cleaned_text),
    })
    view = {
        "task": task_name,
        "raw_text_ref": raw_text_ref,
        "cleaned_text_ref": cleaned_text_ref,
        "cleaned_text": cleaned_text,
        "removed_summary": [],
    }
    _task_view_store(sample)[task_name] = view
    return view


def _parse_plain_pruning_response(text: str) -> tuple[str, list[str]]:
    """Parse the fixed plain-text pruning response format.

    New definition:
    - The pruning LLM no longer returns JSON.
    - It must return a `CLEANED_TEXT:` block followed by a `REMOVED_SUMMARY:`
      block so we can keep the persisted artifact schema unchanged while
      avoiding provider-fragile JSON-plus-large-echo responses.
    """

    raw = str(text or "").replace("\r\n", "\n").strip()
    if not raw:
        return "", []
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    summary_marker = "\nREMOVED_SUMMARY:"
    if raw.startswith("CLEANED_TEXT:"):
        body = raw[len("CLEANED_TEXT:"):].lstrip("\n")
    else:
        return "", []
    if summary_marker in body:
        cleaned_part, summary_part = body.split(summary_marker, 1)
    else:
        cleaned_part, summary_part = body, ""
    cleaned_text = _cleanup_whitespace(cleaned_part)

    removed_summary: list[str] = []
    for line in str(summary_part or "").splitlines():
        item = str(line).strip()
        if not item:
            continue
        if item.startswith("- "):
            item = item[2:].strip()
        elif item.startswith("-"):
            item = item[1:].strip()
        else:
            continue
        if not item or item.lower() == "none":
            continue
        removed_summary.append(item)
    return cleaned_text, _normalize_removed_summary(removed_summary)


def _persist_raw_text(sample: "MTBenchSample") -> str:
    raw_text = _normalize_text(sample.text)
    if not raw_text:
        return ""
    ref = _raw_text_ref(sample)
    path = _artifact_file(RAW_TEXT_DIR, ref)
    if not path.exists():
        _json_write(path, {
            "ref": ref,
            "dataset": sample.dataset,
            "file": sample.file,
            "domain": sample.domain,
            "text": raw_text,
        })
    return ref


def load_raw_text_artifact(raw_text_ref: str) -> dict[str, Any]:
    ref = str(raw_text_ref or "").strip()
    if not ref:
        raise KeyError("raw_text_ref is required")
    path = _artifact_file(RAW_TEXT_DIR, ref)
    if not path.exists():
        raise KeyError(f"raw text artifact not found: {ref}")
    payload = _json_read(path)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid raw text artifact: {ref}")
    return payload


async def prepare_task_text(sample: "MTBenchSample", task: str, *, llm: Any) -> dict[str, Any]:
    from timeclaw.core.retry_control import (
        RetryLater,
        current_work_item_id,
        should_defer_retry_to_scheduler,
    )
    from timeclaw.llm.base import (
        ProviderAccountQuotaError,
        ProviderContentPolicyError,
        is_supplier_pool_retry_message,
        is_retryable_error_message,
        retry_delay_seconds_for_attempt,
    )

    task_name = str(task).strip()
    if not task_name:
        raise ValueError("task is required")

    existing = task_text_view(sample, task_name)
    if existing:
        return existing

    raw_text = _normalize_text(sample.text)
    if not raw_text:
        view = {
            "task": task_name,
            "raw_text_ref": "",
            "cleaned_text_ref": "",
            "cleaned_text": "",
            "removed_summary": [],
        }
        _task_view_store(sample)[task_name] = view
        return view

    raw_text_ref = _persist_raw_text(sample)
    pruning_source_text = _prepare_pruning_source_text(raw_text)
    task_prompt = _task_prompt_text(sample, task_name)
    cleaned_text_ref = _cleaned_text_ref(raw_text_ref, task_name, task_prompt)
    cache_path = _artifact_file(CLEANED_TEXT_DIR, cleaned_text_ref)
    if cache_path.exists():
        cached = _json_read(cache_path)
        view = {
            "task": task_name,
            "raw_text_ref": raw_text_ref,
            "cleaned_text_ref": cleaned_text_ref,
            "cleaned_text": _normalize_text(cached.get("cleaned_text")),
            "removed_summary": _normalize_removed_summary(cached.get("removed_summary", [])),
        }
        _task_view_store(sample)[task_name] = view
        return view

    if llm is None:
        raise RuntimeError("task text pruning requires an llm backend")

    work_item_id = current_work_item_id()
    retry_key = (
        f"{work_item_id}:{cleaned_text_ref}"
        if should_defer_retry_to_scheduler() and work_item_id
        else None
    )
    retry_state = _TEXT_PRUNING_DEFERRED_RETRY_STATE.get(retry_key, {}) if retry_key else {}
    supplier_retry_count = int(retry_state.get("supplier_retry_count", 0) or 0)
    retryable_retry_count = int(retry_state.get("retryable_retry_count", 0) or 0)
    while True:
        try:
            response = await llm.complete(
                [
                    {"role": "system", "content": _PRUNING_PLAIN_RESPONSE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Task prompt:\n{task_prompt}\n\n"
                            f"Original text:\n{pruning_source_text}\n"
                        ),
                    },
                ],
                temperature=0.1,
                max_retries=_TEXT_PRUNING_LLM_MAX_RETRIES,
            )
        except RetryLater:
            raise
        except (ProviderContentPolicyError, ProviderAccountQuotaError):
            if retry_key:
                _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
            raise
        except Exception as exc:
            raw_error = str(exc).strip()
            if is_supplier_pool_retry_message(raw_error):
                supplier_retry_count += 1
                delay = retry_delay_seconds_for_attempt(supplier_retry_count - 1)
                if retry_key:
                    _TEXT_PRUNING_DEFERRED_RETRY_STATE[retry_key] = {
                        "supplier_retry_count": supplier_retry_count,
                        "retryable_retry_count": retryable_retry_count,
                    }
                    raise RetryLater(
                        delay_s=delay,
                        reason=raw_error,
                        layer="text_pruning_supplier_pool",
                    ) from exc
                await asyncio.sleep(delay)
                continue
            if is_retryable_error_message(raw_error):
                retryable_retry_count += 1
                if retryable_retry_count < _TEXT_PRUNING_MAX_RETRY_ATTEMPTS:
                    delay = retry_delay_seconds_for_attempt(retryable_retry_count - 1)
                    if retry_key:
                        _TEXT_PRUNING_DEFERRED_RETRY_STATE[retry_key] = {
                            "supplier_retry_count": supplier_retry_count,
                            "retryable_retry_count": retryable_retry_count,
                        }
                        raise RetryLater(
                            delay_s=delay,
                            reason=raw_error,
                            layer="text_pruning",
                        ) from exc
                    await asyncio.sleep(delay)
                    continue
                if _is_incomplete_error_message(raw_error):
                    if retry_key:
                        _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
                    return _build_original_text_fallback_view(
                        sample=sample,
                        task_name=task_name,
                        raw_text_ref=raw_text_ref,
                        cleaned_text_ref=cleaned_text_ref,
                        raw_text=raw_text,
                        pruning_source_text=pruning_source_text,
                        cache_path=cache_path,
                    )
            if retry_key:
                _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
            raise RuntimeError(f"text pruning llm failed: {raw_error}") from exc
        raw_payload = getattr(response, "raw", None)
        raw_error = ""
        supplier_pool_retry = False
        if isinstance(raw_payload, dict):
            raw_error = str(raw_payload.get("error") or "").strip()
            supplier_pool_retry = bool(raw_payload.get("supplier_pool_retry"))
        if raw_error:
            if supplier_pool_retry or is_supplier_pool_retry_message(raw_error):
                supplier_retry_count += 1
                delay = retry_delay_seconds_for_attempt(supplier_retry_count - 1)
                if retry_key:
                    _TEXT_PRUNING_DEFERRED_RETRY_STATE[retry_key] = {
                        "supplier_retry_count": supplier_retry_count,
                        "retryable_retry_count": retryable_retry_count,
                    }
                    raise RetryLater(
                        delay_s=delay,
                        reason=raw_error,
                        layer="text_pruning_supplier_pool",
                    )
                await asyncio.sleep(delay)
                continue
            if is_retryable_error_message(raw_error):
                retryable_retry_count += 1
                if retryable_retry_count < _TEXT_PRUNING_MAX_RETRY_ATTEMPTS:
                    delay = retry_delay_seconds_for_attempt(retryable_retry_count - 1)
                    if retry_key:
                        _TEXT_PRUNING_DEFERRED_RETRY_STATE[retry_key] = {
                            "supplier_retry_count": supplier_retry_count,
                            "retryable_retry_count": retryable_retry_count,
                        }
                        raise RetryLater(
                            delay_s=delay,
                            reason=raw_error,
                            layer="text_pruning",
                        )
                    await asyncio.sleep(delay)
                    continue
                if _is_incomplete_error_message(raw_error):
                    if retry_key:
                        _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
                    return _build_original_text_fallback_view(
                        sample=sample,
                        task_name=task_name,
                        raw_text_ref=raw_text_ref,
                        cleaned_text_ref=cleaned_text_ref,
                        raw_text=raw_text,
                        pruning_source_text=pruning_source_text,
                        cache_path=cache_path,
                    )
            if retry_key:
                _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
            raise RuntimeError(f"text pruning llm failed: {raw_error}")

        response_text = getattr(response, "text", None) or ""
        cleaned_text, removed_summary = _parse_plain_pruning_response(response_text)
        cleaned_text = _normalize_text(cleaned_text)
        if not cleaned_text:
            retryable_retry_count += 1
            if retryable_retry_count < _TEXT_PRUNING_MAX_RETRY_ATTEMPTS:
                delay = retry_delay_seconds_for_attempt(retryable_retry_count - 1)
                if retry_key:
                    _TEXT_PRUNING_DEFERRED_RETRY_STATE[retry_key] = {
                        "supplier_retry_count": supplier_retry_count,
                        "retryable_retry_count": retryable_retry_count,
                    }
                    raise RetryLater(
                        delay_s=delay,
                        reason="text pruning returned empty cleaned_text",
                        layer="text_pruning",
                    )
                await asyncio.sleep(delay)
                continue
            if not str(response_text).strip() or "CLEANED_TEXT:" not in str(response_text):
                if retry_key:
                    _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
                return _build_original_text_fallback_view(
                    sample=sample,
                    task_name=task_name,
                    raw_text_ref=raw_text_ref,
                    cleaned_text_ref=cleaned_text_ref,
                    raw_text=raw_text,
                    pruning_source_text=pruning_source_text,
                    cache_path=cache_path,
                )
            if retry_key:
                _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
            raise RuntimeError("text pruning returned empty cleaned_text for non-empty source text")
        break

    cleaned_text = _cleanup_whitespace(cleaned_text)
    if retry_key:
        _TEXT_PRUNING_DEFERRED_RETRY_STATE.pop(retry_key, None)
    _json_write(cache_path, {
        "ref": cleaned_text_ref,
        "task": task_name,
        "raw_text_ref": raw_text_ref,
        "cleaned_text": cleaned_text,
        "removed_summary": removed_summary,
        "raw_chars": len(raw_text),
        "pruning_source_chars": len(pruning_source_text),
        "cleaned_chars": len(cleaned_text),
    })
    view = {
        "task": task_name,
        "raw_text_ref": raw_text_ref,
        "cleaned_text_ref": cleaned_text_ref,
        "cleaned_text": cleaned_text,
        "removed_summary": removed_summary,
    }
    _task_view_store(sample)[task_name] = view
    return view
