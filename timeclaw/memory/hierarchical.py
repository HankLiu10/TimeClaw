"""Agent Memory System — OpenClaw-inspired file-first architecture.

Five components (aligned with OpenClaw naming):

SOUL.md (fixed, human-written):
    Agent identity and behavioral guidelines.  Corresponds to OpenClaw SOUL.md.
    Not learned — maintained by developers.

Notes (auto-recorded):
    Natural-language summaries of each pipeline run + outcome.
    Append-only, stored as scope-sharded raw experience records.

MEMORY (auto-distilled from notes):
    LLM-distilled causal attributions explaining WHY certain patterns of
    success/failure occur.  Scoped to ``scope_key``.  Additionally holds
    task-bucket ``cross_domain`` scopes for universal operating strategies
    distilled across corresponding scopes.
    Corresponds to OpenClaw MEMORY.md, but auto-distilled rather than
    manually curated.

TOOLS.md (auto-distilled from notes):
    Per-scope per-tool usage notes learned from experience — when each
    tool works well or poorly for that scope, helping the agent distinguish
    between functionally similar tools under the current task boundary.

SKILLS/ (auto-distilled from MEMORY):
    Scene-specific SOPs (Standard Operating Procedures) per (domain, task).
    Step-by-step operational guides.  Corresponds to OpenClaw SKILLS.md.

Distillation chain:
    Notes  ──→  MEMORY (per-scope + task-bucket cross_domain)
                 └──→  TOOLS.md (per-scope cards)
                 └──→  SKILLS/

New definition:
    ``scope_key`` is the canonical storage and retrieval key for one
    benchmark scope. It should match the benchmark's dataset-level scope
    name directly, for example ``weather_aligned_short_forecast``.

New definition:
    ``Notes`` are now a distillation-source layer only. They remain
    append-only raw records used to distill MEMORY / TOOLS / SKILLS, but
    they are no longer injected directly into the agent prompt and no
    longer rely on a vector-store retrieval path.
"""

from __future__ import annotations

import asyncio
from contextvars import copy_context
from difflib import SequenceMatcher
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from timeclaw.core import trace_logger as trace
from timeclaw.core.learning_text import normalize_learning_clause

MEMORY_ROOT = Path(".timeclaw/hierarchical_memory")
NOTES_DIR = MEMORY_ROOT / "notes"
MEMORY_DIR = MEMORY_ROOT / "memory"
TOOLS_FILE = MEMORY_ROOT / "TOOLS.md"
SKILLS_DIR = MEMORY_ROOT / "skills"
STATE_DIR = MEMORY_ROOT / "state"
DISTILL_STATE_FILE = STATE_DIR / "distill_state.json"
HISTORY_DIR = MEMORY_ROOT / "history"
SOUL_FILE = Path("timeclaw/memory/SOUL.md")

CROSS_DOMAIN_SCOPE = "cross_domain"
CROSS_DOMAIN_SCOPE_PREFIX = "cross_domain__"
_FRAMEWORK_TOOLS = {
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
    "spawn_subagent",
    "none",
}
_NOTE_VALIDATION_TOOLS = {
    "consistency_check",
    "bound_check",
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
}
_TOOL_NAME_ALIASES = {
    "multimodal_classifier": "llm_reasoning_classification",
    "reasoning_based_multimodal_classification": "llm_reasoning_classification",
    "direct_indicator_reasoning": "llm_reasoning_indicator",
    "reasoning_based_indicator_analysis": "llm_reasoning_indicator",
    "context_aware_forecast": "llm_reasoning_forecast",
    "multimodal_qa": "llm_reasoning_qa",
}

# New definition:
# - `scoped note count`: number of note blocks written for one exact scope_key
# - `global note count`: total number of note blocks across all scopes
#
# Distillation cadence:
# - Notes -> MEMORY: whenever one scope accumulates at least 5 undistilled notes
# - MEMORY -> SKILLS / TOOLS / cross_domain: rebuild from the latest full MEMORY
#   view when their source fingerprint changes
_DISTILL_L1_TO_L2 = 5
_MEMORY_RULE_CAP = 30
_L1_RETRIEVAL_K = 5


def _task_bucket_from_scope(scope: str) -> str:
    scope_name = str(scope).strip()
    if not scope_name:
        return ""
    if scope_name.startswith(CROSS_DOMAIN_SCOPE_PREFIX):
        return scope_name[len(CROSS_DOMAIN_SCOPE_PREFIX):].strip()
    parts = scope_name.split("_", 1)
    if len(parts) == 2 and parts[0] in {"weather", "finance"}:
        return parts[1].strip()
    return scope_name


def _cross_domain_scope_for_bucket(bucket: str) -> str:
    bucket_name = str(bucket).strip()
    return f"{CROSS_DOMAIN_SCOPE_PREFIX}{bucket_name}" if bucket_name else CROSS_DOMAIN_SCOPE


def _cross_domain_scope_for_scope(scope: str) -> str:
    return _cross_domain_scope_for_bucket(_task_bucket_from_scope(scope))


def _is_cross_domain_scope(scope: str) -> bool:
    scope_name = str(scope).strip()
    return scope_name == CROSS_DOMAIN_SCOPE or scope_name.startswith(CROSS_DOMAIN_SCOPE_PREFIX)


def _available_tool_names() -> set[str]:
    try:
        from timeclaw.clawhub.tool_registry import discover_and_register, list_tool_instances

        tools = list_tool_instances()
        if not tools:
            discover_and_register()
            tools = list_tool_instances()
        return {t.name for t in tools}
    except Exception:
        return set()


def _canonicalize_tool_list(tool_names: list[str]) -> list[str]:
    available = _available_tool_names()
    out: list[str] = []
    seen: set[str] = set()
    for raw_name in tool_names:
        name = str(raw_name).strip()
        if not name:
            continue
        key = re.sub(r"\s+", "_", name.lower())
        canonical = _TOOL_NAME_ALIASES.get(key, name)
        if canonical not in available:
            continue
        if canonical in _FRAMEWORK_TOOLS or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Notes (append-only experience log, distillation source only)
# ═══════════════════════════════════════════════════════════════════════

class Notes:
    """Append-only natural-language note store used as distillation source.

    New definition:
    - `Notes` is the raw note collection/layer name
    - one individual record is still called a `note`
    - storage is scope-sharded on disk under `notes/<scope>.md`
    """

    def __init__(self):
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        self._total_count = self._count_notes_on_disk()
        self._scope_counts = self._count_scope_notes_on_disk()
        self._pending: list[dict[str, Any]] = []
        self._scope_locks: dict[str, asyncio.Lock] = {}
        self._lock_registry_lock = asyncio.Lock()
        self._counts_lock = asyncio.Lock()
        self._note_llm = None

    @property
    def total(self) -> int:
        return self._total_count

    def scoped_total(self, scope: str) -> int:
        return int(self._scope_counts.get(scope, 0) or 0)

    async def _get_llm(self):
        if self._note_llm is None:
            from timeclaw.core.config import TimeClawConfig
            from timeclaw.llm.router import ModelRouter
            cfg = TimeClawConfig.load()
            router = ModelRouter(cfg.llm)
            self._note_llm = router.get_backend("reasoner")
        return self._note_llm

    async def add(self, run_context: dict[str, Any]) -> int:
        note_text = self._build_note(run_context)
        scope = str(run_context.get("scope_key", "")).strip()
        scope_lock = await self._scope_lock(scope)

        async with scope_lock:
            self._append_to_scope_log(note_text, run_context)
            async with self._counts_lock:
                self._total_count += 1
                if scope:
                    self._scope_counts[scope] = self._scope_counts.get(scope, 0) + 1
                return self._total_count

    async def _scope_lock(self, scope: str) -> asyncio.Lock:
        scope_name = scope or "__default__"
        async with self._lock_registry_lock:
            lock = self._scope_locks.get(scope_name)
            if lock is None:
                lock = asyncio.Lock()
                self._scope_locks[scope_name] = lock
            return lock

    @staticmethod
    def _scope_storage_name(scope: str) -> str:
        scope_name = str(scope).strip() or "__default__"
        return re.sub(r"[^A-Za-z0-9_.-]+", "__", scope_name)

    @classmethod
    def _scope_log_path(cls, scope: str) -> Path:
        return NOTES_DIR / f"{cls._scope_storage_name(scope)}.md"

    @staticmethod
    def _count_notes_on_disk() -> int:
        total = 0
        for path in NOTES_DIR.glob("*.md"):
            text = path.read_text()
            total += sum(1 for block in text.split("\n### ") if block.strip())
        return total

    @classmethod
    def _count_scope_notes_on_disk(cls) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in NOTES_DIR.glob("*.md"):
            for item in cls._note_items_from_text(path.name, path.read_text()):
                scope = cls._extract_scope_from_note_block(item["text"])
                if scope:
                    counts[scope] = counts.get(scope, 0) + 1
        return counts

    @staticmethod
    def _stable_note_id(file_name: str, note: str) -> str:
        payload = f"{file_name}\n{note}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:16]

    @classmethod
    def _note_items_from_text(cls, file_name: str, text: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for raw in reversed(text.split("\n### ")):
            note = raw.strip()
            if not note:
                continue
            items.append({
                "id": cls._stable_note_id(file_name, note),
                "text": note,
            })
        return items

    @staticmethod
    def _extract_scope_from_note_block(note: str) -> str:
        lines = note.splitlines()
        for idx, line in enumerate(lines):
            if line.strip() == "Scope:" and idx + 1 < len(lines):
                scope = lines[idx + 1].strip()
                if scope:
                    return scope
        first_content = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_content = stripped
                break
        match = re.match(r"(?:###\s*)?(?:\[[^\]]+\]\s+)?([^\n]+?)\s+[—-]\s+(OK|FAIL|SCORED|LOGGED)\s*$", first_content)
        if match:
            return match.group(1).strip()
        inline = re.match(r"\[([^\]]+)\]\s+(?:OK|FAIL|SCORED|LOGGED)\b", first_content)
        if inline:
            return inline.group(1).strip()
        return ""

    def _build_note(self, ctx: dict[str, Any]) -> str:
        scope = ctx.get("scope_key", "")
        domain = ctx.get("domain", "unknown")
        task = ctx.get("task", "unknown")
        metrics = ctx.get("metrics") or {}
        outcome = self._outcome_label(ctx.get("correct"), metrics)
        predicted = str(ctx.get("predicted", ""))
        _GT_KEYS = {"ground_truth_label", "ground_truth", "correct"}
        metric_items = [
            (k, v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and k not in _GT_KEYS
        ] if metrics else []
        fingerprint = ctx.get("data_fingerprint", "")[:500]
        description = ctx.get("description", "")
        pipeline_summary = ctx.get("pipeline_summary", "")
        analysis_structured = ctx.get("analysis_structured") or {}

        candidate_traces = ctx.get("candidate_traces") or []
        tools = self._winner_tools_for_note(ctx.get("tool_chain", []), candidate_traces)
        winner_generators, winner_validators = self._split_tool_roles(tools)
        approaches = self._format_approaches(candidate_traces)
        candidates = self._format_candidates(candidate_traces)
        task_prompt, numeric_block, _text_block = self._task_prompt_blocks(description)
        source_lines = self._source_lines(ctx)

        scope_label = scope or f"{domain}/{task}"
        lines = [f"[{scope_label}] {outcome}", "", "Scope:", scope_label]
        if source_lines:
            lines.extend(["", "Source:"])
            lines.extend(source_lines)
        if task_prompt:
            lines.extend([
                "",
                "Task Prompt:",
                "[TASK_PROMPT_BEGIN]",
                task_prompt,
                "[TASK_PROMPT_END]",
            ])
        if numeric_block:
            lines.extend([
                "",
                "Numeric Time Series:",
                "[NUMERIC_TIME_SERIES_BEGIN]",
                numeric_block,
                "[NUMERIC_TIME_SERIES_END]",
            ])
        text_signals = self._select_text_signals(
            description=description,
            task=task,
            analysis_structured=analysis_structured,
        )
        if text_signals:
            lines.extend(["", "Text Signals:"])
            lines.extend(f"- {item}" for item in text_signals)
        lines.extend(["", "Winner Tools:"])
        winner_tools = winner_generators + winner_validators
        if winner_tools:
            lines.extend(f"- {tool}" for tool in winner_tools[:20])
        else:
            lines.append("- none")
        lines.extend(["", "Prediction:", predicted])
        if metric_items:
            lines.extend(["", "Metrics:"])
            lines.extend(
                f"- {key}: {value:.3f}" for key, value in metric_items
            )
        if fingerprint:
            lines.extend(["", "Fingerprint:", fingerprint])
        if approaches:
            lines.extend(["", "Approaches:"])
            lines.extend(f"- {item}" for item in approaches[:5])
        if candidates:
            lines.extend(["", "Candidates:", "[CANDIDATES_BEGIN]"])
            lines.extend(candidates[:3])
            lines.append("[CANDIDATES_END]")
        rendered_summary = self._analysis_without_text_signals(pipeline_summary, bool(text_signals))
        if rendered_summary:
            lines.extend(["", "Analysis:", "[ANALYSIS_BEGIN]", rendered_summary, "[ANALYSIS_END]"])
        return "\n".join(lines)

    @staticmethod
    def _source_lines(ctx: dict[str, Any]) -> list[str]:
        source = ctx.get("source_metadata") or {}
        if not isinstance(source, dict):
            source = {}
        dataset_name = str(source.get("dataset_name", "")).strip()
        sample_ref = str(source.get("sample_ref", "")).strip()
        source_file = str(source.get("source_file", "")).strip()
        sample_idx = source.get("sample_idx")
        task_name = str(source.get("task_name", "")).strip()
        trace_id = str(ctx.get("trace_id") or source.get("trace_id") or "").strip()

        lines: list[str] = []
        if dataset_name:
            lines.append(f"- Dataset: {dataset_name}")
        if sample_ref:
            lines.append(f"- Sample Ref: {sample_ref}")
        if source_file:
            lines.append(f"- Source File: {source_file}")
        if isinstance(sample_idx, int):
            lines.append(f"- Sample Index: {sample_idx}")
        if task_name:
            lines.append(f"- Task: {task_name}")
        if trace_id:
            lines.append(f"- Trace ID: {trace_id}")
            lines.append(f"- Trace Path: .timeclaw/traces/{trace_id}.jsonl")
        return lines

    @classmethod
    def _split_tool_roles(cls, tools: list[str]) -> tuple[list[str], list[str]]:
        generators: list[str] = []
        validators: list[str] = []
        for raw_tool in tools:
            tool = str(raw_tool).strip()
            if not tool:
                continue
            if tool in _NOTE_VALIDATION_TOOLS:
                validators.append(tool)
            else:
                generators.append(tool)
        return generators, validators

    @classmethod
    def _task_prompt_blocks(cls, description: str) -> tuple[str, str, str]:
        compact = cls._task_without_long_text(description).strip()
        if not compact:
            return "", "", ""

        numeric_match = cls._extract_numeric_series_block(description)
        text_match = cls._extract_text_context_block(description)

        if numeric_match:
            placeholder = f"{numeric_match['label']}\n[see Numeric Time Series block below]\n"
            compact = compact.replace(str(numeric_match["full_text"]), placeholder, 1)
        if text_match:
            placeholder = f"{text_match['label']}\n[see Text Context block below]\n"
            compact = compact.replace(str(text_match["full_text"]), placeholder, 1)

        compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
        numeric_block = cls._render_extracted_block(numeric_match)
        task_prompt = cls._compact_task_prompt(compact)
        return task_prompt, numeric_block, ""

    @staticmethod
    def _compact_task_prompt(compact_prompt: str, max_lines: int = 6) -> str:
        if not isinstance(compact_prompt, str) or not compact_prompt.strip():
            return ""

        text = compact_prompt.replace("\r", "\n")
        if "\n\nUser prompt:\n" in text:
            _system, user_text = text.split("\n\nUser prompt:\n", 1)
        else:
            user_text = text
        user_text = re.sub(r"\[see Text Context block below\]", "[see Text Signals below]", user_text)
        user_text = re.sub(r"\s+", " ", user_text).strip()
        if not user_text:
            return ""

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", user_text)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [user_text]

        selected: list[str] = []
        seen: set[str] = set()
        priority_patterns = [
            r"\byour task is\b",
            r"\binput time series\b|\bstock price\b|\btemperature readings\b|\btime series spans\b|\bestimation period\b",
            r"\bevery 24\b|\bcompare\b|\bcalculate\b|\bclassify\b|\bthreshold\b|\bnext[- ]day\b",
            r"\breturn\b|\boutput\b|\bensure the output\b|\bnumber of\b|\bjson\b",
        ]
        for pattern in priority_patterns:
            for sentence in sentences:
                lower = sentence.lower()
                if re.search(pattern, lower) and sentence not in seen:
                    selected.append(sentence)
                    seen.add(sentence)
                    if len(selected) >= max_lines:
                        break
            if len(selected) >= max_lines:
                break

        for sentence in sentences:
            if len(selected) >= max_lines:
                break
            if sentence not in seen:
                selected.append(sentence)
                seen.add(sentence)

        return "\n".join(f"- {item}" for item in selected[:max_lines])

    @staticmethod
    def _render_extracted_block(block: dict[str, str] | None) -> str:
        if not isinstance(block, dict):
            return ""
        label = str(block.get("label", "")).strip()
        body = str(block.get("body", "")).strip()
        if label and body:
            return f"{label}\n{body}"
        return body or label

    @classmethod
    def _extract_numeric_series_block(cls, description: str) -> dict[str, str] | None:
        if not isinstance(description, str) or not description.strip():
            return None

        explicit_patterns = [
            r"(?P<label>The input prices are:)\s*(?P<body>\[(?:[^\[\]]|\n)+?\])",
            r"(?P<label>The input temperature readings are:)\s*(?P<body>\[(?:[^\[\]]|\n)+?\])",
            r"(?P<label>(?:\d+-day )?temperature time series[^\n:]*:)\s*(?P<body>\[(?:[^\[\]]|\n)+?\])",
            r"(?P<label>stock price of .*?:)\s*(?P<body>\[(?:[^\[\]]|\n)+?\])",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, description, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return {
                    "label": str(match.group("label")).strip(),
                    "body": str(match.group("body")).strip(),
                    "full_text": str(match.group(0)),
                }

        match = re.search(
            r"\[(?:\s*-?\d+(?:\.\d+)?\s*,){4,}\s*-?\d+(?:\.\d+)?\s*\]",
            description,
            flags=re.DOTALL,
        )
        if not match:
            return None
        return {
            "label": "Numeric time series:",
            "body": str(match.group(0)).strip(),
            "full_text": str(match.group(0)),
        }

    @staticmethod
    def _extract_text_context_block(description: str) -> dict[str, str] | None:
        if not isinstance(description, str) or not description.strip():
            return None

        patterns = [
            (
                r"(?P<label>The news article states:)\s*(?P<body>.*?)(?=(?:\n\nPlease return|\n\nBased on|\n\nReturn|\n\nNow, you need|Please return|Return your|Question:| Answer:|$))",
                re.DOTALL,
            ),
            (
                r"(?P<label>Weather report:)\s*(?P<body>.*?)(?=(?:\n\nYour task is|\n\nReturn ONLY|\n\nQuestion:|Return ONLY|$))",
                re.DOTALL,
            ),
            (
                r"(?P<label>News:)\s*(?P<body>.*?)(?=(?:\n\nYour task is|\n\nReturn ONLY|\n\nQuestion:|Return ONLY|$))",
                re.DOTALL,
            ),
            (
                r"(?P<label>Text:)\s*(?P<body>.*?)(?=(?:\n\nYour task is|\n\nReturn ONLY|\n\nQuestion:|Return ONLY|$))",
                re.DOTALL,
            ),
            (
                r"(?P<label>Context:)\s*(?P<body>.*?)(?=(?:\n\nQuestion:| Question:| Answer:|Return your answer|$))",
                re.DOTALL,
            ),
            (
                r"(?P<label>weather event report:)\s*(?P<body>.*?)(?=(?:Question:| Answer:|$))",
                re.IGNORECASE | re.DOTALL,
            ),
            (
                r"(?P<label>News published at [^:]+:)\s*(?P<body>.*?)(?=(?:Question:| Give your answer| Answer:|$))",
                re.DOTALL,
            ),
        ]
        for pattern, flags in patterns:
            match = re.search(pattern, description, flags=flags)
            if match:
                body = str(match.group("body")).strip()
                if not body:
                    continue
                return {
                    "label": str(match.group("label")).strip(),
                    "body": body,
                    "full_text": str(match.group(0)),
                }
        return None

    @classmethod
    def _select_text_signals(
        cls,
        description: str,
        task: str,
        analysis_structured: dict[str, Any],
    ) -> list[str]:
        llm_signals = analysis_structured.get("text_signals")
        if isinstance(llm_signals, list):
            cleaned = [str(item).strip() for item in llm_signals if str(item).strip()]
            if cleaned:
                return cleaned[:6]
        return cls._extract_text_signals(description, task=task, task_description=description)

    @staticmethod
    def _analysis_without_text_signals(pipeline_summary: str, has_top_level_text_signals: bool) -> str:
        if not has_top_level_text_signals or not isinstance(pipeline_summary, str):
            return pipeline_summary
        marker = "Text Signals:\n"
        if not pipeline_summary.startswith(marker):
            return pipeline_summary
        split_marker = "\n\nSample Conditions:\n"
        idx = pipeline_summary.find(split_marker)
        if idx < 0:
            return pipeline_summary
        return "Sample Conditions:\n" + pipeline_summary[idx + len(split_marker):].lstrip()

    @staticmethod
    def _round_display_numbers(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return [Notes._round_display_numbers(item) for item in value]
        if isinstance(value, tuple):
            return [Notes._round_display_numbers(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): Notes._round_display_numbers(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _extract_text_signals(
        description: str,
        task: str = "",
        task_description: str = "",
        max_sentences: int = 5,
        max_chars: int = 1200,
    ) -> list[str]:
        if not isinstance(description, str) or not description.strip():
            return []
        text = description
        if "\n\nUser prompt:\n" in text:
            text = text.split("\n\nUser prompt:\n", 1)[1]
        signal_blocks: list[str] = []
        for label in ("Weather report:", "News:", "Text:", "Context:", "The news article states:"):
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
                "\n\nPlease return",
                "\nPlease return",
                "Please return",
                "Return your",
                "\n\nBased on",
                "\nBased on",
                "\n\nNow, you need",
                "\nNow, you need",
            ):
                stop_idx = snippet.find(stop)
                if stop_idx >= 0:
                    snippet = snippet[:stop_idx].strip()
                    break
            snippet = " ".join(snippet.replace("||", " ").split())
            if snippet:
                signal_blocks.append(snippet)
            break
        if not signal_blocks:
            return []
        cleaned = signal_blocks[0]
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]
        ranked_sentences = Notes._rank_text_signals_for_task(
            sentences,
            task=task,
            task_description=task_description or description,
        )
        selected: list[str] = []
        used = 0
        for sentence in ranked_sentences:
            if len(selected) >= max_sentences:
                break
            if used + len(sentence) > max_chars and selected:
                break
            selected.append(sentence)
            used += len(sentence)
        return selected

    @staticmethod
    def _rank_text_signals_for_task(
        sentences: list[str],
        task: str,
        task_description: str = "",
    ) -> list[str]:
        """Rank text sentences by task relevance before truncation.

        New definition:
            `task focus` means the small set of content words extracted from
            the full task description by a lightweight local prompt template.
            It is only used to re-rank original text sentences; it does not
            rewrite, summarize, or introduce task-specific hardcoded rules.
        """
        if not sentences:
            return []

        task_focus = Notes._task_focus_terms(task_description, task=task)

        scored: list[tuple[int, int, str]] = []
        for idx, sentence in enumerate(sentences):
            lower = sentence.lower()
            score = 0
            for term in task_focus:
                if term in lower:
                    score += 1
            if any(ch.isdigit() for ch in sentence):
                score += 1
            if any(token in lower for token in ("between", "near", "caused", "resulted", "damage", "injur", "fatal")):
                score += 1
            scored.append((-score, idx, sentence))

        scored.sort()
        return [sentence for _, _, sentence in scored]

    @staticmethod
    def _task_focus_terms(task_description: str, task: str = "", limit: int = 12) -> list[str]:
        if not isinstance(task_description, str) or not task_description.strip():
            return []

        focus_text = task_description
        lower = focus_text.lower()
        for marker in (
            "your task is to",
            "your task is",
            "task:",
            "question:",
            "return only",
        ):
            idx = lower.find(marker)
            if idx >= 0:
                focus_text = focus_text[idx + len(marker):]
                break

        prompt_prefix = (
            f"Task type: {task}. Focus on the concrete entities, targets, variables, "
            f"and requested outputs that matter for solving this task.\n"
        )
        focus_text = prompt_prefix + focus_text
        focus_text = focus_text.replace("\n", " ")
        focus_text = re.sub(r"[^a-zA-Z0-9_\- ]+", " ", focus_text)
        tokens = [tok.lower() for tok in focus_text.split() if tok.strip()]

        stop = {
            "the", "a", "an", "and", "or", "to", "of", "for", "with", "in", "on", "at",
            "from", "this", "that", "these", "those", "is", "are", "be", "as", "by",
            "only", "return", "nothing", "else", "using", "based", "given", "predict",
            "identify", "answer", "choose", "best", "next", "past", "values", "value",
            "json", "array", "number", "numbers",
        }
        selected: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in stop or len(token) < 3 or token.isdigit():
                continue
            if token not in seen:
                seen.add(token)
                selected.append(token)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _task_without_long_text(description: str) -> str:
        if not isinstance(description, str) or not description.strip():
            return ""
        text = description
        if "\n\nUser prompt:\n" not in text:
            return text
        system_part, user_part = text.split("\n\nUser prompt:\n", 1)
        for label in ("Weather report:", "News:", "Text:", "Context:", "The news article states:"):
            idx = user_part.find(label)
            if idx < 0:
                continue
            prefix = user_part[:idx].rstrip()
            suffix = user_part[idx + len(label):]
            for stop in (
                "\n\nYour task is",
                "\nYour task is",
                "\n\nReturn ONLY",
                "\nReturn ONLY",
                "\n\nQuestion:",
                "\nQuestion:",
                "\n\nPlease return",
                "\nPlease return",
                "Please return",
                "Return your",
                "\n\nBased on",
                "\nBased on",
                "\n\nNow, you need",
                "\nNow, you need",
            ):
                stop_idx = suffix.find(stop)
                if stop_idx >= 0:
                    suffix = suffix[stop_idx:].lstrip()
                    break
            user_part = f"{prefix}\n\n{label}\n[see Text Signals below]\n\n{suffix}".strip()
            break
        return f"{system_part}\n\nUser prompt:\n{user_part}".strip()

    @staticmethod
    def _visible_tool_chain(tool_chain: list[str]) -> list[str]:
        """Filter out orchestration / supervision tools from learned tool traces."""
        return [t for t in tool_chain if t and t not in _FRAMEWORK_TOOLS]

    @classmethod
    def _winner_tools_for_note(
        cls,
        tool_chain: list[str],
        candidate_traces: list[dict[str, Any]],
    ) -> list[str]:
        visible = cls._visible_tool_chain(tool_chain)
        if visible:
            return visible
        winner = cls._select_best_candidate_trace(candidate_traces)
        if not winner:
            return []
        return cls._visible_tool_chain(winner.get("tool_chain", []))

    @staticmethod
    def _candidate_score(metrics: dict[str, Any]) -> tuple[int, float]:
        if isinstance(metrics.get("accuracy"), (int, float)):
            return (0, -float(metrics["accuracy"]))
        if isinstance(metrics.get("correct"), bool):
            return (0, 0.0 if metrics["correct"] else 1.0)
        for key in ("mae", "avg_mae", "rmse", "mse", "max_mae"):
            if isinstance(metrics.get(key), (int, float)):
                return (1, float(metrics[key]))
        return (9, 0.0)

    @classmethod
    def _select_best_candidate_trace(
        cls,
        candidate_traces: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        valid = [
            trace for trace in candidate_traces
            if isinstance(trace, dict)
            and isinstance(trace.get("metrics"), dict)
            and trace.get("metrics")
        ]
        if not valid:
            return None
        return min(valid, key=lambda trace: cls._candidate_score(trace["metrics"]))

    @staticmethod
    def _outcome_label(correct: bool | None, metrics: dict[str, Any]) -> str:
        """Map MTBench-style training outcomes to note labels.

        New definition:
            - ``OK`` / ``FAIL`` only for tasks with explicit binary correctness
              (classification / QA / correlation style metrics).
            - ``SCORED`` for continuous-score tasks that only provide graded
              errors such as MAE / RMSE / component MAE.
            - Explicit parse/type/tool errors still map to ``FAIL``.
        """
        if correct is True:
            return "OK"
        if correct is False:
            return "FAIL"
        if not metrics:
            return "LOGGED"
        if bool(metrics.get("parse_error")) or bool(metrics.get("type_error")):
            return "FAIL"
        if any(
            bool(v) for k, v in metrics.items()
            if k.endswith("_error") and k not in {"parse_error", "type_error"}
        ):
            return "FAIL"
        return "SCORED"

    @staticmethod
    def _format_approaches(candidate_traces: list[dict[str, Any]]) -> list[str]:
        """Build ranked candidate summaries directly from candidate traces."""
        if not candidate_traces:
            return []
        ranked = Notes._ranked_candidate_summaries(candidate_traces)
        return ranked[:5]

    @staticmethod
    def _format_candidates(candidate_traces: list[dict[str, Any]]) -> list[str]:
        if not candidate_traces:
            return []
        formatted: list[str] = []
        for idx, trace in enumerate(candidate_traces[:3]):
            name = Notes._display_candidate_name(trace, idx)
            metrics = trace.get("metrics") or {}
            _, score_label, _ = Notes._candidate_score(metrics)
            header = f"- Candidate: {name}"
            if score_label:
                header += f" ({score_label})"
            block = [header]
            if metrics:
                metric_text = Notes._stringify_metrics(metrics)
                if metric_text:
                    block.append(f"  Metrics: {metric_text}")
            tool_chain = [t for t in trace.get("tool_chain", []) if t and t not in _FRAMEWORK_TOOLS]
            generator_tools, validation_tools = Notes._split_tool_roles(tool_chain)
            tool_parts: list[str] = []
            if generator_tools:
                tool_parts.append(" -> ".join(generator_tools))
            if validation_tools:
                tool_parts.append(f"Validation: {' -> '.join(validation_tools)}")
            if tool_parts:
                block.append(f"  Tools: {' | '.join(tool_parts)}")
            prediction = Notes._short_prediction_value(trace.get("prediction"))
            if prediction:
                block.append(f"  Prediction: {prediction}")
            reasoning = Notes._compact_candidate_reasoning(trace)
            if reasoning:
                block.append(f"  Reasoning: {reasoning}")
            weakness = Notes._compact_candidate_weakness(trace)
            if weakness:
                block.append(f"  Weakness: {weakness}")
            formatted.append("\n".join(block))
        return formatted

    @staticmethod
    def _short_prediction_value(prediction: Any) -> str:
        if prediction is None or isinstance(prediction, (list, tuple, dict)):
            return ""
        rendered = Notes._stringify_value(prediction).strip()
        if not rendered or len(rendered) > 80 or "\n" in rendered:
            return ""
        return rendered

    @staticmethod
    def _sanitize_candidate_reasoning(text: str) -> str:
        cleaned = str(text).strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"^\s*\d+[\.\)\-:]\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\breason:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\baction:\s*.*$", "", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = normalize_learning_clause(cleaned)
        framework_patterns = (
            r"(?i)\b(spawn_subagent|evaluate_against_gt|evaluate_batch_against_gt|batch_evaluate_against_gt)\b",
            r"(?i)\b(execution context|assigned(?: exploration)? path|assigned branch|branch(?: instructions?)?|training mode|learning-mode|ground truth)\b.*\b(require|requires|required|prioritiz|mandat|must|before finishing)\b",
            r"(?i)\b(required|prioritized?|mandated)\b[^.]{0,80}\bpath\b",
            r"(?i)\bmust use at least one tool\b",
        )
        if any(re.search(pattern, cleaned) for pattern in framework_patterns):
            return ""
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;|-")
        lower = cleaned.lower()
        banned_markers = (
            "spawn_subagent",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
            "execution context",
            "assigned path",
            "assigned branch",
            "branch explicitly prioritizes",
            "before finishing",
            "ground truth",
            "training mode",
            "learning-mode",
        )
        if any(marker in lower for marker in banned_markers):
            return ""
        # Keep short but meaningful tool intentions such as "run ets" instead
        # of dropping them purely due to length.
        if len(cleaned) < 4:
            return ""
        return cleaned

    @staticmethod
    def _compact_candidate_reasoning(trace: dict[str, Any]) -> str:
        steps = trace.get("steps")
        if isinstance(steps, list):
            snippets: list[str] = []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                raw = str(step.get("thought") or "").strip()
                if not raw:
                    continue
                snippet = Notes._sanitize_candidate_reasoning(raw)
                if not snippet:
                    continue
                snippet = snippet[0].upper() + snippet[1:] if snippet else snippet
                if not snippet.endswith("."):
                    snippet += "."
                snippets.append(snippet)
                if len(snippets) >= 2:
                    break
            if snippets:
                return " ".join(snippets)
        return ""

    @staticmethod
    def _compact_candidate_weakness(trace: dict[str, Any]) -> str:
        tool_chain = [t for t in trace.get("tool_chain", []) if t and t not in _FRAMEWORK_TOOLS]
        if tool_chain == ["basic_stats"]:
            return "Relied on global summary statistics without an explicit future-path estimate."

        steps = trace.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                tool = str(step.get("tool") or "").strip()
                obs = step.get("observation")
                if tool == "arima_forecast" and isinstance(obs, dict):
                    forecast = obs.get("forecast")
                    if isinstance(forecast, list):
                        numeric = [float(v) for v in forecast if isinstance(v, (int, float))]
                        if len(numeric) >= 3 and max(numeric) - min(numeric) < 0.5:
                            return "Forecast path was nearly flat, which can miss structured seasonality."
        _, score_label, _ = Notes._candidate_score(trace.get("metrics") or {})
        if score_label:
            return f"This path recorded {score_label} on the sample."
        return ""

    @staticmethod
    def _ranked_candidate_summaries(candidate_traces: list[dict[str, Any]]) -> list[str]:
        ranked: list[tuple[tuple[int, float], str]] = []
        fallback: list[str] = []
        seen: set[str] = set()
        for idx, trace in enumerate(candidate_traces):
            if not isinstance(trace, dict):
                continue
            name = Notes._display_candidate_name(trace, idx)
            if name in seen:
                continue
            seen.add(name)
            metrics = trace.get("metrics") or {}
            score_value, score_label, sort_key = Notes._candidate_score(metrics)
            if score_label:
                ranked.append((sort_key, f"{name}({score_label})"))
            else:
                fallback.append(name)
        ranked.sort(key=lambda item: item[0])
        return [text for _, text in ranked] + fallback

    @staticmethod
    def _format_candidate_steps(steps: Any, role: str = "all") -> list[str]:
        if not isinstance(steps, list):
            return []
        rendered: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "").strip()
            if not tool:
                continue
            is_validation = tool in _NOTE_VALIDATION_TOOLS
            if role == "generator" and is_validation:
                continue
            if role == "validation" and not is_validation:
                continue
            rendered.append(f"  - Tool: {tool}")
            reason = str(step.get("thought") or "").strip()
            if reason:
                rendered.append(f"    Reason/Action: {reason}")
            observation = step.get("observation")
            if observation is not None:
                rendered.append(f"    Observation: {Notes._stringify_observation(observation)}")
        return rendered

    @staticmethod
    def _display_candidate_name(trace: dict[str, Any], index: int) -> str:
        name = str(trace.get("name") or "").strip()
        tool_chain = [t for t in trace.get("tool_chain", []) if t and t not in _FRAMEWORK_TOOLS]
        if name and not re.fullmatch(r"(slot|candidate|approach)_\d+", name):
            return name[:80]
        if tool_chain:
            return " → ".join(tool_chain[:3])[:80]
        if name:
            return name[:80]
        return f"candidate_{index}"

    @staticmethod
    def _candidate_score(metrics: dict[str, Any]) -> tuple[float | None, str, tuple[int, float]]:
        if not isinstance(metrics, dict) or not metrics:
            return None, "", (9, 0.0)
        for key in ("mae", "avg_mae", "rmse", "mse", "max_mae"):
            if isinstance(metrics.get(key), (int, float)):
                value = float(metrics[key])
                label_parts = [f"{key}={value:.3f}"]
                if key != "rmse" and isinstance(metrics.get("rmse"), (int, float)):
                    label_parts.append(f"rmse={float(metrics['rmse']):.3f}")
                return value, ", ".join(label_parts), (1, value)
        if isinstance(metrics.get("accuracy"), (int, float)):
            value = float(metrics["accuracy"])
            return value, f"acc={value:.3f}", (0, -value)
        return None, "", (9, 0.0)

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(
                Notes._round_display_numbers(value),
                ensure_ascii=False,
                default=str,
            )
        except TypeError:
            return str(value)

    @staticmethod
    def _stringify_metrics(metrics: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("mae", "avg_mae", "rmse", "mse", "mape", "max_mae", "accuracy"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.3f}")
        if not parts:
            return ""
        return ", ".join(parts)

    @staticmethod
    def _stringify_observation(observation: Any) -> str:
        payload = observation
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return payload
        if isinstance(payload, dict):
            filtered = dict(payload)
            if "quantiles" in filtered:
                filtered.pop("quantiles", None)
            if "fitted_values" in filtered:
                filtered.pop("fitted_values", None)
            if "order" in filtered:
                filtered.pop("order", None)
            try:
                return json.dumps(
                    Notes._round_display_numbers(filtered),
                    ensure_ascii=False,
                    default=str,
                )
            except TypeError:
                return str(filtered)
        return Notes._stringify_value(payload)

    @staticmethod
    def _observation_contains_prediction(steps: Any, prediction: Any) -> bool:
        if not isinstance(steps, list):
            return False
        prediction_text = Notes._stringify_value(prediction)
        for step in steps:
            if not isinstance(step, dict):
                continue
            observation = step.get("observation")
            if observation is None:
                continue
            if isinstance(observation, str):
                try:
                    observation = json.loads(observation)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(observation, dict):
                for key in ("point_forecast", "forecast", "prediction", "values", "answer"):
                    if key in observation:
                        return True
            observation_text = (
                observation if isinstance(observation, str)
                else Notes._stringify_value(observation)
            )
            if prediction_text and prediction_text in observation_text:
                return True
        return False

    def _append_to_scope_log(self, note: str, ctx: dict[str, Any]) -> None:
        scope = ctx.get("scope_key", "")
        domain = ctx.get("domain", "")
        task = ctx.get("task", "")
        correct = self._outcome_label(ctx.get("correct"), ctx.get("metrics") or {})
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self._scope_log_path(str(scope).strip())

        label = scope or f"{domain}/{task}"
        entry = f"\n### [{timestamp}] {label} — {correct}\n{note}\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(entry)

    def search(self, query: str, k: int = _L1_RETRIEVAL_K,
               domain: str | None = None, task: str | None = None,
               scope: str | None = None,
               return_ids: bool = False) -> list[str] | list[dict[str, Any]]:
        tokens = [token.lower() for token in re.findall(r"\w+", query or "") if token.strip()]
        matches: list[dict[str, Any]] = []
        files = sorted(NOTES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            text = path.read_text()
            for item in self._note_items_from_text(path.name, text):
                note = item["text"]
                parsed = DistillationAgent._parse_note_sections(note)
                note_scope = str(parsed["sections"].get("Scope") or parsed.get("header") or "").strip()
                if scope and note_scope != scope:
                    continue
                if domain and parsed.get("domain") != domain:
                    continue
                if task and parsed.get("task") != task:
                    continue
                lower = note.lower()
                score = sum(lower.count(token) for token in tokens) if tokens else 0
                if tokens and score == 0 and scope is None and domain is None and task is None:
                    continue
                matches.append({
                    "id": item["id"],
                    "text": note,
                    "score": score,
                })
        matches.sort(key=lambda item: (-int(item["score"]), item["id"]), reverse=False)
        trimmed = matches[:k]
        if return_ids:
            return [{"id": item["id"], "text": item["text"]} for item in trimmed]
        return [item["text"] for item in trimmed]

    def delete_notes(self, note_ids: list[str]) -> int:
        logger.info("[Notes] delete_notes is disabled; notes are kept as raw distillation sources")
        return 0

    def get_recent_notes(self, n: int = 10) -> list[str]:
        return [item["text"] for item in self.get_recent_note_items(n)]

    def get_recent_note_items(self, n: int = 10) -> list[dict[str, str]]:
        files = sorted(NOTES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        notes: list[dict[str, str]] = []
        for f in files:
            if len(notes) >= n:
                break
            for item in self._note_items_from_text(f.name, f.read_text()):
                notes.append(item)
                if len(notes) >= n:
                    break
        return notes[:n]

    def get_scope_note_items(self, scope: str) -> list[dict[str, Any]]:
        """Return one scope's notes in append order with a stable 1-based sequence."""
        scope_name = str(scope).strip()
        if not scope_name:
            return []
        path = NOTES_DIR / f"{scope_name}.md"
        if not path.exists():
            return []
        items = self._note_items_from_text(path.name, path.read_text())
        items.reverse()
        for idx, item in enumerate(items, start=1):
            item["seq"] = idx
        return items

    async def resolve_conflicts(self, query: str,
                                domain: str | None = None,
                                task: str | None = None,
                                scope: str | None = None) -> list[str]:
        notes = self.search(
            query, k=_L1_RETRIEVAL_K * 3,
            domain=domain, task=task, scope=scope,
        )
        return notes if isinstance(notes, list) else [item["text"] for item in notes]

    def get_prompt_section(self, query: str,
                           domain: str | None = None,
                           task: str | None = None,
                           scope: str | None = None) -> str:
        return ""


# ═══════════════════════════════════════════════════════════════════════
# MEMORY.md — Causal insights distilled from notes
# ═══════════════════════════════════════════════════════════════════════

class MemoryEntry:
    """Structured MEMORY rule stored per scope.

    New definition:
        A MEMORY entry is now a scope-local rule object rather than a single
        free-form insight string. `conditions` are descriptive only; they are
        not used as retrieval keys because scope is already the hard boundary.
    """

    def __init__(
        self,
        scope: str,
        rule_id: str,
        kind: str,
        summary: str,
        conditions: list[str] | None = None,
        preferred_tools: list[str] | None = None,
        avoid_tools: list[str] | None = None,
        rationale: str = "",
        evidence: dict[str, Any] | None = None,
        source_rule_ids: list[str] | None = None,
        confidence: float = 0.5,
        injectable: bool = True,
        status: str = "active",
        timestamp: float = 0.0,
    ):
        self.scope = scope
        self.rule_id = rule_id
        self.kind = kind
        self.summary = summary
        self.conditions = conditions or []
        self.preferred_tools = preferred_tools or []
        self.avoid_tools = avoid_tools or []
        self.rationale = rationale
        self.evidence = evidence or {}
        self.source_rule_ids = source_rule_ids or []
        self.confidence = confidence
        self.injectable = injectable
        self.status = status
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "rule_id": self.rule_id,
            "kind": self.kind,
            "summary": self.summary,
            "conditions": list(self.conditions),
            "preferred_tools": list(self.preferred_tools),
            "avoid_tools": list(self.avoid_tools),
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
            "source_rule_ids": list(self.source_rule_ids),
            "confidence": float(self.confidence),
            "injectable": bool(self.injectable),
            "status": self.status,
            "timestamp": self.timestamp,
        }


class MemoryStore:
    """MEMORY.md — causal insights partitioned by scope (domain_task)."""

    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, list[dict]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for f in MEMORY_DIR.glob("*.json"):
            try:
                raw = json.loads(f.read_text())
                if isinstance(raw, list):
                    normalized = [
                        self._normalize_entry(f.stem, entry, idx)
                        for idx, entry in enumerate(raw)
                    ]
                    normalized = self._dedupe_rule_ids(normalized)
                    self._entries[f.stem] = normalized
                    if normalized != raw:
                        f.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))
            except Exception:
                pass

    def get_entries(self, scope: str) -> list[dict]:
        return self._entries.get(scope, [])

    def get_all_entries(self) -> list[dict]:
        all_entries = []
        for entries in self._entries.values():
            all_entries.extend(entries)
        return all_entries

    def total(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def add_entries(self, scope: str, new_entries: list[dict],
                    removed_rule_ids: list[str] | None = None) -> None:
        if removed_rule_ids:
            removed_set = {str(item).strip() for item in removed_rule_ids if str(item).strip()}
            existing = [
                e for e in self._entries.get(scope, [])
                if str(e.get("rule_id", "")).strip() not in removed_set
            ]
            n_removed = len(self._entries.get(scope, [])) - len(existing)
            if n_removed:
                logger.info(f"[MEMORY] {scope}: removed {n_removed} contradicted rules")
            self._entries[scope] = existing

        existing = self._entries.get(scope, [])
        existing_ids = {str(e.get("rule_id", "")) for e in existing}
        existing_summaries = {str(e.get("summary", "")) for e in existing}
        for idx, entry in enumerate(new_entries):
            normalized = self._normalize_entry(scope, entry, len(existing) + idx)
            rule_id = str(normalized.get("rule_id", "")).strip()
            summary = str(normalized.get("summary", "")).strip()
            if rule_id and rule_id not in existing_ids and summary not in existing_summaries:
                existing.append(normalized)
                existing_ids.add(rule_id)
                existing_summaries.add(summary)
        self._entries[scope] = existing
        path = MEMORY_DIR / f"{scope}.json"
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    def replace_entries(self, scope: str, entries: list[dict]) -> None:
        normalized = [self._normalize_entry(scope, entry, idx) for idx, entry in enumerate(entries)]
        normalized = self._dedupe_rule_ids(normalized)
        self._entries[scope] = normalized
        path = MEMORY_DIR / f"{scope}.json"
        path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))

    def get_cross_domain_entries(self, bucket: str = "") -> list[dict]:
        if bucket:
            return self.get_entries(_cross_domain_scope_for_bucket(bucket))
        entries: list[dict] = []
        for scope, scoped_entries in self._entries.items():
            if _is_cross_domain_scope(scope):
                entries.extend(scoped_entries)
        return entries

    @staticmethod
    def _normalize_note_refs(refs: Any) -> list[str]:
        if not isinstance(refs, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in refs:
            ref = str(item).strip()
            if not ref or ref in seen:
                continue
            seen.add(ref)
            out.append(ref)
        return out[:200]

    @classmethod
    def _entry_note_refs(cls, entry: dict[str, Any]) -> list[str]:
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        return cls._normalize_note_refs(evidence.get("sample_refs", []))

    @classmethod
    def _entry_support_count(cls, entry: dict[str, Any]) -> int:
        refs = cls._entry_note_refs(entry)
        return len(refs)

    @staticmethod
    def _normalize_rule_payload(scope: str, payload: dict[str, Any], index: int) -> dict[str, Any]:
        kind = str(payload.get("kind", "")).strip() or "domain_heuristic"
        summary = str(payload.get("summary", "")).strip()
        conditions = [str(item).strip() for item in payload.get("conditions", []) if str(item).strip()]
        preferred_tools = _canonicalize_tool_list(
            [str(item).strip() for item in payload.get("preferred_tools", []) if str(item).strip()]
        )
        avoid_tools = _canonicalize_tool_list(
            [str(item).strip() for item in payload.get("avoid_tools", []) if str(item).strip()]
        )
        rationale = str(payload.get("rationale", "")).strip()
        source_rule_ids = [
            str(item).strip()
            for item in payload.get("source_rule_ids", [])
            if str(item).strip()
        ]
        return MemoryEntry(
            scope=scope,
            rule_id=str(payload.get("rule_id", "")).strip() or f"{scope}::{index + 1:03d}",
            kind=kind,
            summary=summary,
            conditions=conditions,
            preferred_tools=preferred_tools,
            avoid_tools=avoid_tools,
            rationale=rationale,
            evidence={"support_count": 0, "sample_refs": []},
            source_rule_ids=source_rule_ids,
            status="active",
        ).to_dict()

    def get_prompt_section(self, domain: str = "", task: str = "", scope: str = "") -> str:
        scope = scope or f"{domain}_{task}"
        entries = [entry for entry in self.get_entries(scope) if bool(entry.get("injectable", True))]
        domain_entries = [
            entry for entry in self.get_entries(domain)
            if bool(entry.get("injectable", True))
        ] if domain else []
        cross_entries: list[dict] = []
        all_entries = entries + domain_entries
        if not all_entries and not cross_entries:
            return ""
        lines = ["## MEMORY — Learned Insights\n"]
        if cross_entries:
            lines.append("### Cross-Domain Strategies")
            for e in cross_entries[-6:]:
                rendered = self._render_entry(e)
                if rendered:
                    lines.append(f"- {rendered}")
            lines.append("")
        if all_entries:
            lines.append(f"### {scope} Specific")
            for e in all_entries[-8:]:
                rendered = self._render_entry(e)
                if rendered:
                    lines.append(f"- {rendered}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_entry(entry: dict[str, Any]) -> str:
        summary = str(entry.get("summary", "")).strip()
        conditions = [str(item).strip() for item in entry.get("conditions", []) if str(item).strip()]
        preferred = [str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()]
        avoid = [str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()]
        rationale = str(entry.get("rationale", "")).strip()
        support = MemoryStore._entry_support_count(entry)

        parts: list[str] = []
        if summary:
            parts.append(f"Rule: {summary}")
        if conditions:
            parts.append(f"When: {', '.join(conditions[:3])}")
        if preferred:
            parts.append(f"Prefer: {', '.join(preferred[:3])}")
        if avoid:
            parts.append(f"Avoid: {', '.join(avoid[:3])}")
        if rationale:
            parts.append(f"Why: {rationale}")
        if support > 0:
            parts.append(f"Evidence: {support} note(s)")
        return " | ".join(parts)

    @staticmethod
    def _normalize_entry(scope: str, entry: dict[str, Any], index: int) -> dict[str, Any]:
        if not isinstance(entry, dict):
            entry = {}

        # Migrate legacy single-string insight objects into the new rule schema.
        if "summary" not in entry and "insight" in entry:
            insight = str(entry.get("insight", "")).strip()
            evidence_count = int(entry.get("evidence_count", 0) or 0)
            entry = MemoryEntry(
                scope=scope,
                rule_id=f"{scope}::{index + 1:03d}",
                kind="domain_heuristic",
                summary=insight,
                conditions=[],
                preferred_tools=[],
                avoid_tools=[],
                rationale="",
                evidence={"support_count": evidence_count, "sample_refs": []},
                source_rule_ids=[
                    str(item).strip()
                    for item in entry.get("source_rule_ids", [])
                    if str(item).strip()
                ],
                status="active",
                timestamp=float(entry.get("timestamp", 0.0) or time.time()),
            ).to_dict()
            return entry

        summary = str(entry.get("summary", "")).strip()
        rule_id = str(entry.get("rule_id", "")).strip() or f"{scope}::{index + 1:03d}"
        kind = str(entry.get("kind", "")).strip() or "domain_heuristic"
        conditions = [str(item).strip() for item in entry.get("conditions", []) if str(item).strip()]
        preferred_tools = _canonicalize_tool_list(
            [str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()]
        )
        avoid_tools = _canonicalize_tool_list(
            [str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()]
        )
        rationale = str(entry.get("rationale", "")).strip()
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        evidence = {
            **evidence,
            "sample_refs": MemoryStore._normalize_note_refs(evidence.get("sample_refs", [])),
        }
        evidence["support_count"] = len(evidence["sample_refs"])
        confidence = float(entry.get("confidence", 0.5) or 0.5)
        injectable = bool(entry.get("injectable", True))
        status = str(entry.get("status", "active")).strip() or "active"
        timestamp = float(entry.get("timestamp", 0.0) or time.time())
        source_rule_ids = [
            str(item).strip()
            for item in entry.get("source_rule_ids", [])
            if str(item).strip()
        ]

        return MemoryEntry(
            scope=scope,
            rule_id=rule_id,
            kind=kind,
            summary=summary,
            conditions=conditions,
            preferred_tools=preferred_tools,
            avoid_tools=avoid_tools,
            rationale=rationale,
            evidence=evidence,
            source_rule_ids=source_rule_ids,
            confidence=confidence,
            injectable=injectable,
            status=status,
            timestamp=timestamp,
        ).to_dict()

    @staticmethod
    def _dedupe_rule_ids(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only the strongest entry for each non-empty rule_id."""
        deduped: dict[str, dict[str, Any]] = {}
        ordered_fallback: list[dict[str, Any]] = []

        def _rank(entry: dict[str, Any]) -> tuple[float, int, float]:
            return (
                float(entry.get("confidence", 0.0) or 0.0),
                MemoryStore._entry_support_count(entry),
                float(entry.get("timestamp", 0.0) or 0.0),
            )

        for entry in entries:
            rule_id = str(entry.get("rule_id", "")).strip()
            if not rule_id:
                ordered_fallback.append(entry)
                continue
            existing = deduped.get(rule_id)
            if existing is None or _rank(entry) > _rank(existing):
                deduped[rule_id] = entry

        unique_entries = list(deduped.values()) + ordered_fallback
        unique_entries.sort(
            key=lambda entry: (
                not bool(entry.get("injectable", True)),
                -float(entry.get("confidence", 0.0) or 0.0),
                -MemoryStore._entry_support_count(entry),
            )
        )
        return unique_entries[:_MEMORY_RULE_CAP]


# ═══════════════════════════════════════════════════════════════════════
# SOUL.md — Fixed agent identity (human-written, not learned)
# ═══════════════════════════════════════════════════════════════════════

def load_soul() -> str:
    """Load the fixed SOUL.md content."""
    if SOUL_FILE.exists():
        return SOUL_FILE.read_text(encoding="utf-8").strip()
    return ""


# ═══════════════════════════════════════════════════════════════════════
# TOOLS.md — Per-tool usage notes learned from experience
# ═══════════════════════════════════════════════════════════════════════

class ToolsStore:
    """TOOLS.md — per-scope per-tool experience notes, distilled from MEMORY.

    New definition:
    - tool cards are keyed by exact ``scope_key``
    - usage counters are also keyed by exact ``scope_key``
    """

    _MIN_RUNS = 5
    _MAX_PROMPT_TOOLS = 15

    def __init__(self):
        MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
        self._notes_by_scope: dict[str, dict[str, dict[str, Any]]] = {}
        self._scope_counts: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if TOOLS_FILE.exists():
            try:
                raw = json.loads(TOOLS_FILE.read_text())
                if isinstance(raw, dict) and "notes_by_scope" in raw:
                    self._notes_by_scope = {
                        str(scope): {
                            str(tool_name): self._normalize_note_payload(payload)
                            for tool_name, payload in (tool_map or {}).items()
                        }
                        for scope, tool_map in (raw.get("notes_by_scope", {}) or {}).items()
                    }
                    self._scope_counts = raw.get("_scope_counts", {})
                elif isinstance(raw, dict) and "_scope_counts" in raw:
                    self._notes_by_scope = {}
                    self._scope_counts = raw.get("_scope_counts", {})
                else:
                    self._notes_by_scope = {}
            except Exception:
                self._notes_by_scope = {}

    def _save(self) -> None:
        payload = {"notes_by_scope": self._notes_by_scope, "_scope_counts": self._scope_counts}
        TOOLS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def get_notes(self, scope: str = "") -> dict[str, dict[str, Any]]:
        if scope:
            return dict(self._notes_by_scope.get(scope, {}))
        merged: dict[str, dict[str, Any]] = {}
        for tool_map in self._notes_by_scope.values():
            for tool_name, payload in tool_map.items():
                merged.setdefault(tool_name, payload)
        return merged

    def update(self, scope: str, notes: dict[str, Any]) -> None:
        scope_name = str(scope).strip()
        if not scope_name:
            return
        scoped = self._notes_by_scope.setdefault(scope_name, {})
        scoped.update({
            str(tool_name): self._normalize_note_payload(payload)
            for tool_name, payload in notes.items()
        })
        self._save()
        logger.info(f"[TOOLS] Updated usage notes for {len(notes)} tools in {scope_name}")

    def replace(self, scope: str, notes: dict[str, Any]) -> None:
        scope_name = str(scope).strip()
        if not scope_name:
            return
        self._notes_by_scope[scope_name] = {
            str(tool_name): self._normalize_note_payload(payload)
            for tool_name, payload in notes.items()
        }
        self._save()
        logger.info(f"[TOOLS] Rebuilt usage notes for {len(notes)} tools in {scope_name}")

    def record_scope_usage(self, scope: str, tool_counts: dict[str, int]) -> None:
        """Replace per-tool usage counts for a scope from the latest scan."""
        self._scope_counts[scope] = dict(tool_counts)
        self._save()

    def increment_scope_usage(self, scope: str, tool_names: list[str]) -> None:
        """Increment per-tool usage counts for one scope from a finished run."""
        if not scope or not tool_names:
            return
        bucket = self._scope_counts.setdefault(scope, {})
        for raw_name in tool_names:
            name = str(raw_name).strip()
            if not name:
                continue
            bucket[name] = int(bucket.get(name, 0) or 0) + 1
        self._save()

    def get_scope_usage(self, scope: str) -> dict[str, int]:
        """Return a copy of per-tool usage counts for one scope."""
        return dict(self._scope_counts.get(scope, {}))

    def total(self) -> int:
        return sum(len(tool_map) for tool_map in self._notes_by_scope.values())

    def get_tool_note(self, tool_name: str, scope: str) -> str:
        payload = self._notes_by_scope.get(scope, {}).get(tool_name)
        return self._render_note_payload(payload) if payload else ""

    def get_prompt_for_tools(
        self, tool_names: list[str], scope: str = "",
    ) -> dict[str, str]:
        """Return notes for the given tools, limited to a small relevant subset."""
        scope_data = self._scope_counts.get(scope, {}) if scope else {}
        scoped_notes = self._notes_by_scope.get(scope, {}) if scope else {}
        ranked: list[tuple[int, str, str]] = []
        for name in tool_names:
            payload = scoped_notes.get(name)
            note = self._render_note_payload(payload) if payload else ""
            if note:
                ranked.append((scope_data.get(name, 0), name, note))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        result: dict[str, str] = {}
        for _, name, note in ranked[: self._MAX_PROMPT_TOOLS]:
            result[name] = note
        return result

    @staticmethod
    def _normalize_note_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, str):
            return {
                "when_it_helps": [payload.strip()] if payload.strip() else [],
                "when_to_avoid": [],
                "output_risks": [],
            }
        if not isinstance(payload, dict):
            return {"when_it_helps": [], "when_to_avoid": [], "output_risks": []}

        def _listify(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            text = str(value).strip()
            return [text] if text else []

        return {
            "when_it_helps": _listify(payload.get("when_it_helps")),
            "when_to_avoid": _listify(payload.get("when_to_avoid")),
            "output_risks": _listify(payload.get("output_risks")),
        }

    @staticmethod
    def _render_note_payload(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        parts: list[str] = []
        helps = [str(item).strip() for item in payload.get("when_it_helps", []) if str(item).strip()]
        avoid = [str(item).strip() for item in payload.get("when_to_avoid", []) if str(item).strip()]
        risks = [str(item).strip() for item in payload.get("output_risks", []) if str(item).strip()]
        if helps:
            parts.append("When it helps: " + "; ".join(helps[:2]))
        if avoid:
            parts.append("When to avoid: " + "; ".join(avoid[:2]))
        if risks:
            parts.append("Output risks: " + "; ".join(risks[:2]))
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# SKILLS/ — Scene-specific SOPs distilled from MEMORY insights
# ═══════════════════════════════════════════════════════════════════════

class SkillsStore:
    """SKILLS/ — per (domain, task) operational guides.

    Each skill is a natural-language SOP describing how to approach
    a specific task type: steps, tool combinations, and pitfalls.
    Corresponds to OpenClaw SKILLS.md.
    """

    def __init__(self):
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        for f in SKILLS_DIR.glob("*.md"):
            try:
                self._skills[f.stem] = f.read_text()
            except Exception:
                pass

    def get_skill(self, scope: str) -> str:
        return self._skills.get(scope, "")

    def total(self) -> int:
        return len(self._skills)

    def update_skill(self, scope: str, content: str) -> None:
        self._skills[scope] = content
        path = SKILLS_DIR / f"{scope}.md"
        path.write_text(content)
        logger.info(f"[SKILLS] Updated SOP for {scope}")

    def delete_skill(self, scope: str) -> None:
        if scope in self._skills:
            del self._skills[scope]
        path = SKILLS_DIR / f"{scope}.md"
        if path.exists():
            path.unlink()
        logger.info(f"[SKILLS] Removed SOP for {scope}")

    def get_prompt_section(self, domain: str = "", task: str = "", scope: str = "") -> str:
        scope = scope or f"{domain}_{task}"
        skill = self._skills.get(scope, "")
        if not skill:
            return ""
        lines = [f"## SKILL — Operational Guide for {scope}\n"]
        lines.append(skill)
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Distillation Prompts
# ═══════════════════════════════════════════════════════════════════════

NOTES_TO_MEMORY_PROMPT = """\
You are updating scope-local MEMORY for a multimodal time-series agent.
Your job is to convert raw experience notes into reusable task-solving rules.

## Context
Scope: {scope}
Total runs analyzed: {total}

## Statistics
{statistics}

## Sample Experience Notes (recent, each with stable note_id)
{notes}

## Existing Rules (review and update these)
{existing_memory}

## What You Must Decide In This One Response
1. Which existing rules gained additional support from these notes, and by how much
2. Which existing rules are duplicates and should be merged
3. Which existing rules are genuine alternatives and should remain as conflicts
4. Which existing rules should remain stored but not be injected because a stronger conflicting rule exists
5. Which truly new rules must be added

## Rule Scope And Content
Every rule must be:
- specific to this scope
- about solving the task itself
- grounded in repeated evidence from the notes/statistics
- usable at test time without access to ground truth

Allowed `kind` values only:
- tool_selection
- tool_failure
- domain_heuristic
- result_constraint

Use these meanings:
- `tool_selection`: when one tool family is preferable under certain input conditions
- `tool_failure`: when a tool often breaks, misformats, or underperforms under certain conditions
- `domain_heuristic`: stable task/domain pattern that affects how to solve the problem
- `result_constraint`: output-shape or validity constraint that good results should satisfy

## Hard Prohibitions
Do NOT create rules about:
- spawn_subagent
- evaluate_against_gt
- evaluate_batch_against_gt
- batch_evaluate_against_gt
- retry loops
- training workflow
- learning mode
- exploration policy
- orchestration behavior

Do NOT use:
- ground-truth values
- ground-truth distributions
- label frequencies
- sample-specific answer content

Rules must be based on:
- input characteristics
- task requirements
- output validity patterns
- observed relative tool behavior from the notes/statistics

## How To Write A Good New Rule
`summary`:
- must be one short reusable sentence
- plain text only
- no brackets like [WHEN]
- no arrow notation like -> or =>
- no evidence clause inside summary
- no semicolon-packed mega-sentence
- do not recap a single sample

`conditions`:
- 0 to 3 short phrases
- describe observable input/task conditions only
- descriptive only, not retrieval keys

`preferred_tools` and `avoid_tools`:
- only real task-solving tools
- do not include framework actions or evaluators
- leave empty if not applicable

`rationale`:
- 1 to 2 short sentences
- explain why this rule is supported by the notes/statistics
- may mention relative metrics or repeated failures
- do not mention hidden labels or ground truth values

## How To Decide Between Existing Rules And New Rules
- Prefer updating an existing rule when the new evidence clearly supports it
- Merge rules only when they express the same operational guidance
- Mark rules as conflicting only when both can be valid under different conditions or when evidence is genuinely split
- Create a new rule only if the evidence is not already covered by an existing rule family
- Keep the total active rule set at or below {memory_rule_cap} after the update
- Before adding any new rule, you MUST compare it against every existing rule that could express the same operational guidance
- Treat `domain_heuristic` and `result_constraint` as a shared duplicate-check pool; do not create a second rule just because the better label could be argued either way
- If a candidate rule leads to the same test-time decision or same output-validity check as an existing rule, treat it as a duplicate even if the wording is different
- Wording changes, abstraction changes, and scope-local paraphrases are NOT enough to justify a new rule
- When in doubt, update the closest existing rule instead of creating a new one
- For `tool_failure`, if several tools fail for the same underlying reason or produce the same operational warning, prefer one broader rule and list all relevant concrete tools in `avoid_tools`
- For `tool_selection`, if multiple tools support the same workflow under the same conditions, prefer one workflow-level rule and list the interchangeable concrete tools in `preferred_tools`
- Do not split one workflow into multiple rules only because the concrete tool names differ

## Strong Boundary Between `domain_heuristic` And `result_constraint`
- Use `result_constraint` only when the rule is about validity, consistency, shape, format, or internal correctness of the output itself
- Use `domain_heuristic` only when the rule is about input patterns, task structure, or domain regularities that influence strategy choice
- If the rule says what a good output must satisfy, it is `result_constraint`, not `domain_heuristic`

## Output Requirements
Return JSON only with this exact top-level structure:
{{
 "support_updates": [
    {{
      "rule_id": "existing_rule_id_1",
      "support_note_ids": ["note_id_1", "note_id_2"]
    }}
  ],
  "merge_groups": [["existing_rule_id_1", "existing_rule_id_2"]],
  "conflict_groups": [["existing_rule_id_3", "existing_rule_id_4"]],
  "non_injectable_rule_ids": ["existing_rule_id_4"],
  "new_rules": [
    {{
      "kind": "tool_selection | tool_failure | domain_heuristic | result_constraint",
      "summary": "short reusable sentence",
      "conditions": ["condition 1", "condition 2"],
      "preferred_tools": ["tool_a"],
      "avoid_tools": ["tool_b"],
      "rationale": "brief evidence-grounded rationale",
      "support_note_ids": ["note_id_3", "note_id_4"],
      "closest_existing_rule_id": "existing_rule_id_5",
      "novelty_reason": "why this is not already covered by the closest existing rule"
    }}
  ]
}}

If no items belong in a section, return an empty list for that section.
Every returned `support_note_ids` must use only note ids that appear in the
provided experience notes.
Every returned `new_rule` MUST include `closest_existing_rule_id` and `novelty_reason`.
If a valid `novelty_reason` cannot be stated concretely, do not create the new rule.
"""

MEMORY_TO_CROSS_DOMAIN_PROMPT = """\
You are synthesizing domain-specific decision rules into universal
cross-domain insights for a multimodal time-series analysis agent.

## Allowed Concrete Tool Names
{allowed_tool_names}

## All Domain-Task Rules
{all_memory_entries}

## Existing Cross-Domain Insights (review and update)
{existing_cross_domain}

## Your Task
Produce an UPDATED set of cross-domain rules in the same JSON schema,
but with summary/conditions/rationale phrased at a cross-domain level.

RULES:
- Max 6 rules total
- Allowed kinds only:
  - tool_selection
  - tool_failure
  - domain_heuristic
  - result_constraint
- Cross-domain rules must still avoid any training/orchestration behavior
- If you mention tools, you MUST use exact names from the allowed tool list above
- If no exact concrete tool name is well-supported, leave `preferred_tools` / `avoid_tools` empty
- **CRITICAL**: Insights must NEVER reference ground truth values or label
  distributions.  Base all guidance on input data characteristics and tool
  performance metrics, not on output label frequencies.

Return JSON: {{
  "rules": [
    {{
      "kind": "tool_selection | tool_failure | domain_heuristic | result_constraint",
      "summary": "short reusable rule",
      "conditions": ["condition 1", "condition 2"],
      "preferred_tools": ["tool_a"],
      "avoid_tools": ["tool_b"],
      "rationale": "why this rule is useful",
      "source_rule_ids": ["scope_rule_id_1", "scope_rule_id_2"]
    }}
  ],
  "removed_rule_ids": ["existing_rule_id_1", "existing_rule_id_2"]
}}

Every `source_rule_ids` item must reference a rule id that appears in
`All Domain-Task Rules`.
"""

MEMORY_TO_TOOLS_PROMPT = """\
You are distilling TOOLS.md for a time-series agent from already-distilled MEMORY rules.
Produce concise single-tool cards for all tools supported by the evidence below.

## Tool Evidence From MEMORY
{tool_evidence}

## Existing Tool Notes (review and update)
{existing_tools}

## Rules
- Use ONLY the provided MEMORY evidence
- Generate all tool cards in this one response
- Keep cards short and operational
- Each card must be about that tool as the subject
- Avoid repeating the exact same boundary across many tools when the wording can be made tool-specific
- Do not mechanically restate the same comparison sentence for multiple tools
- Prefer tool-specific boundaries over shared comparison phrasing
- If two tools share the same broad comparison evidence, phrase each card from that tool's own failure/suitability perspective
- Prefer unique risks, failure modes, or usage limits for each tool when available
- Do NOT restate scope definitions, task names, or generic facts like "this is a forecasting tool"
- `when_it_helps`: only when the evidence clearly supports this tool
- `when_to_avoid`: only when the evidence clearly warns against this tool
- `output_risks`: only concrete output / failure risks tied to this tool
- Omit weak or empty sections
- Omit tools that have no meaningful boundary in the evidence
- If a tool has only weak evidence compared with another tool, do not just write "prefer X instead"; explain the boundary from the current tool's perspective
- Keep each bullet self-contained and specific

Return JSON: {{
  "tool_notes": {{
    "tool_name": {{
      "when_it_helps": ["short boundary"],
      "when_to_avoid": ["short boundary"],
      "output_risks": ["short risk"]
    }}
  }}
}}
"""

MEMORY_TO_SKILLS_PROMPT = """\
You are creating an operational guide (skill) for a time-series agent.
Based on the structured MEMORY rules learned for this specific scope,
produce a small set of executable workflow skills the agent can follow at test time.

## Scope: {scope}

## Structured MEMORY Rules
{memory_entries}

## Existing Skill (review and update)
{existing_skill}

## Your Task
Write a concise operational guide in Markdown.

Use the MEMORY rules as the source of truth.
- Prefer rules with higher `support_count`
- Prefer rules where `injectable=true`
- Use `conditions` to decide when a workflow skill applies
- Use `kind` to organize content:
  - `tool_selection` + `domain_heuristic` -> skill condition / why / workflow
  - `result_constraint` + `tool_failure` -> guardrails when strongly supported
- Do not invent new knowledge that is not supported by the rules
- Do not flatten all rules into generic advice; preserve meaningful conditionality when present
- Do not restate the scope definition, task name, horizon, or other information that is already implied by the scope itself
- Output at most 5 skills
- Each skill must represent a distinct workflow, not a duplicate paraphrase
- If the MEMORY rules only support 1 or 2 real workflows, output only 1 or 2
- If evidence is weak, keep the workflow conservative and short
- Do not output two skills whose conditions substantially overlap and whose workflows are materially the same
- Merge nearby conditions into one broader skill when they lead to the same workflow choice and same guardrails
- Prefer fewer, cleaner skills over many slightly reworded variants

**CRITICAL**: The guide must NEVER reference ground truth values, ground
truth distributions, or label base rates.  The agent will NOT have access
to ground truth at test time.  All guidance must be based on input data
characteristics and observed tool behavior, not on output label statistics.

FORMAT: Plain Markdown (not JSON). Keep it under 400 words.

Required structure:
For each skill, use exactly this template:

### Skill N
Condition:
...

Why:
...

Workflow:
1. ...
2. ...

Guardrails:
- ...

Rules:
- Every skill must have `Condition`, `Why`, and `Workflow`
- `Workflow` should have 2-4 concrete steps
- `Guardrails` is optional; include it only when MEMORY rules clearly support hard constraints or failure warnings for that workflow
- `Condition` must describe the trigger for choosing this workflow, not the general task definition
- `Why` must explain the rationale from MEMORY rules, not generic prose
- `Workflow` must describe what to do, not just what tool is preferred
- Do not create a catch-all skill that simply restates all rules
- Before writing, deduplicate candidate skills by condition overlap + workflow overlap
- Do not include an introduction or conclusion outside the skill list
"""


# ═══════════════════════════════════════════════════════════════════════
# Distillation State
# ═══════════════════════════════════════════════════════════════════════

class DistillStateStore:
    """Persist per-path distillation watermarks independently from artifacts.

    New definition:
    - `notes_to_memory_done_seq`: highest scope-local note sequence already
      consumed by Notes -> MEMORY.
    - `*_done_fingerprint`: hash of the exact full-memory view used to build
      a downstream artifact.
    - `memory_history_seq`: append-only snapshot counter for one scope's
      MEMORY evolution across training.
    """

    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, dict[str, Any]]] = {
            "scopes": {},
            "buckets": {},
        }
        self._load()

    def _load(self) -> None:
        if not DISTILL_STATE_FILE.exists():
            return
        try:
            raw = json.loads(DISTILL_STATE_FILE.read_text())
            if isinstance(raw, dict):
                self._data["scopes"] = {
                    str(scope): dict(payload or {})
                    for scope, payload in (raw.get("scopes", {}) or {}).items()
                }
                self._data["buckets"] = {
                    str(bucket): dict(payload or {})
                    for bucket, payload in (raw.get("buckets", {}) or {}).items()
                }
        except Exception:
            self._data = {"scopes": {}, "buckets": {}}
        self.reset_inflight_flags()

    def save(self) -> None:
        DISTILL_STATE_FILE.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    def reset_inflight_flags(self) -> None:
        for payload in self._data["scopes"].values():
            payload["notes_to_memory_inflight"] = False
            payload["memory_to_skills_inflight"] = False
            payload["memory_to_tools_inflight"] = False
        for payload in self._data["buckets"].values():
            payload["memory_to_cross_inflight"] = False

    def get_scope(self, scope: str) -> dict[str, Any]:
        scope_name = str(scope).strip()
        payload = dict(self._data["scopes"].get(scope_name, {}))
        payload.setdefault("notes_total_seq", 0)
        payload.setdefault("notes_to_memory_done_seq", 0)
        payload.setdefault("notes_to_memory_inflight", False)
        payload.setdefault("memory_fingerprint", "")
        payload.setdefault("memory_history_seq", 0)
        payload.setdefault("memory_to_skills_done_fingerprint", "")
        payload.setdefault("memory_to_skills_inflight", False)
        payload.setdefault("memory_to_tools_done_fingerprint", "")
        payload.setdefault("memory_to_tools_inflight", False)
        return payload

    def update_scope(self, scope: str, **fields: Any) -> dict[str, Any]:
        scope_name = str(scope).strip()
        payload = self.get_scope(scope_name)
        payload.update(fields)
        self._data["scopes"][scope_name] = payload
        self.save()
        return dict(payload)

    def get_bucket(self, bucket: str) -> dict[str, Any]:
        bucket_name = str(bucket).strip()
        payload = dict(self._data["buckets"].get(bucket_name, {}))
        payload.setdefault("memory_fingerprint", "")
        payload.setdefault("memory_to_cross_done_fingerprint", "")
        payload.setdefault("memory_to_cross_inflight", False)
        return payload

    def update_bucket(self, bucket: str, **fields: Any) -> dict[str, Any]:
        bucket_name = str(bucket).strip()
        payload = self.get_bucket(bucket_name)
        payload.update(fields)
        self._data["buckets"][bucket_name] = payload
        self.save()
        return dict(payload)


# ═══════════════════════════════════════════════════════════════════════
# Distillation Agent
# ═══════════════════════════════════════════════════════════════════════

class DistillationAgent:
    """Orchestrates bottom-up knowledge distillation.

    Distillation chain:
        Notes  ──→  MEMORY (per-scope + cross_domain)
                     └──→  TOOLS.md
                     └──→  SKILLS/
    """

    def __init__(self, notes: Notes, memory: MemoryStore,
                 tools: ToolsStore, skills: SkillsStore):
        self.notes = notes
        self.memory = memory
        self.tools = tools
        self.skills = skills
        self._state = DistillStateStore()
        self._state_lock = asyncio.Lock()
        self._scope_workers: dict[str, asyncio.Task] = {}
        self._bucket_workers: dict[str, asyncio.Task] = {}
        self._llm = None

    async def _get_llm(self):
        if self._llm is None:
            from timeclaw.core.config import TimeClawConfig
            from timeclaw.llm.router import ModelRouter
            cfg = TimeClawConfig.load()
            router = ModelRouter(cfg.llm)
            self._llm = router.get_backend("reasoner")
        return self._llm

    @staticmethod
    def _threshold_reached(current: int, last: int, threshold: int) -> bool:
        return max(0, int(current) - int(last)) >= int(threshold)

    async def _get_scope_state(self, scope: str) -> dict[str, Any]:
        async with self._state_lock:
            return self._state.get_scope(scope)

    async def _update_scope_state(self, scope: str, **fields: Any) -> dict[str, Any]:
        async with self._state_lock:
            return self._state.update_scope(scope, **fields)

    async def _get_bucket_state(self, bucket: str) -> dict[str, Any]:
        async with self._state_lock:
            return self._state.get_bucket(bucket)

    async def _update_bucket_state(self, bucket: str, **fields: Any) -> dict[str, Any]:
        async with self._state_lock:
            return self._state.update_bucket(bucket, **fields)

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _scope_memory_fingerprint(self, scope: str) -> str:
        entries = self.memory.get_entries(scope)
        if not entries:
            return ""
        normalized = [
            {
                "rule_id": str(entry.get("rule_id", "")).strip(),
                "kind": str(entry.get("kind", "")).strip(),
                "summary": str(entry.get("summary", "")).strip(),
                "conditions": [str(item).strip() for item in entry.get("conditions", []) if str(item).strip()],
                "preferred_tools": [str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()],
                "avoid_tools": [str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()],
                "rationale": str(entry.get("rationale", "")).strip(),
                "evidence": {
                    "sample_refs": self._entry_note_refs(entry),
                    "support_count": self._entry_support_count(entry),
                },
                "injectable": bool(entry.get("injectable", True)),
                "status": str(entry.get("status", "active")).strip() or "active",
            }
            for entry in entries
        ]
        normalized.sort(key=lambda item: (item["rule_id"], item["summary"], item["kind"]))
        return self._hash_payload(normalized)

    def _bucket_memory_fingerprint(self, bucket: str) -> str:
        entries = [
            {
                "scope": str(entry.get("scope", "")).strip(),
                "rule_id": str(entry.get("rule_id", "")).strip(),
                "kind": str(entry.get("kind", "")).strip(),
                "summary": str(entry.get("summary", "")).strip(),
                "conditions": [str(item).strip() for item in entry.get("conditions", []) if str(item).strip()],
                "preferred_tools": [str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()],
                "avoid_tools": [str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()],
                "injectable": bool(entry.get("injectable", True)),
                "status": str(entry.get("status", "active")).strip() or "active",
                "support_count": self._entry_support_count(entry),
            }
            for entry in self.memory.get_all_entries()
            if _task_bucket_from_scope(str(entry.get("scope", "")).strip()) == bucket
            and not _is_cross_domain_scope(str(entry.get("scope", "")).strip())
        ]
        if not entries:
            return ""
        entries.sort(key=lambda item: (item["scope"], item["rule_id"], item["summary"], item["kind"]))
        return self._hash_payload(entries)

    @staticmethod
    def _scope_history_dir(scope: str) -> Path:
        path = HISTORY_DIR / str(scope).strip()
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _append_memory_history_snapshot(
        self,
        *,
        scope: str,
        notes_from_seq: int,
        notes_to_seq: int,
        memory_fingerprint: str,
    ) -> None:
        """Persist one scope-local MEMORY snapshot after a successful rebuild."""
        scope_name = str(scope).strip()
        if not scope_name:
            return

        state = await self._get_scope_state(scope_name)
        snapshot_seq = int(state.get("memory_history_seq", 0) or 0) + 1
        created_at = time.time()
        entries = self.memory.get_entries(scope_name)
        payload = {
            "scope": scope_name,
            "snapshot_seq": snapshot_seq,
            "created_at": created_at,
            "source_stage": "notes_to_memory",
            "notes_from_seq": int(notes_from_seq),
            "notes_to_seq": int(notes_to_seq),
            "notes_covered_count": max(0, int(notes_to_seq) - int(notes_from_seq) + 1),
            "memory_fingerprint": str(memory_fingerprint or ""),
            "rule_count": len(entries),
            "entries": entries,
        }

        scope_dir = self._scope_history_dir(scope_name)
        snapshot_path = scope_dir / f"{snapshot_seq:06d}.json"
        snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        manifest_path = scope_dir / "manifest.jsonl"
        manifest_row = {
            "scope": scope_name,
            "snapshot_seq": snapshot_seq,
            "created_at": created_at,
            "source_stage": "notes_to_memory",
            "notes_from_seq": int(notes_from_seq),
            "notes_to_seq": int(notes_to_seq),
            "memory_fingerprint": str(memory_fingerprint or ""),
            "rule_count": len(entries),
            "path": str(snapshot_path.relative_to(MEMORY_ROOT)),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

        await self._update_scope_state(scope_name, memory_history_seq=snapshot_seq)

    @staticmethod
    def _create_background_task(coro: Any, label: str) -> asyncio.Task:
        ctx = copy_context()

        def _spawn() -> asyncio.Task:
            trace.detach_context()
            return asyncio.create_task(coro)

        task = ctx.run(_spawn)

        def _log_failure(done_task: asyncio.Task) -> None:
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(f"[Distill] Background task {label} failed: {exc}")

        task.add_done_callback(_log_failure)
        return task

    def _ensure_scope_worker(self, scope: str, *, force: bool = False) -> None:
        scope_name = str(scope).strip()
        if not scope_name:
            return
        worker = self._scope_workers.get(scope_name)
        if worker and not worker.done():
            return
        self._scope_workers[scope_name] = self._create_background_task(
            self._scope_worker(scope_name, force=force),
            label=f"scope:{scope_name}",
        )

    def _ensure_bucket_worker(self, bucket: str, *, force: bool = False) -> None:
        bucket_name = str(bucket).strip()
        if not bucket_name:
            return
        worker = self._bucket_workers.get(bucket_name)
        if worker and not worker.done():
            return
        self._bucket_workers[bucket_name] = self._create_background_task(
            self._bucket_worker(bucket_name, force=force),
            label=f"bucket:{bucket_name}",
        )

    async def on_sample_commit(self, scope_keys: list[str]) -> None:
        await self.on_commit(scope_keys)

    async def on_commit(self, scope_keys: list[str]) -> None:
        """Record new note availability and trigger non-blocking distillation workers."""
        seen: set[str] = set()
        for raw_scope in scope_keys:
            scope = str(raw_scope).strip()
            if not scope or scope in seen or _is_cross_domain_scope(scope):
                continue
            seen.add(scope)
            await self._update_scope_state(scope, notes_total_seq=self.notes.scoped_total(scope))
            self._ensure_scope_worker(scope)

    async def on_scope_commit(self, scope_keys: list[str]) -> None:
        await self.on_commit(scope_keys)

    async def on_global_commit(self, scope_keys: list[str]) -> None:
        seen_buckets: set[str] = set()
        for raw_scope in scope_keys:
            bucket = _task_bucket_from_scope(raw_scope)
            if not bucket or bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            self._ensure_bucket_worker(bucket)

    async def _scope_worker(self, scope: str, *, force: bool = False) -> None:
        try:
            while True:
                progressed = False
                notes_total = self.notes.scoped_total(scope)
                state = await self._update_scope_state(scope, notes_total_seq=notes_total)

                should_run_notes = (
                    notes_total > int(state.get("notes_to_memory_done_seq", 0) or 0)
                    and (
                        force
                        or self._threshold_reached(
                            notes_total,
                            int(state.get("notes_to_memory_done_seq", 0) or 0),
                            _DISTILL_L1_TO_L2,
                        )
                    )
                )
                if should_run_notes:
                    target_seq = notes_total
                    await self._update_scope_state(scope, notes_to_memory_inflight=True)
                    success = await self.maybe_distill_notes_to_memory(scope, target_seq=target_seq, force=force)
                    await self._update_scope_state(scope, notes_to_memory_inflight=False)
                    if success:
                        memory_fp = self._scope_memory_fingerprint(scope)
                        await self._append_memory_history_snapshot(
                            scope=scope,
                            notes_from_seq=int(state.get("notes_to_memory_done_seq", 0) or 0) + 1,
                            notes_to_seq=target_seq,
                            memory_fingerprint=memory_fp,
                        )
                        await self._update_scope_state(
                            scope,
                            notes_to_memory_done_seq=target_seq,
                            memory_fingerprint=memory_fp,
                        )
                        bucket = _task_bucket_from_scope(scope)
                        if bucket:
                            self._ensure_bucket_worker(bucket, force=force)
                        progressed = True
                    else:
                        break

                state = await self._get_scope_state(scope)
                memory_fp = self._scope_memory_fingerprint(scope)
                if memory_fp != str(state.get("memory_fingerprint", "") or ""):
                    state = await self._update_scope_state(scope, memory_fingerprint=memory_fp)

                if memory_fp and memory_fp != str(state.get("memory_to_skills_done_fingerprint", "") or ""):
                    await self._update_scope_state(scope, memory_to_skills_inflight=True)
                    success = await self.maybe_distill_memory_to_skills(scope)
                    await self._update_scope_state(scope, memory_to_skills_inflight=False)
                    if success:
                        await self._update_scope_state(scope, memory_to_skills_done_fingerprint=memory_fp)
                        progressed = True
                    else:
                        break
                elif not memory_fp and self.skills.get_skill(scope):
                    self.skills.delete_skill(scope)
                    await self._update_scope_state(scope, memory_to_skills_done_fingerprint="")
                    progressed = True

                state = await self._get_scope_state(scope)
                if memory_fp and memory_fp != str(state.get("memory_to_tools_done_fingerprint", "") or ""):
                    await self._update_scope_state(scope, memory_to_tools_inflight=True)
                    success = await self.maybe_distill_memory_to_tools(scope)
                    await self._update_scope_state(scope, memory_to_tools_inflight=False)
                    if success:
                        await self._update_scope_state(scope, memory_to_tools_done_fingerprint=memory_fp)
                        progressed = True
                    else:
                        break
                elif not memory_fp and self.tools.get_notes(scope):
                    self.tools.replace(scope, {})
                    await self._update_scope_state(scope, memory_to_tools_done_fingerprint="")
                    progressed = True

                if not progressed:
                    break
        finally:
            current = asyncio.current_task()
            if self._scope_workers.get(scope) is current:
                self._scope_workers.pop(scope, None)

    async def _bucket_worker(self, bucket: str, *, force: bool = False) -> None:
        try:
            while True:
                memory_fp = self._bucket_memory_fingerprint(bucket)
                state = await self._update_bucket_state(bucket, memory_fingerprint=memory_fp)
                if memory_fp and memory_fp != str(state.get("memory_to_cross_done_fingerprint", "") or ""):
                    await self._update_bucket_state(bucket, memory_to_cross_inflight=True)
                    success = await self.maybe_distill_memory_to_cross_domain(bucket)
                    await self._update_bucket_state(bucket, memory_to_cross_inflight=False)
                    if success:
                        await self._update_bucket_state(bucket, memory_to_cross_done_fingerprint=memory_fp)
                        continue
                    break
                if not memory_fp:
                    cross_scope = _cross_domain_scope_for_bucket(bucket)
                    if self.memory.get_entries(cross_scope):
                        self.memory.replace_entries(cross_scope, [])
                        await self._update_bucket_state(bucket, memory_to_cross_done_fingerprint="")
                    break
                break
        finally:
            current = asyncio.current_task()
            if self._bucket_workers.get(bucket) is current:
                self._bucket_workers.pop(bucket, None)

    @staticmethod
    def _normalize_ref_list(refs: Any, allowed: set[str] | None = None) -> list[str]:
        if not isinstance(refs, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in refs:
            ref = str(item).strip()
            if not ref or ref in seen:
                continue
            if allowed is not None and ref not in allowed:
                continue
            seen.add(ref)
            out.append(ref)
        return out

    def _note_ref_count(self, refs: list[str]) -> int:
        return len(self._normalize_ref_list(refs))

    def _entry_note_refs(self, entry: dict[str, Any]) -> list[str]:
        return self.memory._entry_note_refs(entry)

    def _entry_support_count(self, entry: dict[str, Any]) -> int:
        return self.memory._entry_support_count(entry)

    def _source_rule_note_refs(self, source_rule_ids: list[str]) -> list[str]:
        if not source_rule_ids:
            return []
        refs: list[str] = []
        seen: set[str] = set()
        by_id = {
            str(entry.get("rule_id", "")).strip(): entry
            for entry in self.memory.get_all_entries()
            if isinstance(entry, dict)
        }
        for rule_id in source_rule_ids:
            entry = by_id.get(str(rule_id).strip())
            if not entry:
                continue
            for ref in self._entry_note_refs(entry):
                if ref in seen:
                    continue
                seen.add(ref)
                refs.append(ref)
        return refs

    @staticmethod
    def _score_rule_confidence(
        support_count: int,
        last_seen_at: float,
        max_conflict_support_count: int = 0,
    ) -> float:
        now = time.time()
        age_days = max(0.0, (now - float(last_seen_at or now)) / 86400.0)
        support_component = min(0.45, 0.12 * max(0, support_count))
        freshness_component = max(0.05, 0.35 - min(0.30, age_days * 0.02))
        conflict_penalty = min(0.35, 0.08 * max(0, max_conflict_support_count))
        score = 0.15 + support_component + freshness_component - conflict_penalty
        return max(0.05, min(0.95, score))

    @staticmethod
    def _max_conflict_support(rule_id: str, conflict_groups: list[list[str]], entries_by_id: dict[str, dict[str, Any]]) -> int:
        max_support = 0
        for group in conflict_groups:
            if rule_id not in group:
                continue
            for other_id in group:
                if other_id == rule_id:
                    continue
                other = entries_by_id.get(other_id)
                if not other:
                    continue
                support = MemoryStore._entry_support_count(other)
                max_support = max(max_support, support)
        return max_support

    @staticmethod
    def _canonical_rule_text(text: str) -> str:
        normalized = str(text or "").lower()
        replacements = {
            "self-consistency": "internal consistency",
            "self consistency": "internal consistency",
            "internally consistent": "internal consistency",
            "internally coherent": "internal consistency",
            "numerically consistent": "numeric consistency",
            "numerical consistency": "numeric consistency",
            "consistency-preserving": "consistency preserving",
            "outputs": "output",
            "results": "result",
            "predictions": "prediction",
            "answers": "answer",
            "constraints": "constraint",
            "heuristics": "heuristic",
        }
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
        normalized = re.sub(r"\bresults?\b", "output", normalized)
        normalized = re.sub(r"\bpredictions?\b", "output", normalized)
        normalized = re.sub(r"\banswers?\b", "output", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def _rule_text_tokens(cls, text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "do", "for",
            "from", "good", "if", "in", "into", "is", "it", "of", "on",
            "or", "must", "preserve", "remain", "same", "should", "stay",
            "that", "the", "their", "them", "then", "this", "to", "under",
            "use", "when", "with",
        }
        tokens: set[str] = set()
        for token in cls._canonical_rule_text(text).split():
            if len(token) <= 2 or token in stopwords:
                continue
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            tokens.add(token)
        return tokens

    @classmethod
    def _rule_similarity_signature(cls, entry: dict[str, Any]) -> dict[str, Any]:
        summary = str(entry.get("summary", "")).strip()
        conditions = [
            str(item).strip()
            for item in entry.get("conditions", [])
            if str(item).strip()
        ]
        return {
            "kind": str(entry.get("kind", "")).strip(),
            "summary_text": cls._canonical_rule_text(summary),
            "summary_tokens": cls._rule_text_tokens(summary),
            "condition_tokens": cls._rule_text_tokens(" ".join(conditions)),
            "preferred_tools": tuple(
                str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()
            ),
            "avoid_tools": tuple(
                str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()
            ),
        }

    @staticmethod
    def _token_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @classmethod
    def _is_near_duplicate_rule(
        cls,
        existing_entry: dict[str, Any],
        candidate_entry: dict[str, Any],
    ) -> bool:
        target_kinds = {"result_constraint", "domain_heuristic", "tool_failure"}
        existing_sig = cls._rule_similarity_signature(existing_entry)
        candidate_sig = cls._rule_similarity_signature(candidate_entry)

        if existing_sig["kind"] not in target_kinds or candidate_sig["kind"] not in target_kinds:
            return False
        same_kind = existing_sig["kind"] == candidate_sig["kind"]
        if not same_kind and {existing_sig["kind"], candidate_sig["kind"]} != {"result_constraint", "domain_heuristic"}:
            return False
        existing_preferred = set(existing_sig["preferred_tools"])
        candidate_preferred = set(candidate_sig["preferred_tools"])
        existing_avoid = set(existing_sig["avoid_tools"])
        candidate_avoid = set(candidate_sig["avoid_tools"])

        if same_kind and existing_sig["kind"] == "tool_failure":
            same_tool_family = (
                not existing_avoid and not candidate_avoid
            ) or bool(existing_avoid & candidate_avoid)
            if not same_tool_family:
                return False
            if existing_preferred != candidate_preferred:
                return False
        else:
            if existing_preferred != candidate_preferred:
                return False
            if existing_avoid != candidate_avoid:
                return False

        summary_overlap = cls._token_overlap(
            existing_sig["summary_tokens"], candidate_sig["summary_tokens"]
        )
        condition_overlap = cls._token_overlap(
            existing_sig["condition_tokens"], candidate_sig["condition_tokens"]
        )
        summary_ratio = SequenceMatcher(
            None,
            existing_sig["summary_text"],
            candidate_sig["summary_text"],
        ).ratio()

        condition_match = (
            not existing_sig["condition_tokens"] and not candidate_sig["condition_tokens"]
        ) or condition_overlap >= 0.6

        if condition_match and summary_overlap >= 0.65:
            return True
        if condition_match and summary_ratio >= 0.82:
            return True
        if same_kind and existing_sig["kind"] == "tool_failure":
            if condition_overlap >= 0.65 and (summary_ratio >= 0.62 or summary_overlap >= 0.33):
                return True
        if not same_kind:
            cross_kind_condition_match = (
                not existing_sig["condition_tokens"] and not candidate_sig["condition_tokens"]
            ) or condition_overlap >= 0.55
            if cross_kind_condition_match and (summary_ratio >= 0.60 or summary_overlap >= 0.30):
                return True
        return False

    @classmethod
    def _find_near_duplicate_rule_id(
        cls,
        entries: list[dict[str, Any]],
        candidate_entry: dict[str, Any],
        hinted_rule_id: str = "",
    ) -> str:
        by_id = {
            str(entry.get("rule_id", "")).strip(): entry
            for entry in entries
            if str(entry.get("rule_id", "")).strip()
        }
        hint = str(hinted_rule_id).strip()
        if hint and hint in by_id and cls._is_near_duplicate_rule(by_id[hint], candidate_entry):
            return hint

        best_rule_id = ""
        best_score = 0.0
        candidate_sig = cls._rule_similarity_signature(candidate_entry)
        for entry in entries:
            rule_id = str(entry.get("rule_id", "")).strip()
            if not rule_id:
                continue
            if not cls._is_near_duplicate_rule(entry, candidate_entry):
                continue
            existing_sig = cls._rule_similarity_signature(entry)
            overlap = cls._token_overlap(
                existing_sig["summary_tokens"] | existing_sig["condition_tokens"],
                candidate_sig["summary_tokens"] | candidate_sig["condition_tokens"],
            )
            ratio = SequenceMatcher(
                None,
                existing_sig["summary_text"],
                candidate_sig["summary_text"],
            ).ratio()
            score = max(overlap, ratio)
            if score > best_score:
                best_score = score
                best_rule_id = rule_id
        return best_rule_id

    def _apply_memory_update_decision(
        self,
        scope: str,
        existing_entries: list[dict[str, Any]],
        decision: dict[str, Any],
        available_note_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        entries = self.memory._dedupe_rule_ids([dict(entry) for entry in existing_entries])
        by_id = {str(entry.get("rule_id", "")).strip(): entry for entry in entries if str(entry.get("rule_id", "")).strip()}
        now = time.time()

        available_note_ids = available_note_ids or set()
        support_updates: dict[str, list[str]] = {}
        for item in decision.get("support_updates", []):
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("rule_id", "")).strip()
            if not rule_id or rule_id not in by_id:
                continue
            note_ids = self._normalize_ref_list(item.get("support_note_ids", []), allowed=available_note_ids)
            if note_ids:
                support_updates.setdefault(rule_id, [])
                for note_id in note_ids:
                    if note_id not in support_updates[rule_id]:
                        support_updates[rule_id].append(note_id)

        # Backward compatibility for older LLM outputs.
        for rule_id in decision.get("support_rule_ids", []):
            rid = str(rule_id).strip()
            if rid and rid in by_id:
                support_updates.setdefault(rid, [])
        merge_groups = [
            [str(rule_id).strip() for rule_id in group if str(rule_id).strip() in by_id]
            for group in decision.get("merge_groups", [])
            if isinstance(group, list)
        ]
        merge_groups = [group for group in merge_groups if len(group) >= 2]
        conflict_groups = [
            [str(rule_id).strip() for rule_id in group if str(rule_id).strip() in by_id]
            for group in decision.get("conflict_groups", [])
            if isinstance(group, list)
        ]
        conflict_groups = [group for group in conflict_groups if len(group) >= 2]
        non_injectable_ids = {
            str(rule_id).strip()
            for rule_id in decision.get("non_injectable_rule_ids", [])
            if str(rule_id).strip() in by_id
        }

        # Apply support increments before confidence recomputation.
        for rule_id, added_note_ids in support_updates.items():
            entry = by_id[rule_id]
            evidence = dict(entry.get("evidence") or {})
            existing_refs = self._normalize_ref_list(evidence.get("sample_refs", []))
            merged_refs = existing_refs[:]
            for note_id in added_note_ids:
                if note_id not in merged_refs:
                    merged_refs.append(note_id)
            evidence["sample_refs"] = merged_refs[:200]
            evidence["support_count"] = len(evidence["sample_refs"])
            entry["evidence"] = evidence
            entry["timestamp"] = now

        # Merge duplicate/overlapping rules by keeping the strongest first id.
        merged_away: set[str] = set()
        for group in merge_groups:
            survivors = [by_id[rule_id] for rule_id in group if rule_id in by_id and rule_id not in merged_away]
            if len(survivors) < 2:
                continue
            survivors.sort(key=lambda e: (self._entry_support_count(e), float(e.get("timestamp", 0.0) or 0.0)), reverse=True)
            keeper = survivors[0]
            keeper_evidence = dict(keeper.get("evidence") or {})
            sample_refs = self._normalize_ref_list(keeper_evidence.get("sample_refs", []))
            for entry in survivors[1:]:
                merged_away.add(str(entry.get("rule_id", "")).strip())
                other_evidence = dict(entry.get("evidence") or {})
                for note_id in self._normalize_ref_list(other_evidence.get("sample_refs", [])):
                    if note_id not in sample_refs:
                        sample_refs.append(note_id)
            keeper_evidence["sample_refs"] = sample_refs[:200]
            keeper_evidence["support_count"] = len(keeper_evidence["sample_refs"])
            keeper["evidence"] = keeper_evidence
            keeper["timestamp"] = now

        entries = [entry for entry in entries if str(entry.get("rule_id", "")).strip() not in merged_away]
        by_id = {str(entry.get("rule_id", "")).strip(): entry for entry in entries if str(entry.get("rule_id", "")).strip()}

        # Add brand-new rules.
        base_index = len(entries)
        for idx, payload in enumerate(decision.get("new_rules", [])):
            if not isinstance(payload, dict):
                continue
            note_ids = self._normalize_ref_list(payload.get("support_note_ids", []), allowed=available_note_ids)
            if not note_ids:
                continue
            normalized = self.memory._normalize_rule_payload(scope, payload, base_index + idx)
            normalized["timestamp"] = now
            normalized["injectable"] = True
            normalized["status"] = "active"
            duplicate_rule_id = self._find_near_duplicate_rule_id(
                entries,
                normalized,
                hinted_rule_id=str(payload.get("closest_existing_rule_id", "")).strip(),
            )
            if duplicate_rule_id and duplicate_rule_id in by_id:
                target = by_id[duplicate_rule_id]
                evidence = dict(target.get("evidence") or {})
                merged_refs = self._normalize_ref_list(evidence.get("sample_refs", []))
                for note_id in note_ids:
                    if note_id not in merged_refs:
                        merged_refs.append(note_id)
                evidence["sample_refs"] = merged_refs[:200]
                evidence["support_count"] = len(evidence["sample_refs"])
                target["evidence"] = evidence
                target["timestamp"] = now
                logger.info(
                    f"[Distill Notes→MEMORY] {scope}: merged near-duplicate new rule into {duplicate_rule_id}"
                )
                continue
            evidence = dict(normalized.get("evidence") or {})
            evidence["sample_refs"] = note_ids[:200]
            evidence["support_count"] = len(evidence["sample_refs"])
            normalized["evidence"] = evidence
            entries.append(normalized)
            rule_id = str(normalized.get("rule_id", "")).strip()
            if rule_id:
                by_id[rule_id] = normalized

        by_id = {str(entry.get("rule_id", "")).strip(): entry for entry in entries if str(entry.get("rule_id", "")).strip()}

        # Recompute injectable/confidence from conflict decisions.
        for entry in entries:
            rule_id = str(entry.get("rule_id", "")).strip()
            max_conflict_support = self._max_conflict_support(rule_id, conflict_groups, by_id)
            evidence = dict(entry.get("evidence") or {})
            support_count = self._entry_support_count(entry)
            entry["confidence"] = self._score_rule_confidence(
                support_count=support_count,
                last_seen_at=float(entry.get("timestamp", now) or now),
                max_conflict_support_count=max_conflict_support,
            )
            entry["injectable"] = rule_id not in non_injectable_ids
            entry["status"] = "active"

        entries.sort(
            key=lambda entry: (
                not bool(entry.get("injectable", True)),
                -float(entry.get("confidence", 0.0) or 0.0),
                -self._entry_support_count(entry),
            )
        )
        return self.memory._dedupe_rule_ids(entries)

    @staticmethod
    def _normalize_task_for_notes(task: str) -> str:
        """Map dataset task names to the task labels stored in notes."""
        return {
            "mcqa": "qa",
            "trend": "classification",
        }.get(task, task)

    @staticmethod
    def _denormalize_task_from_notes(task: str) -> str:
        """Map note task labels back to canonical memory scope names."""
        return {
            "qa": "mcqa",
            "classification": "trend",
        }.get(task, task)

    @staticmethod
    def _parse_note_sections(note: str) -> dict[str, Any]:
        """Parse either the new multi-section note format or legacy single-line notes."""
        import re

        titles = {
            "Scope",
            "Source",
            "Task",
            "Task Prompt",
            "Numeric Time Series",
            "Text Context",
            "Winner Tools",
            "Tools",
            "Winner Tool",
            "Winner Generator Tools",
            "Winner Validation Tools",
            "Prediction",
            "Metrics",
            "Fingerprint",
            "Approaches",
            "Candidates",
            "Analysis",
            "Pred",
            "GT",
        }
        sections: dict[str, Any] = {"header": "", "status": "", "sections": {}}

        raw_lines = note.splitlines()
        lines = [line.rstrip() for line in raw_lines]
        first_content = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_content = stripped
                break

        header_match = re.match(
            r"(?:###\s*)?(?:\[[^\]]+\]\s+)?([^\n]+?)\s+[—-]\s+(OK|FAIL|SCORED|LOGGED)\s*$",
            first_content,
        )
        if header_match:
            sections["header"] = header_match.group(1).strip()
            sections["status"] = header_match.group(2).strip()
        else:
            inline_match = re.match(r"\[([^\]]+)\]\s+(OK|FAIL|SCORED|LOGGED)\b", first_content)
            if inline_match:
                sections["header"] = inline_match.group(1).strip()
                sections["status"] = inline_match.group(2).strip()

        current_title: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal current_title, buffer
            if current_title is None:
                return
            content = "\n".join(buffer).strip()
            sections["sections"][current_title] = content
            current_title = None
            buffer = []

        for line in lines:
            stripped = line.strip()
            if stripped in {
                first_content,
                f"### {first_content}",
            } and sections["status"]:
                continue
            if stripped.endswith(":") and stripped[:-1] in titles:
                flush()
                current_title = stripped[:-1]
                continue
            if current_title is not None:
                buffer.append(line)
        flush()
        return sections

    @staticmethod
    def _extract_tools_from_note(note: str) -> list[str]:
        import re

        parsed = DistillationAgent._parse_note_sections(note)
        tools_section = parsed["sections"].get("Tools", "").strip()
        if not tools_section:
            tools_section = parsed["sections"].get("Winner Tools", "").strip()
        if not tools_section:
            tools_section = parsed["sections"].get("Winner Tool", "").strip()
        if not tools_section:
            tools_section = parsed["sections"].get("Winner Generator Tools", "").strip()
        if tools_section:
            tools: list[str] = []
            for line in tools_section.splitlines():
                item = line.strip()
                if item.startswith("- "):
                    item = item[2:].strip()
                if item and item not in _FRAMEWORK_TOOLS and item != "none":
                    tools.append(item)
            return tools

        legacy_match = re.search(r"(?:Tools|Winner Tools|Winner Tool|Winner Generator Tools):\s*(.+?)(?:\s*\|)", note)
        if not legacy_match:
            return []
        chain = [t.strip() for t in legacy_match.group(1).split("→")]
        return [t for t in chain if t and t not in _FRAMEWORK_TOOLS and t != "none"]

    @staticmethod
    def _extract_metrics_from_note(note: str) -> dict[str, float]:
        import re

        parsed = DistillationAgent._parse_note_sections(note)
        metrics_section = parsed["sections"].get("Metrics", "").strip()
        metrics: dict[str, float] = {}
        if metrics_section:
            for line in metrics_section.splitlines():
                item = line.strip()
                if item.startswith("- "):
                    item = item[2:].strip()
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    continue
            return metrics

        legacy_match = re.search(r"Metrics:\s*(.+?)(?:\s*\||$)", note)
        if not legacy_match:
            return {}
        for metric in legacy_match.group(1).split(","):
            item = metric.strip()
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
        return metrics

    @staticmethod
    def _extract_prediction_and_gt(note: str) -> tuple[str, str]:
        import re

        parsed = DistillationAgent._parse_note_sections(note)
        pred = parsed["sections"].get("Prediction")
        if pred is not None:
            return pred.strip(), parsed["sections"].get("GT", "").strip()

        legacy_match = re.search(r"Pred:\s*(.+?)\s*\|\s*GT:\s*(.+?)(?:\s*\||$)", note)
        if legacy_match:
            return legacy_match.group(1).strip(), legacy_match.group(2).strip()
        return "", ""

    @staticmethod
    def _compute_statistics(notes: list[str]) -> str:
        from collections import Counter

        tool_ok: Counter = Counter()
        tool_fail: Counter = Counter()
        labels_pred: Counter = Counter()
        labels_gt: Counter = Counter()
        metric_sums: dict[str, list[float]] = {}
        total_ok = 0
        total_fail = 0

        for note in notes:
            parsed = DistillationAgent._parse_note_sections(note)
            status = parsed.get("status", "")
            is_ok = status in {"OK", "SCORED"}
            if is_ok:
                total_ok += 1
            else:
                total_fail += 1

            for tool in DistillationAgent._extract_tools_from_note(note):
                (tool_ok if is_ok else tool_fail)[tool] += 1

            pred, gt = DistillationAgent._extract_prediction_and_gt(note)
            if pred or gt:
                p, g = pred, gt
                if len(p) < 30:
                    labels_pred[p] += 1
                if len(g) < 30:
                    labels_gt[g] += 1

            for key, value in DistillationAgent._extract_metrics_from_note(note).items():
                metric_sums.setdefault(key, []).append(value)

        lines: list[str] = []
        lines.append(f"Total runs: {total_ok + total_fail} (OK={total_ok}, FAIL={total_fail})")

        all_tools = set(tool_ok) | set(tool_fail)
        if all_tools:
            lines.append("\nTool success rates:")
            for t in sorted(all_tools, key=lambda x: -(tool_ok[x] + tool_fail[x])):
                ok_n, fail_n = tool_ok[t], tool_fail[t]
                total = ok_n + fail_n
                rate = ok_n / total * 100 if total else 0
                lines.append(f"  {t}: {ok_n}/{total} ({rate:.0f}%)")

        if labels_gt and len(labels_gt) <= 10:
            lines.append("\nGround truth label distribution:")
            for label, cnt in labels_gt.most_common():
                lines.append(f"  '{label}': {cnt}")
            lines.append("Predicted label distribution:")
            for label, cnt in labels_pred.most_common():
                lines.append(f"  '{label}': {cnt}")

        if metric_sums:
            lines.append("\nMetric averages:")
            for k, vals in sorted(metric_sums.items()):
                avg = sum(vals) / len(vals)
                lines.append(f"  {k}: avg={avg:.3f} (n={len(vals)})")

        return "\n".join(lines) if lines else "(no statistics available)"

    @staticmethod
    def _compute_tool_stats(notes: list[str]) -> str:
        """Extract per-tool performance statistics for TOOLS.md distillation."""
        from collections import Counter

        tool_ok: Counter = Counter()
        tool_fail: Counter = Counter()
        tool_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for note in notes:
            parsed = DistillationAgent._parse_note_sections(note)
            is_ok = parsed.get("status", "") in {"OK", "SCORED"}
            chain = DistillationAgent._extract_tools_from_note(note)
            if not chain:
                continue

            metric_vals = DistillationAgent._extract_metrics_from_note(note)

            for tool in chain:
                (tool_ok if is_ok else tool_fail)[tool] += 1
                for metric_key, metric_value in metric_vals.items():
                    tool_metrics[tool][metric_key].append(metric_value)

        lines = []
        all_tools = sorted(set(tool_ok) | set(tool_fail),
                           key=lambda x: -(tool_ok[x] + tool_fail[x]))
        for t in all_tools:
            ok_n, fail_n = tool_ok[t], tool_fail[t]
            total = ok_n + fail_n
            rate = ok_n / total * 100 if total else 0
            line = f"- **{t}**: {total} runs, {rate:.0f}% success"
            if t in tool_metrics:
                metric_parts = []
                for mk, mvs in sorted(tool_metrics[t].items()):
                    avg = sum(mvs) / len(mvs)
                    metric_parts.append(f"{mk}={avg:.3f}")
                if metric_parts:
                    line += f" | {', '.join(metric_parts)}"
            lines.append(line)

        return "\n".join(lines) if lines else "(no tool data)"

    @staticmethod
    def _tool_boundary_text(entry: dict[str, Any]) -> str:
        summary = str(entry.get("summary", "")).strip().rstrip(".")
        conditions = [str(item).strip().rstrip(".") for item in entry.get("conditions", []) if str(item).strip()]
        kind = str(entry.get("kind", "")).strip()
        preferred = [str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()]
        avoid = [str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()]

        cond_text = "; ".join(conditions[:2]) if conditions else ""

        action_text = ""
        if kind == "tool_selection":
            if preferred:
                action_text = f"prefer {preferred[0]}"
            elif summary:
                action_text = summary
        elif kind == "tool_failure":
            if avoid:
                action_text = f"avoid {', '.join(avoid[:2])}"
            elif summary:
                action_text = summary
        elif summary:
            action_text = summary

        if cond_text and action_text:
            return f"{cond_text}: {action_text}"[:160]
        if action_text:
            return action_text[:160]
        return cond_text[:160]

    def _build_tool_evidence_from_memory(self, scope: str) -> dict[str, dict[str, Any]]:
        """Collect scope-local tool-centered evidence bundles from MEMORY rules."""
        from collections import defaultdict

        all_entries = [
            entry for entry in self.memory.get_all_entries()
            if str(entry.get("scope", "")).strip() == scope
        ]
        grouped: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(
            lambda: {
                "when_it_helps": [],
                "when_to_avoid": [],
                "output_risks": [],
            }
        )

        for entry in all_entries:
            support = self._entry_support_count(entry)
            if support <= 0:
                continue
            kind = str(entry.get("kind", "")).strip()
            boundary_text = self._tool_boundary_text(entry)
            if not boundary_text:
                continue

            preferred = [str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()]
            avoid = [str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()]

            for tool_name in preferred:
                grouped[tool_name]["when_it_helps"].append((support, boundary_text))

            for tool_name in avoid:
                grouped[tool_name]["when_to_avoid"].append((support, boundary_text))
                if kind == "tool_failure":
                    grouped[tool_name]["output_risks"].append((support, boundary_text))

        tool_evidence: dict[str, dict[str, Any]] = {}
        for tool_name, buckets in grouped.items():
            normalized: dict[str, list[str]] = {}
            for field, items in buckets.items():
                dedup: list[str] = []
                seen: set[str] = set()
                for _, text in sorted(items, key=lambda item: (-item[0], item[1])):
                    if text in seen:
                        continue
                    seen.add(text)
                    dedup.append(text)
                if dedup:
                    normalized[field] = dedup[:2]
            if normalized.get("when_it_helps") or normalized.get("when_to_avoid") or normalized.get("output_risks"):
                tool_evidence[tool_name] = {
                    "when_it_helps": normalized.get("when_it_helps", []),
                    "when_to_avoid": normalized.get("when_to_avoid", []),
                    "output_risks": normalized.get("output_risks", []),
                }

        return tool_evidence

    # ── Notes → MEMORY.md ─────────────────────────────────────────────

    async def maybe_distill_notes_to_memory(self, scope: str, *, target_seq: int | None = None, force: bool = False) -> bool:
        key = scope
        state = await self._get_scope_state(key)
        done_seq = int(state.get("notes_to_memory_done_seq", 0) or 0)
        note_items = self.notes.get_scope_note_items(key)
        current = len(note_items)
        target = min(int(target_seq or current), current)
        if target <= done_seq:
            return False
        if not force and not self._threshold_reached(target, done_seq, _DISTILL_L1_TO_L2):
            return False

        pending_notes = note_items[done_seq:target]
        if len(pending_notes) < 2:
            return False

        logger.info(f"[Distill Notes→MEMORY] {key}: processing notes {done_seq + 1}-{target} of {current}")

        await self.notes.resolve_conflicts(query=scope, scope=scope)

        note_texts = [item["text"] for item in pending_notes]
        available_note_ids = {item["id"] for item in pending_notes}
        statistics = self._compute_statistics(note_texts)
        notes_text = "\n\n".join(
            f"- note_id={item['id']}\n{item['text']}"
            for item in pending_notes
        )

        existing_memory = self.memory.get_entries(key)
        existing_blocks: list[str] = []
        for e in existing_memory:
            existing_blocks.append(
                "\n".join(
                    [
                        f"  - rule_id={e.get('rule_id', '')}",
                        f"    kind={e.get('kind', '')}",
                        f"    summary={e.get('summary', '')}",
                        f"    conditions={json.dumps(e.get('conditions', []), ensure_ascii=False)}",
                        f"    preferred_tools={json.dumps(e.get('preferred_tools', []), ensure_ascii=False)}",
                        f"    avoid_tools={json.dumps(e.get('avoid_tools', []), ensure_ascii=False)}",
                        f"    rationale={e.get('rationale', '')}",
                        f"    support_count={self._entry_support_count(e)}",
                        f"    support_note_ids={json.dumps(self._entry_note_refs(e), ensure_ascii=False)}",
                        f"    last_seen_at={e.get('timestamp', 0.0)}",
                        f"    confidence={e.get('confidence', 0.0)}",
                        f"    injectable={e.get('injectable', True)}",
                    ]
                )
            )
        existing_text = "\n\n".join(existing_blocks) if existing_blocks else "(none yet)"

        prompt = NOTES_TO_MEMORY_PROMPT.format(
            scope=key, total=len(pending_notes),
            statistics=statistics, notes=notes_text,
            memory_rule_cap=_MEMORY_RULE_CAP,
            existing_memory=existing_text,
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": (
                      "You are a strategy analyst updating structured MEMORY rules. "
                      "Return JSON only. Prefer supporting or merging existing rules over inventing duplicates. "
                      "Rules must describe task-solving behavior only, never framework/orchestration behavior."
                  )},
                 {"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed and isinstance(resp.parsed, dict):
                updated_entries = self._apply_memory_update_decision(
                    key, existing_memory, resp.parsed, available_note_ids=available_note_ids
                )
                self.memory.replace_entries(key, updated_entries)
                logger.info(
                    f"[Distill Notes→MEMORY] {key}: updated to {len(self.memory.get_entries(key))} rules "
                )
                return True
        except Exception as e:
            logger.warning(f"[Distill Notes→MEMORY] {key} failed: {e}")
        return False

    # ── Notes → TOOLS.md ──────────────────────────────────────────────

    async def maybe_distill_memory_to_tools(self, scope: str) -> bool:
        """Rebuild scope-local tool cards from the current full MEMORY view."""
        full_notes = self.notes.get_recent_notes(10000)
        if full_notes:
            self._update_scope_counts(full_notes)

        tool_evidence = self._build_tool_evidence_from_memory(scope)
        if not tool_evidence:
            self.tools.replace(scope, {})
            logger.info(f"[Distill MEMORY→TOOLS] Rebuilt 0 tool notes for {scope}")
            return True

        existing_text = "\n".join(
            f"- **{k}**: {self.tools._render_note_payload(v)}"
            for k, v in self.tools.get_notes(scope).items()
        ) if self.tools.get_notes(scope) else "(none yet)"
        tool_evidence_text = json.dumps(tool_evidence, ensure_ascii=False, indent=2)

        prompt = MEMORY_TO_TOOLS_PROMPT.format(
            tool_evidence=tool_evidence_text,
            existing_tools=existing_text,
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [
                    {"role": "system", "content": "You write concise single-tool boundary cards. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format="json",
            )
            if resp.parsed and isinstance(resp.parsed, dict) and isinstance(resp.parsed.get("tool_notes"), dict):
                self.tools.replace(scope, resp.parsed["tool_notes"])
                logger.info(f"[Distill MEMORY→TOOLS] Rebuilt {len(resp.parsed['tool_notes'])} tool notes for {scope}")
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→TOOLS] {scope} failed: {e}")
        return False

    def _update_scope_counts(self, notes: list[str]) -> None:
        """Parse notes and rebuild per-scope tool usage counters.

        Notes now prefer canonical ``[scope_key]`` headers, but we still
        accept the legacy ``[domain/task]`` format for backward
        compatibility during migration.
        """
        scope_tool_counts: dict[str, dict[str, int]] = {}

        for note in notes:
            scope = self._extract_scope_from_note(note)
            if not scope:
                continue
            bucket = scope_tool_counts.setdefault(scope, {})
            for tool in self._extract_tools_from_note(note):
                bucket[tool] = bucket.get(tool, 0) + 1

        for scope, counts in scope_tool_counts.items():
            self.tools.record_scope_usage(scope, counts)

    def _extract_scope_from_note(self, note: str) -> str:
        """Extract the canonical scope key from a note block."""
        import re

        parsed = self._parse_note_sections(note)
        scope = str(parsed["sections"].get("Scope") or parsed.get("header") or "").strip()
        if not scope:
            match = re.search(r"(?:^|\n)\[([^\]]+)\]\s+(?:OK|FAIL|SCORED|LOGGED)\b", note)
            if not match:
                return ""
            scope = match.group(1).strip()
        if not scope:
            return ""
        if "/" in scope:
            domain, task = scope.split("/", 1)
            return f"{domain}_{self._denormalize_task_from_notes(task)}"
        return scope

    # ── MEMORY → cross_domain scope ──────────────────────────────────────

    async def maybe_distill_memory_to_cross_domain(self, bucket: str) -> bool:
        cross_scope = _cross_domain_scope_for_bucket(bucket)
        all_memory = [
            entry for entry in self.memory.get_all_entries()
            if _task_bucket_from_scope(str(entry.get("scope", "")).strip()) == bucket
            and not _is_cross_domain_scope(str(entry.get("scope", "")).strip())
        ]
        if len(all_memory) < 4:
            self.memory.replace_entries(cross_scope, [])
            logger.info(f"[Distill MEMORY→cross_domain] {bucket}: cleared cross-domain rules")
            return True

        logger.info(f"[Distill MEMORY→cross_domain] {bucket}: rebuilding from {len(all_memory)} rules")

        memory_lines = []
        for entry in all_memory:
            preferred = ", ".join(entry.get("preferred_tools", [])[:4])
            avoid = ", ".join(entry.get("avoid_tools", [])[:4])
            parts = [
                f"  rule_id={entry.get('rule_id', '')}",
                f"scope={entry.get('scope', '?')}",
                f"summary={entry.get('summary', '')}",
                f"support_count={self._entry_support_count(entry)}",
            ]
            if preferred:
                parts.append(f"prefer_tools={preferred}")
            if avoid:
                parts.append(f"avoid_tools={avoid}")
            memory_lines.append(" | ".join(parts))
        memory_text = "\n".join(memory_lines)

        existing_cross = self.memory.get_cross_domain_entries(bucket)
        existing_text = "\n".join(
            f"  - {e.get('summary', '')}" for e in existing_cross
        ) if existing_cross else "(none yet)"

        prompt = MEMORY_TO_CROSS_DOMAIN_PROMPT.format(
            allowed_tool_names=", ".join(sorted(_available_tool_names())),
            all_memory_entries=memory_text,
            existing_cross_domain=existing_text,
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You synthesize cross-domain MEMORY rules from per-scope rules. Return JSON."},
                 {"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed and "rules" in resp.parsed:
                removed_rule_ids = resp.parsed.get("removed_rule_ids", [])
                new_entries = []
                for i, rule in enumerate(resp.parsed["rules"]):
                    if not isinstance(rule, dict):
                        continue
                    normalized = self.memory._normalize_rule_payload(cross_scope, rule, i)
                    source_rule_ids = self._normalize_ref_list(
                        rule.get("source_rule_ids", []),
                        allowed={
                            str(entry.get("rule_id", "")).strip()
                            for entry in all_memory
                            if str(entry.get("rule_id", "")).strip()
                        },
                    )
                    note_refs = self._source_rule_note_refs(source_rule_ids)
                    evidence = normalized.get("evidence", {})
                    evidence["sample_refs"] = note_refs
                    evidence["support_count"] = len(note_refs)
                    normalized["evidence"] = evidence
                    normalized["source_rule_ids"] = source_rule_ids
                    new_entries.append(normalized)
                if removed_rule_ids:
                    allowed_removed = {str(item).strip() for item in removed_rule_ids if str(item).strip()}
                    new_entries = [
                        entry for entry in new_entries
                        if str(entry.get("rule_id", "")).strip() not in allowed_removed
                    ]
                self.memory.replace_entries(cross_scope, new_entries)
                n_cross = len(self.memory.get_cross_domain_entries(bucket))
                logger.info(
                    f"[Distill MEMORY→cross_domain] {bucket}: updated to {n_cross} cross-domain rules "
                    f"(removed {len(removed_rule_ids)})"
                )
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→cross_domain] {bucket} failed: {e}")
        return False

    # ── MEMORY → PLAYBOOKS/ ────────────────────────────────────────────

    async def maybe_distill_memory_to_skills(self, scope: str) -> bool:
        """Rebuild a SKILLS/ SOP from the current full MEMORY view for one scope."""
        entries = self.memory.get_entries(scope)
        if len(entries) < 3:
            self.skills.delete_skill(scope)
            return True

        skill_entries: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            evidence = entry.get("evidence") or {}
            conditions = entry.get("conditions") or []
            if not isinstance(conditions, list):
                conditions = [str(conditions)]
            preferred_tools = entry.get("preferred_tools") or []
            avoid_tools = entry.get("avoid_tools") or []
            skill_entries.append(
                "\n".join(
                    [
                        f"- rule_id: {entry.get('rule_id', '')}",
                        f"  kind: {entry.get('kind', '')}",
                        f"  summary: {entry.get('summary', '')}",
                        f"  conditions: {json.dumps([str(c) for c in conditions if str(c).strip()], ensure_ascii=False)}",
                        f"  rationale: {entry.get('rationale', '')}",
                        f"  preferred_tools: {json.dumps(preferred_tools, ensure_ascii=False, default=str)}",
                        f"  avoid_tools: {json.dumps(avoid_tools, ensure_ascii=False, default=str)}",
                        f"  support_count: {self._entry_support_count(entry)}",
                        f"  injectable: {entry.get('injectable', True)}",
                    ]
                )
            )
        entries_text = "\n\n".join(skill_entries)
        existing_skill = self.skills.get_skill(scope)

        prompt = MEMORY_TO_SKILLS_PROMPT.format(
            scope=scope,
            memory_entries=entries_text,
            existing_skill=existing_skill or "(none yet)",
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You write concise operational guides for a time-series agent. Return plain Markdown."},
                 {"role": "user", "content": prompt}],
            )
            if resp.text and len(resp.text.strip()) > 50:
                self.skills.update_skill(scope, resp.text.strip())
                logger.info(f"[Distill MEMORY→SKILLS] Updated SOP for {scope}")
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→SKILLS] {scope} failed: {e}")
        return False

    # ── Orchestration ─────────────────────────────────────────────────

    async def on_new_note(self, scope: str) -> None:
        """Backward-compatible wrapper for single-scope commits."""
        await self.on_scope_commit([scope])
        await self.on_global_commit([scope])

    async def force_distill_all(self) -> None:
        """Force a full rebuild from current notes/memory sources."""
        scopes = set()
        for f in MEMORY_DIR.glob("*.json"):
            scopes.add(f.stem)
        notes = self.notes.get_recent_notes(10000)
        for note in notes:
            scope = self._extract_scope_from_note(note)
            if scope:
                scopes.add(scope)

        for scope in scopes:
            if _is_cross_domain_scope(scope):
                continue
            note_total = self.notes.scoped_total(scope)
            await self._update_scope_state(
                scope,
                notes_total_seq=note_total,
                notes_to_memory_done_seq=0,
                notes_to_memory_inflight=False,
                memory_fingerprint="",
                memory_to_skills_done_fingerprint="",
                memory_to_skills_inflight=False,
                memory_to_tools_done_fingerprint="",
                memory_to_tools_inflight=False,
            )
            self._ensure_scope_worker(scope, force=True)

        buckets = sorted({_task_bucket_from_scope(scope) for scope in scopes if not _is_cross_domain_scope(scope)})
        for bucket in buckets:
            if not bucket:
                continue
            await self._update_bucket_state(
                bucket,
                memory_fingerprint="",
                memory_to_cross_done_fingerprint="",
                memory_to_cross_inflight=False,
            )
            self._ensure_bucket_worker(bucket, force=True)
        await self.distill_pending_all()

    async def distill_pending_all(self) -> None:
        """Drain all currently pending distillation work, including short tails."""
        scopes = set()
        for f in MEMORY_DIR.glob("*.json"):
            scopes.add(f.stem)
        notes = self.notes.get_recent_notes(10000)
        for note in notes:
            scope = self._extract_scope_from_note(note)
            if scope:
                scopes.add(scope)

        scope_list = [scope for scope in scopes if not _is_cross_domain_scope(scope)]
        bucket_list = sorted({_task_bucket_from_scope(scope) for scope in scope_list})

        for scope in scope_list:
            await self._update_scope_state(scope, notes_total_seq=self.notes.scoped_total(scope))

        while True:
            for scope in scope_list:
                self._ensure_scope_worker(scope, force=True)

            for bucket in bucket_list:
                if not bucket:
                    continue
                self._ensure_bucket_worker(bucket, force=True)

            scope_tasks = [task for task in self._scope_workers.values() if not task.done()]
            bucket_tasks = [task for task in self._bucket_workers.values() if not task.done()]
            pending = scope_tasks + bucket_tasks
            if not pending:
                break
            await asyncio.gather(*pending)

            tails_left = False
            for scope in scope_list:
                state = await self._get_scope_state(scope)
                if self.notes.scoped_total(scope) > int(state.get("notes_to_memory_done_seq", 0) or 0):
                    tails_left = True
                    break
                memory_fp = self._scope_memory_fingerprint(scope)
                if memory_fp != str(state.get("memory_to_skills_done_fingerprint", "") or ""):
                    tails_left = True
                    break
                if memory_fp != str(state.get("memory_to_tools_done_fingerprint", "") or ""):
                    tails_left = True
                    break

            if tails_left:
                continue

            cross_left = False
            for bucket in bucket_list:
                if not bucket:
                    continue
                state = await self._get_bucket_state(bucket)
                if self._bucket_memory_fingerprint(bucket) != str(state.get("memory_to_cross_done_fingerprint", "") or ""):
                    cross_left = True
                    break
            if not cross_left:
                break


# ═══════════════════════════════════════════════════════════════════════
# Unified Memory Interface
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalMemory:
    """Unified interface for TimeClaw's five-component memory system.

    Components (aligned with OpenClaw naming):
        soul_text   — Fixed SOUL.md (agent identity)
        notes       — Notes (raw experience log)
        memory      — MemoryStore (causal insights, per-scope + cross_domain)
        tools       — ToolsStore (per-tool usage experience notes)
        skills      — SkillsStore (scene-specific SOPs)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
            self.soul_text = load_soul()
            self.notes = Notes()
            self.memory = MemoryStore()
            self.tools = ToolsStore()
            self.skills = SkillsStore()
            self.distiller = DistillationAgent(
                self.notes, self.memory,
                self.tools, self.skills,
            )
            n_notes = self.notes.total
            n_memory = self.memory.total()
            n_cross = len(self.memory.get_cross_domain_entries())
            n_tools = self.tools.total()
            n_skills = self.skills.total()
            if n_notes + n_memory > 0:
                logger.info(
                    f"[AgentMemory] Loaded: "
                    f"notes={n_notes}, MEMORY={n_memory} (cross_domain={n_cross}), "
                    f"TOOLS={n_tools}, SKILLS={n_skills}"
                )
        else:
            self.soul_text = ""
            self.notes = None
            self.memory = None
            self.tools = None
            self.skills = None
            self.distiller = None

    # Backward compatibility aliases
    @property
    def l1(self):
        return self.notes

    @property
    def l2(self):
        return self.memory

    @property
    def principles(self):
        """Backward compat: returns memory store (cross_domain is inside)."""
        return self.memory

    @property
    def playbooks(self):
        """Backward compat: renamed to skills."""
        return self.skills

    @staticmethod
    def _normalized_scope_keys(run_context: dict[str, Any]) -> list[str]:
        scope_keys = run_context.get("scope_keys") or []
        if not scope_keys:
            scope_key = run_context.get("scope_key", "")
            if scope_key:
                scope_keys = [scope_key]
        if not scope_keys:
            domain = run_context.get("domain", "")
            task = run_context.get("task", "")
            if domain and task:
                scope_keys = [f"{domain}_{task}"]

        ordered: list[str] = []
        seen: set[str] = set()
        for raw_scope in scope_keys:
            scope = str(raw_scope).strip()
            if not scope or scope in seen:
                continue
            seen.add(scope)
            ordered.append(scope)
        return ordered

    async def record(self, **run_context) -> None:
        if not self.enabled:
            return
        scope_keys = self._normalized_scope_keys(run_context)
        if not scope_keys:
            return

        committed_scopes: list[str] = []
        for scope_key in scope_keys:
            await self.notes.add({**run_context, "scope_key": scope_key})
            if scope_key:
                committed_scopes.append(scope_key)
        if committed_scopes:
            await self.distiller.on_commit(committed_scopes)

    def get_memory(
        self, query: str, domain: str = "", task: str = "",
        scope_keys: list[str] | None = None,
    ) -> str:
        if not self.enabled:
            return ""
        parts = []
        if self.soul_text:
            parts.append(self.soul_text)
        if scope_keys:
            for scope in scope_keys:
                mem = self.memory.get_prompt_section(scope=scope)
                if mem:
                    parts.append(mem)
        else:
            mem = self.memory.get_prompt_section(domain, task) if domain and task else ""
            if mem:
                parts.append(mem)
        return "\n".join(parts)

    def get_tool_notes(
        self, domain: str = "", task: str = "",
        tool_names: list[str] | None = None,
        scope_keys: list[str] | None = None,
    ) -> dict[str, str]:
        if not self.enabled or not self.tools:
            return {}
        if scope_keys:
            merged: dict[str, str] = {}
            for scope in scope_keys:
                names = tool_names or list(self.tools.get_notes(scope).keys())
                scoped_notes = self.tools.get_prompt_for_tools(names, scope)
                for tool_name, note in scoped_notes.items():
                    existing = merged.get(tool_name, "")
                    if existing == note:
                        continue
                    if existing and note and note not in existing:
                        merged[tool_name] = f"{existing}\n{note}"
                    else:
                        merged[tool_name] = note or existing
            return merged

        scope = f"{domain}_{task}" if domain and task else ""
        names = tool_names or list(self.tools.get_notes(scope).keys())
        return self.tools.get_prompt_for_tools(names, scope)

    def get_skill(self, domain: str, task: str, scope_keys: list[str] | None = None) -> str:
        if not self.enabled or not self.skills:
            return ""
        if scope_keys:
            parts = [self.skills.get_prompt_section(scope=scope) for scope in scope_keys]
            return "\n".join(p for p in parts if p)
        return self.skills.get_prompt_section(domain, task)

    def get_playbook(self, domain: str, task: str) -> str:
        """Backward compatibility alias for get_skill."""
        return self.get_skill(domain, task)

    async def flush(self) -> None:
        if self.enabled and self.distiller:
            await self.distiller.distill_pending_all()

    async def force_distill(self) -> None:
        if self.enabled and self.distiller:
            await self.distiller.force_distill_all()

    async def distill_pending(self) -> None:
        if self.enabled and self.distiller:
            await self.distiller.distill_pending_all()

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "notes": self.notes.total,
            "memory_insights": self.memory.total(),
            "cross_domain_insights": len(self.memory.get_cross_domain_entries()),
            "tool_notes": self.tools.total(),
            "skills": self.skills.total(),
        }
