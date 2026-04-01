"""Agent Memory System — OpenClaw-inspired file-first architecture.

Five components (aligned with OpenClaw naming):

SOUL.md (fixed, human-written):
    Agent identity and behavioral guidelines.  Corresponds to OpenClaw SOUL.md.
    Not learned — maintained by developers.

Daily Notes (auto-recorded):
    Natural-language summaries of each pipeline run + outcome.
    Append-only, stored as daily log files as raw experience records.
    Corresponds to OpenClaw memory/YYYY-MM-DD.md.

MEMORY (auto-distilled from daily notes):
    LLM-distilled causal attributions explaining WHY certain patterns of
    success/failure occur.  Scoped to (domain, task).  Additionally holds
    a special ``cross_domain`` scope for universal operating strategies
    distilled across all scopes (replaces the former PRINCIPLES layer).
    Corresponds to OpenClaw MEMORY.md, but auto-distilled rather than
    manually curated.

TOOLS.md (auto-distilled from daily notes):
    Per-tool usage notes learned from experience — when each tool works
    well or poorly, helping the agent distinguish between functionally
    similar tools.  Corresponds to OpenClaw TOOLS.md.

SKILLS/ (auto-distilled from MEMORY):
    Scene-specific SOPs (Standard Operating Procedures) per (domain, task).
    Step-by-step operational guides.  Corresponds to OpenClaw SKILLS.md.

Distillation chain:
    Daily Notes  ──→  MEMORY (per-scope + cross_domain)
                 └──→  TOOLS.md
                 └──→  SKILLS/

New definition:
    ``scope_key`` is the canonical storage and retrieval key for one
    benchmark scope. It should match the benchmark's dataset-level scope
    name directly, for example ``weather_aligned_short_forecast``.

New definition:
    ``Daily Notes`` are now a distillation-source layer only. They remain
    append-only raw records used to distill MEMORY / TOOLS / SKILLS, but
    they are no longer injected directly into the agent prompt and no
    longer rely on a vector-store retrieval path.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

MEMORY_ROOT = Path(".timeclaw/hierarchical_memory")
DAILY_DIR = MEMORY_ROOT / "daily"
MEMORY_DIR = MEMORY_ROOT / "memory"
TOOLS_FILE = MEMORY_ROOT / "TOOLS.md"
SKILLS_DIR = MEMORY_ROOT / "skills"
SOUL_FILE = Path("timeclaw/memory/SOUL.md")

CROSS_DOMAIN_SCOPE = "cross_domain"
_FRAMEWORK_TOOLS = {
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
    "spawn_subagent",
    "none",
}

_DISTILL_L1_TO_L2 = 2
_DISTILL_L2_TO_CROSS = 20
_L1_RETRIEVAL_K = 5


# ═══════════════════════════════════════════════════════════════════════
# Daily Notes (append-only experience log, distillation source only)
# ═══════════════════════════════════════════════════════════════════════

class DailyNotes:
    """Append-only natural-language note store used as distillation source."""

    def __init__(self):
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        self._total_count = self._count_notes_on_disk()
        self._pending: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._note_llm = None

    @property
    def total(self) -> int:
        return self._total_count

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
        self._append_to_daily_log(note_text, run_context)

        async with self._lock:
            self._total_count += 1
            return self._total_count

    @staticmethod
    def _count_notes_on_disk() -> int:
        total = 0
        for path in DAILY_DIR.glob("*.md"):
            text = path.read_text()
            total += sum(1 for block in text.split("\n### ") if block.strip())
        return total

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
        approaches = self._format_approaches(candidate_traces)
        candidates = self._format_candidates(candidate_traces)

        scope_label = scope or f"{domain}/{task}"
        lines = [f"[{scope_label}] {outcome}", "", "Scope:", scope_label]
        if description:
            lines.extend(["", "Task:", self._task_without_long_text(description)])
        text_signals = self._select_text_signals(
            description=description,
            task=task,
            analysis_structured=analysis_structured,
        )
        if text_signals:
            lines.extend(["", "Text Signals:"])
            lines.extend(f"- {item}" for item in text_signals)
        lines.extend(["", "Winner Tool:"])
        if tools:
            lines.extend(f"- {tool}" for tool in tools[:20])
        else:
            lines.append("- none")
        lines.extend(["", "Prediction:", predicted])
        if metric_items:
            lines.extend(["", "Metrics:"])
            lines.extend(
                f"- {key}: {value:.4f}" for key, value in metric_items
            )
        if fingerprint:
            lines.extend(["", "Fingerprint:", fingerprint])
        if approaches:
            lines.extend(["", "Approaches:"])
            lines.extend(f"- {item}" for item in approaches[:5])
        if candidates:
            lines.extend(["", "Candidates:"])
            lines.extend(candidates[:4])
        rendered_summary = self._analysis_without_text_signals(pipeline_summary, bool(text_signals))
        if rendered_summary:
            lines.extend(["", "Analysis:", rendered_summary])
        return "\n".join(lines)

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
        for label in ("Weather report:", "News:", "Text:", "Context:"):
            idx = text.find(label)
            if idx < 0:
                continue
            snippet = text[idx + len(label):].strip()
            for stop in ("\n\nYour task is", "\nYour task is", "\n\nReturn ONLY", "\nReturn ONLY", "\n\nQuestion:", "\nQuestion:"):
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
        ranked_sentences = DailyNotes._rank_text_signals_for_task(
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

        task_focus = DailyNotes._task_focus_terms(task_description, task=task)

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
        for label in ("Weather report:", "News:", "Text:", "Context:"):
            idx = user_part.find(label)
            if idx < 0:
                continue
            prefix = user_part[:idx].rstrip()
            suffix = user_part[idx + len(label):]
            for stop in ("\n\nYour task is", "\nYour task is", "\n\nReturn ONLY", "\nReturn ONLY", "\n\nQuestion:", "\nQuestion:"):
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
        ranked = DailyNotes._ranked_candidate_summaries(candidate_traces)
        return ranked[:5]

    @staticmethod
    def _format_candidates(candidate_traces: list[dict[str, Any]]) -> list[str]:
        if not candidate_traces:
            return []
        formatted: list[str] = []
        for idx, trace in enumerate(candidate_traces[:4]):
            name = DailyNotes._display_candidate_name(trace, idx)
            metrics = trace.get("metrics") or {}
            _, score_label, _ = DailyNotes._candidate_score(metrics)
            header = f"- Candidate: {name}"
            if score_label:
                header += f" ({score_label})"
            block = [header]
            if metrics:
                metric_text = DailyNotes._stringify_metrics(metrics)
                if metric_text:
                    block.append(f"  Metrics: {metric_text}")
            tool_chain = [t for t in trace.get("tool_chain", []) if t and t not in _FRAMEWORK_TOOLS]
            if tool_chain:
                block.append(f"  Tool chain: {' → '.join(tool_chain)}")
            prediction = trace.get("prediction")
            if prediction is not None and not DailyNotes._observation_contains_prediction(trace.get("steps"), prediction):
                block.append(f"  Prediction: {DailyNotes._stringify_value(prediction)}")
            steps = DailyNotes._format_candidate_steps(trace.get("steps"))
            if steps:
                block.append("  Steps:")
                block.extend(steps)
            formatted.append("\n".join(block))
        return formatted

    @staticmethod
    def _ranked_candidate_summaries(candidate_traces: list[dict[str, Any]]) -> list[str]:
        ranked: list[tuple[tuple[int, float], str]] = []
        fallback: list[str] = []
        seen: set[str] = set()
        for idx, trace in enumerate(candidate_traces):
            if not isinstance(trace, dict):
                continue
            name = DailyNotes._display_candidate_name(trace, idx)
            if name in seen:
                continue
            seen.add(name)
            metrics = trace.get("metrics") or {}
            score_value, score_label, sort_key = DailyNotes._candidate_score(metrics)
            if score_label:
                ranked.append((sort_key, f"{name}({score_label})"))
            else:
                fallback.append(name)
        ranked.sort(key=lambda item: item[0])
        return [text for _, text in ranked] + fallback

    @staticmethod
    def _format_candidate_steps(steps: Any) -> list[str]:
        if not isinstance(steps, list):
            return []
        rendered: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "").strip()
            if not tool:
                continue
            rendered.append(f"  - Tool: {tool}")
            reason = str(step.get("thought") or "").strip()
            if reason:
                rendered.append(f"    Reason/Action: {reason}")
            observation = step.get("observation")
            if observation is not None:
                rendered.append(f"    Observation: {DailyNotes._stringify_observation(observation)}")
        return rendered

    @staticmethod
    def _display_candidate_name(trace: dict[str, Any], index: int) -> str:
        name = str(trace.get("name") or "").strip()
        if name and not name.startswith("approach_"):
            return name[:80]
        tool_chain = [t for t in trace.get("tool_chain", []) if t and t not in _FRAMEWORK_TOOLS]
        if tool_chain:
            return " → ".join(tool_chain[:3])[:80]
        return f"candidate_{index}"

    @staticmethod
    def _candidate_score(metrics: dict[str, Any]) -> tuple[float | None, str, tuple[int, float]]:
        if not isinstance(metrics, dict) or not metrics:
            return None, "", (9, 0.0)
        for key in ("mae", "avg_mae", "rmse", "mse", "max_mae"):
            if isinstance(metrics.get(key), (int, float)):
                value = float(metrics[key])
                label_parts = [f"{key}={value:.2f}"]
                if key != "rmse" and isinstance(metrics.get("rmse"), (int, float)):
                    label_parts.append(f"rmse={float(metrics['rmse']):.2f}")
                return value, ", ".join(label_parts), (1, value)
        if isinstance(metrics.get("accuracy"), (int, float)):
            value = float(metrics["accuracy"])
            return value, f"acc={value:.2f}", (0, -value)
        return None, "", (9, 0.0)

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _stringify_metrics(metrics: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("mae", "avg_mae", "rmse", "mse", "mape", "max_mae", "accuracy"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.4f}")
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
                return json.dumps(filtered, ensure_ascii=False, default=str)
            except TypeError:
                return str(filtered)
        return DailyNotes._stringify_value(payload)

    @staticmethod
    def _observation_contains_prediction(steps: Any, prediction: Any) -> bool:
        if not isinstance(steps, list):
            return False
        prediction_text = DailyNotes._stringify_value(prediction)
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
                else DailyNotes._stringify_value(observation)
            )
            if prediction_text and prediction_text in observation_text:
                return True
        return False

    def _append_to_daily_log(self, note: str, ctx: dict[str, Any]) -> None:
        today = time.strftime("%Y-%m-%d")
        path = DAILY_DIR / f"{today}.md"
        scope = ctx.get("scope_key", "")
        domain = ctx.get("domain", "")
        task = ctx.get("task", "")
        correct = self._outcome_label(ctx.get("correct"), ctx.get("metrics") or {})
        timestamp = time.strftime("%H:%M:%S")

        label = scope or f"{domain}/{task}"
        entry = f"\n### [{timestamp}] {label} — {correct}\n{note}\n"
        with path.open("a") as f:
            f.write(entry)

    def search(self, query: str, k: int = _L1_RETRIEVAL_K,
               domain: str | None = None, task: str | None = None,
               scope: str | None = None,
               return_ids: bool = False) -> list[str] | list[dict[str, Any]]:
        tokens = [token.lower() for token in re.findall(r"\w+", query or "") if token.strip()]
        matches: list[dict[str, Any]] = []
        files = sorted(DAILY_DIR.glob("*.md"), reverse=True)
        for path in files:
            text = path.read_text()
            for idx, block in enumerate(reversed(text.split("\n### "))):
                note = block.strip()
                if not note:
                    continue
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
                    "id": f"{path.name}#{idx}",
                    "text": note,
                    "score": score,
                })
        matches.sort(key=lambda item: (-int(item["score"]), item["id"]), reverse=False)
        trimmed = matches[:k]
        if return_ids:
            return [{"id": item["id"], "text": item["text"]} for item in trimmed]
        return [item["text"] for item in trimmed]

    def delete_notes(self, note_ids: list[str]) -> int:
        logger.info("[DailyNotes] delete_notes is disabled; daily notes are kept as raw distillation sources")
        return 0

    def get_recent_notes(self, n: int = 10) -> list[str]:
        files = sorted(DAILY_DIR.glob("*.md"), reverse=True)
        notes = []
        for f in files:
            if len(notes) >= n:
                break
            for block in reversed(f.read_text().split("\n### ")):
                if block.strip():
                    notes.append(block.strip())
                    if len(notes) >= n:
                        break
        return notes[:n]

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
# MEMORY.md — Causal insights distilled from daily notes
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
                    self._entries[f.stem] = [self._normalize_entry(f.stem, entry, idx) for idx, entry in enumerate(raw)]
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

    def get_cross_domain_entries(self) -> list[dict]:
        return self.get_entries(CROSS_DOMAIN_SCOPE)

    @staticmethod
    def _normalize_rule_payload(scope: str, payload: dict[str, Any], index: int) -> dict[str, Any]:
        kind = str(payload.get("kind", "")).strip() or "domain_heuristic"
        summary = str(payload.get("summary", "")).strip()
        conditions = [str(item).strip() for item in payload.get("conditions", []) if str(item).strip()]
        preferred_tools = [str(item).strip() for item in payload.get("preferred_tools", []) if str(item).strip()]
        avoid_tools = [str(item).strip() for item in payload.get("avoid_tools", []) if str(item).strip()]
        rationale = str(payload.get("rationale", "")).strip()
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
            status="active",
        ).to_dict()

    def get_prompt_section(self, domain: str = "", task: str = "", scope: str = "") -> str:
        scope = scope or f"{domain}_{task}"
        entries = self.get_entries(scope)
        domain_entries = self.get_entries(domain) if domain else []
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

        parts: list[str] = []
        if summary:
            parts.append(summary)
        if conditions:
            parts.append(f"Conditions: {', '.join(conditions[:3])}")
        if preferred:
            parts.append(f"Prefer: {', '.join(preferred[:3])}")
        if avoid:
            parts.append(f"Avoid: {', '.join(avoid[:3])}")
        if rationale:
            parts.append(f"Why: {rationale}")
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
                status="active",
                timestamp=float(entry.get("timestamp", 0.0) or time.time()),
            ).to_dict()
            return entry

        summary = str(entry.get("summary", "")).strip()
        rule_id = str(entry.get("rule_id", "")).strip() or f"{scope}::{index + 1:03d}"
        kind = str(entry.get("kind", "")).strip() or "domain_heuristic"
        conditions = [str(item).strip() for item in entry.get("conditions", []) if str(item).strip()]
        preferred_tools = [str(item).strip() for item in entry.get("preferred_tools", []) if str(item).strip()]
        avoid_tools = [str(item).strip() for item in entry.get("avoid_tools", []) if str(item).strip()]
        rationale = str(entry.get("rationale", "")).strip()
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        confidence = float(entry.get("confidence", 0.5) or 0.5)
        injectable = bool(entry.get("injectable", True))
        status = str(entry.get("status", "active")).strip() or "active"
        timestamp = float(entry.get("timestamp", 0.0) or time.time())

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
                int((entry.get("evidence") or {}).get("support_count", 0) or 0),
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
                -int((entry.get("evidence") or {}).get("support_count", 0) or 0),
            )
        )
        return unique_entries[:20]


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
    """TOOLS.md — per-tool experience notes, distilled from daily notes.

    Global notes describe each tool's general behaviour.  Per-scope
    counters track how many times each tool has been used in a given
    (domain, task) scope so we can flag under-explored tools.
    """

    _MIN_RUNS = 5
    _MAX_PROMPT_TOOLS = 5

    def __init__(self):
        MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
        self._notes: dict[str, dict[str, Any]] = {}
        self._scope_counts: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if TOOLS_FILE.exists():
            try:
                raw = json.loads(TOOLS_FILE.read_text())
                if isinstance(raw, dict) and "_scope_counts" in raw:
                    self._notes = {
                        str(k): self._normalize_note_payload(v)
                        for k, v in (raw.get("notes", {}) or {}).items()
                    }
                    self._scope_counts = raw.get("_scope_counts", {})
                else:
                    self._notes = {
                        str(k): self._normalize_note_payload(v)
                        for k, v in (raw.items() if isinstance(raw, dict) else [])
                    }
            except Exception:
                self._notes = {}

    def _save(self) -> None:
        payload = {"notes": self._notes, "_scope_counts": self._scope_counts}
        TOOLS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def get_notes(self) -> dict[str, dict[str, Any]]:
        return dict(self._notes)

    def update(self, notes: dict[str, Any]) -> None:
        self._notes.update({
            str(tool_name): self._normalize_note_payload(payload)
            for tool_name, payload in notes.items()
        })
        self._save()
        logger.info(f"[TOOLS] Updated usage notes for {len(notes)} tools")

    def record_scope_usage(self, scope: str, tool_counts: dict[str, int]) -> None:
        """Replace per-tool usage counts for a scope from the latest scan."""
        self._scope_counts[scope] = dict(tool_counts)
        self._save()

    def total(self) -> int:
        return len(self._notes)

    def get_tool_note(self, tool_name: str) -> str:
        payload = self._notes.get(tool_name)
        return self._render_note_payload(payload) if payload else ""

    def get_prompt_for_tools(
        self, tool_names: list[str], scope: str = "",
    ) -> dict[str, str]:
        """Return notes for the given tools, limited to a small relevant subset."""
        scope_data = self._scope_counts.get(scope, {}) if scope else {}
        ranked: list[tuple[int, str, str]] = []
        for name in tool_names:
            payload = self._notes.get(name)
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

## Sample Experience Notes (recent)
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
- Keep the total active rule set at or below 20 after the update

## Output Requirements
Return JSON only with this exact top-level structure:
{{
  "support_updates": [
    {{"rule_id": "existing_rule_id_1", "add_support": 2}}
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
      "initial_support": 2
    }}
  ]
}}

If no items belong in a section, return an empty list for that section.
"""

MEMORY_TO_CROSS_DOMAIN_PROMPT = """\
You are synthesizing domain-specific decision rules into universal
cross-domain insights for a multimodal time-series analysis agent.

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
      "rationale": "why this rule is useful"
    }}
  ],
  "removed_rule_ids": ["existing_rule_id_1", "existing_rule_id_2"]
}}
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
# Distillation Agent
# ═══════════════════════════════════════════════════════════════════════

class DistillationAgent:
    """Orchestrates bottom-up knowledge distillation.

    Distillation chain:
        Daily Notes  ──→  MEMORY (per-scope + cross_domain)
                     └──→  TOOLS.md
                     └──→  SKILLS/
    """

    def __init__(self, daily: DailyNotes, memory: MemoryStore,
                 tools: ToolsStore, skills: SkillsStore):
        self.daily = daily
        self.memory = memory
        self.tools = tools
        self.skills = skills
        self._notes_distilled_at: dict[str, int] = {}
        self._memory_distilled_at: int = 0
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
                support = int((other.get("evidence") or {}).get("support_count", 0) or 0)
                max_support = max(max_support, support)
        return max_support

    def _apply_memory_update_decision(
        self,
        scope: str,
        existing_entries: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        entries = self.memory._dedupe_rule_ids([dict(entry) for entry in existing_entries])
        by_id = {str(entry.get("rule_id", "")).strip(): entry for entry in entries if str(entry.get("rule_id", "")).strip()}
        now = time.time()

        support_updates: dict[str, int] = {}
        for item in decision.get("support_updates", []):
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("rule_id", "")).strip()
            if not rule_id or rule_id not in by_id:
                continue
            try:
                add_support = int(item.get("add_support", 0) or 0)
            except Exception:
                add_support = 0
            if add_support > 0:
                support_updates[rule_id] = support_updates.get(rule_id, 0) + add_support

        # Backward compatibility for older LLM outputs.
        for rule_id in decision.get("support_rule_ids", []):
            rid = str(rule_id).strip()
            if rid and rid in by_id:
                support_updates[rid] = support_updates.get(rid, 0) + 1
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
        for rule_id, add_support in support_updates.items():
            entry = by_id[rule_id]
            evidence = dict(entry.get("evidence") or {})
            evidence["support_count"] = int(evidence.get("support_count", 0) or 0) + int(add_support)
            entry["evidence"] = evidence
            entry["timestamp"] = now

        # Merge duplicate/overlapping rules by keeping the strongest first id.
        merged_away: set[str] = set()
        for group in merge_groups:
            survivors = [by_id[rule_id] for rule_id in group if rule_id in by_id and rule_id not in merged_away]
            if len(survivors) < 2:
                continue
            survivors.sort(key=lambda e: (int((e.get("evidence") or {}).get("support_count", 0) or 0), float(e.get("timestamp", 0.0) or 0.0)), reverse=True)
            keeper = survivors[0]
            keeper_evidence = dict(keeper.get("evidence") or {})
            sample_refs = list(keeper_evidence.get("sample_refs", []) or [])
            total_support = int(keeper_evidence.get("support_count", 0) or 0)
            for entry in survivors[1:]:
                merged_away.add(str(entry.get("rule_id", "")).strip())
                other_evidence = dict(entry.get("evidence") or {})
                total_support += int(other_evidence.get("support_count", 0) or 0)
                sample_refs.extend(other_evidence.get("sample_refs", []) or [])
            keeper_evidence["support_count"] = total_support
            keeper_evidence["sample_refs"] = sample_refs[:20]
            keeper["evidence"] = keeper_evidence
            keeper["timestamp"] = now

        entries = [entry for entry in entries if str(entry.get("rule_id", "")).strip() not in merged_away]
        by_id = {str(entry.get("rule_id", "")).strip(): entry for entry in entries if str(entry.get("rule_id", "")).strip()}

        # Add brand-new rules.
        base_index = len(entries)
        for idx, payload in enumerate(decision.get("new_rules", [])):
            if not isinstance(payload, dict):
                continue
            normalized = self.memory._normalize_rule_payload(scope, payload, base_index + idx)
            normalized["timestamp"] = now
            normalized["injectable"] = True
            normalized["status"] = "active"
            evidence = dict(normalized.get("evidence") or {})
            try:
                initial_support = int(payload.get("initial_support", 1) or 1)
            except Exception:
                initial_support = 1
            evidence["support_count"] = max(1, initial_support)
            evidence["sample_refs"] = []
            normalized["evidence"] = evidence
            entries.append(normalized)

        by_id = {str(entry.get("rule_id", "")).strip(): entry for entry in entries if str(entry.get("rule_id", "")).strip()}

        # Recompute injectable/confidence from conflict decisions.
        for entry in entries:
            rule_id = str(entry.get("rule_id", "")).strip()
            max_conflict_support = self._max_conflict_support(rule_id, conflict_groups, by_id)
            evidence = dict(entry.get("evidence") or {})
            support_count = int(evidence.get("support_count", 0) or 0)
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
                -int((entry.get("evidence") or {}).get("support_count", 0) or 0),
            )
        )
        return self.memory._dedupe_rule_ids(entries)

    @staticmethod
    def _normalize_task_for_notes(task: str) -> str:
        """Map dataset task names to the task labels stored in daily notes."""
        return {
            "mcqa": "qa",
            "trend": "classification",
        }.get(task, task)

    @staticmethod
    def _denormalize_task_from_notes(task: str) -> str:
        """Map daily-note task labels back to canonical memory scope names."""
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
            "Task",
            "Tools",
            "Winner Tool",
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
            tools_section = parsed["sections"].get("Winner Tool", "").strip()
        if tools_section:
            tools: list[str] = []
            for line in tools_section.splitlines():
                item = line.strip()
                if item.startswith("- "):
                    item = item[2:].strip()
                if item and item not in _FRAMEWORK_TOOLS and item != "none":
                    tools.append(item)
            return tools

        legacy_match = re.search(r"(?:Tools|Winner Tool):\s*(.+?)(?:\s*\|)", note)
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
                lines.append(f"  {k}: avg={avg:.4f} (n={len(vals)})")

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
                    metric_parts.append(f"{mk}={avg:.4f}")
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

    def _build_tool_evidence_from_memory(self) -> dict[str, dict[str, Any]]:
        """Collect tool-centered evidence bundles from MEMORY rules."""
        from collections import defaultdict

        all_entries = [
            entry for entry in self.memory.get_all_entries()
            if str(entry.get("scope", "")).strip() != CROSS_DOMAIN_SCOPE
        ]
        grouped: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(
            lambda: {
                "when_it_helps": [],
                "when_to_avoid": [],
                "output_risks": [],
            }
        )

        for entry in all_entries:
            support = int((entry.get("evidence") or {}).get("support_count", 0) or 0)
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

    async def maybe_distill_notes_to_memory(self, scope: str) -> bool:
        key = scope
        current = self.daily.total
        last = self._notes_distilled_at.get(key, 0)
        if current - last < _DISTILL_L1_TO_L2:
            return False

        logger.info(f"[Distill Notes→MEMORY] {key}: {current} notes (+{current - last})")
        self._notes_distilled_at[key] = current

        await self.daily.resolve_conflicts(query=scope, scope=scope)

        notes = self.daily.search(
            query=f"{scope} error failure wrong",
            k=30, scope=scope,
        )
        success_notes = self.daily.search(
            query=f"{scope} correct success",
            k=15, scope=scope,
        )
        all_notes = notes + success_notes
        if len(all_notes) < 2:
            return False

        statistics = self._compute_statistics(all_notes)
        notes_text = "\n\n".join(f"- {n}" for n in all_notes[:30])

        existing_memory = self.memory.get_entries(key)
        existing_text = "\n".join(
            f"  - rule_id={e.get('rule_id','')}; kind={e.get('kind','')}; "
            f"summary={e.get('summary','')}; support_count={(e.get('evidence') or {}).get('support_count', 0)}; "
            f"last_seen_at={e.get('timestamp', 0.0)}; confidence={e.get('confidence', 0.0)}; "
            f"injectable={e.get('injectable', True)}"
            for e in existing_memory
        ) if existing_memory else "(none yet)"

        prompt = NOTES_TO_MEMORY_PROMPT.format(
            scope=key, total=len(all_notes),
            statistics=statistics, notes=notes_text,
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
                updated_entries = self._apply_memory_update_decision(key, existing_memory, resp.parsed)
                self.memory.replace_entries(key, updated_entries)
                logger.info(
                    f"[Distill Notes→MEMORY] {key}: updated to {len(self.memory.get_entries(key))} rules "
                )
                return True
        except Exception as e:
            logger.warning(f"[Distill Notes→MEMORY] {key} failed: {e}")
        return False

    # ── Notes → TOOLS.md ──────────────────────────────────────────────

    async def maybe_distill_notes_to_tools(self) -> bool:
        """Derive TOOLS.md from tool-related MEMORY rules.

        Scope usage counters still come from daily notes, but tool cards
        themselves are derived from already-distilled MEMORY rather than
        exploration winners in raw notes.
        """
        full_notes = self.daily.get_recent_notes(10000)
        if full_notes:
            self._update_scope_counts(full_notes)

        tool_evidence = self._build_tool_evidence_from_memory()
        if not tool_evidence:
            self.tools.update({})
            logger.info("[Distill MEMORY→TOOLS] Updated 0 tool notes")
            return True

        existing_text = "\n".join(
            f"- **{k}**: {self.tools._render_note_payload(v)}"
            for k, v in self.tools.get_notes().items()
        ) if self.tools.total() > 0 else "(none yet)"
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
                self.tools.update(resp.parsed["tool_notes"])
                logger.info(f"[Distill MEMORY→TOOLS] Updated {len(resp.parsed['tool_notes'])} tool notes")
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→TOOLS] failed: {e}")
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
        """Extract the canonical scope key from a daily-note block."""
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

    async def maybe_distill_memory_to_cross_domain(self) -> bool:
        current = self.memory.total()
        if current - self._memory_distilled_at < _DISTILL_L2_TO_CROSS:
            return False

        logger.info(f"[Distill MEMORY→cross_domain] {current} entries (+{current - self._memory_distilled_at})")
        self._memory_distilled_at = current

        all_memory = self.memory.get_all_entries()
        if len(all_memory) < 4:
            return False

        memory_text = "\n".join(
            f"  [{e.get('scope', '?')}] {e.get('summary', '')}"
            for e in all_memory[-60:]
        )

        existing_cross = self.memory.get_cross_domain_entries()
        existing_text = "\n".join(
            f"  - {e.get('summary', '')}" for e in existing_cross
        ) if existing_cross else "(none yet)"

        prompt = MEMORY_TO_CROSS_DOMAIN_PROMPT.format(
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
                    normalized = self.memory._normalize_rule_payload(CROSS_DOMAIN_SCOPE, rule, i)
                    evidence = normalized.get("evidence", {})
                    evidence["support_count"] = len(all_memory)
                    normalized["evidence"] = evidence
                    new_entries.append(normalized)
                self.memory.add_entries(
                    CROSS_DOMAIN_SCOPE, new_entries,
                    removed_rule_ids=removed_rule_ids,
                )
                n_cross = len(self.memory.get_cross_domain_entries())
                logger.info(
                    f"[Distill MEMORY→cross_domain] Updated to {n_cross} cross-domain rules "
                    f"(removed {len(removed_rule_ids)})"
                )
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→cross_domain] failed: {e}")
        return False

    # ── MEMORY → PLAYBOOKS/ ────────────────────────────────────────────

    async def maybe_distill_memory_to_skills(self, scope: str) -> bool:
        """Generate/update a SKILLS/ SOP from MEMORY insights for a scope."""
        entries = self.memory.get_entries(scope)
        if len(entries) < 3:
            return False

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
                        f"  support_count: {evidence.get('support_count', 0)}",
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
        """Called after each daily note is added. Triggers distillation cascade."""
        did_memory = await self.maybe_distill_notes_to_memory(scope)
        if did_memory:
            await self.maybe_distill_memory_to_cross_domain()
            await self.maybe_distill_memory_to_skills(scope)
            await self.maybe_distill_notes_to_tools()

    async def force_distill_all(self) -> None:
        """Force distillation for all scopes."""
        scopes = set()
        for f in MEMORY_DIR.glob("*.json"):
            scopes.add(f.stem)
        notes = self.daily.get_recent_notes(10000)
        for note in notes:
            scope = self._extract_scope_from_note(note)
            if scope:
                scopes.add(scope)

        for scope in scopes:
            if scope == CROSS_DOMAIN_SCOPE:
                continue
            self._notes_distilled_at[scope] = 0
            await self.maybe_distill_notes_to_memory(scope)
            await self.maybe_distill_memory_to_skills(scope)

        self._memory_distilled_at = 0
        await self.maybe_distill_memory_to_cross_domain()
        await self.maybe_distill_notes_to_tools()

    async def distill_pending_all(self) -> None:
        """Finish only pending incremental distillation without resetting counters."""
        scopes = set()
        for f in MEMORY_DIR.glob("*.json"):
            scopes.add(f.stem)
        notes = self.daily.get_recent_notes(10000)
        for note in notes:
            scope = self._extract_scope_from_note(note)
            if scope:
                scopes.add(scope)

        did_memory_any = False
        for scope in scopes:
            if scope == CROSS_DOMAIN_SCOPE:
                continue
            did_memory = await self.maybe_distill_notes_to_memory(scope)
            if did_memory:
                did_memory_any = True
                await self.maybe_distill_memory_to_skills(scope)

        if did_memory_any:
            await self.maybe_distill_memory_to_cross_domain()
            await self.maybe_distill_notes_to_tools()


# ═══════════════════════════════════════════════════════════════════════
# Unified Memory Interface
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalMemory:
    """Unified interface for TimeClaw's five-component memory system.

    Components (aligned with OpenClaw naming):
        soul_text   — Fixed SOUL.md (agent identity)
        daily       — DailyNotes (raw experience log)
        memory      — MemoryStore (causal insights, per-scope + cross_domain)
        tools       — ToolsStore (per-tool usage experience notes)
        skills      — SkillsStore (scene-specific SOPs)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._pending_distill_tasks: set[asyncio.Task[Any]] = set()
        if enabled:
            MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
            self.soul_text = load_soul()
            self.daily = DailyNotes()
            self.memory = MemoryStore()
            self.tools = ToolsStore()
            self.skills = SkillsStore()
            self.distiller = DistillationAgent(
                self.daily, self.memory,
                self.tools, self.skills,
            )
            n_daily = self.daily.total
            n_memory = self.memory.total()
            n_cross = len(self.memory.get_cross_domain_entries())
            n_tools = self.tools.total()
            n_skills = self.skills.total()
            if n_daily + n_memory > 0:
                logger.info(
                    f"[AgentMemory] Loaded: "
                    f"daily={n_daily}, MEMORY={n_memory} (cross_domain={n_cross}), "
                    f"TOOLS={n_tools}, SKILLS={n_skills}"
                )
        else:
            self.soul_text = ""
            self.daily = None
            self.memory = None
            self.tools = None
            self.skills = None
            self.distiller = None

    # Backward compatibility aliases
    @property
    def l1(self):
        return self.daily

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

    async def record(self, **run_context) -> None:
        if not self.enabled:
            return
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

        for scope_key in scope_keys:
            count = await self.daily.add({**run_context, "scope_key": scope_key})
            if count % _DISTILL_L1_TO_L2 == 0 and scope_key:
                task = asyncio.create_task(self.distiller.on_new_note(scope_key))
                self._pending_distill_tasks.add(task)
                task.add_done_callback(self._pending_distill_tasks.discard)

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
        names = tool_names or list(self.tools.get_notes().keys())
        if scope_keys:
            merged: dict[str, str] = {}
            for scope in scope_keys:
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
        if not self._pending_distill_tasks:
            return
        pending = tuple(self._pending_distill_tasks)
        await asyncio.gather(*pending, return_exceptions=True)

    async def force_distill(self) -> None:
        if self.enabled and self.distiller:
            await self.distiller.force_distill_all()

    async def distill_pending(self) -> None:
        if not self.enabled or not self.distiller:
            return
        await self.flush()
        await self.distiller.distill_pending_all()

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "daily_notes": self.daily.total,
            "memory_insights": self.memory.total(),
            "cross_domain_insights": len(self.memory.get_cross_domain_entries()),
            "tool_notes": self.tools.total(),
            "skills": self.skills.total(),
        }
