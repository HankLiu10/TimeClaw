"""Train loop: explorative learning on the **official MTBench train list**
(``train_files.json`` + ``data/processed/`` via ``load_split``, not the
independent corpus unless paths are changed).

In training mode the agent uses the same prompt as testing, with a short
learning-mode hint appended.  It also receives an ``evaluate_against_gt``
tool that returns metrics + ground truth for one-step analysis (GT is
sanitised from the conversation afterwards).

The loop:
  1. Extract ground truth from the sample
  2. Call ``pipeline.run(mode="train", ground_truth=gt)``
  3. The agent explores autonomously, evaluates against GT, summarises
  4. Learning outcomes are saved to Notes (L1) + MEMORY (L2)
  5. Notes are recorded immediately; distillation drains asynchronously by scope

All tools are visible to the agent from the start — it explores freely.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from loguru import logger

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.mtbench_adapter import (
    BENCHMARK_EXPANDED_TASK_TARGET,
    DATASET_REGISTRY,
    MTBenchSample,
    datasets_with_benchmark_tasks,
    expanded_benchmark_task_specs,
)
from timeclaw.benchmark.mtbench_support import (
    build_task_description,
    expand_dataset_tasks,
    load_sample_indices,
    scope_keys_for_dataset_task,
    select_samples_with_original_indices,
)
from timeclaw.core.config import TimeClawConfig
from timeclaw.core import trace_logger as trace
from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.benchmark.baseline_eval import (
    _weather_trend_3way_label,
    _weather_trend_past_3way_label,
)

TRAIN_RESULTS_DIR = Path(".timeclaw/train_results")
TRAIN_SUMMARY_PATH = TRAIN_RESULTS_DIR / "train_summary.json"
TRAIN_RUN_MANIFEST_PATH = TRAIN_RESULTS_DIR / "train_run_manifest.json"
_TRAIN_RESULT_FRAMEWORK_TOOLS = {
    "spawn_subagent",
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
}


def _task_train_checkpoint_path(ds_name: str, task: str) -> Path:
    return TRAIN_RESULTS_DIR / f"train_{ds_name}__{task}.json"


def _task_note_view_path(ds_name: str, task: str) -> Path:
    return TRAIN_RESULTS_DIR / f"train_notes_{ds_name}__{task}.md"


def _load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 10:
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [
        _normalize_train_result_row(r)
        for r in data
        if isinstance(r, dict)
    ]


def _load_train_checkpoint(ds_name: str, task: str | None = None) -> list[dict[str, Any]]:
    if task is not None:
        return _load_checkpoint_rows(_task_train_checkpoint_path(ds_name, task))

    task_paths = sorted(TRAIN_RESULTS_DIR.glob(f"train_{ds_name}__*.json"))
    rows: list[dict[str, Any]] = []
    for path in task_paths:
        rows.extend(_load_checkpoint_rows(path))
    return rows


def _checkpoint_exists(ds_name: str) -> bool:
    return any(
        path.stat().st_size > 10
        for path in TRAIN_RESULTS_DIR.glob(f"train_{ds_name}__*.json")
    )


def _normalize_train_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize persisted train-result rows to a stable top-level schema.

    New definition:
        ``train_results`` rows should always expose the same locating fields at
        the top level, even if older checkpoints were written before those
        aliases existed. This keeps filtering, retrying, and winner-chain
        analysis stable across historical and newly appended samples.
    """
    normalized = dict(row)

    dataset_name = str(
        normalized.get("dataset_name")
        or normalized.get("dataset")
        or ""
    ).strip()
    if dataset_name:
        normalized["dataset_name"] = dataset_name
        normalized["dataset"] = dataset_name

    idx_value = normalized.get("idx", normalized.get("abs_idx"))
    try:
        idx_int = int(idx_value) if idx_value is not None else None
    except (TypeError, ValueError):
        idx_int = None
    if idx_int is not None:
        normalized["idx"] = idx_int
        normalized["abs_idx"] = idx_int

    substantive_tool_chain = _extract_train_result_tool_chain(normalized)
    if substantive_tool_chain:
        normalized["tool_chain"] = substantive_tool_chain
        normalized["winner_tool_chain"] = list(substantive_tool_chain)
    else:
        normalized["winner_tool_chain"] = _filter_train_result_tool_chain(
            normalized.get("winner_tool_chain")
        )

    train_run_id = str(normalized.get("train_run_id") or "").strip()
    if train_run_id:
        normalized["train_run_id"] = train_run_id

    return normalized


def _train_done_keys(rows: list[dict[str, Any]]) -> set[tuple[int, str]]:
    out: set[tuple[int, str]] = set()
    for r in rows:
        status = str(r.get("status", "")).lower()
        if status in {"error", "timeout", "failed", "max_steps_exhausted", "cancelled"}:
            continue
        if r.get("error"):
            continue
        # Backward-compatibility: older checkpoints recorded timeout/error rows
        # without a status field, leaving only an empty shell row.
        if (
            not status
            and "answer_type" in r
            and "has_prediction" in r
            and "tools_run" in r
            and not r.get("answer_type")
            and not bool(r.get("has_prediction"))
            and int(r.get("tools_run", 0) or 0) == 0
        ):
            continue
        if "idx" in r and "task" in r:
            try:
                out.add((int(r["idx"]), str(r["task"])))
            except (TypeError, ValueError):
                continue
    return out


def _merge_train_rows(
    existing: list[dict[str, Any]], new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer newer rows for the same (idx, task)."""
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for r in existing:
        if "idx" not in r or "task" not in r:
            continue
        try:
            k = (int(r["idx"]), str(r["task"]))
        except (TypeError, ValueError):
            continue
        by_key[k] = r
    for r in new_rows:
        if "idx" not in r or "task" not in r:
            continue
        try:
            k = (int(r["idx"]), str(r["task"]))
        except (TypeError, ValueError):
            continue
        by_key[k] = r
    merged = list(by_key.values())
    merged.sort(key=lambda x: (int(x.get("idx", 0)), str(x.get("task", ""))))
    return merged


def _preview_value(value: Any, limit: int | None = 400) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text)} chars]"


def _sanitize_learning_artifact_preview(value: Any, limit: int | None = 400) -> str:
    """Sanitize visible training previews down to evidence-facing language.

    New definition:
        ``reasoning_preview`` is meant for humans to quickly inspect what
        evidence or candidate comparison happened. It should not expose
        orchestration mechanics such as branch contracts, "task-valid answer
        already exists", or finish/return-control guidance from the executor.
    """
    if isinstance(value, str):
        text = value
    else:
        preview_limit = None if limit is None else limit * 2
        text = _preview_value(value, limit=preview_limit)
    if not text:
        return text
    # Numbered LLM reasoning sometimes arrives as one long line:
    # "1. ... 2. ... 3. ..."
    # Split those inline steps first so banned control-flow phrases only drop
    # the affected step instead of erasing the whole preview.
    text = re.sub(r"(?<!^)\s+(?=\d+\.\s)", "\n", text.strip())
    hard_banned_markers = (
        "spawn_subagent",
        "evaluate_against_gt",
        "evaluate_batch_against_gt",
        "batch_evaluate_against_gt",
    )
    full_line_banned_markers = (
        "current branch state confirms",
        "available from a real tool result",
        "returned unchanged as the final",
        "may be returned unchanged",
        "corrected final answer",
        "repaired summary",
        "retained in the branch",
    )
    soft_banned_markers = (
        "execution context",
        "assigned path",
        "before finishing",
        "task-valid",
        "learning mode",
        "training mode",
        "branch comparison",
        "most efficient next step",
        "most efficient way",
        "most direct next step",
        "available from a real tool result",
        "returned unchanged as the final",
        "may be returned unchanged",
        "current branch state confirms",
        "corrected final answer",
        "repaired summary",
        "retained in the branch",
    )
    cleaned_lines: list[str] = []
    soft_phrase_patterns = (
        (r"(?i)\btask-valid\s+", ""),
        (r"(?i)\bcurrent branch state confirms(?: that)?\b", ""),
        (r"(?i)\bavailable from a real tool result\b", ""),
        (r"(?i)\breturned unchanged as the final\b", ""),
        (r"(?i)\bmay be returned unchanged\b", ""),
        (r"(?i)\bbefore finishing\b", ""),
        (r"(?i)\bexecution context\b", ""),
        (r"(?i)\bassigned path\b", ""),
        (r"(?i)\blearning mode\b", ""),
        (r"(?i)\btraining mode\b", ""),
        (r"(?i)\bbranch comparison\b", ""),
        (r"(?i)\bmost efficient next step\b", ""),
        (r"(?i)\bmost efficient way\b", ""),
        (r"(?i)\bmost direct next step\b", ""),
        (r"(?i)\bcorrected final answer\b", ""),
        (r"(?i)\brepaired summary\b", ""),
        (r"(?i)\bretained in the branch\b", ""),
    )
    for line in text.splitlines():
        lower = line.lower()
        if any(marker in lower for marker in hard_banned_markers):
            continue
        if any(marker in lower for marker in full_line_banned_markers):
            continue
        cleaned_line = line
        for pattern, repl in soft_phrase_patterns:
            cleaned_line = re.sub(pattern, repl, cleaned_line)
        cleaned_line = re.sub(r"\s+", " ", cleaned_line).strip()
        cleaned_line = re.sub(r"[ ,;\-]+$", "", cleaned_line)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    cleaned = "\n".join(cleaned_lines).strip()
    lower_cleaned = cleaned.lower()
    if any(marker in lower_cleaned for marker in hard_banned_markers):
        return ""
    if any(marker in lower_cleaned for marker in full_line_banned_markers):
        return ""
    if any(marker in lower_cleaned for marker in soft_banned_markers):
        return ""
    if limit is None or len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + f"... [{len(cleaned)} chars]"


def _extract_train_result_reasoning_preview(
    result: dict[str, Any],
    learning_summary: dict[str, Any],
) -> str:
    """Return a readable reasoning preview instead of raw final-answer JSON.

    New definition:
        ``train_results.reasoning_preview`` should prefer the actual natural-
        language reasoning string. If the executor stored the final envelope as
        JSON text, extract nested ``reasoning`` first and only fall back to the
        raw string when no cleaner explanation exists.
    """
    candidates: list[str] = []
    execution_log = result.get("execution_log", [])
    raw_answer = result.get("raw_answer")
    if isinstance(execution_log, list) and execution_log:
        candidate_traces = PipelineRunner._extract_candidate_traces(execution_log)
        if candidate_traces:
            evaluated_metrics = [
                trace.get("metrics")
                for trace in candidate_traces
                if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
            ]
            best_metrics = PipelineRunner._select_best_metrics(evaluated_metrics)
            winner_trace, _winner_source = PipelineRunner._resolve_learning_summary_winner_trace(
                raw_answer if isinstance(raw_answer, dict) else {},
                candidate_traces,
                best_metrics,
            )
            winner_prediction = (winner_trace or {}).get("prediction")
            if winner_prediction is None and isinstance(raw_answer, dict):
                winner_prediction = raw_answer.get("winner_prediction")
            substantive_tool_chain = _extract_train_result_tool_chain(result)
            preview_trace = winner_trace
            winner_chain = _filter_train_result_tool_chain((winner_trace or {}).get("tool_chain"))
            if (
                preview_trace is None
                or (substantive_tool_chain and winner_chain and winner_chain != substantive_tool_chain)
            ):
                preview_trace = _select_train_result_preview_trace_by_chain(
                    candidate_traces,
                    substantive_tool_chain,
                    winner_prediction=winner_prediction,
                )
            if preview_trace is None and winner_trace is not None:
                preview_trace = winner_trace
            preview = _format_train_result_reasoning_preview_from_trace(preview_trace)
            if preview:
                return preview

    summary_reasoning = learning_summary.get("reasoning")
    if isinstance(summary_reasoning, str) and summary_reasoning.strip():
        candidates.append(summary_reasoning)

    execution_context = (
        result.get("execution_context")
        if isinstance(result.get("execution_context"), dict)
        else {}
    )
    public_reasoning = execution_context.get("public_reasoning")
    if isinstance(public_reasoning, list):
        joined_public = "\n".join(
            str(item).strip()
            for item in public_reasoning
            if str(item).strip()
        ).strip()
        if joined_public:
            candidates.append(joined_public)
    evidence_summary = str(execution_context.get("evidence_summary") or "").strip()
    if evidence_summary:
        candidates.append(evidence_summary)

    raw_reasoning = result.get("reasoning", "")
    parsed_reasoning = PipelineRunner._parse_json_like_payload(raw_reasoning)
    if isinstance(parsed_reasoning, dict):
        nested_answer = parsed_reasoning.get("answer")
        if isinstance(nested_answer, dict):
            nested_reasoning = nested_answer.get("reasoning")
            if isinstance(nested_reasoning, str) and nested_reasoning.strip():
                candidates.append(nested_reasoning)
        parsed_text_reasoning = parsed_reasoning.get("reasoning")
        if isinstance(parsed_text_reasoning, str) and parsed_text_reasoning.strip():
            candidates.append(parsed_text_reasoning)
        parsed_thought = parsed_reasoning.get("thought")
        if isinstance(parsed_thought, str) and parsed_thought.strip():
            reason_only = _preview_reason_only(parsed_thought)
            if reason_only:
                candidates.append(reason_only)
            candidates.append(parsed_thought)
    if isinstance(raw_reasoning, str) and raw_reasoning.strip():
        candidates.append(raw_reasoning)

    for candidate in candidates:
        preview = _sanitize_learning_artifact_preview(candidate, limit=None)
        if preview:
            return preview
    return ""


def _format_train_result_reasoning_preview_from_trace(trace: dict[str, Any] | None) -> str:
    if not isinstance(trace, dict):
        return ""
    steps = trace.get("steps")
    step_list = steps if isinstance(steps, list) else []
    lines: list[str] = []
    for idx, step in enumerate(step_list[:4], 1):
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "").strip() or "unknown_tool"
        observation = PipelineRunner._parse_json_like_payload(step.get("observation"))
        if observation is None or observation == "":
            continue
        prediction = PipelineRunner._prediction_from_observation(observation)
        payload = prediction if prediction is not None else observation
        preview = _preview_value(payload, limit=180)
        lines.append(f"{len(lines) + 1}. {tool} produced {preview}.")
    if not lines:
        snippets = PipelineRunner._reasoning_snippets(trace)
        for idx, snippet in enumerate(snippets[:4], 1):
            if not isinstance(snippet, str) or not snippet.strip():
                continue
            lines.append(
                f"{idx}. {snippet if str(snippet).rstrip().endswith(('.', '!', '?')) else str(snippet).rstrip() + '.'}"
            )
    if not lines:
        return ""
    return _sanitize_learning_artifact_preview("\n".join(lines), limit=None)


def _filter_train_result_tool_chain(chain: list[Any] | tuple[Any, ...] | None) -> list[str]:
    out: list[str] = []
    for tool in chain or []:
        name = str(tool or "").strip()
        if not name or name in _TRAIN_RESULT_FRAMEWORK_TOOLS:
            continue
        out.append(name)
    return out


def _extract_train_result_tool_chain(result: dict[str, Any]) -> list[str]:
    """Return the substantive winner chain for train_results rows.

    New definition:
        ``train_results.tool_chain`` should reflect the actual candidate tools
        that produced the recorded learning outcome, not framework orchestration
        tools such as ``spawn_subagent`` or ``evaluate_*``.

    Example:
        Old row: ``["spawn_subagent", "evaluate_batch_against_gt"]``
        New row: ``["hurst_exponent", "timesfm2_forecast"]``
    """
    raw_answer = result.get("raw_answer")
    if isinstance(raw_answer, dict):
        execution_context = (
            raw_answer.get("execution_context")
            if isinstance(raw_answer.get("execution_context"), dict)
            else {}
        )
        winner_chain = _filter_train_result_tool_chain(
            raw_answer.get("winner_tool_chain") or execution_context.get("tool_chain")
        )
        if winner_chain:
            return winner_chain

    execution_context = (
        result.get("execution_context")
        if isinstance(result.get("execution_context"), dict)
        else {}
    )
    winner_chain = _filter_train_result_tool_chain(
        execution_context.get("tool_chain")
    )
    if winner_chain:
        return winner_chain

    execution_log = result.get("execution_log", [])
    if isinstance(execution_log, list) and execution_log:
        candidate_traces = PipelineRunner._extract_candidate_traces(execution_log)
        evaluated_metrics = [
            trace.get("metrics")
            for trace in candidate_traces
            if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
        ]
        best_metrics = PipelineRunner._select_best_metrics(evaluated_metrics)
        winner_chain = _filter_train_result_tool_chain(
            PipelineRunner._select_winner_tool_chain(candidate_traces, best_metrics)
        )
        if winner_chain:
            return winner_chain

        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            trace_chain = _filter_train_result_tool_chain(trace.get("tool_chain"))
            if trace_chain:
                return trace_chain

    top_level_chain = _filter_train_result_tool_chain(result.get("tool_chain"))
    if top_level_chain:
        return top_level_chain

    if isinstance(execution_log, list):
        return _filter_train_result_tool_chain(
            [entry.get("tool") for entry in execution_log if isinstance(entry, dict)]
        )
    return []


def _build_train_result_evidence_preview(
    result: dict[str, Any],
    substantive_tool_chain: list[str],
) -> str:
    """Build a winner-aligned evidence preview for train_results rows.

    New definition:
        ``train_results.evidence_summary_preview`` should summarize the actual
        candidate chain that produced the recorded learning outcome, not the
        top-level executor evidence log. This keeps the reported step count
        aligned with ``tool_chain`` and avoids mixing framework orchestration
        steps with the winner branch preview.
    """
    raw_summary = result.get("evidence_summary", "")
    execution_log = result.get("execution_log", [])
    raw_answer = result.get("raw_answer")

    if not isinstance(execution_log, list) or not execution_log:
        trace_preview = _build_train_result_evidence_preview_from_trace(
            trace_id=str(result.get("trace_id") or "").strip(),
            substantive_tool_chain=substantive_tool_chain,
        )
        if trace_preview:
            return trace_preview
        return _sanitize_learning_artifact_preview(raw_summary, limit=400)

    candidate_traces = PipelineRunner._extract_candidate_traces(execution_log)
    if not candidate_traces:
        return _sanitize_learning_artifact_preview(raw_summary, limit=400)

    evaluated_metrics = [
        trace.get("metrics")
        for trace in candidate_traces
        if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
    ]
    best_metrics = PipelineRunner._select_best_metrics(evaluated_metrics)
    winner_trace, _winner_source = PipelineRunner._resolve_learning_summary_winner_trace(
        raw_answer if isinstance(raw_answer, dict) else {},
        candidate_traces,
        best_metrics,
    )
    winner_prediction = (winner_trace or {}).get("prediction")
    if winner_prediction is None and isinstance(raw_answer, dict):
        winner_prediction = raw_answer.get("winner_prediction")

    winner_chain = _filter_train_result_tool_chain((winner_trace or {}).get("tool_chain"))
    if winner_trace is not None and (
        not substantive_tool_chain or winner_chain == substantive_tool_chain
    ):
        preview = _format_train_result_evidence_preview_from_steps(winner_trace.get("steps"))
        if _preview_has_substantive_step_detail(preview):
            return preview
        trace_preview = _build_train_result_evidence_preview_from_trace(
            trace_id=str(result.get("trace_id") or "").strip(),
            substantive_tool_chain=winner_chain or substantive_tool_chain,
        )
        if trace_preview:
            return trace_preview
        return preview

    exact_match = _select_train_result_preview_trace_by_chain(
        candidate_traces,
        substantive_tool_chain,
        winner_prediction=winner_prediction,
    )
    if exact_match is not None:
        preview = _format_train_result_evidence_preview_from_steps(exact_match.get("steps"))
        if _preview_has_substantive_step_detail(preview):
            return preview
        trace_preview = _build_train_result_evidence_preview_from_trace(
            trace_id=str(result.get("trace_id") or "").strip(),
            substantive_tool_chain=substantive_tool_chain,
        )
        if trace_preview:
            return trace_preview
        return preview

    if winner_trace is None:
        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            trace_chain = _filter_train_result_tool_chain(trace.get("tool_chain"))
            if trace_chain:
                winner_trace = trace
                break

    if winner_trace is None:
        return _sanitize_learning_artifact_preview(raw_summary, limit=400)

    preview = _format_train_result_evidence_preview_from_steps(winner_trace.get("steps"))
    if _preview_has_substantive_step_detail(preview):
        return preview
    trace_preview = _build_train_result_evidence_preview_from_trace(
        trace_id=str(result.get("trace_id") or "").strip(),
        substantive_tool_chain=substantive_tool_chain,
    )
    if trace_preview:
        return trace_preview
    return preview


def _select_train_result_preview_trace_by_chain(
    candidate_traces: list[dict[str, Any]],
    substantive_tool_chain: list[str],
    winner_prediction: Any | None = None,
) -> dict[str, Any] | None:
    if not substantive_tool_chain:
        return None
    exact_matches: list[dict[str, Any]] = []
    for trace in candidate_traces:
        if not isinstance(trace, dict):
            continue
        trace_chain = _filter_train_result_tool_chain(trace.get("tool_chain"))
        if trace_chain == substantive_tool_chain:
            exact_matches.append(trace)
    if not exact_matches:
        return None
    if winner_prediction is not None:
        prediction_matches = [
            trace
            for trace in exact_matches
            if PipelineRunner._predictions_match(trace.get("prediction"), winner_prediction)
        ]
        if len(prediction_matches) == 1:
            return prediction_matches[0]
    return max(
        exact_matches,
        key=lambda item: (
            len(item.get("steps") or []),
            len(_filter_train_result_tool_chain(item.get("tool_chain"))),
        ),
    )


def _format_train_result_evidence_preview_from_steps(steps: Any) -> str:
    step_list = steps if isinstance(steps, list) else []
    formatted_lines = [f"Evidence Log ({len(step_list)} steps):"]
    successful_steps = 0
    for idx, step in enumerate(step_list):
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "").strip() or "unknown_tool"
        observation = str(step.get("observation") or "").strip()
        detail = _build_train_result_step_detail(step)
        status = "OK" if observation else "FAIL"
        if observation:
            successful_steps += 1
        formatted_lines.append(f"  [{status}] Step {idx}: {tool} -> {detail}")
    if step_list:
        formatted_lines.append(f"  Successful steps: {successful_steps}/{len(step_list)}")
    return _sanitize_learning_artifact_preview("\n".join(formatted_lines), limit=None)


def _build_train_result_step_detail(step: dict[str, Any]) -> str:
    """Format one preview step from actual execution, not intended next actions.

    New definition:
        ``evidence_summary_preview`` is an execution summary of what actually
        happened. If a step has an observation, that observation is the preview
        payload; we do not append ReAct-style ``Reason:`` text, because that
        text often describes control flow or search intent rather than grounded
        evidence. Thought text is only used when no observation was recorded.
    """
    observation = str(step.get("observation") or "").strip()
    raw_thought = str(step.get("thought") or "").strip()
    cleaned_fallback = PipelineRunner._sanitize_reasoning_clause(raw_thought)

    if observation:
        return f"Observed: {observation}"
    if cleaned_fallback:
        return cleaned_fallback
    return "no detail recorded"


def _preview_reason_only(thought: str) -> str:
    text = str(thought or "").strip()
    if not text:
        return ""
    reason_match = re.search(r"\breason\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if reason_match is None:
        return PipelineRunner._sanitize_reasoning_clause(text)
    reason_text = reason_match.group(1).strip()
    parts = re.split(r"\baction\s*:", reason_text, maxsplit=1, flags=re.IGNORECASE)
    return PipelineRunner._sanitize_reasoning_clause(parts[0].strip())


def _preview_has_substantive_step_detail(preview: str) -> bool:
    text = str(preview or "").strip()
    if not text:
        return False
    if "no detail recorded" in text:
        return False
    return bool(re.search(r"\bStep\s+\d+:", text))


def _build_train_result_evidence_preview_from_trace(
    *,
    trace_id: str,
    substantive_tool_chain: list[str],
) -> str:
    """Rebuild preview from the persisted sample trace when row-local logs are absent.

    New definition:
        ``train_results`` rows are intentionally lightweight and may omit the
        full ``execution_log``. In that case, we recover the candidate step
        summary from ``.timeclaw/traces/<trace_id>.jsonl`` so historical rows
        can still be backfilled without rerunning the sample.
    """
    if not trace_id or not substantive_tool_chain:
        return ""
    trace_path = Path(".timeclaw/traces") / f"{trace_id}.jsonl"
    if not trace_path.exists():
        return ""

    top_level_steps: list[dict[str, str]] = []
    subagent_steps: dict[int, list[dict[str, str]]] = defaultdict(list)
    candidate_traces: list[dict[str, Any]] = []
    final_tool = str(substantive_tool_chain[-1] if substantive_tool_chain else "").strip()

    try:
        for raw_line in trace_path.read_text().splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            if record.get("event") != "agent_step":
                continue
            payload = record.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            summary = payload.get("summary") or {}
            if not isinstance(summary, dict):
                continue
            phase = str(payload.get("phase") or "").strip()

            if phase == "subagent_start":
                try:
                    sub_idx = int(summary.get("sub_agent_idx"))
                except (TypeError, ValueError):
                    continue
                subagent_steps[sub_idx] = []
                continue

            if phase.startswith("react_step_"):
                tool = str(summary.get("tool") or "").strip()
                if not tool or tool in _TRAIN_RESULT_FRAMEWORK_TOOLS:
                    continue
                step_payload = {
                    "tool": tool,
                    "thought": str(summary.get("thought") or "").strip(),
                    "observation": str(summary.get("observation_preview") or "").strip(),
                }
                top_level_steps.append(step_payload)
                candidate_traces.append({
                    "tool_chain": _filter_train_result_tool_chain([tool]),
                    "steps": [dict(step_payload)],
                })
                if step_payload["observation"] and final_tool and tool == final_tool:
                    candidate_traces.append({
                        "tool_chain": _filter_train_result_tool_chain(
                            [step.get("tool") for step in top_level_steps if isinstance(step, dict)]
                        ),
                        "steps": list(top_level_steps),
                    })
                continue

            if phase == "subagent_tool_result":
                try:
                    sub_idx = int(summary.get("sub_agent_idx"))
                except (TypeError, ValueError):
                    continue
                tool = str(summary.get("tool") or "").strip()
                if not tool or tool in _TRAIN_RESULT_FRAMEWORK_TOOLS:
                    continue
                subagent_steps[sub_idx].append({
                    "tool": tool,
                    "observation": _trace_preview_text(summary.get("observation")),
                })
                continue

            if phase != "subagent_complete":
                continue

            try:
                sub_idx = int(summary.get("sub_agent_idx"))
            except (TypeError, ValueError):
                continue

            preview_chain = _filter_train_result_tool_chain(
                _trace_summary_preview_list(summary.get("tool_chain"))
            )
            observed_chain = _filter_train_result_tool_chain(
                [step.get("tool") for step in subagent_steps.get(sub_idx, []) if isinstance(step, dict)]
            )
            tool_chain = preview_chain
            if observed_chain and (
                not tool_chain
                or len(observed_chain) > len(tool_chain)
                or (final_tool and observed_chain[-1] == final_tool and tool_chain[-1:] != [final_tool])
            ):
                tool_chain = observed_chain
            candidate_traces.append({
                "tool_chain": tool_chain,
                "steps": list(subagent_steps.get(sub_idx, [])),
            })
    except (OSError, json.JSONDecodeError):
        return ""

    trace_match = _select_train_result_preview_trace_by_chain(
        candidate_traces,
        substantive_tool_chain,
    )
    if trace_match is None:
        return ""
    return _format_train_result_evidence_preview_from_steps(trace_match.get("steps"))


def _trace_summary_preview_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        preview = value.get("preview")
        if isinstance(preview, list):
            return [str(item).strip() for item in preview if str(item).strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _trace_preview_text(value: Any) -> str:
    if isinstance(value, dict):
        preview = value.get("preview")
        if isinstance(preview, str):
            return preview.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value)


def _recover_failed_result_from_trace(trace_id: str | None) -> dict[str, Any]:
    """Recover the last solve-ready payload from a persisted trace.

    Why this exists:
        Some failed train rows are not execution failures in the usual sense.
        The agent may already have produced a final `learning_summary`, but the
        terminal reuse gate rejects the packaging and the run then exhausts the
        remaining step budget. In that case the trace still contains the final
        attempted payload, and train_results should preserve it for debugging
        instead of recording an empty shell row.
    """
    trace_name = str(trace_id or "").strip()
    if not trace_name:
        return {}
    trace_path = Path(".timeclaw/traces") / f"{trace_name}.jsonl"
    if not trace_path.exists():
        return {}

    recovered: dict[str, Any] = {}
    last_done_payload: dict[str, Any] = {}
    try:
        for raw_line in trace_path.read_text().splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            event = str(record.get("event") or "").strip()
            payload = record.get("payload") or {}
            if not isinstance(payload, dict):
                continue

            if event == "content_block":
                kind = str(payload.get("kind") or "").strip()
                if kind != "prompt_message/assistant":
                    continue
                content = str(payload.get("content") or "").strip()
                if not content.startswith("{"):
                    continue
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                if parsed.get("done") is True and parsed.get("answer_type"):
                    last_done_payload = parsed
                continue

            if event == "run_end":
                recovered["trace_status"] = str(payload.get("status") or "").strip()
                recovered["trace_error"] = str(payload.get("error") or "").strip()
                recovered["trace_error_type"] = str(payload.get("error_type") or "").strip()
                recovered["trace_failure_reason"] = str(payload.get("failure_reason") or "").strip()

        if last_done_payload:
            recovered.update(
                {
                    "answer_type": str(last_done_payload.get("answer_type") or "").strip(),
                    "raw_answer": last_done_payload.get("answer"),
                    "confidence": last_done_payload.get("confidence"),
                    "reasoning": str(last_done_payload.get("reasoning") or "").strip(),
                    "execution_context": (
                        last_done_payload.get("execution_context")
                        if isinstance(last_done_payload.get("execution_context"), dict)
                        else {}
                    ),
                    "control_context": (
                        last_done_payload.get("control_context")
                        if isinstance(last_done_payload.get("control_context"), dict)
                        else {}
                    ),
                    "trace_recovered": True,
                }
            )
    except (OSError, json.JSONDecodeError):
        return {}
    return recovered


def _build_failed_result_payload(exc: Exception, recovered: dict[str, Any]) -> dict[str, Any]:
    """Merge direct exception payloads with trace fallback for failed samples.

    New definition:
        Failed train rows should keep the richest execution-side debug evidence
        available immediately: prefer executor/pipeline `partial_result`
        payloads, then explicit exception attributes, then trace recovery.
        This avoids empty-shell `train_results` rows when the run had already
        executed tools or produced a near-final learning payload before failing.
    """
    partial_result = getattr(exc, "partial_result", None)
    partial = partial_result if isinstance(partial_result, dict) else {}

    def _pick(*values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, dict) and not value:
                continue
            if isinstance(value, list) and not value:
                continue
            return value
        return None

    raw_answer = _pick(
        partial.get("answer"),
        getattr(exc, "raw_answer", None),
        recovered.get("raw_answer"),
    )
    execution_log = _pick(
        partial.get("execution_log"),
        getattr(exc, "execution_log", None),
    ) or []
    execution_context = _pick(
        partial.get("execution_context"),
        getattr(exc, "execution_context", None),
        recovered.get("execution_context"),
    ) or {}
    control_context = _pick(
        partial.get("control_context"),
        getattr(exc, "control_context", None),
        recovered.get("control_context"),
    ) or {}

    return {
        "answer_type": str(_pick(partial.get("answer_type"), getattr(exc, "answer_type", None), recovered.get("answer_type"), "") or ""),
        "execution_log": execution_log if isinstance(execution_log, list) else [],
        "prediction": _pick(partial.get("prediction"), raw_answer),
        "status": str(getattr(exc, "failure_reason", "") or "error"),
        "error": str(exc),
        "error_type": str(getattr(exc, "error_type", type(exc).__name__)),
        "failure_reason": str(getattr(exc, "failure_reason", "") or "error"),
        "trace_id": getattr(exc, "trace_id", None),
        "total_ms": getattr(exc, "total_ms", 0),
        "raw_answer": raw_answer,
        "confidence": _pick(partial.get("confidence"), getattr(exc, "confidence", None), recovered.get("confidence")),
        "reasoning": str(_pick(partial.get("reasoning"), getattr(exc, "reasoning", None), recovered.get("reasoning"), "") or ""),
        "execution_context": execution_context if isinstance(execution_context, dict) else {},
        "control_context": control_context if isinstance(control_context, dict) else {},
        "trace_recovered": bool(recovered.get("trace_recovered", False)),
    }


def _build_train_result_row(
    *,
    ds_name: str,
    sample_idx: int,
    sample: MTBenchSample,
    task: str,
    pipeline_task: str,
    scope_keys: list[str],
    result: dict[str, Any],
    train_run_id: str = "",
) -> dict[str, Any]:
    """Persist one train-result row with both preview and full debug payloads.

    New definition:
        `train_results` rows keep the compact previews for quick scanning, but
        also store the exact executor outputs (`raw_answer`,
        `execution_context`, `control_context`, `execution_log`) so debugging
        no longer needs an immediate second hop into traces.
    """
    raw_answer = result.get("raw_answer")
    learning_summary = raw_answer if isinstance(raw_answer, dict) else {}
    sample_ref = f"{ds_name}:{sample.file}:{sample_idx}:{task}"
    substantive_tool_chain = _extract_train_result_tool_chain(result)

    row = {
        "idx": sample_idx,
        "abs_idx": sample_idx,
        "dataset": ds_name,
        "dataset_name": ds_name,
        "sample_ref": sample_ref,
        "file": sample.file,
        "task": task,
        "pipeline_task": pipeline_task,
        "scope_keys": list(scope_keys),
        "train_run_id": str(train_run_id or "").strip(),
        "domain": sample.domain,
        "status": result.get("status", "completed"),
        "answer_type": result.get("answer_type", ""),
        "has_prediction": result.get("prediction") is not None,
        "trace_id": result.get("trace_id"),
        "trace_recovered": bool(result.get("trace_recovered", False)),
        "gate": "PASS" if result.get("gate_accepted") else "FAIL",
        "gate_accepted": result.get("gate_accepted"),
        "gate_rejections": list(result.get("gate_rejections", [])),
        "confidence": result.get("confidence"),
        "tools_run": len(substantive_tool_chain),
        "tool_chain": substantive_tool_chain,
        "winner_tool_chain": list(substantive_tool_chain),
        "total_ms": result.get("total_ms", 0),
        "error": result.get("error"),
        "error_type": result.get("error_type"),
        "failure_reason": result.get("failure_reason"),
        "reasoning_preview": _extract_train_result_reasoning_preview(result, learning_summary),
        "raw_answer": raw_answer,
        "raw_answer_preview": _sanitize_learning_artifact_preview(raw_answer, limit=500),
        "execution_context": result.get("execution_context"),
        "control_context": result.get("control_context"),
        "execution_log": result.get("execution_log", []),
        "learning_insight": learning_summary.get("insight"),
        "learning_recommendation": learning_summary.get("recommendation"),
        "evidence_summary_preview": _build_train_result_evidence_preview(
            result,
            substantive_tool_chain,
        ),
    }
    return _normalize_train_result_row(row)


# ── Ground truth extraction ──────────────────────────────────────────

def _extract_ground_truth(sample: MTBenchSample, task: str) -> Any:
    """Extract the ground truth value for a given sample and task type.

    Returns ``None`` when ground truth is unavailable (sample will be skipped).
    """
    if task == "forecast":
        return (
            sample.output_window.tolist()
            if sample.output_window is not None
            else None
        )

    if task == "indicator":
        if sample.domain == "weather" and sample.output_window is not None:
            from timeclaw.benchmark import mtbench_prompts as mt_prompts

            return mt_prompts.weather_indicator_ground_truth(sample.output_window)
        gt: dict[str, Any] = {}
        if sample.labels.get("out_macd") is not None:
            gt["macd"] = list(np.asarray(sample.labels["out_macd"], dtype=np.float64).ravel())
        if sample.labels.get("out_upper_bb") is not None:
            gt["upper_bb"] = list(np.asarray(sample.labels["out_upper_bb"], dtype=np.float64).ravel())
        return gt if gt else None

    if task == "indicator_macd":
        if sample.labels.get("out_macd") is None:
            return None
        return {"macd": list(np.asarray(sample.labels["out_macd"], dtype=np.float64).ravel())}

    if task == "indicator_upper_bb":
        if sample.labels.get("out_upper_bb") is None:
            return None
        return {"upper_bb": list(np.asarray(sample.labels["out_upper_bb"], dtype=np.float64).ravel())}

    if task == "trend":
        if sample.domain == "weather":
            return _weather_trend_3way_label(sample)
        label = sample.labels.get("trend_label", "")
        if label:
            return label
        if sample.output_window is not None:
            diff = float(sample.output_window[-1] - sample.output_window[0])
            pct = diff / (abs(float(sample.output_window[0])) + 1e-8) * 100
            if pct > 4:
                return "strong_up"
            elif pct > 1:
                return "up"
            elif pct < -4:
                return "strong_down"
            elif pct < -1:
                return "down"
            return "stable"
        return None

    if task == "trend_past":
        if sample.domain == "weather":
            return _weather_trend_past_3way_label(sample)
        return None

    if task == "mcqa":
        return sample.answer if sample.question and sample.answer else None

    if task == "correlation":
        return sample.labels.get("correlation") or None

    return None


# ── Main train loop ──────────────────────────────────────────────────

async def _process_one_sample(
    ds_name: str,
    sample_idx: int,
    sample: MTBenchSample,
    supported_tasks: list[str],
    pipeline: PipelineRunner,
    sem: asyncio.Semaphore,
    train_run_id: str,
) -> list[dict[str, Any]]:
    """Process a single sample across all its tasks (semaphore-gated)."""
    async with sem:
        results: list[dict[str, Any]] = []
        for task in supported_tasks:
            gt = _extract_ground_truth(sample, task)
            if gt is None:
                continue

            desc = build_task_description(sample, task)
            text_data = None
            if sample.text:
                text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

            horizon = (
                len(sample.output_window) if task == "forecast" and sample.output_window is not None
                else None
            )

            task_type_map = {
                "mcqa": "qa",
                "trend": "classification",
                "trend_past": "classification",
                "indicator_macd": "indicator",
                "indicator_upper_bb": "indicator",
            }
            pipeline_task = task_type_map.get(task, task)

            try:
                scope_keys = scope_keys_for_dataset_task(ds_name, task)
                source_metadata = {
                    "dataset_name": ds_name,
                    "sample_ref": f"{ds_name}:{sample.file}:{sample_idx}:{task}",
                    "source_file": sample.file,
                    "sample_idx": sample_idx,
                    "task_name": task,
                    "pipeline_task": pipeline_task,
                    "domain": sample.domain,
                    "train_run_id": train_run_id,
                }
                result = await asyncio.wait_for(
                    pipeline.run(
                        data=sample.input_window,
                        task_type=pipeline_task,
                        description=desc,
                        horizon=horizon,
                        text_data=text_data,
                        domain=sample.domain,
                        scope_keys=scope_keys,
                        mode="train",
                        ground_truth=gt,
                        source_metadata=source_metadata,
                    ),
                    timeout=1200,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: timeout")
                result = {
                    "answer_type": "",
                    "execution_log": [],
                    "prediction": None,
                    "status": "timeout",
                    "error_type": "TimeoutError",
                    "failure_reason": "timeout",
                }
            except Exception as e:
                logger.error(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: {e}")
                trace_id = getattr(e, "trace_id", None)
                recovered = _recover_failed_result_from_trace(trace_id)
                result = _build_failed_result_payload(e, recovered)

            results.append(_build_train_result_row(
                ds_name=ds_name,
                sample_idx=sample_idx,
                sample=sample,
                task=task,
                pipeline_task=pipeline_task,
                scope_keys=scope_keys,
                train_run_id=train_run_id,
                result=result,
            ))
        return results


async def _process_one_task_sample(
    ds_name: str,
    sample_idx: int,
    sample: MTBenchSample,
    task: str,
    pipeline: PipelineRunner,
    sem: asyncio.Semaphore,
    task_sem: asyncio.Semaphore | None = None,
    train_run_id: str = "",
) -> dict[str, Any] | None:
    """Process one (sample, task) pair with optional per-task gating."""
    gt = _extract_ground_truth(sample, task)
    if gt is None:
        return None

    desc = build_task_description(sample, task)
    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    horizon = (
        len(sample.output_window) if task == "forecast" and sample.output_window is not None
        else None
    )

    task_type_map = {
        "mcqa": "qa",
        "trend": "classification",
        "trend_past": "classification",
        "indicator_macd": "indicator",
        "indicator_upper_bb": "indicator",
    }
    pipeline_task = task_type_map.get(task, task)

    if task_sem is not None:
        await task_sem.acquire()
    scope_keys = scope_keys_for_dataset_task(ds_name, task)
    source_metadata = {
        "dataset_name": ds_name,
        "sample_ref": f"{ds_name}:{sample.file}:{sample_idx}:{task}",
        "source_file": sample.file,
        "sample_idx": sample_idx,
        "task_name": task,
        "pipeline_task": pipeline_task,
        "domain": sample.domain,
        "train_run_id": train_run_id,
    }
    try:
        async with sem:
            try:
                result = await asyncio.wait_for(
                    pipeline.run(
                        data=sample.input_window,
                        task_type=pipeline_task,
                        description=desc,
                        horizon=horizon,
                        text_data=text_data,
                        domain=sample.domain,
                        scope_keys=scope_keys,
                        mode="train",
                        ground_truth=gt,
                        source_metadata=source_metadata,
                    ),
                    timeout=1200,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: timeout")
                result = {
                    "answer_type": "",
                    "execution_log": [],
                    "prediction": None,
                    "status": "timeout",
                    "error_type": "TimeoutError",
                    "failure_reason": "timeout",
                }
            except Exception as e:
                logger.error(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: {e}")
                trace_id = getattr(e, "trace_id", None)
                recovered = _recover_failed_result_from_trace(trace_id)
                result = _build_failed_result_payload(e, recovered)

        return _build_train_result_row(
            ds_name=ds_name,
            sample_idx=sample_idx,
            sample=sample,
            task=task,
            pipeline_task=pipeline_task,
            scope_keys=scope_keys,
            train_run_id=train_run_id,
            result=result,
        )
    finally:
        if task_sem is not None:
            task_sem.release()


async def _run_task_work_item(
    ds_name: str,
    sample_idx: int,
    sample: MTBenchSample,
    task: str,
    pipeline: PipelineRunner,
    sem: asyncio.Semaphore,
    task_sem: asyncio.Semaphore | None = None,
    train_run_id: str = "",
) -> tuple[str, list[dict[str, Any]], Exception | None]:
    try:
        row = await _process_one_task_sample(
            ds_name=ds_name,
            sample_idx=sample_idx,
            sample=sample,
            task=task,
            pipeline=pipeline,
            sem=sem,
            task_sem=task_sem,
            train_run_id=train_run_id,
        )
    except Exception as e:  # pragma: no cover - catastrophic wrapper fallback
        return ds_name, [], e
    return ds_name, ([row] if row is not None else []), None


async def _run_sample_work_item(
    ds_name: str,
    sample_idx: int,
    sample: MTBenchSample,
    supported_tasks: list[str],
    pipeline: PipelineRunner,
    sem: asyncio.Semaphore,
    train_run_id: str,
) -> tuple[str, list[dict[str, Any]], Exception | None]:
    try:
        rows = await _process_one_sample(
            ds_name=ds_name,
            sample_idx=sample_idx,
            sample=sample,
            supported_tasks=supported_tasks,
            pipeline=pipeline,
            sem=sem,
            train_run_id=train_run_id,
        )
    except Exception as e:  # pragma: no cover - catastrophic wrapper fallback
        return ds_name, [], e
    return ds_name, rows, None


async def run_train_loop(
    dataset_name: str | None = None,
    task_filter: str | None = None,
    config: TimeClawConfig | None = None,
    max_samples: int | None = None,
    sample_indices: dict[str, list[int]] | str | Path | None = None,
    resume_from: int = 0,
    verbose: bool = False,
    concurrency: int = 8,
    per_task_concurrency: int | None = None,
    indicator_concurrency: int | None = None,
    merge_checkpoint: bool = True,
) -> dict[str, Any]:
    """Run explorative training loop over one or all MTBench datasets.

    Each sample is processed through ``pipeline.run(mode="train")``.
    The agent explores autonomously, evaluates against ground truth using
    the ``evaluate_against_gt`` tool, and saves learning outcomes to memory.

    ``merge_checkpoint`` (default True): if ``.timeclaw/train_results/train_<ds>.json``
    exists, load it and only run (sample_idx, task) pairs not already present,
    then merge and save. Set False for a clean re-run that ignores existing
    checkpoint contents and executes the requested dataset/task selection again.
    """
    cfg = config or TimeClawConfig.load()
    # In-process tool calls are faster under high concurrency; MCP stdio
    # transport is single-threaded per server and would serialize calls.
    cfg.mcp.enabled = False
    trace.configure(verbose)

    n_specs = len(expanded_benchmark_task_specs())
    if n_specs != BENCHMARK_EXPANDED_TASK_TARGET:
        logger.warning(
            f"[TrainLoop] Expanded benchmark task count is {n_specs}, "
            f"expected {BENCHMARK_EXPANDED_TASK_TARGET} (check DATASET_REGISTRY)."
        )
    else:
        logger.info(
            f"[TrainLoop] {BENCHMARK_EXPANDED_TASK_TARGET} scope tasks × "
            f"{len(datasets_with_benchmark_tasks())} datasets (see registry)"
        )

    train_run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:8]}"
    started_at = time.time()
    pipeline = PipelineRunner(cfg, disable_memory=False)
    sample_indices_map = (
        load_sample_indices(sample_indices)
        if isinstance(sample_indices, (str, Path))
        else sample_indices
    )

    datasets_to_run = (
        [dataset_name]
        if dataset_name
        else datasets_with_benchmark_tasks()
    )
    all_results: dict[str, Any] = {}
    _save_train_summary(all_results)
    sem = asyncio.Semaphore(concurrency)
    dataset_work_items: list[tuple[str, int, MTBenchSample, str]] = []
    dataset_samples: dict[str, list[MTBenchSample]] = {}
    dataset_tasks: dict[str, list[str]] = {}
    dataset_existing_rows: dict[str, list[dict[str, Any]]] = {}
    run_rows_by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for ds_name in datasets_to_run:
        info = DATASET_REGISTRY.get(ds_name)
        if not info:
            logger.warning(f"[TrainLoop] Unknown dataset: {ds_name}")
            continue

        existing_rows = _load_train_checkpoint(ds_name) if merge_checkpoint else []
        dataset_existing_rows[ds_name] = existing_rows
        done_keys = _train_done_keys(existing_rows)

        try:
            samples = load_split(ds_name, "train", config=cfg)
        except FileNotFoundError:
            logger.warning(f"[TrainLoop] No split for {ds_name}, skipping")
            continue

        selected_samples = select_samples_with_original_indices(samples, ds_name, sample_indices_map)
        if max_samples:
            selected_samples = selected_samples[:max_samples]
        if resume_from > 0:
            selected_samples = selected_samples[resume_from:]

        supported_tasks = expand_dataset_tasks(ds_name, list(info["tasks"]))
        if task_filter:
            supported_tasks = [t for t in supported_tasks if t == task_filter]
        if not supported_tasks:
            continue

        total = len(selected_samples)
        dataset_samples[ds_name] = [sample for _, sample in selected_samples]
        dataset_tasks[ds_name] = supported_tasks

        ptc_str = f", per_task_concurrency={per_task_concurrency}" if per_task_concurrency else ""
        ic_str = f", indicator_concurrency={indicator_concurrency}" if indicator_concurrency else ""
        pending_all = merge_checkpoint or per_task_concurrency is not None or indicator_concurrency is not None

        if pending_all:
            for abs_idx, sample in selected_samples:
                for task in supported_tasks:
                    if merge_checkpoint and (abs_idx, task) in done_keys:
                        continue
                    if _extract_ground_truth(sample, task) is None:
                        continue
                    dataset_work_items.append((ds_name, abs_idx, sample, task))
        else:
            for abs_idx, sample in selected_samples:
                dataset_work_items.append((ds_name, abs_idx, sample, "__all__"))

        n_pending = sum(1 for x in dataset_work_items if x[0] == ds_name)
        logger.info(
            f"[TrainLoop] {ds_name}: {total} samples, tasks={supported_tasks}, "
            f"pending_runs={n_pending}, checkpoint_had={len(done_keys)}, "
            f"concurrency={concurrency}{ptc_str}{ic_str}"
        )

    task_sems_global: dict[tuple[str, str], asyncio.Semaphore] = {}
    for ds_name, tasks in dataset_tasks.items():
        for task in tasks:
            limit = (
                indicator_concurrency
                if task.startswith("indicator") and indicator_concurrency is not None
                else (per_task_concurrency or concurrency)
            )
            task_sems_global[(ds_name, task)] = asyncio.Semaphore(limit)

    items_this_ds: dict[str, list[tuple]] = defaultdict(list)
    for item in dataset_work_items:
        items_this_ds[item[0]].append(item)

    merged_rows_by_ds: dict[str, list[dict[str, Any]]] = {
        ds_name: list(dataset_existing_rows.get(ds_name, []))
        for ds_name in dataset_samples
    }
    merged_rows_by_ds_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ds_name, tasks in dataset_tasks.items():
        for task in tasks:
            merged_rows_by_ds_task[(ds_name, task)] = list(
                _load_train_checkpoint(ds_name, task) if merge_checkpoint else []
            )
    work_count_by_ds = {
        ds_name: len(items_this_ds.get(ds_name, []))
        for ds_name in dataset_samples
    }
    completed_work_by_ds: dict[str, int] = defaultdict(int)

    if dataset_work_items:
        use_per_task = dataset_work_items[0][3] != "__all__"
        if use_per_task:
            pending_tasks = [
                asyncio.create_task(_run_task_work_item(
                    ds_name=a,
                    sample_idx=si,
                    sample=s,
                    task=t,
                    pipeline=pipeline,
                    sem=sem,
                    task_sem=task_sems_global.get((a, t))
                    if (per_task_concurrency is not None or indicator_concurrency is not None)
                    else None,
                    train_run_id=train_run_id,
                ))
                for (a, si, s, t) in dataset_work_items
            ]
        else:
            batch_work = [w for w in dataset_work_items if w[3] == "__all__"]
            pending_tasks = [
                asyncio.create_task(_run_sample_work_item(
                    a,
                    si,
                    s,
                    dataset_tasks[a],
                    pipeline,
                    sem,
                    train_run_id,
                ))
                for (a, si, s, _) in batch_work
            ]

        total_pending = len(pending_tasks)
        for finished_count, fut in enumerate(asyncio.as_completed(pending_tasks), start=1):
            ds_name, rows, err = await fut
            completed_work_by_ds[ds_name] += 1

            if err is not None:
                logger.error(f"[TrainLoop] {ds_name}: task error: {err}")
            elif rows:
                merged_rows_by_ds[ds_name] = _merge_train_rows(
                    merged_rows_by_ds.get(ds_name, []),
                    rows,
                )
                rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    task_name = str(row.get("task", "unknown"))
                    rows_by_task[task_name].append(row)
                for task_name, task_rows in rows_by_task.items():
                    key = (ds_name, task_name)
                    run_rows_by_task[key].extend(task_rows)
                    merged_rows_by_ds_task[key] = _merge_train_rows(
                        merged_rows_by_ds_task.get(key, []),
                        task_rows,
                    )
                    _save_task_checkpoint(merged_rows_by_ds_task[key], ds_name, task_name)

            if (
                finished_count % 10 == 0
                or finished_count == total_pending
                or completed_work_by_ds[ds_name] == work_count_by_ds.get(ds_name, 0)
            ):
                logger.info(
                    f"[TrainLoop] Progress {finished_count}/{total_pending}; "
                    f"{ds_name} finished {completed_work_by_ds[ds_name]}/"
                    f"{work_count_by_ds.get(ds_name, 0)} work items, "
                    f"checkpoint_rows={len(merged_rows_by_ds.get(ds_name, []))}"
                )

    for ds_name, samples in dataset_samples.items():
        existing_rows = dataset_existing_rows.get(ds_name, [])
        work = items_this_ds.get(ds_name, [])

        if not work:
            if not existing_rows:
                logger.warning(f"[TrainLoop] {ds_name}: no pending work and no checkpoint (missing GT for all tasks?)")
            else:
                summary = _summarize_results(existing_rows)
                all_results[ds_name] = {"total": len(existing_rows), "summary": summary}
                _save_train_summary(all_results)
                logger.info(f"[TrainLoop] {ds_name}: nothing pending, kept {len(existing_rows)} checkpoint rows")
            continue

        merged = merged_rows_by_ds.get(ds_name, existing_rows)
        summary = _summarize_results(merged)
        all_results[ds_name] = {"total": len(merged), "summary": summary}
        _save_train_summary(all_results)
        logger.info(f"[TrainLoop] {ds_name} done: {len(merged)} results")

    note_view_paths = _export_train_note_views(
        pipeline=pipeline,
        train_run_id=train_run_id,
        run_rows_by_task=run_rows_by_task,
    )
    manifest = _build_run_manifest(
        train_run_id=train_run_id,
        started_at=started_at,
        finished_at=time.time(),
        all_results=all_results,
        run_rows_by_task=run_rows_by_task,
        note_view_paths=note_view_paths,
    )
    _write_json_atomically(TRAIN_RUN_MANIFEST_PATH, manifest)
    _save_train_summary(all_results)
    await pipeline.shutdown()
    return all_results


def _save_task_checkpoint(results: list[dict[str, Any]], ds_name: str, task: str) -> None:
    TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _task_train_checkpoint_path(ds_name, task)
    normalized_rows = [
        _normalize_train_result_row({
            **row,
            "dataset_name": row.get("dataset_name") or ds_name,
            "dataset": row.get("dataset") or ds_name,
            "task": row.get("task") or task,
        })
        for row in results
        if isinstance(row, dict)
    ]
    _write_json_atomically(path, normalized_rows)


def _write_json_atomically(path: Path, payload: Any) -> None:
    """Write JSON to a sibling temp file, then replace the target atomically.

    New definition:
        ``atomic JSON write`` here means readers either observe the previous
        complete file or the new complete file, but not a partially written
        intermediate state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str))
    tmp_path.replace(path)


def _save_train_summary(summary: dict[str, Any]) -> None:
    _write_json_atomically(TRAIN_SUMMARY_PATH, summary)


def _render_note_block(note_text: str) -> str:
    stripped = str(note_text or "").strip()
    if not stripped:
        return ""
    return stripped if stripped.startswith("### [") else f"### {stripped}"


def _note_matches_run(note_text: str, *, train_run_id: str, sample_refs: set[str]) -> bool:
    text = str(note_text or "")
    if not text:
        return False
    if train_run_id and f"- Train Run ID: {train_run_id}" in text:
        return True
    return any(sample_ref and f"- Sample Ref: {sample_ref}" in text for sample_ref in sample_refs)


def _export_train_note_views(
    *,
    pipeline: PipelineRunner,
    train_run_id: str,
    run_rows_by_task: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, str]:
    exports: dict[str, str] = {}
    notes = getattr(getattr(pipeline, "memory", None), "notes", None)
    if notes is None:
        return exports

    for (ds_name, task), rows in sorted(run_rows_by_task.items()):
        sample_refs = {
            str(row.get("sample_ref") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("sample_ref") or "").strip()
        }
        if not sample_refs:
            continue
        blocks: list[str] = []
        seen: set[str] = set()
        for scope in scope_keys_for_dataset_task(ds_name, task):
            for item in notes.get_scope_note_items(scope):
                note_text = str(item.get("text") or "")
                note_id = str(item.get("id") or "")
                if note_id in seen or not _note_matches_run(
                    note_text,
                    train_run_id=train_run_id,
                    sample_refs=sample_refs,
                ):
                    continue
                block = _render_note_block(note_text)
                if not block:
                    continue
                seen.add(note_id)
                blocks.append(block)
        if not blocks:
            continue
        path = _task_note_view_path(ds_name, task)
        path.write_text("\n\n".join(blocks).rstrip() + "\n", encoding="utf-8")
        exports[f"{ds_name}::{task}"] = str(path)
    return exports


def _build_run_manifest(
    *,
    train_run_id: str,
    started_at: float,
    finished_at: float,
    all_results: dict[str, Any],
    run_rows_by_task: dict[tuple[str, str], list[dict[str, Any]]],
    note_view_paths: dict[str, str],
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for (ds_name, task), rows in sorted(run_rows_by_task.items()):
        status_counts: dict[str, int] = defaultdict(int)
        failed_sample_refs: list[str] = []
        for row in rows:
            status = str(row.get("status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status and status != "completed":
                failed_sample_refs.append(str(row.get("sample_ref") or ""))
        task_key = f"{ds_name}::{task}"
        tasks[task_key] = {
            "dataset": ds_name,
            "task": task,
            "rows": len(rows),
            "status_counts": dict(status_counts),
            "failed_sample_refs": failed_sample_refs,
            "note_view_path": note_view_paths.get(task_key, ""),
            "result_path": str(_task_train_checkpoint_path(ds_name, task)),
        }
    return {
        "train_run_id": train_run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": round(max(0.0, finished_at - started_at), 3),
        "dataset_summary": all_results,
        "tasks": tasks,
    }


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "learning_summaries": 0})

    for r in results:
        task = r.get("task", "unknown")
        by_task[task]["count"] += 1
        if r.get("answer_type") == "learning_summary":
            by_task[task]["learning_summaries"] += 1

    return dict(by_task)


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="TimeClaw Explorative Train Loop")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-index-file", type=str, default=None)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--per-task-concurrency", type=int, default=None)
    parser.add_argument("--indicator-concurrency", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-merge-checkpoint",
        action="store_true",
        help="Do not merge train_*.json; if checkpoint exists, skip whole dataset (legacy).",
    )
    args = parser.parse_args()

    result = await run_train_loop(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        sample_indices=args.sample_index_file,
        resume_from=args.resume_from,
        verbose=args.verbose,
        concurrency=args.concurrency,
        per_task_concurrency=args.per_task_concurrency,
        indicator_concurrency=args.indicator_concurrency,
        merge_checkpoint=not args.no_merge_checkpoint,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
