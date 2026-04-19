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
from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.benchmark.mtbench_support import (
    build_task_description,
    expand_dataset_tasks,
    load_sample_indices,
    scope_keys_for_dataset_task,
    select_samples_with_original_indices,
)
from timeclaw.core.config import TimeClawConfig
from timeclaw.core.confidence import coerce_confidence
from timeclaw.core.candidate_labels import canonical_candidate_label, canonicalize_candidate_mentions
from timeclaw.core import trace_logger as trace
from timeclaw.core.learning_text import sanitize_learning_evidence_text
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


def _pipeline_task_for_task_name(task: str) -> str:
    """Map dataset-facing train task ids to the runner's core task type."""
    return {
        "mcqa": "qa",
        "trend": "classification",
        "trend_past": "classification",
        "indicator_macd": "indicator",
        "indicator_upper_bb": "indicator",
    }.get(str(task or "").strip(), str(task or "").strip())


def _train_result_task_context(result: dict[str, Any]) -> tuple[str, list[str]]:
    """Resolve task-aware candidate extraction context from a train-result row."""
    task_name = str(result.get("task") or "").strip()
    pipeline_task = str(result.get("pipeline_task") or "").strip() or _pipeline_task_for_task_name(task_name)
    scope_keys = result.get("scope_keys") if isinstance(result.get("scope_keys"), list) else None
    domain = str(result.get("domain") or "").strip()
    indicator_output_keys = PipelineRunner._indicator_output_keys(
        scope_keys,
        domain,
        pipeline_task,
    )
    return pipeline_task, indicator_output_keys


def _match_correlation_label(prediction: str) -> str | None:
    """Canonicalize the official 5-way finance correlation label space."""
    lowered = str(prediction or "").strip().lower()
    if not lowered:
        return None
    explicit = {
        "strong positive correlation": "Strong Positive Correlation",
        "moderate positive correlation": "Moderate Positive Correlation",
        "no correlation": "No Correlation",
        "moderate negative correlation": "Moderate Negative Correlation",
        "strong negative correlation": "Strong Negative Correlation",
    }
    if lowered in explicit:
        return explicit[lowered]
    for short, full in (
        ("strong positive", "Strong Positive Correlation"),
        ("strong negative", "Strong Negative Correlation"),
        ("moderate positive", "Moderate Positive Correlation"),
        ("moderate negative", "Moderate Negative Correlation"),
        ("no correlation", "No Correlation"),
        ("weak", "No Correlation"),
        ("none", "No Correlation"),
    ):
        if short in lowered:
            return full
    return None


def _match_finance_trend_label(prediction: str) -> str | None:
    """Canonicalize the official 5-bucket finance trend label space."""
    lowered = str(prediction or "").strip().lower()
    if not lowered:
        return None
    explicit = {
        "<-4%": "<-4%",
        "-2% ~ -4%": "-2% ~ -4%",
        "-2% ~ +2%": "-2% ~ +2%",
        "+2% ~ +4%": "+2% ~ +4%",
        ">+4%": ">+4%",
    }
    if lowered in explicit:
        return explicit[lowered]
    compact = lowered.replace(" ", "")
    compact_map = {
        "<-4%": "<-4%",
        "-2%~-4%": "-2% ~ -4%",
        "-2%~+2%": "-2% ~ +2%",
        "+2%~+4%": "+2% ~ +4%",
        ">+4%": ">+4%",
    }
    return compact_map.get(compact)


def _canonicalize_train_task_prediction(
    prediction: Any,
    *,
    task_name: str,
    dataset_name: str = "",
    domain: str = "",
    pipeline_task: str = "",
    indicator_output_keys: list[str] | None = None,
) -> Any:
    """Canonicalize benchmark-facing predictions into the task's label space.

    New definition:
        Training-summary candidate comparison should only compare predictions
        that live in the benchmark task's real answer space. Intermediate tool
        outputs like ``up``/``down`` may still be useful evidence, but they are
        not valid final candidates for a bucketed finance-trend task.
    """
    task_name = str(task_name or "").strip()
    dataset_name = str(dataset_name or "").strip()
    domain = str(domain or "").strip()
    pipeline_task = str(pipeline_task or "").strip()

    if pipeline_task == "indicator" or task_name.startswith("indicator"):
        return PipelineRunner._canonicalize_task_prediction(
            prediction,
            task_type="indicator",
            indicator_output_keys=indicator_output_keys,
        )

    if task_name == "mcqa":
        if not isinstance(prediction, str):
            return None
        letter = prompts.extract_mcqa_letter(prediction)
        return letter or None

    if task_name == "correlation":
        if not isinstance(prediction, str):
            return None
        return _match_correlation_label(prediction)

    if task_name in {"trend", "trend_past"}:
        if not isinstance(prediction, str):
            return None
        if task_name == "trend_past" or domain == "weather" or dataset_name.startswith("weather_"):
            return prompts.parse_weather_trend_output(prediction)
        return _match_finance_trend_label(prediction)

    return prediction


def _normalize_train_candidate_traces(
    candidate_traces: list[dict[str, Any]],
    *,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep only candidates whose predictions belong to the task answer space."""
    task_name = str(result.get("task") or "").strip()
    dataset_name = str(result.get("dataset_name") or result.get("dataset") or "").strip()
    domain = str(result.get("domain") or "").strip()
    pipeline_task, indicator_output_keys = _train_result_task_context(result)
    normalized: list[dict[str, Any]] = []
    for trace_payload in candidate_traces:
        if not isinstance(trace_payload, dict):
            continue
        canonical_prediction = _canonicalize_train_task_prediction(
            trace_payload.get("prediction"),
            task_name=task_name,
            dataset_name=dataset_name,
            domain=domain,
            pipeline_task=pipeline_task,
            indicator_output_keys=indicator_output_keys,
        )
        if trace_payload.get("prediction") is not None and canonical_prediction is None:
            continue
        normalized_trace = dict(trace_payload)
        normalized_trace["prediction"] = canonical_prediction
        normalized.append(normalized_trace)
    return normalized


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
        analysis stable across historical and newly appended samples. Failed
        legacy learning-summary rows are also re-normalized from their stored
        ``execution_log`` so stale winner metadata and leaked learning text do
        not persist forever after a bug fix.
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

    answer_type = str(normalized.get("answer_type") or "").strip()
    status = str(normalized.get("status") or "completed").strip()
    raw_answer = normalized.get("raw_answer")
    execution_log = normalized.get("execution_log")
    skip_learning_summary_renormalize = bool(
        row.pop("_skip_learning_summary_renormalize", False)
    )

    if (
        answer_type == "learning_summary"
        and isinstance(raw_answer, dict)
        and isinstance(execution_log, list)
    ):
        if skip_learning_summary_renormalize:
            normalized_raw_answer = raw_answer
        else:
            try:
                normalized_raw_answer = PipelineRunner._normalize_learning_summary_answer(
                    raw_answer,
                    execution_log,
                    task_type=str(normalized.get("pipeline_task") or "").strip() or _pipeline_task_for_task_name(normalized.get("task", "")),
                    indicator_output_keys=PipelineRunner._indicator_output_keys(
                        normalized.get("scope_keys") if isinstance(normalized.get("scope_keys"), list) else None,
                        str(normalized.get("domain") or "").strip(),
                        str(normalized.get("pipeline_task") or "").strip() or _pipeline_task_for_task_name(normalized.get("task", "")),
                    ),
                )
            except Exception:
                normalized_raw_answer = raw_answer
        normalized["raw_answer"] = normalized_raw_answer
        normalized["raw_answer_preview"] = _sanitize_learning_artifact_preview(
            normalized_raw_answer,
            limit=500,
        )
        if status == "completed":
            normalized["learning_insight"] = normalized_raw_answer.get("insight")
            normalized["learning_recommendation"] = normalized_raw_answer.get("recommendation")
        else:
            normalized["learning_insight"] = None
            normalized["learning_recommendation"] = None

    if answer_type == "learning_summary" and isinstance(normalized.get("raw_answer"), dict):
        normalized_raw_answer = normalized["raw_answer"]
        if isinstance(normalized_raw_answer.get("execution_context"), dict):
            normalized["execution_context"] = dict(normalized_raw_answer.get("execution_context") or {})
        winner_state = str(normalized_raw_answer.get("winner_state") or "").strip().lower()
        normalized["winner_state"] = winner_state or None
        summary_kind = str(normalized_raw_answer.get("summary_kind") or "").strip()
        normalized["summary_kind"] = summary_kind or None
        normalized_raw_answer["insight"] = canonicalize_candidate_mentions(
            normalized_raw_answer.get("insight")
        ).strip()
        normalized_raw_answer["recommendation"] = canonicalize_candidate_mentions(
            normalized_raw_answer.get("recommendation")
        ).strip()
        normalized["winner_candidate"] = canonical_candidate_label(
            candidate_key=normalized_raw_answer.get("winner_candidate"),
            tool_chain=normalized_raw_answer.get("winner_tool_chain"),
        )
        normalized["winner_prediction"] = normalized_raw_answer.get("winner_prediction")
        winner_metrics = normalized_raw_answer.get("winner_metrics")
        normalized["winner_metrics"] = (
            dict(winner_metrics)
            if isinstance(winner_metrics, dict)
            else {}
        )
        compared_candidates: list[dict[str, Any]] = []
        raw_compared = normalized_raw_answer.get("compared_candidates")
        if isinstance(raw_compared, list):
            for item in raw_compared:
                if not isinstance(item, dict):
                    continue
                metrics = item.get("metrics")
                compared_candidates.append({
                    "name": canonical_candidate_label(
                        name=item.get("name"),
                        tool_chain=item.get("tool_chain"),
                    ),
                    "tool_chain": _filter_train_result_tool_chain(item.get("tool_chain")),
                    "prediction": item.get("prediction"),
                    "metrics": dict(metrics) if isinstance(metrics, dict) else {},
                })
        normalized["compared_candidates"] = compared_candidates
        # Backfill legacy top-level fields from the normalized learning summary so
        # downstream readers do not see empty `prediction` / `metrics` / `learning_summary`
        # while the winner-aligned fields are populated under the newer schema.
        normalized["learning_summary"] = dict(normalized_raw_answer)
        winner_prediction = normalized.get("winner_prediction")
        winner_metrics = normalized.get("winner_metrics")
        if winner_prediction is not None:
            normalized["prediction"] = winner_prediction
            normalized["metrics"] = (
                dict(winner_metrics)
                if isinstance(winner_metrics, dict) and winner_metrics
                else {}
            )
            normalized["has_prediction"] = True
        else:
            normalized["prediction"] = None
            normalized["metrics"] = None
            normalized["has_prediction"] = False

    substantive_tool_chain = _extract_train_result_tool_chain(normalized)
    if substantive_tool_chain:
        normalized["tool_chain"] = substantive_tool_chain
        normalized["winner_tool_chain"] = list(substantive_tool_chain)
    else:
        normalized["winner_tool_chain"] = _filter_train_result_tool_chain(
            normalized.get("winner_tool_chain")
        )

    if answer_type == "learning_summary" and isinstance(normalized.get("raw_answer"), dict):
        normalized_raw_answer = normalized["raw_answer"]
        if str(normalized_raw_answer.get("winner_state") or "").strip().lower() == "none":
            normalized["winner_state"] = "none"
            normalized["winner_candidate"] = ""
            normalized["winner_prediction"] = None
            normalized["winner_metrics"] = {}
            normalized["tool_chain"] = []
            normalized["winner_tool_chain"] = []
            normalized["has_prediction"] = False
            normalized["prediction"] = None
            normalized["metrics"] = None
            normalized["gate_accepted"] = False
            normalized["gate"] = "FAIL"
            rejections = [
                str(item).strip()
                for item in (normalized.get("gate_rejections") or [])
                if str(item).strip()
            ]
            execution_context = (
                normalized_raw_answer.get("execution_context")
                if isinstance(normalized_raw_answer.get("execution_context"), dict)
                else None
            )
            if execution_context is not None:
                execution_context["tool_chain"] = []
                execution_context["public_reasoning"] = []
                execution_context["evidence_summary"] = ""

    normalized["tools_run"] = _count_train_result_tools_run(normalized)
    substantive_tool_chain = list(normalized.get("tool_chain") or [])

    train_run_id = str(normalized.get("train_run_id") or "").strip()
    if train_run_id:
        normalized["train_run_id"] = train_run_id

    for key in (
        "learning_insight",
        "learning_recommendation",
        "reasoning_preview",
        "raw_answer_preview",
        "evidence_summary_preview",
    ):
        if key in normalized:
            value = normalized.get(key)
            if value is None:
                normalized[key] = None
            else:
                normalized[key] = canonicalize_candidate_mentions(value).strip()

    should_rebuild_trace = isinstance(execution_log, list)
    if should_rebuild_trace or not isinstance(normalized.get("trace"), dict):
        trace_payload = _build_train_result_trace_payload(
            normalized,
            substantive_tool_chain=substantive_tool_chain,
        )
        if trace_payload:
            normalized["trace"] = trace_payload

    if answer_type == "learning_summary" and str(normalized.get("winner_state") or "").strip().lower() == "none":
        candidate_traces = (
            normalized.get("trace", {}).get("candidate_traces")
            if isinstance(normalized.get("trace"), dict)
            else []
        )
        has_candidate_prediction = any(
            isinstance(trace_payload, dict) and trace_payload.get("prediction") is not None
            for trace_payload in (candidate_traces or [])
        )
        normalized["has_candidate_prediction"] = has_candidate_prediction
        rejection_reason = (
            "no validated winner"
            if has_candidate_prediction
            else "no candidate prediction"
        )
        rejections = [
            str(item).strip()
            for item in (normalized.get("gate_rejections") or [])
            if str(item).strip() and str(item).strip() != "prediction is None"
        ]
        if rejection_reason not in rejections:
            rejections.append(rejection_reason)
        normalized["gate_rejections"] = rejections

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
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        prefix_match = re.match(r"^(\d+\.\s+)(.*)$", line.strip())
        prefix = prefix_match.group(1) if prefix_match else ""
        body = prefix_match.group(2) if prefix_match else line
        cleaned_line = str(body).strip()
        if (
            cleaned_line
            and body.rstrip().endswith((".", "!", "?", ":"))
            and not cleaned_line.endswith((".", "!", "?", ":"))
        ):
            cleaned_line += body.rstrip()[-1]
        if cleaned_line:
            cleaned_lines.append(f"{prefix}{cleaned_line}".strip())
    cleaned = "\n".join(cleaned_lines).strip()
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
        task_type, indicator_output_keys = _train_result_task_context(result)
        candidate_traces = PipelineRunner._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
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


def _count_train_result_tools_run(result: dict[str, Any]) -> int:
    """Count substantive tools actually executed for one train-result row.

    New definition:
        ``train_results.tools_run`` counts substantive executed tools from the
        row's execution log. It is not the winner path length. Legacy rows that
        have no execution log fall back to their stored substantive tool chain.
    """
    execution_log = result.get("execution_log")
    if isinstance(execution_log, list) and execution_log:
        task_type, indicator_output_keys = _train_result_task_context(result)
        candidate_traces = PipelineRunner._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
        if candidate_traces:
            return sum(
                len(_filter_train_result_tool_chain(trace.get("tool_chain")))
                for trace in candidate_traces
                if isinstance(trace, dict)
            )
        return len(_filter_train_result_tool_chain([
            entry.get("tool")
            for entry in execution_log
            if isinstance(entry, dict)
        ]))
    return len(_filter_train_result_tool_chain(result.get("tool_chain")))


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
    execution_log = result.get("execution_log", [])
    task_type, indicator_output_keys = _train_result_task_context(result)
    candidate_traces = (
        PipelineRunner._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
        if isinstance(execution_log, list) and execution_log
        else []
    )
    evaluated_metrics = [
        trace.get("metrics")
        for trace in candidate_traces
        if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
    ]
    best_metrics = PipelineRunner._select_best_metrics(evaluated_metrics)
    if isinstance(raw_answer, dict):
        if str(raw_answer.get("winner_state") or "").strip().lower() == "none":
            return []
        if candidate_traces:
            winner_trace, _winner_source = PipelineRunner._resolve_learning_summary_winner_trace(
                raw_answer,
                candidate_traces,
                best_metrics,
            )
            resolved_chain = _filter_train_result_tool_chain((winner_trace or {}).get("tool_chain"))
            if resolved_chain:
                return resolved_chain
        execution_context = (
            raw_answer.get("execution_context")
            if isinstance(raw_answer.get("execution_context"), dict)
            else {}
        )
        winner_chain = _filter_train_result_tool_chain(
            raw_answer.get("winner_tool_chain") or execution_context.get("tool_chain")
        )
        if winner_chain and (
            not candidate_traces
            or any(
                _filter_train_result_tool_chain(trace.get("tool_chain")) == winner_chain
                for trace in candidate_traces
                if isinstance(trace, dict)
            )
        ):
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

    if candidate_traces:
        no_validated_winner = PipelineRunner._all_binary_scored_candidates_failed(candidate_traces)
        winner_chain = _filter_train_result_tool_chain(
            PipelineRunner._select_winner_tool_chain(candidate_traces, best_metrics)
        )
        if winner_chain:
            return winner_chain

        for trace in ([] if no_validated_winner else candidate_traces):
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

    task_type, indicator_output_keys = _train_result_task_context(result)
    candidate_traces = PipelineRunner._extract_candidate_traces(
        execution_log,
        task_type=task_type,
        indicator_output_keys=indicator_output_keys,
    )
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
    no_validated_winner = PipelineRunner._all_binary_scored_candidates_failed(candidate_traces)
    winner_prediction = (winner_trace or {}).get("prediction")
    if winner_prediction is None and isinstance(raw_answer, dict):
        winner_prediction = raw_answer.get("winner_prediction")
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

    if winner_trace is None and not no_validated_winner:
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

    def _capture_done_payload(parsed: Any) -> None:
        nonlocal last_done_payload
        if not isinstance(parsed, dict):
            return
        if parsed.get("done") is not True:
            return
        answer_type = str(parsed.get("answer_type") or "").strip()
        if not answer_type:
            return
        last_done_payload = parsed

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
                _capture_done_payload(parsed)
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
                    "confidence": coerce_confidence(last_done_payload.get("confidence")),
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


def _trace_path_for_id(trace_id: str | None) -> Path | None:
    trace_name = str(trace_id or "").strip()
    if not trace_name:
        return None
    trace_path = Path(".timeclaw/traces") / f"{trace_name}.jsonl"
    if not trace_path.exists():
        return None
    return trace_path


def _compact_train_result_trace_step(step: Any) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    tool = str(step.get("tool") or "").strip()
    raw_observation = step.get("observation")
    parsed_observation = PipelineRunner._parse_json_like_payload(raw_observation)
    preview_source = parsed_observation if parsed_observation not in (None, "") else _trace_preview_text(raw_observation)
    observation = _preview_value(preview_source, limit=240).strip()
    if not tool and not observation:
        return None
    compact: dict[str, Any] = {}
    if tool:
        compact["tool"] = tool
    if observation:
        compact["observation"] = observation
    return compact


def _compact_train_result_candidate_trace(trace: Any) -> dict[str, Any] | None:
    if not isinstance(trace, dict):
        return None

    compact: dict[str, Any] = {}
    name = canonical_candidate_label(
        candidate_key=trace.get("candidate_key"),
        name=trace.get("name"),
        tool_chain=trace.get("tool_chain"),
    )
    if name:
        compact["name"] = name
    candidate_key = str(trace.get("candidate_key") or "").strip()
    if candidate_key:
        compact["candidate_key"] = candidate_key

    tool_chain = _filter_train_result_tool_chain(trace.get("tool_chain"))
    if tool_chain:
        compact["tool_chain"] = tool_chain

    if trace.get("prediction") is not None:
        compact["prediction"] = trace.get("prediction")

    metrics = trace.get("metrics")
    if isinstance(metrics, dict) and metrics:
        compact["metrics"] = metrics

    snippets_source = trace.get("public_reasoning_snippets")
    snippets = snippets_source if isinstance(snippets_source, list) else PipelineRunner._public_reasoning_snippets(trace)
    cleaned_snippets = [
        cleaned
        for item in snippets
        for cleaned in [str(item).strip()]
        if cleaned
    ]
    if cleaned_snippets:
        compact["public_reasoning"] = cleaned_snippets

    compact_steps = [
        compact_step
        for step in (trace.get("steps") if isinstance(trace.get("steps"), list) else [])
        for compact_step in [_compact_train_result_trace_step(step)]
        if compact_step
    ]
    if compact_steps:
        compact["steps"] = compact_steps

    return compact or None


def _merge_train_result_summary_candidates(
    result: dict[str, Any],
    candidate_traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align trace-facing candidates with the repaired row-level summary."""
    normalized_candidates = _normalize_train_candidate_traces(
        candidate_traces,
        result=result,
    )
    deduped_execution = PipelineRunner._dedupe_candidate_traces_by_path(normalized_candidates)
    raw_answer = result.get("raw_answer")
    raw_compared = (
        raw_answer.get("compared_candidates")
        if isinstance(raw_answer, dict)
        else None
    )
    if not isinstance(raw_compared, list) or not raw_compared:
        return deduped_execution

    merged: list[dict[str, Any]] = []
    for item in raw_compared:
        if not isinstance(item, dict):
            continue
        tool_chain = _filter_train_result_tool_chain(item.get("tool_chain"))
        prediction = _canonicalize_train_task_prediction(
            item.get("prediction"),
            task_name=str(result.get("task") or "").strip(),
            dataset_name=str(result.get("dataset_name") or result.get("dataset") or "").strip(),
            domain=str(result.get("domain") or "").strip(),
            pipeline_task=str(result.get("pipeline_task") or "").strip(),
            indicator_output_keys=_train_result_task_context(result)[1],
        )
        if item.get("prediction") is not None and prediction is None:
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}

        matched_trace: dict[str, Any] | None = None
        for trace in deduped_execution:
            if not isinstance(trace, dict):
                continue
            trace_chain = _filter_train_result_tool_chain(trace.get("tool_chain"))
            if trace_chain != tool_chain:
                continue
            if prediction is not None and trace.get("prediction") == prediction:
                matched_trace = trace
                break
            if matched_trace is None:
                matched_trace = trace

        merged_trace = dict(matched_trace or {})
        merged_trace["name"] = canonical_candidate_label(
            name=item.get("name"),
            candidate_key=item.get("name"),
            tool_chain=tool_chain,
        )
        merged_trace["candidate_key"] = str(
            merged_trace.get("candidate_key")
            or item.get("name")
            or canonical_candidate_label(tool_chain=tool_chain)
            or ""
        ).strip()
        merged_trace["tool_chain"] = tool_chain
        merged_trace["prediction"] = prediction
        merged_trace["metrics"] = dict(metrics)
        merged.append(merged_trace)
    return merged


def _recover_train_result_trace_payload_from_file(trace_id: str | None) -> dict[str, Any]:
    trace_path = _trace_path_for_id(trace_id)
    if trace_path is None:
        return {}

    payload: dict[str, Any] = {
        "trace_id": trace_path.stem,
        "trace_path": str(trace_path),
    }
    last_done_payload: dict[str, Any] | None = None
    run_end_payload: dict[str, Any] | None = None

    try:
        for raw_line in trace_path.read_text().splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            event = str(record.get("event") or "").strip()
            body = record.get("payload") or {}
            if not isinstance(body, dict):
                continue

            if event == "content_block":
                if str(body.get("kind") or "").strip() != "prompt_message/assistant":
                    continue
                content = str(body.get("content") or "").strip()
                if not content.startswith("{"):
                    continue
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get("answer_type"):
                    last_done_payload = parsed
                continue

            if event == "run_end":
                run_end_payload = {
                    "status": str(body.get("status") or "").strip(),
                    "error": str(body.get("error") or "").strip(),
                    "error_type": str(body.get("error_type") or "").strip(),
                    "failure_reason": str(body.get("failure_reason") or "").strip(),
                }
    except (OSError, json.JSONDecodeError):
        return payload

    if last_done_payload:
        compact_done: dict[str, Any] = {
            "answer_type": str(last_done_payload.get("answer_type") or "").strip(),
        }
        answer = last_done_payload.get("answer")
        if answer is not None:
            compact_done["answer"] = answer
        execution_context = last_done_payload.get("execution_context")
        if isinstance(execution_context, dict) and execution_context:
            compact_done["execution_context"] = execution_context
        control_context = last_done_payload.get("control_context")
        if isinstance(control_context, dict) and control_context:
            compact_done["control_context"] = control_context
        payload["last_done_payload"] = compact_done
    if run_end_payload:
        payload["run_end"] = run_end_payload
    return payload


def _build_train_result_trace_payload(
    result: dict[str, Any],
    *,
    substantive_tool_chain: list[str] | None = None,
) -> dict[str, Any]:
    """Build direct debug evidence for one train_results row.

    New definition:
        ``train_results.trace`` is a compact, row-local trace artifact. It
        keeps the stable locator (`trace_id`, `trace_path`) plus the resolved
        candidate paths and winner path, so debugging does not require opening
        the raw `.jsonl` trace file first.
    """
    trace_id = str(result.get("trace_id") or "").strip()
    trace_path = _trace_path_for_id(trace_id)
    payload: dict[str, Any] = {}
    if trace_id:
        payload["trace_id"] = trace_id
    if trace_path is not None:
        payload["trace_path"] = str(trace_path)

    execution_log = result.get("execution_log")
    if not isinstance(execution_log, list) or not execution_log:
        recovered_payload = _recover_train_result_trace_payload_from_file(trace_id)
        if recovered_payload:
            payload.update(recovered_payload)
        return payload

    task_type, indicator_output_keys = _train_result_task_context(result)
    candidate_traces = PipelineRunner._extract_candidate_traces(
        execution_log,
        task_type=task_type,
        indicator_output_keys=indicator_output_keys,
    )
    candidate_traces = _normalize_train_candidate_traces(
        candidate_traces,
        result=result,
    )
    PipelineRunner._propagate_metrics_across_equivalent_predictions(candidate_traces)
    summary_candidate_traces = _merge_train_result_summary_candidates(
        result,
        candidate_traces,
    )
    compact_candidates = [
        compact_trace
        for trace in summary_candidate_traces
        for compact_trace in [_compact_train_result_candidate_trace(trace)]
        if compact_trace
    ]
    if compact_candidates:
        payload["candidate_traces"] = compact_candidates

    raw_answer = result.get("raw_answer")
    evaluated_metrics = [
        trace.get("metrics")
        for trace in summary_candidate_traces
        if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
    ]
    best_metrics = PipelineRunner._select_best_metrics(evaluated_metrics)
    winner_trace, winner_source = PipelineRunner._resolve_learning_summary_winner_trace(
        raw_answer if isinstance(raw_answer, dict) else {},
        summary_candidate_traces,
        best_metrics,
    )
    winner_prediction = raw_answer.get("winner_prediction") if isinstance(raw_answer, dict) else None
    resolved_tool_chain = list(substantive_tool_chain or _extract_train_result_tool_chain(result))
    no_validated_winner = PipelineRunner._all_binary_scored_candidates_failed(summary_candidate_traces)
    winner_chain = _filter_train_result_tool_chain((winner_trace or {}).get("tool_chain"))
    if resolved_tool_chain and (winner_trace is None or winner_chain != resolved_tool_chain):
        exact_match = _select_train_result_preview_trace_by_chain(
            summary_candidate_traces,
            resolved_tool_chain,
            winner_prediction=winner_prediction,
        )
        if exact_match is not None:
            winner_trace = exact_match
            winner_source = "tool_chain"
    if winner_trace is None and summary_candidate_traces and not no_validated_winner:
        winner_trace = summary_candidate_traces[0]
        winner_source = "first_candidate"

    compact_winner = _compact_train_result_candidate_trace(winner_trace)
    if compact_winner:
        payload["winner_trace"] = compact_winner
    if winner_source:
        payload["winner_trace_source"] = winner_source

    recovered_payload = _recover_train_result_trace_payload_from_file(trace_id)
    run_end = recovered_payload.get("run_end")
    if isinstance(run_end, dict) and run_end:
        payload["run_end"] = run_end

    return payload


def _build_failed_result_payload(
    exc: BaseException,
    recovered: dict[str, Any],
    *,
    task_type: str = "",
    indicator_output_keys: list[str] | None = None,
) -> dict[str, Any]:
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
    answer_type = str(
        _pick(
            partial.get("answer_type"),
            getattr(exc, "answer_type", None),
            recovered.get("answer_type"),
            "",
        ) or ""
    )

    if answer_type == "learning_summary" and isinstance(raw_answer, dict):
        recovered_raw_answer = recovered.get("raw_answer")
        if isinstance(recovered_raw_answer, dict):
            merged_raw_answer = dict(recovered_raw_answer)
            merged_raw_answer.update(raw_answer)
            raw_answer = merged_raw_answer
        try:
            raw_answer = PipelineRunner._normalize_learning_summary_answer(
                raw_answer,
                execution_log if isinstance(execution_log, list) else [],
                task_type=task_type,
                indicator_output_keys=indicator_output_keys,
            )
        except Exception:
            pass
        if (
            isinstance(raw_answer, dict)
            and not str(raw_answer.get("summary_kind") or "").strip()
            and not str(raw_answer.get("winner_candidate") or "").strip()
            and not _filter_train_result_tool_chain(raw_answer.get("winner_tool_chain"))
            and raw_answer.get("winner_prediction") is None
        ):
            raw_answer = dict(raw_answer)
            raw_answer["summary_kind"] = "failure_summary"
            raw_answer["winner_state"] = "none"
            raw_answer["winner_candidate"] = ""
            raw_answer["winner_prediction"] = None
            raw_answer["winner_tool_chain"] = []
            raw_answer["winner_metrics"] = {}

    return {
        "answer_type": answer_type,
        "execution_log": execution_log if isinstance(execution_log, list) else [],
        "prediction": _pick(partial.get("prediction"), raw_answer),
        "status": str(getattr(exc, "failure_reason", "") or "error"),
        "error": str(exc),
        "error_type": str(getattr(exc, "error_type", type(exc).__name__)),
        "failure_reason": str(getattr(exc, "failure_reason", "") or "error"),
        "trace_id": getattr(exc, "trace_id", None),
        "total_ms": getattr(exc, "total_ms", 0),
        "raw_answer": raw_answer,
        "confidence": coerce_confidence(
            _pick(partial.get("confidence"), getattr(exc, "confidence", None), recovered.get("confidence"))
        ),
        "reasoning": str(_pick(partial.get("reasoning"), getattr(exc, "reasoning", None), recovered.get("reasoning"), "") or ""),
        "execution_context": execution_context if isinstance(execution_context, dict) else {},
        "control_context": control_context if isinstance(control_context, dict) else {},
        "trace_recovered": bool(recovered.get("trace_recovered", False)),
    }


async def _run_pipeline_train_call_with_timeout(
    awaitable: Any,
    *,
    timeout_s: float,
) -> tuple[bool, Any]:
    """Run one pipeline call and preserve trace metadata when the timeout fires.

    New definition:
        A train-loop timeout should still keep the cancelled pipeline run's
        `trace_id` and any recoverable trace evidence. `asyncio.wait_for()`
        drops that context by default, so we wrap the call in an explicit task,
        cancel it ourselves, and then await the cancellation result once.
    """
    async def _capture_outcome() -> tuple[bool, Any]:
        try:
            return True, await awaitable
        except BaseException as e:  # noqa: BLE001 - the caller needs the original exception object
            return False, e

    task = asyncio.create_task(_capture_outcome())
    done, _pending = await asyncio.wait({task}, timeout=timeout_s)
    if task in done:
        completed, payload = await task
        if completed:
            return True, payload
        raise payload

    if not done:
        task.cancel()
        cancelled_exc: BaseException | None = None
        try:
            completed, payload = await task
            if not completed and isinstance(payload, BaseException):
                cancelled_exc = payload
        except BaseException as e:  # noqa: BLE001 - we need trace metadata from cancellation
            cancelled_exc = e

        timeout_exc = asyncio.TimeoutError("pipeline run timed out")
        setattr(timeout_exc, "trace_id", getattr(cancelled_exc, "trace_id", None))
        setattr(timeout_exc, "total_ms", getattr(cancelled_exc, "total_ms", 0))
        setattr(timeout_exc, "error_type", "TimeoutError")
        setattr(timeout_exc, "failure_reason", "timeout")
        partial_result = getattr(cancelled_exc, "partial_result", None)
        if isinstance(partial_result, dict):
            setattr(timeout_exc, "partial_result", partial_result)
            setattr(timeout_exc, "execution_log", partial_result.get("execution_log", []))
            setattr(timeout_exc, "raw_answer", partial_result.get("answer"))
            setattr(timeout_exc, "answer_type", partial_result.get("answer_type", ""))
            setattr(timeout_exc, "confidence", partial_result.get("confidence"))
            setattr(timeout_exc, "reasoning", partial_result.get("reasoning", ""))
            setattr(timeout_exc, "execution_context", partial_result.get("execution_context", {}))
            setattr(timeout_exc, "control_context", partial_result.get("control_context", {}))
        return False, timeout_exc


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
    gt = _extract_ground_truth(sample, task)
    if isinstance(raw_answer, dict):
        raw_answer = _repair_learning_summary_with_ground_truth(
            raw_answer=raw_answer,
            execution_log=result.get("execution_log", []),
            ground_truth=gt,
            task_type=pipeline_task,
            indicator_output_keys=_indicator_output_keys_for_task(task),
        )
    learning_summary = (
        raw_answer
        if str(result.get("status") or "completed") == "completed" and isinstance(raw_answer, dict)
        else {}
    )
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
        "tools_run": _count_train_result_tools_run(result),
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
        "trace": _build_train_result_trace_payload(
            result,
            substantive_tool_chain=substantive_tool_chain,
        ),
        "_skip_learning_summary_renormalize": (
            str(result.get("answer_type") or "") == "learning_summary"
            and isinstance(raw_answer, dict)
        ),
    }
    return _normalize_train_result_row(row)


def _indicator_output_keys_for_task(task: str) -> list[str]:
    if task == "indicator_macd":
        return ["macd"]
    if task == "indicator_upper_bb":
        return ["upper_bb"]
    return []


def _repair_learning_summary_with_ground_truth(
    *,
    raw_answer: dict[str, Any],
    execution_log: Any,
    ground_truth: Any,
    task_type: str,
    indicator_output_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Delegate GT-backed learning-summary repair to the shared runner logic.

    New definition:
        Train-result rows, in-run memory writes, and any later checkpoint
        normalization must all use the same GT-repair implementation. This
        wrapper keeps the train loop on the shared `PipelineRunner` path so the
        winner is not resolved differently in different persistence layers.
    """
    if not isinstance(raw_answer, dict) or not isinstance(execution_log, list):
        return raw_answer
    return PipelineRunner._repair_learning_summary_with_ground_truth(
        raw_answer,
        execution_log,
        ground_truth=ground_truth,
        task_type=task_type,
        indicator_output_keys=indicator_output_keys,
    )


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

            pipeline_task = _pipeline_task_for_task_name(task)

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
                completed, payload = await _run_pipeline_train_call_with_timeout(
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
                    timeout_s=1200,
                )
                if completed:
                    result = payload
                else:
                    logger.warning(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: timeout")
                    trace_id = getattr(payload, "trace_id", None)
                    recovered = _recover_failed_result_from_trace(trace_id)
                    result = _build_failed_result_payload(
                        payload,
                        recovered,
                        task_type=pipeline_task,
                        indicator_output_keys=PipelineRunner._indicator_output_keys(
                            scope_keys,
                            sample.domain,
                            pipeline_task,
                        ),
                    )
            except Exception as e:
                logger.error(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: {e}")
                trace_id = getattr(e, "trace_id", None)
                recovered = _recover_failed_result_from_trace(trace_id)
                result = _build_failed_result_payload(
                    e,
                    recovered,
                    task_type=pipeline_task,
                    indicator_output_keys=PipelineRunner._indicator_output_keys(
                        scope_keys,
                        sample.domain,
                        pipeline_task,
                    ),
                )

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

    pipeline_task = _pipeline_task_for_task_name(task)

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
                completed, payload = await _run_pipeline_train_call_with_timeout(
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
                    timeout_s=1200,
                )
                if completed:
                    result = payload
                else:
                    logger.warning(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: timeout")
                    trace_id = getattr(payload, "trace_id", None)
                    recovered = _recover_failed_result_from_trace(trace_id)
                    result = _build_failed_result_payload(
                        payload,
                        recovered,
                        task_type=pipeline_task,
                        indicator_output_keys=PipelineRunner._indicator_output_keys(
                            scope_keys,
                            sample.domain,
                            pipeline_task,
                        ),
                    )
            except Exception as e:
                logger.error(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: {e}")
                trace_id = getattr(e, "trace_id", None)
                recovered = _recover_failed_result_from_trace(trace_id)
                result = _build_failed_result_payload(
                    e,
                    recovered,
                    task_type=pipeline_task,
                    indicator_output_keys=PipelineRunner._indicator_output_keys(
                        scope_keys,
                        sample.domain,
                        pipeline_task,
                    ),
                )

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
        max_samples=max_samples,
        concurrency=concurrency,
        per_task_concurrency=per_task_concurrency,
        indicator_concurrency=indicator_concurrency,
        merge_checkpoint=merge_checkpoint,
        current_run_results=all_results,
        current_run_task_rows_by_key=merged_rows_by_ds_task,
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


def _load_active_task_rows_by_key() -> dict[tuple[str, str], list[dict[str, Any]]]:
    active_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(TRAIN_RESULTS_DIR.glob("train_*.json")):
        if path.name in {"train_summary.json", "train_run_manifest.json"}:
            continue
        rows = _load_checkpoint_rows(path)
        if not rows:
            continue
        first = rows[0]
        ds_name = str(first.get("dataset_name") or first.get("dataset") or "").strip()
        task = str(first.get("task") or "").strip()
        if not ds_name or not task:
            continue
        active_rows[(ds_name, task)] = rows
    return active_rows


def _build_active_train_summary() -> dict[str, Any]:
    dataset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (ds_name, _task), rows in _load_active_task_rows_by_key().items():
        dataset_rows[ds_name].extend(rows)
    summary: dict[str, Any] = {}
    for ds_name, rows in sorted(dataset_rows.items()):
        summary[ds_name] = {
            "total": len(rows),
            "summary": _summarize_results(rows),
        }
    return summary


def _save_train_summary(summary: dict[str, Any] | None = None) -> None:
    active_summary = _build_active_train_summary()
    _write_json_atomically(TRAIN_SUMMARY_PATH, active_summary or dict(summary or {}))


def _render_note_block(note_text: str) -> str:
    stripped = canonicalize_candidate_mentions(note_text).strip()
    if not stripped:
        return ""
    return stripped if stripped.startswith("### [") else f"### {stripped}"


def _note_train_run_id(note_text: str) -> str:
    text = str(note_text or "")
    if not text:
        return ""
    match = re.search(r"^\s*-\s*Train Run ID:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return str(match.group(1) if match else "").strip()


def _note_matches_run(note_text: str, *, train_run_id: str, sample_refs: set[str]) -> bool:
    text = str(note_text or "")
    if not text:
        return False
    note_run_id = _note_train_run_id(text)
    if train_run_id:
        return note_run_id == train_run_id
    return any(sample_ref and f"- Sample Ref: {sample_ref}" in text for sample_ref in sample_refs)


def _split_rendered_note_blocks(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    matches = list(re.finditer(r"(?m)^###\s", raw))
    if not matches:
        block = _render_note_block(raw)
        return [block] if block else []
    blocks: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        block = _render_note_block(raw[start:end])
        if block:
            blocks.append(block)
    return blocks


def _note_sample_ref(note_text: str) -> str:
    match = re.search(r"^\s*-\s*Sample Ref:\s*(.+?)\s*$", str(note_text or ""), flags=re.MULTILINE)
    return str(match.group(1) if match else "").strip()


def _note_sample_index(note_text: str) -> int:
    match = re.search(r"^\s*-\s*Sample Index:\s*(\d+)\s*$", str(note_text or ""), flags=re.MULTILINE)
    try:
        return int(match.group(1)) if match else 10**9
    except (TypeError, ValueError):
        return 10**9


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
        touched_sample_refs = {
            str(row.get("sample_ref") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("sample_ref") or "").strip()
        }
        if not touched_sample_refs:
            continue
        current_blocks_by_ref: dict[str, str] = {}
        seen: set[str] = set()
        for scope in scope_keys_for_dataset_task(ds_name, task):
            for item in notes.get_scope_note_items(scope):
                note_text = str(item.get("text") or "")
                note_id = str(item.get("id") or "")
                if note_id in seen or not _note_matches_run(
                    note_text,
                    train_run_id=train_run_id,
                    sample_refs=touched_sample_refs,
                ):
                    continue
                block = _render_note_block(note_text)
                if not block:
                    continue
                seen.add(note_id)
                sample_ref = _note_sample_ref(block)
                if sample_ref:
                    current_blocks_by_ref[sample_ref] = block

        path = _task_note_view_path(ds_name, task)
        existing_blocks = _split_rendered_note_blocks(path.read_text(encoding="utf-8")) if path.exists() else []
        merged_blocks: list[str] = []
        replaced_refs: set[str] = set()
        for block in existing_blocks:
            sample_ref = _note_sample_ref(block)
            if sample_ref and sample_ref in touched_sample_refs:
                replaced_refs.add(sample_ref)
                replacement = current_blocks_by_ref.get(sample_ref)
                if replacement:
                    merged_blocks.append(replacement)
                continue
            merged_blocks.append(block)
        pending_new = [
            block
            for sample_ref, block in current_blocks_by_ref.items()
            if sample_ref and sample_ref not in replaced_refs
        ]
        pending_new.sort(key=lambda block: (_note_sample_index(block), _note_sample_ref(block)))
        merged_blocks.extend(pending_new)
        if not merged_blocks:
            if path.exists():
                path.unlink()
            continue
        path.write_text("\n\n".join(merged_blocks).rstrip() + "\n", encoding="utf-8")
        exports[f"{ds_name}::{task}"] = str(path)
    return exports


def _build_run_manifest(
    *,
    train_run_id: str,
    started_at: float,
    finished_at: float,
    max_samples: int | None = None,
    concurrency: int | None = None,
    per_task_concurrency: int | None = None,
    indicator_concurrency: int | None = None,
    merge_checkpoint: bool = False,
    current_run_results: dict[str, Any],
    current_run_task_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    note_view_paths: dict[str, str],
) -> dict[str, Any]:
    active_summary = _build_active_train_summary()
    active_task_rows_by_key = _load_active_task_rows_by_key()
    tasks: dict[str, Any] = {}
    for (ds_name, task), rows in sorted(active_task_rows_by_key.items()):
        status_counts: dict[str, int] = defaultdict(int)
        failed_sample_refs: list[str] = []
        for row in rows:
            status = str(row.get("status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if _row_counts_as_failed_sample_ref(row):
                failed_sample_refs.append(str(row.get("sample_ref") or ""))
        task_key = f"{ds_name}::{task}"
        tasks[task_key] = {
            "dataset": ds_name,
            "task": task,
            "rows": len(rows),
            "status_counts": dict(status_counts),
            "aligned_counts": _aligned_train_counts(rows),
            "failed_sample_refs": failed_sample_refs,
            "note_view_path": note_view_paths.get(task_key, str(_task_note_view_path(ds_name, task)) if _task_note_view_path(ds_name, task).exists() else ""),
            "result_path": str(_task_train_checkpoint_path(ds_name, task)),
        }
    return {
        "train_run_id": train_run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": round(max(0.0, finished_at - started_at), 3),
        "max_samples": max_samples,
        "concurrency": concurrency,
        "per_task_concurrency": per_task_concurrency,
        "indicator_concurrency": indicator_concurrency,
        "merge_checkpoint": merge_checkpoint,
        "dataset_summary": active_summary,
        "current_run_dataset_summary": current_run_results,
        "current_run_task_keys": [
            f"{ds_name}::{task}"
            for (ds_name, task) in sorted(current_run_task_rows_by_key.keys())
        ],
        "partial_run": bool(
            active_summary != current_run_results
            or set(active_task_rows_by_key.keys()) != set(current_run_task_rows_by_key.keys())
        ),
        "tasks": tasks,
    }


def _row_counts_as_failed_sample_ref(row: dict[str, Any]) -> bool:
    """Classify rows that should surface in manifest-level failure references.

    New definition:
        `failed_sample_refs` must include both execution failures
        (`status != completed`) and semantic failures that finished running but
        still produced an unusable training outcome, such as `gate=FAIL` or
        `winner_state=none`.
    """
    status = str(row.get("status") or "").strip()
    if status and status != "completed":
        return True
    if row.get("gate_accepted") is False:
        return True
    answer_type = str(row.get("answer_type") or "").strip()
    winner_state = str(row.get("winner_state") or "").strip().lower()
    if answer_type == "learning_summary" and winner_state == "none":
        return True
    return False


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "learning_summaries": 0,
            "aligned_counts": _aligned_train_counts([]),
            "__rows__": [],
        }
    )

    for r in results:
        task = r.get("task", "unknown")
        by_task[task]["count"] += 1
        if r.get("status") == "completed" and r.get("answer_type") == "learning_summary":
            by_task[task]["learning_summaries"] += 1
        by_task[task]["__rows__"].append(r)

    for task_summary in by_task.values():
        task_summary["aligned_counts"] = _aligned_train_counts(task_summary.get("__rows__", []))
        task_summary.pop("__rows__", None)

    return dict(by_task)


def _aligned_train_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Build an explicit training stats glossary for one task's rows.

    New definition:
        ``aligned_counts`` separates execution health from learning quality.
        These counts must not be inferred from one another.

    Examples:
        - ``execution_completed_n`` answers "how many rows ran to completion?"
        - ``gate_pass_n`` answers "how many completed rows passed the quality gate?"
        - ``validated_winner_n`` answers "how many learning summaries selected a
          grounded winner path?"
        - ``continuous_scored_n`` answers "how many winner rows have graded
          metrics only (forecast/indicator style), so they are neither binary OK
          nor binary FAIL?"
    """
    counts = {
        "execution_total_n": 0,
        "execution_completed_n": 0,
        "execution_failed_n": 0,
        "prediction_emitted_n": 0,
        "gate_pass_n": 0,
        "gate_fail_n": 0,
        "learning_summary_n": 0,
        "validated_winner_n": 0,
        "no_winner_n": 0,
        "binary_correct_n": 0,
        "binary_incorrect_n": 0,
        "continuous_scored_n": 0,
    }

    for row in rows:
        if not isinstance(row, dict):
            continue
        counts["execution_total_n"] += 1

        status = str(row.get("status") or "").strip()
        answer_type = str(row.get("answer_type") or "").strip()
        winner_state = str(row.get("winner_state") or "").strip().lower()
        winner_metrics = row.get("winner_metrics")
        if not isinstance(winner_metrics, dict):
            raw_answer = row.get("raw_answer")
            if isinstance(raw_answer, dict) and isinstance(raw_answer.get("winner_metrics"), dict):
                winner_metrics = dict(raw_answer.get("winner_metrics") or {})
            else:
                winner_metrics = {}
        correctness = PipelineRunner._learning_outcome_correctness(winner_metrics)

        if status == "completed":
            counts["execution_completed_n"] += 1
        elif status:
            counts["execution_failed_n"] += 1

        if bool(row.get("has_prediction")):
            counts["prediction_emitted_n"] += 1

        if row.get("gate_accepted") is True:
            counts["gate_pass_n"] += 1
        elif row.get("gate_accepted") is False:
            counts["gate_fail_n"] += 1

        if status == "completed" and answer_type == "learning_summary":
            counts["learning_summary_n"] += 1
            if winner_state == "selected":
                if correctness is True:
                    counts["validated_winner_n"] += 1
                    counts["binary_correct_n"] += 1
                elif correctness is False:
                    counts["binary_incorrect_n"] += 1
                elif winner_metrics:
                    counts["validated_winner_n"] += 1
                    counts["continuous_scored_n"] += 1
            elif winner_state == "none":
                counts["no_winner_n"] += 1

    return counts


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
