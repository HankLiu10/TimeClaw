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
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    select_samples_by_index,
)
from timeclaw.core.config import TimeClawConfig
from timeclaw.core import trace_logger as trace
from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.benchmark.baseline_eval import (
    _weather_trend_3way_label,
    _weather_trend_past_3way_label,
)

TRAIN_RESULTS_DIR = Path(".timeclaw/train_results")
_TRAIN_RESULT_FRAMEWORK_TOOLS = {
    "spawn_subagent",
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
}


def _task_train_checkpoint_path(ds_name: str, task: str) -> Path:
    return TRAIN_RESULTS_DIR / f"train_{ds_name}__{task}.json"


def _load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 10:
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [r for r in data if isinstance(r, dict)]


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


def _train_done_keys(rows: list[dict[str, Any]]) -> set[tuple[int, str]]:
    out: set[tuple[int, str]] = set()
    for r in rows:
        status = str(r.get("status", "")).lower()
        if status in {"error", "timeout", "failed"}:
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


def _preview_value(value: Any, limit: int = 400) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text)} chars]"


def _sanitize_learning_artifact_preview(value: Any, limit: int = 400) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = _preview_value(value, limit=limit * 2)
    if not text:
        return text
    banned_markers = (
        "spawn_subagent",
        "evaluate_against_gt",
        "evaluate_batch_against_gt",
        "batch_evaluate_against_gt",
        "execution context",
        "assigned path",
        "before finishing",
    )
    cleaned_lines = [
        line for line in text.splitlines()
        if not any(marker in line.lower() for marker in banned_markers)
    ]
    cleaned = "\n".join(cleaned_lines).strip()
    lower_cleaned = cleaned.lower()
    if any(marker in lower_cleaned for marker in banned_markers):
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + f"... [{len(cleaned)} chars]"


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
        winner_chain = _filter_train_result_tool_chain(raw_answer.get("winner_tool_chain"))
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


def _build_train_result_row(
    *,
    ds_name: str,
    sample_idx: int,
    sample: MTBenchSample,
    task: str,
    pipeline_task: str,
    scope_keys: list[str],
    result: dict[str, Any],
) -> dict[str, Any]:
    raw_answer = result.get("raw_answer")
    learning_summary = raw_answer if isinstance(raw_answer, dict) else {}
    sample_ref = f"{ds_name}:{sample.file}:{sample_idx}:{task}"
    substantive_tool_chain = _extract_train_result_tool_chain(result)

    return {
        "idx": sample_idx,
        "dataset_name": ds_name,
        "sample_ref": sample_ref,
        "file": sample.file,
        "task": task,
        "pipeline_task": pipeline_task,
        "scope_keys": list(scope_keys),
        "domain": sample.domain,
        "status": result.get("status", "completed"),
        "answer_type": result.get("answer_type", ""),
        "has_prediction": result.get("prediction") is not None,
        "trace_id": result.get("trace_id"),
        "gate": "PASS" if result.get("gate_accepted") else "FAIL",
        "gate_accepted": result.get("gate_accepted"),
        "gate_rejections": list(result.get("gate_rejections", [])),
        "confidence": result.get("confidence"),
        "tools_run": len(substantive_tool_chain),
        "tool_chain": substantive_tool_chain,
        "total_ms": result.get("total_ms", 0),
        "error": result.get("error"),
        "error_type": result.get("error_type"),
        "failure_reason": result.get("failure_reason"),
        "reasoning_preview": _sanitize_learning_artifact_preview(result.get("reasoning", ""), limit=300),
        "raw_answer_preview": _sanitize_learning_artifact_preview(raw_answer, limit=500),
        "learning_insight": learning_summary.get("insight"),
        "learning_recommendation": learning_summary.get("recommendation"),
        "evidence_summary_preview": _sanitize_learning_artifact_preview(result.get("evidence_summary", ""), limit=400),
    }


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
                error_type = type(e).__name__
                failure_reason = (
                    "max_steps_exhausted"
                    if error_type == "MaxStepsExhaustedError"
                    else "error"
                )
                result = {
                    "answer_type": "",
                    "execution_log": [],
                    "prediction": None,
                    "status": failure_reason,
                    "error": str(e),
                    "error_type": error_type,
                    "failure_reason": failure_reason,
                    "trace_id": getattr(e, "trace_id", None),
                    "total_ms": getattr(e, "total_ms", 0),
                }

            results.append(_build_train_result_row(
                ds_name=ds_name,
                sample_idx=sample_idx,
                sample=sample,
                task=task,
                pipeline_task=pipeline_task,
                scope_keys=scope_keys,
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
                error_type = getattr(e, "error_type", type(e).__name__)
                failure_reason = getattr(
                    e,
                    "failure_reason",
                    "max_steps_exhausted"
                    if error_type == "MaxStepsExhaustedError"
                    else "error",
                )
                result = {
                    "answer_type": "",
                    "execution_log": [],
                    "prediction": None,
                    "status": failure_reason,
                    "error": str(e),
                    "error_type": error_type,
                    "failure_reason": failure_reason,
                    "trace_id": getattr(e, "trace_id", None),
                    "total_ms": getattr(e, "total_ms", 0),
                }

        return _build_train_result_row(
            ds_name=ds_name,
            sample_idx=sample_idx,
            sample=sample,
            task=task,
            pipeline_task=pipeline_task,
            scope_keys=scope_keys,
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
) -> tuple[str, list[dict[str, Any]], Exception | None]:
    try:
        rows = await _process_one_sample(
            ds_name=ds_name,
            sample_idx=sample_idx,
            sample=sample,
            supported_tasks=supported_tasks,
            pipeline=pipeline,
            sem=sem,
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
    sem = asyncio.Semaphore(concurrency)
    dataset_work_items: list[tuple[str, int, MTBenchSample, str]] = []
    dataset_samples: dict[str, list[MTBenchSample]] = {}
    dataset_tasks: dict[str, list[str]] = {}
    dataset_existing_rows: dict[str, list[dict[str, Any]]] = {}

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

        samples = select_samples_by_index(samples, ds_name, sample_indices_map)
        if max_samples:
            samples = samples[:max_samples]
        if resume_from > 0:
            samples = samples[resume_from:]

        supported_tasks = expand_dataset_tasks(ds_name, list(info["tasks"]))
        if task_filter:
            supported_tasks = [t for t in supported_tasks if t == task_filter]
        if not supported_tasks:
            continue

        total = len(samples)
        dataset_samples[ds_name] = samples
        dataset_tasks[ds_name] = supported_tasks

        ptc_str = f", per_task_concurrency={per_task_concurrency}" if per_task_concurrency else ""
        ic_str = f", indicator_concurrency={indicator_concurrency}" if indicator_concurrency else ""
        pending_all = merge_checkpoint or per_task_concurrency is not None or indicator_concurrency is not None

        if pending_all:
            for idx, sample in enumerate(samples):
                abs_idx = resume_from + idx
                for task in supported_tasks:
                    if merge_checkpoint and (abs_idx, task) in done_keys:
                        continue
                    if _extract_ground_truth(sample, task) is None:
                        continue
                    dataset_work_items.append((ds_name, abs_idx, sample, task))
        else:
            for idx, sample in enumerate(samples):
                dataset_work_items.append((ds_name, resume_from + idx, sample, "__all__"))

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
                ))
                for (a, si, s, t) in dataset_work_items
            ]
        else:
            batch_work = [w for w in dataset_work_items if w[3] == "__all__"]
            pending_tasks = [
                asyncio.create_task(_run_sample_work_item(a, si, s, dataset_tasks[a], pipeline, sem))
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
                logger.info(f"[TrainLoop] {ds_name}: nothing pending, kept {len(existing_rows)} checkpoint rows")
            continue

        merged = merged_rows_by_ds.get(ds_name, existing_rows)
        summary = _summarize_results(merged)
        all_results[ds_name] = {"total": len(merged), "summary": summary}
        logger.info(f"[TrainLoop] {ds_name} done: {len(merged)} results")

    await pipeline.shutdown()
    return all_results


def _save_task_checkpoint(results: list[dict[str, Any]], ds_name: str, task: str) -> None:
    TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _task_train_checkpoint_path(ds_name, task)
    path.write_text(json.dumps(results, indent=2, default=str))


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
