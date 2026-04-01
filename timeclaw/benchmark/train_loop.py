"""Train Loop: agent-driven explorative learning on MTBench train splits.

In training mode the agent uses the same prompt as testing, with a short
learning-mode hint appended.  It also receives an ``evaluate_against_gt``
tool that returns metrics + ground truth for one-step analysis (GT is
sanitised from the conversation afterwards).

The loop:
  1. Extract ground truth from the sample
  2. Call ``pipeline.run(mode="train", ground_truth=gt)``
  3. The agent explores autonomously, evaluates against GT, summarises
  4. Learning outcomes are saved to Daily Notes (L1) + MEMORY (L2)
  5. Distillation triggers every 30 notes (existing mechanism)

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
    DATASET_REGISTRY,
    MTBenchSample,
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
            return {
                "max": float(np.max(sample.output_window)),
                "min": float(np.min(sample.output_window)),
                "diff": float(np.max(sample.output_window) - np.min(sample.output_window)),
            }
        gt: dict[str, float] = {}
        if sample.labels.get("out_macd") is not None:
            gt["macd"] = float(sample.labels["out_macd"])
        if sample.labels.get("out_upper_bb") is not None:
            gt["upper_bb"] = float(sample.labels["out_upper_bb"])
        return gt if gt else None

    if task == "indicator_macd":
        if sample.labels.get("out_macd") is None:
            return None
        return {"macd": float(sample.labels["out_macd"])}

    if task == "indicator_upper_bb":
        if sample.labels.get("out_upper_bb") is None:
            return None
        return {"upper_bb": float(sample.labels["out_upper_bb"])}

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
                    ),
                    timeout=1200,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: timeout")
                result = {"answer_type": "", "execution_log": [], "prediction": None}
            except Exception as e:
                logger.error(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: {e}")
                result = {"answer_type": "", "execution_log": [], "prediction": None, "error": str(e)}

            results.append({
                "idx": sample_idx,
                "file": sample.file,
                "task": task,
                "domain": sample.domain,
                "answer_type": result.get("answer_type", ""),
                "has_prediction": result.get("prediction") is not None,
                "tools_run": len(result.get("execution_log", [])),
                "total_ms": result.get("total_ms", 0),
            })
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
    try:
        async with sem:
            try:
                scope_keys = scope_keys_for_dataset_task(ds_name, task)
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
                    ),
                    timeout=1200,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: timeout")
                result = {"answer_type": "", "execution_log": [], "prediction": None}
            except Exception as e:
                logger.error(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: {e}")
                result = {
                    "answer_type": "",
                    "execution_log": [],
                    "prediction": None,
                    "error": str(e),
                }

        return {
            "idx": sample_idx,
            "file": sample.file,
            "task": task,
            "domain": sample.domain,
            "answer_type": result.get("answer_type", ""),
            "has_prediction": result.get("prediction") is not None,
            "tools_run": len(result.get("execution_log", [])),
            "total_ms": result.get("total_ms", 0),
        }
    finally:
        if task_sem is not None:
            task_sem.release()


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
) -> dict[str, Any]:
    """Run explorative training loop over one or all MTBench datasets.

    Each sample is processed through ``pipeline.run(mode="train")``.
    The agent explores autonomously, evaluates against ground truth using
    the ``evaluate_against_gt`` tool, and saves learning outcomes to memory.
    """
    cfg = config or TimeClawConfig.load()
    # In-process tool calls are faster under high concurrency; MCP stdio
    # transport is single-threaded per server and would serialize calls.
    cfg.mcp.enabled = False
    trace.configure(verbose)

    pipeline = PipelineRunner(cfg, disable_memory=False)
    sample_indices_map = (
        load_sample_indices(sample_indices)
        if isinstance(sample_indices, (str, Path))
        else sample_indices
    )

    datasets_to_run = [dataset_name] if dataset_name else list(DATASET_REGISTRY.keys())
    all_results: dict[str, Any] = {}
    sem = asyncio.Semaphore(concurrency)
    dataset_work_items: list[tuple[str, int, MTBenchSample, str]] = []
    dataset_samples: dict[str, list[MTBenchSample]] = {}
    dataset_tasks: dict[str, list[str]] = {}

    for ds_name in datasets_to_run:
        info = DATASET_REGISTRY.get(ds_name)
        if not info:
            logger.warning(f"[TrainLoop] Unknown dataset: {ds_name}")
            continue

        ckpt = TRAIN_RESULTS_DIR / f"train_{ds_name}.json"
        if ckpt.exists() and ckpt.stat().st_size > 10 and resume_from == 0:
            logger.info(f"[TrainLoop] {ds_name}: checkpoint exists, skipping")
            continue

        try:
            samples = load_split(ds_name, "train")
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
        logger.info(
            f"[TrainLoop] {ds_name}: {total} samples, tasks={supported_tasks}, "
            f"concurrency={concurrency}{ptc_str}{ic_str}"
        )

        if per_task_concurrency is None and indicator_concurrency is None:
            for idx, sample in enumerate(samples):
                dataset_work_items.append((ds_name, resume_from + idx, sample, "__all__"))
        else:
            for idx, sample in enumerate(samples):
                for task in supported_tasks:
                    dataset_work_items.append((ds_name, resume_from + idx, sample, task))

    if per_task_concurrency is None and indicator_concurrency is None:
        coros = [
            _process_one_sample(ds_name, sample_idx, sample, dataset_tasks[ds_name], pipeline, sem)
            for ds_name, sample_idx, sample, _ in dataset_work_items
        ]
        flat_results = await asyncio.gather(*coros, return_exceptions=True)
        dataset_result_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item, batch in zip(dataset_work_items, flat_results, strict=False):
            ds_name = item[0]
            if isinstance(batch, Exception):
                logger.error(f"[TrainLoop] {ds_name}: batch error: {batch}")
                continue
            dataset_result_map[ds_name].extend(batch)
    else:
        task_sems: dict[tuple[str, str], asyncio.Semaphore] = {}
        for ds_name, tasks in dataset_tasks.items():
            for task in tasks:
                limit = (
                    indicator_concurrency
                    if task.startswith("indicator") and indicator_concurrency is not None
                    else (per_task_concurrency or concurrency)
                )
                task_sems[(ds_name, task)] = asyncio.Semaphore(limit)

        coros = [
            _process_one_task_sample(
                ds_name=ds_name,
                sample_idx=sample_idx,
                sample=sample,
                task=task,
                pipeline=pipeline,
                sem=sem,
                task_sem=task_sems.get((ds_name, task)),
            )
            for ds_name, sample_idx, sample, task in dataset_work_items
        ]
        flat_results = await asyncio.gather(*coros, return_exceptions=True)
        dataset_result_map = defaultdict(list)
        for item, r in zip(dataset_work_items, flat_results, strict=False):
            ds_name = item[0]
            if isinstance(r, Exception):
                logger.error(f"[TrainLoop] {ds_name}: task error: {r}")
                continue
            if r is not None:
                dataset_result_map[ds_name].append(r)

    for ds_name in dataset_samples:
        ds_results = dataset_result_map.get(ds_name, [])
        _save_checkpoint(ds_results, ds_name)
        summary = _summarize_results(ds_results)
        all_results[ds_name] = {"total": len(ds_results), "summary": summary}
        logger.info(f"[TrainLoop] {ds_name} done: {len(ds_results)} results")

    await pipeline.shutdown()
    return all_results


def _save_checkpoint(results: list[dict[str, Any]], ds_name: str) -> None:
    TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAIN_RESULTS_DIR / f"train_{ds_name}.json"
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
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
