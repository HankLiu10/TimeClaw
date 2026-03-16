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
from timeclaw.core.config import TimeClawConfig
from timeclaw.core import trace_logger as trace
from timeclaw.pipeline.runner import PipelineRunner

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

    if task == "trend":
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

    if task == "mcqa":
        return sample.answer if sample.question and sample.answer else None

    if task == "correlation":
        return sample.labels.get("correlation") or None

    return None


def _build_description(sample: MTBenchSample, task: str) -> str:
    """Build a minimal task description for the pipeline."""
    domain = sample.domain or "unknown"
    if task == "forecast":
        horizon = len(sample.output_window) if sample.output_window is not None else 24
        return (
            f"Forecast the next {horizon} values of this {domain} time series. "
            f"Return a JSON array of {horizon} numbers."
        )
    if task == "indicator":
        if domain == "weather":
            n_hours = len(sample.output_window) if sample.output_window is not None else 24
            n_days = max(1, n_hours // 24)
            return (
                f"Predict the next {n_days} day(s) highest temperature (max), "
                f"lowest temperature (min), and the temperature difference (diff). "
                f'Return JSON: {{"max": number, "min": number, "diff": number}}'
            )
        return (
            "Predict the relevant technical indicators for the next period. "
            'Return JSON with the indicator values.'
        )
    if task == "trend":
        if domain == "weather":
            return (
                "Predict the overall temperature trend for the next period. "
                'Return JSON: {"trend": "one of: up, stable, down"}'
            )
        return (
            "Predict the overall price trend for the next period. "
            'Return JSON: {"trend": "one of: strong_up, up, stable, down, strong_down"}'
        )
    if task == "mcqa":
        q = sample.question or ""
        return (
            f"Multiple choice question:\n{q}\n\n"
            f'Return JSON: {{"answer": "A/B/C/D"}}'
        )
    if task == "correlation":
        return (
            "Predict the correlation between the stock price and the analysis. "
            'Return JSON: {"correlation": "one of: Strong Positive Correlation, '
            'Weak Positive Correlation, No Correlation, '
            'Weak Negative Correlation, Strong Negative Correlation"}'
        )
    return f"{task} task on {domain} data."


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

            desc = _build_description(sample, task)
            text_data = None
            if sample.text:
                text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

            horizon = (
                len(sample.output_window) if task == "forecast" and sample.output_window is not None
                else None
            )

            task_type_map = {"mcqa": "qa", "trend": "classification"}
            pipeline_task = task_type_map.get(task, task)

            try:
                result = await asyncio.wait_for(
                    pipeline.run(
                        data=sample.input_window,
                        task_type=pipeline_task,
                        description=desc,
                        horizon=horizon,
                        text_data=text_data,
                        domain=sample.domain,
                        mode="train",
                        ground_truth=gt,
                    ),
                    timeout=300,
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


async def run_train_loop(
    dataset_name: str | None = None,
    task_filter: str | None = None,
    config: TimeClawConfig | None = None,
    max_samples: int | None = None,
    resume_from: int = 0,
    verbose: bool = False,
    concurrency: int = 8,
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

    datasets_to_run = [dataset_name] if dataset_name else list(DATASET_REGISTRY.keys())
    all_results: dict[str, Any] = {}
    sem = asyncio.Semaphore(concurrency)

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

        if max_samples:
            samples = samples[:max_samples]
        if resume_from > 0:
            samples = samples[resume_from:]

        supported_tasks = info["tasks"]
        if task_filter:
            supported_tasks = [t for t in supported_tasks if t == task_filter]
        if not supported_tasks:
            continue

        total = len(samples)

        logger.info(
            f"[TrainLoop] {ds_name}: {total} samples, tasks={supported_tasks}, "
            f"concurrency={concurrency}"
        )

        ds_results: list[dict[str, Any]] = []
        batch_size = max(concurrency, 10)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_samples = samples[batch_start:batch_end]

            coros = [
                _process_one_sample(
                    ds_name, batch_start + i + resume_from,
                    sample, supported_tasks, pipeline, sem,
                )
                for i, sample in enumerate(batch_samples)
            ]

            batch_results = await asyncio.gather(*coros, return_exceptions=True)

            for batch in batch_results:
                if isinstance(batch, Exception):
                    logger.error(f"[TrainLoop] {ds_name}: batch error: {batch}")
                    continue
                ds_results.extend(batch)

            logger.info(
                f"[TrainLoop] {ds_name}: batch {batch_start}-{batch_end} done "
                f"({len(ds_results)} results so far)"
            )

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
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = await run_train_loop(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        resume_from=args.resume_from,
        verbose=args.verbose,
        concurrency=args.concurrency,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
