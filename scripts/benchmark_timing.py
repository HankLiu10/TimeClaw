"""Benchmark: measure average time per task type across MTBench datasets.

Supports parallel sample execution via --concurrency flag.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY, MTBenchSample
from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner

SAMPLES_PER_TASK = 3


async def run_single(
    pipeline: PipelineRunner,
    sample: MTBenchSample,
    task: str,
    sem: asyncio.Semaphore,
) -> dict:
    """Run a single sample through the pipeline and return timing info."""
    async with sem:
        horizon = len(sample.output_window) if sample.output_window is not None else 24

        text_data = None
        if sample.text:
            text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

        task_map = {
            "forecast": "forecast",
            "trend": "classification",
            "indicator": "forecast",
            "mcqa": "qa",
            "correlation": "qa",
        }
        pipeline_task = task_map.get(task, task)
        desc = sample.question[:200] if sample.question and task in ("mcqa", "correlation") else f"{sample.domain} {task}"

        t0 = time.time()
        try:
            result = await pipeline.run(
                data=sample.input_window,
                task_type=pipeline_task,
                description=desc,
                horizon=horizon if task in ("forecast", "indicator") else None,
                text_data=text_data,
                verbose=False,
            )
            elapsed = time.time() - t0
            timeline = result.get("timeline", {})
            return {
                "success": True,
                "total_s": elapsed,
                "total_ms": result.get("total_ms", int(elapsed * 1000)),
                "plan_ms": timeline.get("plan", {}).get("elapsed_ms", 0),
                "diagnose_ms": timeline.get("diagnose", {}).get("elapsed_ms", 0),
                "route_ms": timeline.get("route", {}).get("elapsed_ms", 0),
                "execute_ms": timeline.get("execute", {}).get("elapsed_ms", 0),
                "verify_ms": timeline.get("verify", {}).get("elapsed_ms", 0),
                "deliver_ms": timeline.get("deliver", {}).get("elapsed_ms", 0),
                "modalities": result.get("modalities", []),
                "n_branches": timeline.get("execute", {}).get("executor_branches", 0),
                "dataset": "",
                "task": task,
                "sample": sample.file,
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {"success": False, "total_s": elapsed, "error": str(e),
                    "dataset": "", "task": task, "sample": sample.file}


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4, help="Max parallel samples")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_TASK, help="Samples per task")
    args = parser.parse_args()

    cfg = TimeClawConfig()
    pipeline = PipelineRunner(cfg)
    sem = asyncio.Semaphore(args.concurrency)

    task_timings: dict[str, list[dict]] = defaultdict(list)
    dataset_task_timings: dict[str, list[dict]] = defaultdict(list)

    all_jobs: list[tuple[str, str, MTBenchSample]] = []
    for ds_name, info in DATASET_REGISTRY.items():
        for task in info["tasks"]:
            try:
                samples = load_split(ds_name, "test")
            except FileNotFoundError:
                continue
            for s in samples[:args.samples]:
                all_jobs.append((ds_name, task, s))

    logger.info(f"[Benchmark] {len(all_jobs)} total jobs, concurrency={args.concurrency}")
    t_wall_start = time.time()

    coros = [run_single(pipeline, s, task, sem) for (_, task, s) in all_jobs]
    results = await asyncio.gather(*coros, return_exceptions=True)

    t_wall = time.time() - t_wall_start

    for (ds_name, task, sample), result in zip(all_jobs, results):
        if isinstance(result, Exception):
            result = {"success": False, "total_s": 0, "error": str(result),
                      "dataset": ds_name, "task": task, "sample": sample.file}
        else:
            result["dataset"] = ds_name
        task_timings[task].append(result)
        dataset_task_timings[f"{ds_name}/{task}"].append(result)

    # Summary
    print("\n" + "=" * 95)
    print(f"TIMING BENCHMARK RESULTS  (concurrency={args.concurrency}, wall_time={t_wall:.1f}s)")
    print("=" * 95)

    print(f"\n{'Task':<15} {'N':>4} {'Avg(s)':>8} {'Min(s)':>8} {'Max(s)':>8} "
          f"{'Plan':>7} {'Diag':>7} {'Exec':>7} {'Verify':>7}")
    print("-" * 95)

    for task in ["forecast", "trend", "indicator", "mcqa", "correlation"]:
        ok = [r for r in task_timings[task] if r.get("success")]
        if not ok:
            print(f"{task:<15} {'0':>4} {'N/A':>8}")
            continue
        times = [r["total_s"] for r in ok]
        plan = [r.get("plan_ms", 0) for r in ok]
        diag = [r.get("diagnose_ms", 0) for r in ok]
        exe = [r.get("execute_ms", 0) for r in ok]
        ver = [r.get("verify_ms", 0) for r in ok]
        print(
            f"{task:<15} {len(ok):>4} {np.mean(times):>8.1f} {min(times):>8.1f} {max(times):>8.1f} "
            f"{np.mean(plan):>6.0f}ms {np.mean(diag):>6.0f}ms {np.mean(exe):>6.0f}ms {np.mean(ver):>6.0f}ms"
        )

    print("\n" + "-" * 95)
    print("Per Dataset x Task:")
    print(f"{'Dataset/Task':<45} {'N':>4} {'Avg(s)':>8} {'Min(s)':>8} {'Max(s)':>8}")
    print("-" * 95)

    for key in sorted(dataset_task_timings.keys()):
        ok = [r for r in dataset_task_timings[key] if r.get("success")]
        if not ok:
            continue
        times = [r["total_s"] for r in ok]
        print(f"{key:<45} {len(ok):>4} {np.mean(times):>8.1f} {min(times):>8.1f} {max(times):>8.1f}")

    total_seq = sum(r["total_s"] for r in sum(task_timings.values(), []) if r.get("success"))
    print(f"\nSequential sum: {total_seq:.1f}s | Wall time: {t_wall:.1f}s | Speedup: {total_seq/t_wall:.1f}x")

    all_results = []
    for v in task_timings.values():
        all_results.extend(v)
    Path(".timeclaw").mkdir(exist_ok=True)
    Path(".timeclaw/benchmark_timing.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    print(f"Detailed results saved to .timeclaw/benchmark_timing.json")


if __name__ == "__main__":
    asyncio.run(main())
