#!/usr/bin/env python3
"""Unified comparison: Baseline (direct LLM) vs TimeClaw on MTBench test sets.

Flow:
  Baseline: directly evaluate on test set (no training needed)
  TimeClaw: train on train set -> distill memory -> evaluate on test set

Usage:
    python scripts/eval_comparison.py --mode both --concurrency 80
    python scripts/eval_comparison.py --mode baseline --concurrency 80
    python scripts/eval_comparison.py --mode timeclaw --concurrency 80
    python scripts/eval_comparison.py --mode report   # generate report from saved results
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

RESULTS_DIR = Path(".timeclaw/test_results")
REPORT_PATH = RESULTS_DIR / "comparison_report.md"


async def run_baseline(
    dataset: str | None, task: str | None, max_samples: int | None, concurrency: int,
) -> dict[str, Any]:
    """Baseline: direct LLM on test set — no training needed."""
    from timeclaw.benchmark.baseline_eval import run_baseline as _run
    return await _run(
        dataset_name=dataset, task_filter=task,
        max_samples=max_samples, concurrency=concurrency,
    )


async def run_timeclaw_full(
    dataset: str | None, task: str | None, max_samples: int | None, concurrency: int,
) -> dict[str, Any]:
    """TimeClaw: train -> distill -> test.

    Training now uses the same PipelineRunner path as testing, ensuring
    that memory injection and recording are consistent.
    """
    from timeclaw.memory.hierarchical import HierarchicalMemory
    from timeclaw.benchmark.pipeline_eval import run_pipeline_eval

    # ── Step 1: Train on train set (same pipeline as test) ────────
    logger.info("=" * 50)
    logger.info("[TimeClaw] Step 1/3: Training on train set...")
    logger.info("=" * 50)
    t0 = time.time()
    train_result = await run_pipeline_eval(
        dataset_name=dataset, task_filter=task,
        max_samples=max_samples, concurrency=concurrency,
        use_skills=True, label="train",
        split="train",
    )
    train_elapsed = time.time() - t0
    logger.info(f"[TimeClaw] Training done in {train_elapsed:.1f}s")

    # ── Step 2: Force distill all memory ──────────────────────────
    logger.info("=" * 50)
    logger.info("[TimeClaw] Step 2/3: Distilling memory...")
    logger.info("=" * 50)
    mem = HierarchicalMemory(enabled=True)
    await mem.force_distill()
    mem_summary = mem.summary()
    n_playbooks = mem_summary.get("playbooks", 0)
    logger.info(f"[TimeClaw] Distillation done: {mem_summary}")

    # ── Step 3: Evaluate on test set ──────────────────────────────
    logger.info("=" * 50)
    logger.info("[TimeClaw] Step 3/3: Evaluating on test set...")
    logger.info("=" * 50)
    t0 = time.time()
    test_result = await run_pipeline_eval(
        dataset_name=dataset, task_filter=task,
        max_samples=max_samples, concurrency=concurrency,
        use_skills=True, label="timeclaw",
        split="test",
    )
    test_elapsed = time.time() - t0
    logger.info(f"[TimeClaw] Test evaluation done in {test_elapsed:.1f}s")

    return {
        "train": train_result,
        "playbooks_count": n_playbooks,
        "test": test_result,
        "train_elapsed_s": round(train_elapsed, 1),
        "test_elapsed_s": round(test_elapsed, 1),
    }


def generate_report() -> str:
    """Generate markdown comparison report from saved result files."""
    lines = ["# MTBench Evaluation: Baseline (Direct LLM) vs TimeClaw\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("- **Baseline**: GPT-5.1 with MTBench official prompts, no tools, no agent pipeline")
    lines.append("- **TimeClaw**: Full agent pipeline (train -> memory distill -> test), with tools and multi-agent reasoning\n")

    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY

    for ds_name, info in DATASET_REGISTRY.items():
        bl_path = RESULTS_DIR / f"baseline_{ds_name}.json"
        tc_path = RESULTS_DIR / f"test_{ds_name}_timeclaw.json"

        bl_data = json.loads(bl_path.read_text()) if bl_path.exists() else []
        tc_data = json.loads(tc_path.read_text()) if tc_path.exists() else []

        if not bl_data and not tc_data:
            continue

        lines.append(f"\n## {ds_name} ({info['domain']})\n")

        for task in info["tasks"]:
            bl_task = [r for r in bl_data if r.get("task") == task]
            tc_task = [r for r in tc_data if r.get("task") == task]

            if not bl_task and not tc_task:
                continue

            lines.append(f"\n### Task: {task}\n")

            bl_metrics = _aggregate_metrics(bl_task)
            tc_metrics = _aggregate_metrics(tc_task)

            all_keys = sorted(set(list(bl_metrics.keys()) + list(tc_metrics.keys())))

            lines.append("| Metric | Baseline (LLM) | TimeClaw | Winner |")
            lines.append("|--------|---------------|----------|--------|")

            for k in all_keys:
                bl_val = bl_metrics.get(k)
                tc_val = tc_metrics.get(k)
                bl_str = f"{bl_val:.4f}" if bl_val is not None else "N/A"
                tc_str = f"{tc_val:.4f}" if tc_val is not None else "N/A"

                winner = ""
                if bl_val is not None and tc_val is not None:
                    if k == "accuracy":
                        winner = "**TimeClaw**" if tc_val > bl_val else ("Baseline" if bl_val > tc_val else "Tie")
                    else:
                        winner = "**TimeClaw**" if tc_val < bl_val else ("Baseline" if bl_val < tc_val else "Tie")

                lines.append(f"| {k} | {bl_str} | {tc_str} | {winner} |")

            bl_n = len([r for r in bl_task if "error" not in r.get("metrics", {})])
            tc_n = len([r for r in tc_task if "error" not in r.get("metrics", {})])
            bl_err = len(bl_task) - bl_n
            tc_err = len(tc_task) - tc_n
            lines.append(f"\nSamples: Baseline={bl_n} (err={bl_err}), TimeClaw={tc_n} (err={tc_err})")

    lines.append("\n\n## Global Summary\n")
    bl_all, tc_all = _global_summary()
    if bl_all or tc_all:
        lines.append("| Domain | Task | Metric | Baseline | TimeClaw | Delta |")
        lines.append("|--------|------|--------|----------|----------|-------|")
        all_entries = sorted(set(list(bl_all.keys()) + list(tc_all.keys())))
        for key in all_entries:
            domain, task, metric = key
            bl_v = bl_all.get(key)
            tc_v = tc_all.get(key)
            bl_s = f"{bl_v:.4f}" if bl_v is not None else "N/A"
            tc_s = f"{tc_v:.4f}" if tc_v is not None else "N/A"
            delta = ""
            if bl_v is not None and tc_v is not None:
                d = tc_v - bl_v
                delta = f"{d:+.4f}"
            lines.append(f"| {domain} | {task} | {metric} | {bl_s} | {tc_s} | {delta} |")

    report = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    logger.info(f"Report saved to {REPORT_PATH}")
    return report


def _aggregate_metrics(results: list[dict]) -> dict[str, float]:
    metrics_lists: dict[str, list[float]] = {}
    for r in results:
        m = r.get("metrics", {})
        for k, v in m.items():
            if isinstance(v, (int, float)):
                metrics_lists.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in metrics_lists.items()}


def _global_summary() -> tuple[dict, dict]:
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY

    bl_agg: dict[tuple, list[float]] = {}
    tc_agg: dict[tuple, list[float]] = {}

    for ds_name, info in DATASET_REGISTRY.items():
        domain = info["domain"]
        bl_path = RESULTS_DIR / f"baseline_{ds_name}.json"
        tc_path = RESULTS_DIR / f"test_{ds_name}_timeclaw.json"

        bl_data = json.loads(bl_path.read_text()) if bl_path.exists() else []
        tc_data = json.loads(tc_path.read_text()) if tc_path.exists() else []

        for task in info["tasks"]:
            bl_task = [r for r in bl_data if r.get("task") == task]
            tc_task = [r for r in tc_data if r.get("task") == task]

            bl_m = _aggregate_metrics(bl_task)
            tc_m = _aggregate_metrics(tc_task)

            for k, v in bl_m.items():
                bl_agg.setdefault((domain, task, k), []).append(v)
            for k, v in tc_m.items():
                tc_agg.setdefault((domain, task, k), []).append(v)

    bl_final = {k: float(np.mean(v)) for k, v in bl_agg.items()}
    tc_final = {k: float(np.mean(v)) for k, v in tc_agg.items()}
    return bl_final, tc_final


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline vs TimeClaw Comparison")
    parser.add_argument("--mode", choices=["baseline", "timeclaw", "both", "report"], default="both")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=80)
    args = parser.parse_args()

    t_total = time.time()

    if args.mode in ("baseline", "both"):
        logger.info("=" * 60)
        logger.info("Phase 1: Running BASELINE evaluation (direct LLM, no training)")
        logger.info("=" * 60)
        bl_result = await run_baseline(args.dataset, args.task, args.max_samples, args.concurrency)
        print("\n=== BASELINE RESULTS ===")
        print(json.dumps(bl_result, indent=2, default=str))

    if args.mode in ("timeclaw", "both"):
        logger.info("=" * 60)
        logger.info("Phase 2: Running TIMECLAW (train -> distill -> test)")
        logger.info("=" * 60)
        tc_result = await run_timeclaw_full(args.dataset, args.task, args.max_samples, args.concurrency)
        print("\n=== TIMECLAW RESULTS ===")
        print(json.dumps(tc_result, indent=2, default=str))

    report = generate_report()
    print("\n" + report)

    total_elapsed = time.time() - t_total
    logger.info(f"Total evaluation time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    asyncio.run(main())
