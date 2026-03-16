"""Full benchmark pipeline: clean → split → train → distill → test → baseline → compare.

Ensures a fair comparison by:
1. Clearing ALL learned state (memory, playbooks, experiences, strategies) before training
2. Splitting datasets 70/30 by time (no overlap for time-series data)
3. Training on train split only (pipeline runs + hierarchical memory recording)
4. Distilling memory (L1→L2→L3) after training
5. Evaluating TimeClaw on test split (with learned memory)
6. Evaluating baseline (direct LLM, no tools) on the same test split
7. Generating a comparison report
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

WORKSPACE = Path(".timeclaw")

DIRS_TO_CLEAN = [
    WORKSPACE / "hierarchical_memory",
    WORKSPACE / "experiences",
    WORKSPACE / "skills",
    WORKSPACE / "task_strategies",
    WORKSPACE / "knowledge",
    WORKSPACE / "train_results",
    WORKSPACE / "test_results",
]
FILES_TO_CLEAN = [
    WORKSPACE / "vector_store.db",
    WORKSPACE / "hierarchical_memory" / "TOOLS.md",
]


def clean_all_state() -> None:
    """Remove ALL learned state to ensure a fair start."""
    for d in DIRS_TO_CLEAN:
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"[Clean] Removed {d}")
    for f in FILES_TO_CLEAN:
        if f.exists():
            f.unlink()
            logger.info(f"[Clean] Removed {f}")
    logger.info("[Clean] All learned state cleared")


def run_split(sample_ratio: float = 1.0) -> dict[str, Any]:
    """Split all datasets 70/30 by time, optionally sub-sampling."""
    from timeclaw.benchmark.dataset_split import split_all_datasets
    results = split_all_datasets(sample_ratio=sample_ratio)
    total_train = sum(r["train"] for r in results.values())
    total_test = sum(r["test"] for r in results.values())
    logger.info(f"[Split] {len(results)} datasets: {total_train} train, {total_test} test"
                f" (sample_ratio={sample_ratio:.0%})")
    for name, r in results.items():
        ratio = r["train"] / (r["train"] + r["test"]) * 100 if (r["train"] + r["test"]) else 0
        logger.info(f"  {name}: train={r['train']}, test={r['test']}, "
                     f"split_year={r['split_year']}, ratio={ratio:.0f}%")
    return results


async def run_train(concurrency: int = 80) -> dict[str, Any]:
    """Train TimeClaw on all train splits using the pipeline.

    Each sample is run through the pipeline, and the outcome (with ground truth)
    is recorded into hierarchical memory for learning.
    """
    from timeclaw.benchmark.pipeline_eval import run_pipeline_eval

    logger.info("[Train] Starting pipeline training on train splits...")
    t0 = time.time()

    result = await run_pipeline_eval(
        use_skills=True,
        verbose=False,
        label="train",
        concurrency=concurrency,
        full_dataset=False,
        split="train",
    )

    elapsed = time.time() - t0
    logger.info(f"[Train] Training completed in {elapsed:.0f}s")
    return result


def _clean_memory_state() -> None:
    """Remove all learned state between datasets for fair per-dataset evaluation."""
    for d in DIRS_TO_CLEAN:
        if d.exists():
            shutil.rmtree(d)
    for f in FILES_TO_CLEAN:
        if f.exists():
            f.unlink()
    logger.info("[Clean] Memory/playbooks/experiences cleared for next dataset")


async def run_train_on_train_split(concurrency: int = 80) -> dict[str, Any]:
    """For each dataset: clean → train → distill → test vs baseline.

    Each dataset starts from a blank memory slate, ensuring fair comparison.
    Uses run_pipeline_eval for both train and test to ensure _record_enabled
    is set correctly and the pipeline is properly initialised.
    """
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
    from timeclaw.benchmark.pipeline_eval import run_pipeline_eval
    from timeclaw.benchmark.baseline_eval import run_baseline as _run_baseline
    from timeclaw.memory.hierarchical import HierarchicalMemory

    all_train: dict[str, Any] = {}
    all_tc_test: dict[str, Any] = {}
    all_bl_test: dict[str, Any] = {}

    for ds_name in list(DATASET_REGISTRY.keys()):
        _clean_memory_state()
        logger.info(f"\n{'='*60}")
        logger.info(f"[Pipeline] {ds_name}: clean → train → distill → test → baseline")
        logger.info(f"{'='*60}")

        # ── Train ──
        try:
            train_result = await run_pipeline_eval(
                dataset_name=ds_name,
                use_skills=True, verbose=False,
                label="train", concurrency=concurrency,
                split="train",
            )
        except Exception as e:
            logger.error(f"[Train] {ds_name} failed: {e}")
            continue
        all_train[ds_name] = train_result.get("results", {}).get(ds_name, {})

        # ── Force distill ──
        try:
            mem = HierarchicalMemory(enabled=True)
            await mem.force_distill()
            logger.info(f"[Distill] {ds_name}: {mem.summary()}")
        except Exception as e:
            logger.error(f"[Distill] {ds_name} error: {e}")

        # ── Test (TimeClaw with learned memory) ──
        try:
            tc_result = await run_pipeline_eval(
                dataset_name=ds_name,
                use_skills=True, verbose=False,
                label="timeclaw", concurrency=concurrency,
                split="test",
            )
            all_tc_test[ds_name] = tc_result.get("results", {}).get(ds_name, {})
        except Exception as e:
            logger.error(f"[Test:TC] {ds_name} failed: {e}")

        # ── Baseline (direct LLM, no training) ──
        try:
            bl_result = await _run_baseline(
                dataset_name=ds_name, concurrency=concurrency,
            )
            all_bl_test[ds_name] = bl_result.get(ds_name, {})
        except Exception as e:
            logger.error(f"[Test:BL] {ds_name} failed: {e}")

        # ── Per-dataset comparison ──
        tc_s = all_tc_test.get(ds_name, {}).get("summary", {})
        bl_s = all_bl_test.get(ds_name, {}).get("summary", {})
        _print_comparison(ds_name, tc_s, bl_s)

    return {
        "label": "train",
        "train_results": all_train,
        "tc_test": all_tc_test,
        "bl_test": all_bl_test,
    }


def _print_comparison(ds_name: str, tc_summary: dict, bl_summary: dict) -> None:
    """Print a formatted per-dataset comparison table."""
    logger.info(f"\n{'='*60}")
    logger.info(f"[Compare] {ds_name}")
    logger.info(f"{'─'*60}")
    all_tasks = sorted(set(list(tc_summary.keys()) + list(bl_summary.keys())))
    for task in all_tasks:
        tc_t = tc_summary.get(task, {})
        bl_t = bl_summary.get(task, {})
        all_keys = sorted(set(
            [k for k in tc_t if k.startswith("avg_")] +
            [k for k in bl_t if k.startswith("avg_")]
        ))
        for key in all_keys:
            metric = key[4:]
            if metric == "error":
                continue
            tc_v = tc_t.get(key, float("nan"))
            bl_v = bl_t.get(key, float("nan"))
            if not isinstance(tc_v, (int, float)):
                tc_v = float("nan")
            if not isinstance(bl_v, (int, float)):
                bl_v = float("nan")
            higher_better = metric in ("accuracy",)
            if higher_better:
                winner = "TC ✓" if tc_v > bl_v + 0.001 else ("BL ✓" if bl_v > tc_v + 0.001 else "Tie")
            else:
                winner = "TC ✓" if tc_v < bl_v - 0.001 else ("BL ✓" if bl_v < tc_v - 0.001 else "Tie")
            logger.info(f"  {task:>12s}/{metric:<12s}: TC={tc_v:.4f}  BL={bl_v:.4f}  {winner}")
    logger.info(f"{'='*60}\n")


async def run_test_timeclaw(concurrency: int = 80) -> dict[str, Any]:
    """Evaluate TimeClaw on test splits (with learned memory)."""
    from timeclaw.benchmark.pipeline_eval import run_pipeline_eval

    logger.info("[Test:TimeClaw] Starting evaluation on test splits...")
    t0 = time.time()

    result = await run_pipeline_eval(
        use_skills=True,
        verbose=False,
        label="timeclaw",
        concurrency=concurrency,
    )

    elapsed = time.time() - t0
    logger.info(f"[Test:TimeClaw] Completed in {elapsed:.0f}s")
    return result


async def run_test_baseline(concurrency: int = 80) -> dict[str, Any]:
    """Evaluate baseline (direct LLM) on test splits."""
    from timeclaw.benchmark.baseline_eval import run_baseline

    logger.info("[Test:Baseline] Starting baseline evaluation on test splits...")
    t0 = time.time()

    result = await run_baseline(concurrency=concurrency)

    elapsed = time.time() - t0
    logger.info(f"[Test:Baseline] Completed in {elapsed:.0f}s")
    return result


def generate_comparison_report(
    tc_results: dict[str, Any],
    bl_results: dict[str, Any],
    split_info: dict[str, Any],
) -> str:
    """Generate a markdown comparison report."""
    lines = [
        "# TimeClaw vs Baseline — Full Benchmark Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Splits (70/30 by time, no overlap)",
        "",
        "| Dataset | Train | Test | Split Year | Train % |",
        "|---------|-------|------|------------|---------|",
    ]

    for name, info in sorted(split_info.items()):
        total = info["train"] + info["test"]
        pct = info["train"] / total * 100 if total else 0
        lines.append(f"| {name} | {info['train']} | {info['test']} | "
                      f"{info['split_year']} | {pct:.0f}% |")

    lines += ["", "## Per-Task Comparison", ""]

    tc_all = tc_results.get("results", tc_results)
    bl_all = bl_results

    task_tc_agg: dict[str, dict[str, list[float]]] = {}
    task_bl_agg: dict[str, dict[str, list[float]]] = {}

    for ds_name in set(list(tc_all.keys()) + list(bl_all.keys())):
        tc_ds = tc_all.get(ds_name, {}).get("summary", {})
        bl_ds = bl_all.get(ds_name, {}).get("summary", {})

        for task in set(list(tc_ds.keys()) + list(bl_ds.keys())):
            if task not in task_tc_agg:
                task_tc_agg[task] = {}
            if task not in task_bl_agg:
                task_bl_agg[task] = {}

            tc_task = tc_ds.get(task, {})
            bl_task = bl_ds.get(task, {})

            for key, val in tc_task.items():
                if key.startswith("avg_") and isinstance(val, (int, float)):
                    metric = key[4:]
                    n = tc_task.get("n", 1)
                    task_tc_agg[task].setdefault(metric, []).extend([val] * n)

            for key, val in bl_task.items():
                if key.startswith("avg_") and isinstance(val, (int, float)):
                    metric = key[4:]
                    n = bl_task.get("n", 1)
                    task_bl_agg[task].setdefault(metric, []).extend([val] * n)

    lines.append("### Overall (aggregated across all datasets)")
    lines.append("")
    lines.append("| Task | Metric | TimeClaw | Baseline | Δ | Winner |")
    lines.append("|------|--------|----------|----------|---|--------|")

    for task in sorted(task_tc_agg.keys()):
        tc_metrics = task_tc_agg.get(task, {})
        bl_metrics = task_bl_agg.get(task, {})
        all_keys = sorted(set(list(tc_metrics.keys()) + list(bl_metrics.keys())))

        for metric in all_keys:
            if metric in ("error",):
                continue
            tc_vals = tc_metrics.get(metric, [])
            bl_vals = bl_metrics.get(metric, [])
            tc_avg = float(np.mean(tc_vals)) if tc_vals else float("nan")
            bl_avg = float(np.mean(bl_vals)) if bl_vals else float("nan")

            if np.isnan(tc_avg) or np.isnan(bl_avg):
                delta_str = "N/A"
                winner = "—"
            else:
                delta = tc_avg - bl_avg
                delta_str = f"{delta:+.4f}"
                higher_better = metric in ("accuracy",)
                if higher_better:
                    winner = "TimeClaw" if delta > 0.001 else ("Baseline" if delta < -0.001 else "Tie")
                else:
                    winner = "TimeClaw" if delta < -0.001 else ("Baseline" if delta > 0.001 else "Tie")

            lines.append(f"| {task} | {metric} | {tc_avg:.4f} | {bl_avg:.4f} | "
                          f"{delta_str} | {winner} |")

    lines += ["", "### Per-Dataset Breakdown", ""]

    for ds_name in sorted(set(list(tc_all.keys()) + list(bl_all.keys()))):
        tc_ds = tc_all.get(ds_name, {}).get("summary", {})
        bl_ds = bl_all.get(ds_name, {}).get("summary", {})
        if not tc_ds and not bl_ds:
            continue

        lines.append(f"#### {ds_name}")
        lines.append("")
        lines.append("| Task | Metric | TimeClaw | Baseline | Winner |")
        lines.append("|------|--------|----------|----------|--------|")

        for task in sorted(set(list(tc_ds.keys()) + list(bl_ds.keys()))):
            tc_task = tc_ds.get(task, {})
            bl_task = bl_ds.get(task, {})
            for key in sorted(tc_task.keys()):
                if not key.startswith("avg_"):
                    continue
                metric = key[4:]
                if metric == "error":
                    continue
                tc_val = tc_task.get(key, float("nan"))
                bl_val = bl_task.get(f"avg_{metric}", float("nan"))

                if isinstance(tc_val, (int, float)) and isinstance(bl_val, (int, float)):
                    higher_better = metric in ("accuracy",)
                    delta = tc_val - bl_val
                    if higher_better:
                        winner = "TimeClaw" if delta > 0.001 else ("Baseline" if delta < -0.001 else "Tie")
                    else:
                        winner = "TimeClaw" if delta < -0.001 else ("Baseline" if delta > 0.001 else "Tie")
                else:
                    winner = "—"
                    tc_val = tc_val if isinstance(tc_val, (int, float)) else "N/A"
                    bl_val = bl_val if isinstance(bl_val, (int, float)) else "N/A"

                tc_str = f"{tc_val:.4f}" if isinstance(tc_val, (int, float)) else str(tc_val)
                bl_str = f"{bl_val:.4f}" if isinstance(bl_val, (int, float)) else str(bl_val)
                lines.append(f"| {task} | {metric} | {tc_str} | {bl_str} | {winner} |")
        lines.append("")

    return "\n".join(lines)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full TimeClaw Benchmark")
    parser.add_argument("--concurrency", type=int, default=80,
                        help="Max concurrent pipeline runs")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing memory)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline evaluation")
    parser.add_argument("--skip-clean", action="store_true",
                        help="Skip cleaning (reuse existing state)")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                        help="Fraction of samples to keep per split (e.g. 0.2 for 20%%)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TimeClaw Full Benchmark Pipeline")
    logger.info("=" * 70)

    overall_t0 = time.time()

    # Phase 0: Clean
    if not args.skip_clean:
        logger.info("\n[Phase 0] Cleaning all learned state...")
        clean_all_state()
    else:
        logger.info("\n[Phase 0] Skipping clean (--skip-clean)")

    # Phase 1: Split
    logger.info(f"\n[Phase 1] Splitting datasets 70/30 by time (sample_ratio={args.sample_ratio:.0%})...")
    split_info = run_split(sample_ratio=args.sample_ratio)

    # Phase 2: Train + per-dataset test/baseline comparison
    if not args.skip_train:
        logger.info("\n[Phase 2] Training + per-dataset test/baseline comparison...")
        train_t0 = time.time()
        combined = await run_train_on_train_split(concurrency=args.concurrency)
        logger.info(f"[Phase 2] Completed in {time.time() - train_t0:.0f}s")

        tc_results = {}
        bl_results = {}
        for ds_name, tc_data in combined.get("tc_test", {}).items():
            tc_results[ds_name] = tc_data
        for ds_name, bl_data in combined.get("bl_test", {}).items():
            bl_results[ds_name] = bl_data
    else:
        logger.info("\n[Phase 2] Skipping training (--skip-train)")
        tc_results = {}
        bl_results = {}

    # Phase 3: Final report
    logger.info("\n[Phase 3] Generating comparison report...")
    report = generate_comparison_report(tc_results, bl_results, split_info)
    report_path = Path("benchmark_report.md")
    report_path.write_text(report)
    logger.info(f"[Phase 3] Report saved to {report_path}")

    results_path = Path(".timeclaw/test_results/full_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "split": split_info,
        "timeclaw": tc_results,
        "baseline": bl_results,
        "total_elapsed_s": round(time.time() - overall_t0, 1),
    }, indent=2, default=str))

    total = time.time() - overall_t0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Full benchmark completed in {total:.0f}s ({total/60:.1f}min)")
    logger.info(f"Report: {report_path.absolute()}")
    logger.info(f"{'=' * 70}")

    print(f"\n{report}")


if __name__ == "__main__":
    asyncio.run(main())
