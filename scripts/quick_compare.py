"""Quick comparison: TimeClaw vs Baseline vs LLM-Reasoning on 3 samples per task per dataset.

No training, no memory — raw capability comparison across three approaches:
  1. Baseline:      Direct LLM with MTBench official prompts (no tools)
  2. TimeClaw:      Full agent pipeline with ReAct loop and tools
  3. LLM-Reasoning: llm_reasoning tool only (raw data → LLM structured answer)
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TIMECLAW_CONFIG", "configs/gpt51.yaml")

import numpy as np
from loguru import logger

SAMPLES_PER_TASK = 3


async def run_comparison():
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
    from timeclaw.benchmark.pipeline_eval import run_pipeline_eval
    from timeclaw.benchmark.baseline_eval import run_baseline, run_llm_reasoning_baseline

    all_datasets = list(DATASET_REGISTRY.keys())
    all_tasks_map = {ds: info["tasks"] for ds, info in DATASET_REGISTRY.items()}

    print("=" * 80)
    print(f"Quick Compare: Baseline vs TimeClaw vs LLM-Reasoning")
    print(f"Samples per task: {SAMPLES_PER_TASK}")
    print(f"Datasets: {len(all_datasets)}")
    print("=" * 80)

    # Phase 1: Baseline
    print("\n[Phase 1] Running Baseline (direct LLM, MTBench prompts)...")
    t0 = time.time()
    baseline_results = await run_baseline(
        max_samples=SAMPLES_PER_TASK,
        concurrency=50,
        full_dataset=True,
    )
    baseline_time = time.time() - t0
    print(f"  Baseline done in {baseline_time:.1f}s")

    # Phase 2: LLM-Reasoning only
    print("\n[Phase 2] Running LLM-Reasoning baseline (llm_reasoning tool only)...")
    t0 = time.time()
    llmr_results = await run_llm_reasoning_baseline(
        max_samples=SAMPLES_PER_TASK,
        concurrency=50,
        full_dataset=True,
    )
    llmr_time = time.time() - t0
    print(f"  LLM-Reasoning done in {llmr_time:.1f}s")

    # Phase 3: TimeClaw
    print("\n[Phase 3] Running TimeClaw (agent pipeline, no memory)...")
    t0 = time.time()
    timeclaw_results = await run_pipeline_eval(
        max_samples=SAMPLES_PER_TASK,
        use_skills=False,
        use_strategy=False,
        label="timeclaw_raw",
        concurrency=20,
        full_dataset=True,
        verbose=False,
    )
    timeclaw_time = time.time() - t0
    print(f"  TimeClaw done in {timeclaw_time:.1f}s")

    # Phase 4: Report
    print("\n" + "=" * 80)
    print("COMPARISON REPORT (3-way)")
    print("=" * 80)

    tc_data = timeclaw_results.get("results", {})
    bl_data = baseline_results
    lr_data = llmr_results

    report_lines = []
    report_lines.append("# Baseline vs TimeClaw vs LLM-Reasoning Quick Comparison")
    report_lines.append(f"\nSamples per task: {SAMPLES_PER_TASK}")
    report_lines.append(f"Baseline time: {baseline_time:.1f}s")
    report_lines.append(f"LLM-Reasoning time: {llmr_time:.1f}s")
    report_lines.append(f"TimeClaw time: {timeclaw_time:.1f}s")
    report_lines.append("")

    for ds_name in all_datasets:
        tc_ds = tc_data.get(ds_name, {})
        bl_ds = bl_data.get(ds_name, {})
        lr_ds = lr_data.get(ds_name, {})
        tc_summary = tc_ds.get("summary", {})
        bl_summary = bl_ds.get("summary", {})
        lr_summary = lr_ds.get("summary", {})

        if not tc_summary and not bl_summary and not lr_summary:
            continue

        tasks = all_tasks_map.get(ds_name, [])
        report_lines.append(f"\n## {ds_name}")
        report_lines.append("| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |")
        report_lines.append("|------|--------|----------|---------------|----------|------|")

        print(f"\n--- {ds_name} ---")

        for task in tasks:
            tc_task = tc_summary.get(task, {})
            bl_task = bl_summary.get(task, {})
            lr_task = lr_summary.get(task, {})

            if not tc_task and not bl_task and not lr_task:
                continue

            all_metrics = set()
            for src in [tc_task, bl_task, lr_task]:
                for k in src:
                    if k.startswith("avg_") and k != "avg_n":
                        all_metrics.add(k)

            for metric in sorted(all_metrics):
                tc_val = tc_task.get(metric)
                bl_val = bl_task.get(metric)
                lr_val = lr_task.get(metric)

                tc_str = f"{tc_val:.4f}" if tc_val is not None else "N/A"
                bl_str = f"{bl_val:.4f}" if bl_val is not None else "N/A"
                lr_str = f"{lr_val:.4f}" if lr_val is not None else "N/A"

                metric_name = metric.replace("avg_", "")
                higher_better = metric_name in ("accuracy",)

                vals = {}
                if tc_val is not None:
                    vals["TimeClaw"] = tc_val
                if bl_val is not None:
                    vals["Baseline"] = bl_val
                if lr_val is not None:
                    vals["LLM-R"] = lr_val

                best = ""
                if len(vals) >= 2:
                    if higher_better:
                        best = max(vals, key=vals.get)
                    else:
                        best = min(vals, key=vals.get)

                print(
                    f"  {task:12s} {metric:20s}  "
                    f"BL={bl_str:>10s}  LR={lr_str:>10s}  TC={tc_str:>10s}  {best}"
                )
                report_lines.append(
                    f"| {task} | {metric_name} | {bl_str} | {lr_str} | {tc_str} | {best} |"
                )

    report_text = "\n".join(report_lines)
    report_path = "reports/03_quick_comparison.md"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline:      {baseline_time:.1f}s")
    print(f"LLM-Reasoning: {llmr_time:.1f}s")
    print(f"TimeClaw:      {timeclaw_time:.1f}s")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    asyncio.run(run_comparison())
