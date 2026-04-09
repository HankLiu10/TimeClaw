"""Unified entry point for the full MTBench learning pipeline.

Usage examples:
    # Download all datasets
    python -m timeclaw.benchmark.run_learning --download

    # Full pipeline on all datasets
    python -m timeclaw.benchmark.run_learning --all

    # Regenerate official MTBench sample-list JSONs only (data/splits/)
    python -m timeclaw.benchmark.run_learning --split

    # Train on specific domain/task
    python -m timeclaw.benchmark.run_learning --train --domain weather --task forecast

    # Distill memory from training data
    python -m timeclaw.benchmark.run_learning --distill

    # Test with/without memory comparison
    python -m timeclaw.benchmark.run_learning --test --compare

    # Full pipeline for a single dataset
    python -m timeclaw.benchmark.run_learning --all --dataset weather_qa_long --max-samples 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from loguru import logger

from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY


def _filter_datasets(
    dataset: str | None, domain: str | None,
) -> list[str]:
    """Return list of dataset names matching filters."""
    if dataset:
        if dataset not in DATASET_REGISTRY:
            logger.error(f"Unknown dataset: {dataset}")
            logger.info(f"Available: {list(DATASET_REGISTRY.keys())}")
            sys.exit(1)
        return [dataset]

    names = list(DATASET_REGISTRY.keys())
    if domain:
        names = [n for n in names if DATASET_REGISTRY[n]["domain"] == domain]
    return names


def cmd_download(args: argparse.Namespace) -> None:
    from scripts.download_mtbench import download_all
    download_all()


def cmd_split(args: argparse.Namespace) -> None:
    from timeclaw.benchmark.dataset_split import split_all_datasets
    from timeclaw.core.config import TimeClawConfig

    results = split_all_datasets(config=TimeClawConfig.load())
    total_train = sum(r["train"] for r in results.values())
    total_test = sum(r["test"] for r in results.values())
    logger.info(f"Split done: {total_train} train, {total_test} test across {len(results)} datasets")
    for name, r in results.items():
        print(f"  {name}: train={r['train']}, test={r['test']}, split_year={r['split_year']}")


async def cmd_train(args: argparse.Namespace) -> None:
    from timeclaw.benchmark.train_loop import run_train_loop

    result = await run_train_loop(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        sample_indices=args.sample_index_file,
        verbose=args.verbose,
        concurrency=args.concurrency,
        per_task_concurrency=args.per_task_concurrency,
        indicator_concurrency=args.indicator_concurrency,
        merge_checkpoint=not getattr(args, "no_merge_checkpoint", False),
    )

    out = Path(".timeclaw/train_results/train_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    logger.info(f"[RunLearning] Train summary saved to {out}")
    print(json.dumps(result, indent=2, default=str))


async def cmd_distill(args: argparse.Namespace) -> None:
    from timeclaw.memory.hierarchical import HierarchicalMemory
    mem = HierarchicalMemory(enabled=True)
    await mem.force_distill()
    result = mem.summary()
    logger.info(f"[RunLearning] Distillation complete: {result}")
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))


async def cmd_test(args: argparse.Namespace) -> None:
    from timeclaw.benchmark.pipeline_eval import run_pipeline_eval

    if args.compare:
        logger.info("[RunLearning] Compare: running with memory...")
        mc = not getattr(args, "no_merge_checkpoint", False)
        with_mem = await run_pipeline_eval(
            dataset_name=args.dataset, task_filter=args.task,
            max_samples=args.max_samples, verbose=args.verbose,
            sample_indices=args.sample_index_file,
            concurrency=args.concurrency,
            per_task_concurrency=args.per_task_concurrency,
            indicator_concurrency=args.indicator_concurrency,
            use_skills=True, label="with_memory", split="test",
            merge_checkpoint=mc,
        )
        logger.info("[RunLearning] Compare: running without memory...")
        without_mem = await run_pipeline_eval(
            dataset_name=args.dataset, task_filter=args.task,
            max_samples=args.max_samples, verbose=args.verbose,
            sample_indices=args.sample_index_file,
            concurrency=args.concurrency,
            per_task_concurrency=args.per_task_concurrency,
            indicator_concurrency=args.indicator_concurrency,
            use_skills=False, label="no_memory", split="test",
            merge_checkpoint=mc,
        )
        result = {"with_memory": with_mem, "no_memory": without_mem}
    else:
        result = await run_pipeline_eval(
            dataset_name=args.dataset,
            task_filter=args.task,
            max_samples=args.max_samples,
            sample_indices=args.sample_index_file,
            use_skills=not args.no_skills,
            verbose=args.verbose,
            label="no_memory" if args.no_skills else "with_memory",
            concurrency=args.concurrency,
            per_task_concurrency=args.per_task_concurrency,
            indicator_concurrency=args.indicator_concurrency,
            split="test",
            merge_checkpoint=not getattr(args, "no_merge_checkpoint", False),
        )

    out = Path(".timeclaw/test_results/test_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    logger.info(f"[RunLearning] Test summary saved to {out}")
    print(json.dumps(result, indent=2, default=str))


async def cmd_all(args: argparse.Namespace) -> None:
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("Step 1/4: Regenerating official sample-list JSONs (data/splits/)")
    logger.info("=" * 60)
    cmd_split(args)

    logger.info("=" * 60)
    logger.info("Step 2/4: Training")
    logger.info("=" * 60)
    await cmd_train(args)

    logger.info("=" * 60)
    logger.info("Step 3/4: Distilling memory")
    logger.info("=" * 60)
    await cmd_distill(args)

    logger.info("=" * 60)
    logger.info("Step 4/4: Testing")
    logger.info("=" * 60)
    await cmd_test(args)

    elapsed = time.time() - t0
    logger.info(f"Full pipeline completed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TimeClaw MTBench Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    steps = parser.add_argument_group("Pipeline steps")
    steps.add_argument("--all", action="store_true", help="Full pipeline: regenerate data/splits lists -> train -> distill -> test")
    steps.add_argument("--download", action="store_true", help="Download all MTBench datasets")
    steps.add_argument("--split", action="store_true", help="Regenerate data/splits train_files/test_files (time-partitioned lists on official processed data)")
    steps.add_argument("--train", action="store_true", help="Run training loop")
    steps.add_argument("--distill", action="store_true", help="Distill skills from experience")
    steps.add_argument("--test", action="store_true", help="Evaluate using official test_files.json lists")

    filters = parser.add_argument_group("Filters")
    filters.add_argument("--dataset", type=str, default=None,
                         help="Specific dataset name (e.g. weather_qa_long)")
    filters.add_argument("--domain", type=str, choices=["weather", "finance"], default=None,
                         help="Filter by domain")
    filters.add_argument("--task", type=str,
                         choices=[
                             "forecast",
                             "trend",
                             "trend_past",
                             "indicator",
                             "indicator_macd",
                             "indicator_upper_bb",
                             "mcqa",
                             "correlation",
                         ],
                         default=None, help="Filter by task type")

    options = parser.add_argument_group("Options")
    options.add_argument("--max-samples", type=int, default=None,
                         help="Max samples per dataset")
    options.add_argument("--sample-index-file", type=str, default=None,
                         help="JSON file mapping dataset names to indices into load_split output for that dataset")
    options.add_argument("--compare", action="store_true",
                         help="Compare with/without memory (test mode)")
    options.add_argument("--no-skills", action="store_true",
                         help="Disable memory during test")
    options.add_argument("--concurrency", type=int, default=8,
                         help="Max parallel samples")
    options.add_argument("--per-task-concurrency", type=int, default=None,
                         help="Max parallel samples per task within each dataset")
    options.add_argument("--indicator-concurrency", type=int, default=None,
                         help="Optional override for indicator-task concurrency")
    options.add_argument("--verbose", action="store_true",
                         help="Enable verbose trace logging")
    options.add_argument(
        "--no-merge-checkpoint",
        action="store_true",
        help="Train/test: do not merge existing JSON checkpoints (full re-eval per dataset).",
    )

    args = parser.parse_args()

    if not any([args.all, args.download, args.split, args.train, args.distill, args.test]):
        parser.print_help()
        sys.exit(0)

    if args.download:
        cmd_download(args)

    if args.split:
        cmd_split(args)

    if args.train:
        asyncio.run(cmd_train(args))

    if args.distill:
        asyncio.run(cmd_distill(args))

    if args.test:
        asyncio.run(cmd_test(args))

    if args.all:
        asyncio.run(cmd_all(args))


if __name__ == "__main__":
    main()
