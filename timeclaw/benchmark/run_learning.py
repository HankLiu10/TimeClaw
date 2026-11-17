"""Unified entry point for independent-corpus learning and data maintenance.

Usage examples:
    # Download all datasets
    python -m timeclaw.benchmark.run_learning --download

    # Full native learning lifecycle (same finalized path as --train)
    python -m timeclaw.benchmark.run_learning --all


    # Train on a specific independent-corpus domain/task
    python -m timeclaw.benchmark.run_learning --train --dataset weather_aligned_short --task forecast

    # Distill memory already collected by --train
    python -m timeclaw.benchmark.run_learning --distill

    # Explore and distill a single independent dataset
    python -m timeclaw.benchmark.run_learning --all --dataset weather_qa_long --max-samples 20

"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

from timeclaw.core.config import TimeClawConfig



def cmd_download(args: argparse.Namespace) -> None:
    from scripts.download_mtbench import download_all
    cfg = TimeClawConfig.load()
    download_all(datasets=[args.dataset] if args.dataset else None, repo_root=Path(cfg.benchmark.repo_root))




async def cmd_train(args: argparse.Namespace) -> None:
    from timeclaw.benchmark.train_loop import run_train_loop

    # Observe the original training entrypoint end to end, including text
    # preparation and background/final memory distillation outside run traces.
    from timeclaw.core.accounting import RequestLedger
    cfg = TimeClawConfig.load()
    ledger = RequestLedger(
        f"exploration:{args.dataset or 'all'}:{args.task or 'all'}",
        Path(cfg.workspace_dir) / "accounting",
    )
    try:
        with ledger.activate():
            result = await run_train_loop(
                dataset_name=args.dataset,
                task_filter=args.task,
                max_samples=args.max_samples,
                sample_indices=args.sample_index_file,
                verbose=args.verbose,
                full_dump=getattr(args, "full_dump", False),
                concurrency=args.concurrency,
                per_task_concurrency=args.per_task_concurrency,
                indicator_concurrency=args.indicator_concurrency,
                merge_checkpoint=not getattr(args, "no_merge_checkpoint", False),
                skip_failed_checkpoint_rows=getattr(
                    args, "skip_failed_checkpoint_rows", False,
                ),
                config=cfg,
            )
    finally:
        accounting = ledger.summary()
        accounting["measurement_scope"] = "whole_learning_job_including_startup_and_memory"
        accounting["job_wall_seconds"] = accounting.pop("online_seconds")
        ledger.path.with_suffix(".summary.json").write_text(
            json.dumps(accounting, ensure_ascii=False, indent=2)
        )
    # The train-loop owner writes and seals this file before returning. A
    # second serialization here would invalidate its finalized source lineage.
    out = Path(cfg.workspace_dir) / "train_results" / "train_summary.json"
    logger.info(f"[RunLearning] Train summary saved to {out}")
    print(json.dumps(result, indent=2, default=str))
    total = sum(int(info.get("total", 0)) for info in result.values())
    failed = sum(
        int((task_info.get("aligned_counts") or {}).get("execution_failed_n", 0))
        for info in result.values() for task_info in (info.get("summary") or {}).values()
    )
    if not total or failed:
        raise RuntimeError(
            f"Training did not complete successfully: saved_rows={total}, execution_failed={failed}. "
            "Outcomes and accounting were preserved; inspect train_results before resuming."
        )


async def cmd_distill(args: argparse.Namespace) -> None:
    from timeclaw.memory.hierarchical import HierarchicalMemory
    from timeclaw.benchmark.train_loop import _configure_workspace_roots
    from timeclaw.llm.router import ModelRouter
    cfg = TimeClawConfig.load()
    from timeclaw.benchmark.source_admission import begin_distillation_lineage, finish_training_lineage, workspace_access
    with workspace_access(cfg, write=True):
        begin_distillation_lineage(cfg)
        _configure_workspace_roots(cfg.workspace_dir)
        router = None
        mem = None
        finalized = False
        try:
            router = ModelRouter(cfg.llm)
            mem = HierarchicalMemory(enabled=True, reasoner_llm=router.get_backend("reasoner"))
            await mem.force_distill()
            result = mem.summary()
            finalized = True
            logger.info(f"[RunLearning] Distillation complete: {result}")
            print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        except BaseException:
            if mem is not None:
                await mem.cancel_pending_distillation()
            raise
        finally:
            try:
                if router is not None:
                    await router.shutdown()
            finally:
                finish_training_lineage(cfg, finalized=finalized)




async def cmd_all(args: argparse.Namespace) -> None:
    """Alias for native exploration with its own final distillation lifecycle."""
    await cmd_train(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TimeClaw independent-corpus learning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    steps = parser.add_argument_group("Pipeline steps")
    steps.add_argument(
        "--all",
        action="store_true",
        help="Full learning pipeline: independent-corpus train/explore -> distill",
    )
    steps.add_argument("--download", action="store_true", help="Download all MTBench datasets")
    steps.add_argument("--train", action="store_true", help="Run training loop on benchmark.training_root")
    steps.add_argument("--distill", action="store_true", help="Distill skills from experience")
    filters = parser.add_argument_group("Filters")
    filters.add_argument("--dataset", type=str, default=None,
                         help="Specific dataset name (e.g. weather_qa_long)")
    filters.add_argument("--task", type=str,
                         choices=[
                             "forecast",
                             "trend",
                             "trend_past",
                             "indicator", "indicator_macd",
                             "mcqa",
                             "correlation",
                         ],
                         default=None, help="Filter by task type")

    options = parser.add_argument_group("Options")
    options.add_argument("--max-samples", type=int, default=None,
                         help="Max samples per dataset")
    options.add_argument(
        "--workspace-dir",
        type=str,
        default=None,
        help="Override the configured artifact root; otherwise honor TIMECLAW_WORKSPACE_DIR and config.workspace_dir.",
    )
    options.add_argument("--sample-index-file", type=str, default=None,
                         help="JSON file mapping dataset names to indices into the independent training corpus for that dataset")
    options.add_argument("--concurrency", type=int, default=8,
                         help="Max parallel samples")
    options.add_argument("--per-task-concurrency", type=int, default=None,
                         help="Max parallel samples per task within each dataset")
    options.add_argument("--indicator-concurrency", type=int, default=None,
                         help="Optional override for indicator-task concurrency")
    options.add_argument("--verbose", action="store_true",
                         help="Enable verbose trace logging")
    options.add_argument(
        "--full-dump",
        action="store_true",
        help="Persist full-fidelity LLM prompts/responses alongside compact traces.",
    )
    options.add_argument(
        "--no-merge-checkpoint",
        action="store_true",
        help="Learning: do not merge existing JSON checkpoints (rerun the requested training samples).",
    )
    options.add_argument(
        "--skip-failed-checkpoint-rows",
        action="store_true",
        help="Keep failed training rows and continue with the remaining samples.",
    )
    args = parser.parse_args()
    if args.all and (args.train or args.distill):
        parser.error("--all already includes training and final distillation; do not combine it with --train or --distill")
    if args.workspace_dir is not None:
        os.environ["TIMECLAW_WORKSPACE_DIR"] = args.workspace_dir

    if not any([args.all, args.download, args.train, args.distill]):
        parser.print_help()
        sys.exit(0)

    if args.download:
        cmd_download(args)

    if args.train:
        asyncio.run(cmd_train(args))

    if args.distill:
        asyncio.run(cmd_distill(args))

    if args.all:
        asyncio.run(cmd_all(args))


if __name__ == "__main__":
    main()
