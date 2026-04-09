#!/usr/bin/env python3
"""小规模实验编排（平行实现）：不修改 pipeline_eval / baseline_eval 等库代码。

流程与 docs/SMALL_SCALE_100_SAMPLES.md 一致：
  1) 读或生成 data/test_indices_100.json（按 test split 下标固定 100 条）
  2) 按该索引跑 baseline → 写入 .timeclaw/eval_results 同 baseline_eval 的文件名
  3) run_train_loop(max_samples=100)
  4) 训练后 force_distill
  5) 按同一索引跑 pipeline 测试，label=trained
  6) 可选：打印 baseline vs trained 摘要对比

评测循环在脚本内自行组装，仅复用各模块已有的内部函数（不改动其源码）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# ── 常量（可按需改）──────────────────────────────────────────────────
DEFAULT_INDICES_PATH = Path("data/test_indices_100.json")
TRAIN_MAX = 100
TEST_MAX = 100


def _stratified_sample_indices(
    samples: list[Any],
    n: int,
    *,
    seed: int = 42,
) -> list[int]:
    """按 start_year 分层，在 test split 内选 n 个下标（与「分层」文档语义一致）。"""
    if n <= 0 or not samples:
        return []
    n = min(n, len(samples))
    rng = np.random.default_rng(seed)
    by_year: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        y = int(getattr(s, "start_year", 0) or 0)
        by_year[y].append(i)
    years = sorted(by_year.keys())
    if len(years) <= 1:
        pick = rng.choice(len(samples), size=n, replace=False).tolist()
        return sorted(int(x) for x in pick)

    total = len(samples)
    picked: list[int] = []
    for yi, y in enumerate(years):
        pool = by_year[y]
        if yi == len(years) - 1:
            need = n - len(picked)
        else:
            need = int(round(n * len(pool) / total))
            need = max(0, min(need, len(pool), n - len(picked)))
        if need > 0:
            sub = rng.choice(pool, size=need, replace=False).tolist()
            picked.extend(int(x) for x in sub)

    picked_set = set(picked)
    if len(picked) < n:
        rest = [i for i in range(len(samples)) if i not in picked_set]
        rng.shuffle(rest)
        for i in rest:
            if len(picked) >= n:
                break
            picked.append(i)
    return picked[:n]


def freeze_test_indices(
    path: Path,
    n: int,
    datasets: list[str] | None,
    *,
    config: Any | None = None,
) -> dict[str, list[int]]:
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        logger.info(f"[SmallScale] 使用已有索引文件: {path}")
        return data

    from timeclaw.benchmark.dataset_split import load_split
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
    from timeclaw.core.config import TimeClawConfig

    cfg = config or TimeClawConfig.load()
    names = datasets if datasets else list(DATASET_REGISTRY.keys())
    out: dict[str, list[int]] = {}
    for ds in names:
        if ds not in DATASET_REGISTRY:
            continue
        try:
            samples = load_split(ds, "test", config=cfg)
        except FileNotFoundError:
            logger.warning(f"[SmallScale] 无 test split: {ds}")
            continue
        k = min(n, len(samples))
        out[ds] = _stratified_sample_indices(samples, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info(f"[SmallScale] 已写入新索引文件: {path}")
    return out


def _subset_by_indices(samples: list[Any], indices: list[int]) -> list[Any]:
    return [samples[i] for i in indices if 0 <= i < len(samples)]


async def phase_baseline_indexed(
    index_map: dict[str, list[int]],
    *,
    task_filter: str | None,
    concurrency: int,
    config: Any,
) -> dict[str, Any]:
    from timeclaw.benchmark.baseline_eval import (
        _eval_one_sample,
        _save_results,
        _summarize,
    )
    from timeclaw.benchmark.dataset_split import load_split
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
    from timeclaw.benchmark.mtbench_support import expand_dataset_tasks
    from timeclaw.core.config import TimeClawConfig
    from timeclaw.llm.router import ModelRouter

    cfg = config or TimeClawConfig.load()
    router = ModelRouter(cfg.llm)
    llm = router.get_backend("executor")
    sem = asyncio.Semaphore(concurrency)
    all_results: dict[str, Any] = {}

    for ds_name in index_map:
        info = DATASET_REGISTRY.get(ds_name)
        if not info:
            continue
        try:
            full = load_split(ds_name, "test", config=cfg)
        except FileNotFoundError:
            logger.warning(f"[SmallScale:Baseline] 无数据 {ds_name}")
            continue
        samples = _subset_by_indices(full, index_map[ds_name])
        if not samples:
            continue
        tasks = expand_dataset_tasks(ds_name, list(info["tasks"]))
        if task_filter:
            tasks = [t for t in tasks if t == task_filter]
        if not tasks:
            continue

        logger.info(
            f"[SmallScale:Baseline] {ds_name}: {len(samples)} 样本 tasks={tasks} "
            f"concurrency={concurrency}"
        )
        t0 = time.time()
        coros = [
            _eval_one_sample(ds_name, idx, s, tasks, llm, sem)
            for idx, s in enumerate(samples)
        ]
        batch = await asyncio.gather(*coros, return_exceptions=True)
        ds_results: list[dict[str, Any]] = []
        for b in batch:
            if isinstance(b, Exception):
                logger.error(f"[SmallScale:Baseline] {ds_name}: {b}")
                continue
            ds_results.extend(b)
        elapsed = time.time() - t0
        _save_results(ds_results, ds_name)
        summary = _summarize(ds_results)
        all_results[ds_name] = {
            "total": len(ds_results),
            "elapsed_s": round(elapsed, 1),
            "summary": summary,
        }
        logger.info(
            f"[SmallScale:Baseline] {ds_name} done {elapsed:.1f}s: {json.dumps(summary, default=str)}"
        )

    return all_results


async def phase_pipeline_indexed(
    index_map: dict[str, list[int]],
    *,
    task_filter: str | None,
    label: str,
    use_skills: bool,
    use_strategy: bool,
    verbose: bool,
    concurrency: int,
    per_task_concurrency: int | None,
    indicator_concurrency: int | None,
    config: Any,
    merge_checkpoint: bool = True,
) -> dict[str, Any]:
    from timeclaw.benchmark.dataset_split import load_split
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
    from timeclaw.benchmark.mtbench_support import expand_dataset_tasks
    from timeclaw.benchmark import pipeline_eval as pe
    from timeclaw.core.config import TimeClawConfig
    from timeclaw.core import trace_logger as trace
    from timeclaw.pipeline.runner import PipelineRunner

    cfg = config or TimeClawConfig.load()
    cfg.mcp.enabled = False
    trace.configure(verbose)
    pe._pipeline_ref = None
    pe._record_enabled = False

    disable_mem = not use_skills and not use_strategy
    pipeline: PipelineRunner | None = None
    try:
        pipeline = PipelineRunner(cfg, disable_memory=disable_mem)
        pe._pipeline_ref = pipeline
        pe._record_enabled = False
        try:
            from timeclaw.sbm.registry import auto_register, get_global_registry

            reg = get_global_registry()
            if not reg.list_available():
                auto_register()
            for name in reg.list_available():
                adapter = reg.get(name)
                if not adapter.is_loaded:
                    try:
                        adapter.load()
                    except Exception as e:
                        logger.warning(f"[SmallScale:Pipeline] 预加载 {name} 失败: {e}")
        except Exception as e:
            logger.warning(f"[SmallScale:Pipeline] 模型预加载跳过: {e}")

        sem = asyncio.Semaphore(concurrency)
        all_results: dict[str, Any] = {}
        task_sems: dict[tuple[str, str], asyncio.Semaphore] = {}

        def _get_task_sem(ds_name: str, task_name: str) -> asyncio.Semaphore | None:
            if per_task_concurrency is None and indicator_concurrency is None:
                return None
            key = (ds_name, task_name)
            if key in task_sems:
                return task_sems[key]
            limit = (
                indicator_concurrency
                if task_name.startswith("indicator") and indicator_concurrency is not None
                else per_task_concurrency
            )
            if limit is None:
                return None
            task_sems[key] = asyncio.Semaphore(limit)
            return task_sems[key]

        for ds_name in index_map:
            info = DATASET_REGISTRY.get(ds_name)
            if not info:
                continue
            try:
                full = load_split(ds_name, "test", config=cfg)
            except FileNotFoundError:
                logger.warning(f"[SmallScale:Pipeline] 无数据 {ds_name}")
                continue
            samples = _subset_by_indices(full, index_map[ds_name])
            if not samples:
                continue
            supported = expand_dataset_tasks(ds_name, list(info["tasks"]))
            if task_filter:
                supported = [t for t in supported if t == task_filter]
            if not supported:
                continue

            existing = pe._load_eval_checkpoint(ds_name, label) if merge_checkpoint else []
            done = pe._eval_done_keys(existing)
            work_specs = [
                (idx, sample, task)
                for idx, sample in enumerate(samples)
                for task in supported
                if not merge_checkpoint or (idx, task) not in done
            ]

            logger.info(
                f"[SmallScale:Pipeline:{label}] {ds_name}: {len(samples)} 样本 "
                f"× {len(supported)} tasks ({len(work_specs)} pending), concurrency={concurrency}, "
                f"per_task_concurrency={per_task_concurrency}, "
                f"indicator_concurrency={indicator_concurrency}"
            )
            t0 = time.time()
            if work_specs:
                coros = [
                    pe._eval_one_task(
                        ds_name, idx, sample, task, pipeline, verbose, sem,
                        _get_task_sem(ds_name, task),
                    )
                    for idx, sample, task in work_specs
                ]
                flat = await asyncio.gather(*coros, return_exceptions=True)
                new_rows: list[dict[str, Any]] = []
                for r in flat:
                    if isinstance(r, Exception):
                        logger.error(f"[SmallScale:Pipeline] {ds_name}: {r}")
                        continue
                    new_rows.append(r)
                ds_results = pe._merge_eval_rows(existing, new_rows)
            else:
                ds_results = list(existing)
            elapsed = time.time() - t0
            pe._save_results(ds_results, ds_name, label)
            summary = pe._summarize(ds_results)
            all_results[ds_name] = {
                "total": len(ds_results),
                "elapsed_s": round(elapsed, 1),
                "summary": summary,
            }
            logger.info(
                f"[SmallScale:Pipeline:{label}] {ds_name} done {elapsed:.1f}s: "
                f"{json.dumps(summary, default=str)}"
            )

        return {"label": label, "results": all_results}
    finally:
        pe._pipeline_ref = None
        pe._record_enabled = False
        if pipeline is not None:
            try:
                await pipeline.shutdown()
            except Exception:
                pass


async def phase_distill(config: Any = None) -> None:
    from timeclaw.core.config import TimeClawConfig
    from timeclaw.pipeline.runner import PipelineRunner

    cfg = config or TimeClawConfig.load()
    cfg.mcp.enabled = False
    p = PipelineRunner(cfg, disable_memory=False)
    try:
        await p.memory.force_distill()
        logger.info(f"[SmallScale] distill 完成: {p.memory.summary()}")
    finally:
        await p.shutdown()


def print_compare(
    index_map: dict[str, list[int]],
    *,
    trained_label: str = "trained",
) -> None:
    from timeclaw.benchmark.baseline_eval import RESULTS_DIR as BL_DIR
    from timeclaw.benchmark.pipeline_eval import RESULTS_DIR as EV_DIR

    bl_dir = Path(BL_DIR)
    ev_dir = Path(EV_DIR)
    print("\n=== Small-scale: baseline vs pipeline ===\n")
    for ds in sorted(index_map.keys()):
        bp = bl_dir / f"baseline_{ds}.json"
        tp = ev_dir / f"{trained_label}_{ds}.json"
        if not bp.exists():
            print(f"{ds}: 缺少 baseline 结果 {bp}")
            continue
        if not tp.exists():
            print(f"{ds}: 缺少 pipeline 结果 {tp}")
            continue
        bj = json.loads(bp.read_text())
        tj = json.loads(tp.read_text())
        from timeclaw.benchmark.baseline_eval import _summarize as bl_sum
        from timeclaw.benchmark.pipeline_eval import _summarize as ev_sum

        bs = bl_sum(bj)
        ts = ev_sum(tj)
        print(f"--- {ds} ---")
        print("  baseline:", json.dumps(bs, default=str)[:500])
        print("  trained: ", json.dumps(ts, default=str)[:500])
        print()


def has_existing_baseline(index_map: dict[str, list[int]]) -> bool:
    """Return True iff all datasets in the current indexed batch already have baseline files.

    New definition:
        ``existing baseline`` means the current batch's datasets already have
        ``baseline_{dataset}.json`` under ``.timeclaw/test_results/`` and can
        be reused directly instead of recomputing the baseline phase.
    """
    from timeclaw.benchmark.baseline_eval import RESULTS_DIR as BL_DIR

    bl_dir = Path(BL_DIR)
    if not index_map:
        return False
    return all((bl_dir / f"baseline_{ds}.json").exists() for ds in index_map)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="小规模实验（平行编排，不改库）")
    parser.add_argument(
        "--indices-path",
        type=Path,
        default=DEFAULT_INDICES_PATH,
        help="固定 test 子集下标 JSON",
    )
    parser.add_argument("--test-n", type=int, default=TEST_MAX)
    parser.add_argument("--train-max", type=int, default=TRAIN_MAX)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--train-concurrency", type=int, default=8)
    parser.add_argument("--eval-concurrency", type=int, default=80)
    parser.add_argument("--per-task-concurrency", type=int, default=None)
    parser.add_argument("--indicator-concurrency", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "freeze", "baseline", "train", "distill", "test", "compare"),
        default="all",
        help="单步或全流程",
    )
    parser.add_argument("--pipeline-label", type=str, default="trained")
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="即使已有 baseline 文件，也强制重跑 baseline",
    )
    parser.add_argument(
        "--no-merge-checkpoint",
        action="store_true",
        help="训练/测试阶段不合并已有 checkpoint，从头写（训练仍会跳过已有 train_*.json 若未传此标志时的旧逻辑见 merge_checkpoint=False）",
    )
    args = parser.parse_args()
    merge_ckpt = not args.no_merge_checkpoint

    phases = (
        {"freeze", "baseline", "train", "distill", "test", "compare"}
        if args.phase == "all"
        else {args.phase}
    )

    from timeclaw.core.config import TimeClawConfig

    cfg = TimeClawConfig.load()
    index_map: dict[str, list[int]] = {}
    if phases & {"freeze", "baseline", "test"}:
        index_map = freeze_test_indices(
            args.indices_path, args.test_n, None, config=cfg,
        )
    elif "compare" in phases:
        ip = Path(args.indices_path)
        if ip.exists():
            index_map = json.loads(ip.read_text())

    if "baseline" in phases:
        if args.force_baseline or not has_existing_baseline(index_map):
            await phase_baseline_indexed(
                index_map,
                task_filter=args.task,
                concurrency=args.eval_concurrency,
                config=cfg,
            )
        else:
            logger.info("[SmallScale] 复用已有 baseline，跳过 baseline 阶段")

    if "train" in phases:
        from timeclaw.benchmark.train_loop import run_train_loop

        await run_train_loop(
            dataset_name=None,
            task_filter=args.task,
            config=cfg,
            max_samples=args.train_max,
            resume_from=0,
            verbose=args.verbose,
            concurrency=args.train_concurrency,
            per_task_concurrency=args.per_task_concurrency,
            indicator_concurrency=args.indicator_concurrency,
            merge_checkpoint=merge_ckpt,
        )

    if "distill" in phases:
        await phase_distill(cfg)

    if "test" in phases:
        await phase_pipeline_indexed(
            index_map,
            task_filter=args.task,
            label=args.pipeline_label,
            use_skills=not args.no_skills,
            use_strategy=True,
            verbose=args.verbose,
            concurrency=args.eval_concurrency,
            per_task_concurrency=args.per_task_concurrency,
            indicator_concurrency=args.indicator_concurrency,
            config=cfg,
            merge_checkpoint=merge_ckpt,
        )

    if "compare" in phases and index_map:
        print_compare(index_map, trained_label=args.pipeline_label)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
