"""Run ONE sample per task type through TimeClaw pipeline with full verbose logging.

Task types: forecast, trend, indicator, mcqa, correlation
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY, load_dataset, MTBenchSample
from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner

TASK_TO_DATASET: dict[str, str] = {
    "forecast":    "weather_aligned_short",
    "trend":       "weather_aligned_short",
    "indicator":   "finance_aligned_short",
    "mcqa":        "weather_qa_short",
    "correlation": "finance_qa_short",
}

TASK_TO_PIPELINE_TYPE: dict[str, str] = {
    "forecast":    "forecast",
    "trend":       "classification",
    "indicator":   "forecast",
    "mcqa":        "qa",
    "correlation": "qa",
}

BANNER = "=" * 80


def _format_sample_info(sample: MTBenchSample, task: str) -> str:
    lines = [
        f"  文件: {sample.file}",
        f"  数据集: {sample.dataset}",
        f"  领域: {sample.domain}",
        f"  输入形状: {sample.input_window.shape}",
    ]
    if sample.output_window is not None:
        lines.append(f"  输出形状: {sample.output_window.shape}")
    if sample.text:
        lines.append(f"  文本: {sample.text[:150]}...")
    if sample.question:
        lines.append(f"  问题: {sample.question[:200]}")
    if sample.answer:
        lines.append(f"  标准答案: {sample.answer}")
    if sample.labels:
        lines.append(f"  标签: {sample.labels}")
    return "\n".join(lines)


def _format_result(result: dict[str, Any], task: str) -> str:
    lines = []

    pred = result.get("prediction")
    if isinstance(pred, np.ndarray):
        if pred.size <= 20:
            lines.append(f"  预测值: {pred.tolist()}")
        else:
            lines.append(f"  预测值: shape={pred.shape}, mean={pred.mean():.4f}, range=[{pred.min():.4f}, {pred.max():.4f}]")
    elif isinstance(pred, (list, tuple)):
        if len(pred) <= 20:
            lines.append(f"  预测值: {pred}")
        else:
            lines.append(f"  预测值: [{pred[0]}, ..., {pred[-1]}] ({len(pred)} items)")
    elif isinstance(pred, dict):
        lines.append(f"  预测值: {pred}")
    else:
        lines.append(f"  预测值: {pred}")

    lines.append(f"  总耗时: {result.get('total_ms', 0)}ms")
    lines.append(f"  质量门控: {'通过' if result.get('gate_accepted') else '未通过'}")
    lines.append(f"  模态: {result.get('modalities', [])}")

    tl = result.get("timeline", {})
    for stage_name in ["plan", "diagnose", "route", "execute", "verify", "deliver"]:
        stage = tl.get(stage_name, {})
        ms = stage.get("elapsed_ms", 0)
        extra = ""
        if stage_name == "plan":
            extra = f" | 意图: {stage.get('intent', {}).get('task_type', '?')}"
        elif stage_name == "diagnose":
            extra = f" | 策略: {stage.get('suggested_strategy', '?')}, 模态: {stage.get('modalities', [])}"
        elif stage_name == "route":
            extra = f" | DAG: {' → '.join(stage.get('dag_nodes', []))}"
        elif stage_name == "execute":
            extra = f" | 分支: {stage.get('executor_branches', 0)}, 有预测: {stage.get('has_prediction', False)}"
        elif stage_name == "verify":
            extra = f" | 接受: {stage.get('accepted', False)}, 重分析: {stage.get('reanalysis_rounds', 0)}"
        lines.append(f"  {stage_name:>10}: {ms:>6}ms{extra}")

    if result.get("evidence_summary"):
        lines.append(f"  证据摘要: {str(result['evidence_summary'])[:300]}")

    if result.get("fusion_report"):
        lines.append(f"  融合报告: {str(result['fusion_report'])[:300]}")

    return "\n".join(lines)


async def run_one_task(pipeline: PipelineRunner, task: str, sample: MTBenchSample) -> dict[str, Any]:
    horizon = len(sample.output_window) if sample.output_window is not None else 24

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    pipeline_task = TASK_TO_PIPELINE_TYPE[task]
    if task in ("mcqa", "correlation"):
        desc = sample.question[:200] if sample.question else f"{sample.domain} {task}"
    else:
        desc = f"{sample.domain} {task}"

    result = await pipeline.run(
        data=sample.input_window,
        task_type=pipeline_task,
        description=desc,
        horizon=horizon if task in ("forecast", "indicator") else None,
        text_data=text_data,
        verbose=True,
    )
    return result


async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | {message}",
        level="DEBUG",
    )

    cfg = TimeClawConfig()
    pipeline = PipelineRunner(cfg, disable_memory=True)

    tasks = ["forecast", "trend", "indicator", "mcqa", "correlation"]

    for task in tasks:
        ds_name = TASK_TO_DATASET[task]
        logger.info(f"\n{BANNER}")
        logger.info(f"  加载数据集: {ds_name} (任务: {task})")
        logger.info(BANNER)

        samples = load_dataset(ds_name)
        if not samples:
            logger.warning(f"数据集 {ds_name} 为空，跳过")
            continue

        sample = samples[0]

        logger.info(f"\n{'─' * 60}")
        logger.info(f"  任务类型: {task.upper()}")
        logger.info(f"  样本信息:")
        logger.info(f"\n{_format_sample_info(sample, task)}")
        logger.info(f"{'─' * 60}\n")

        t0 = time.time()
        try:
            result = await run_one_task(pipeline, task, sample)
            elapsed = time.time() - t0

            logger.info(f"\n{'─' * 60}")
            logger.info(f"  [{task.upper()}] 运行完成 ({elapsed:.1f}s)")
            logger.info(f"\n{_format_result(result, task)}")
            logger.info(f"{'─' * 60}\n")

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"\n  [{task.upper()}] 运行失败 ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{BANNER}")
    logger.info(f"  全部 {len(tasks)} 种任务运行完毕")
    logger.info(BANNER)


if __name__ == "__main__":
    asyncio.run(main())
