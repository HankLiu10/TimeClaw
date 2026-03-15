"""Run ONE sample per task type with FULL LLM prompt/response dump.

Outputs complete LLM interaction details to see exactly how TimeClaw thinks.
Uses the same evaluation flow as pipeline_eval.py (including _final_llm_call
self-correction) so that the logged output matches the real evaluation path.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY, load_dataset, MTBenchSample
from timeclaw.benchmark.pipeline_eval import EVAL_HANDLERS
from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner

TASK_CONFIGS: list[dict[str, str]] = [
    {"task": "forecast",    "dataset": "weather_aligned_short", "eval_key": "forecast"},
    {"task": "trend",       "dataset": "weather_aligned_short", "eval_key": "trend"},
    {"task": "indicator",   "dataset": "finance_aligned_short", "eval_key": "indicator"},
    {"task": "mcqa",        "dataset": "weather_qa_short",      "eval_key": "mcqa"},
    {"task": "correlation", "dataset": "finance_qa_short",      "eval_key": "correlation"},
]


def _format_gt(sample: MTBenchSample, task: str) -> str:
    """Format ground truth for display."""
    parts = []
    if sample.output_window is not None:
        parts.append(f"output_window shape={sample.output_window.shape}")
    if sample.answer:
        parts.append(f"answer={sample.answer}")
    if sample.labels:
        parts.append(f"labels={sample.labels}")
    return ", ".join(parts) if parts else "(无)"


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None,
                        help="Run only this task type (forecast/trend/indicator/mcqa/correlation). Default: all")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path. Default: stdout")
    args = parser.parse_args()

    sink = sys.stderr
    if args.output:
        sink = open(args.output, "w", encoding="utf-8")

    logger.remove()
    logger.add(
        sink,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | {message}",
        level="DEBUG",
    )

    cfg = TimeClawConfig()
    pipeline = PipelineRunner(cfg, disable_memory=True)

    if args.task:
        configs = [c for c in TASK_CONFIGS if c["task"] == args.task]
        if not configs:
            logger.error(f"Unknown task: {args.task}. Choose from: {[c['task'] for c in TASK_CONFIGS]}")
            return
    else:
        configs = TASK_CONFIGS

    for task_cfg in configs:
        task = task_cfg["task"]
        ds_name = task_cfg["dataset"]
        eval_key = task_cfg["eval_key"]

        logger.info(f"\n{'=' * 80}")
        logger.info(f"  任务: {task.upper()} | 数据集: {ds_name}")
        logger.info(f"{'=' * 80}")

        samples = load_dataset(ds_name)
        if not samples:
            logger.warning(f"数据集 {ds_name} 为空")
            continue

        sample = samples[0]
        logger.info(f"  样本: {sample.file}")
        logger.info(f"  输入形状: {sample.input_window.shape}")
        if sample.output_window is not None:
            logger.info(f"  输出形状: {sample.output_window.shape}")
        if sample.question:
            logger.info(f"  问题: {sample.question[:300]}")
        if sample.labels:
            logger.info(f"  标签: {sample.labels}")
        logger.info(f"  Ground Truth: {_format_gt(sample, task)}")

        handler = EVAL_HANDLERS.get(eval_key)
        if handler is None:
            logger.error(f"  未找到评估处理器: {eval_key}")
            continue

        from timeclaw.core import trace_logger as trace
        trace.configure(verbose=True, full_dump=True)

        t0 = time.time()
        try:
            eval_result = await handler(sample, pipeline, verbose=True)
            elapsed = time.time() - t0

            logger.info(f"\n{'─' * 60}")
            logger.info(f"  [{task.upper()}] 评估完成 ({elapsed:.1f}s)")

            metrics = eval_result.get("metrics", {})
            predicted = eval_result.get("predicted", metrics.get("error", "N/A"))
            ground_truth = eval_result.get("ground_truth", "N/A")

            logger.info(f"  预测结果: {str(predicted)[:300]}")
            logger.info(f"  Ground Truth: {str(ground_truth)[:300]}")
            logger.info(f"  评估指标: {json.dumps(metrics, default=str)}")

            if "accuracy" in metrics:
                correct = metrics["accuracy"] == 1.0
                logger.info(f"  正确性: {'✅ 正确' if correct else '❌ 错误'}")
            elif "error" in metrics:
                logger.info(f"  错误: {metrics['error']}")

            logger.info(f"{'─' * 60}")
        except Exception as e:
            logger.error(f"  [{task.upper()}] 失败: {e}")
            import traceback
            traceback.print_exc()

    if args.output and sink is not sys.stderr:
        sink.close()
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.info(f"完整日志已保存到: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
