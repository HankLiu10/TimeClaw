"""Run a full pipeline with real Qwen2.5:32b LLM and verbose=True."""

import asyncio
import sys

import numpy as np

sys.path.insert(0, ".")

from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner


async def main():
    config = TimeClawConfig.from_yaml("configs/qwen25_32b.yaml")
    config.pipeline.verbose = True
    config.reasoning.max_evidence_steps = 5

    runner = PipelineRunner(config)

    np.random.seed(42)
    t = np.arange(100)
    temperature = 20.0 + 5.0 * np.sin(2 * np.pi * t / 25) + np.random.randn(100) * 0.8

    print("=" * 72)
    print("  TimeClaw Real LLM Demo (Qwen2.5:32b, verbose=True)")
    print("  数据: 100天气温, 正弦周期=25天, 噪声=0.8")
    print("=" * 72)

    result = await runner.run(
        data=temperature,
        task_type="forecast",
        description="根据过去100天的气温数据，预测未来7天的气温走势",
        horizon=7,
        verbose=True,
    )

    print("\n" + "=" * 72)
    print("  最终结果")
    print("=" * 72)
    print(f"  trace_id:    {result['trace_id']}")
    print(f"  总耗时:      {result['total_ms']}ms")
    print(f"  模态:        {result['modalities']}")
    print(f"  质量门:      {'通过' if result['gate_accepted'] else '未通过'}")

    pred = result.get("prediction")
    if pred is not None:
        if isinstance(pred, dict) and "forecast" in pred:
            fc = pred["forecast"]
            print(f"  预测值:      {[round(v, 2) for v in fc[:10]]}")
            print(f"  反归一化:    {pred.get('denormalized', False)}")
        else:
            print(f"  预测值:      {str(pred)[:300]}")
    else:
        print("  预测值:      None (未产出预测)")

    tl = result["timeline"]
    print(f"\n  各阶段耗时:")
    for stage, info in tl.items():
        ms = info.get("elapsed_ms", "?")
        print(f"    {stage:12s}  {ms}ms")

    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
