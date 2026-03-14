"""Run a multimodal pipeline (timeseries + text) with real Qwen2.5:32b."""

import asyncio
import sys

import numpy as np

sys.path.insert(0, ".")

from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner


async def main():
    config = TimeClawConfig.from_yaml("configs/gpt5.yaml")
    config.pipeline.verbose = True
    config.reasoning.max_evidence_steps = 5

    runner = PipelineRunner(config)

    np.random.seed(42)
    t = np.arange(60)
    # 前40天平稳在22度左右，后20天因寒潮骤降到15度
    temperature = np.where(
        t < 40,
        22.0 + np.sin(2 * np.pi * t / 10) * 2 + np.random.randn(60) * 0.5,
        15.0 + np.random.randn(60) * 0.8,
    )

    text_data = [
        {
            "source": "weather_news",
            "timestamp": "day_35",
            "content": "气象局预报：未来一周气温将保持稳定，日均温度在21-23度之间。",
        },
        {
            "source": "weather_alert",
            "timestamp": "day_39",
            "content": "紧急预警：强冷空气将于明日到达，预计气温骤降8-10度，最低温度可能降至13度。请注意防寒保暖。",
        },
        {
            "source": "weather_news",
            "timestamp": "day_45",
            "content": "寒潮已持续5天，气温维持在14-16度。气象部门预测寒潮将在未来3天逐步减弱，气温缓慢回升。",
        },
    ]

    print("=" * 72)
    print("  TimeClaw 多模态 Demo (时序 + 文本, Qwen2.5:32b)")
    print("  数据: 60天气温 (含寒潮事件) + 3条气象新闻")
    print("=" * 72)

    result = await runner.run(
        data=temperature,
        task_type="forecast",
        description="根据60天的气温数据和气象新闻报道，预测未来5天的气温走势",
        horizon=5,
        text_data=text_data,
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
        print("  预测值:      None")

    if result.get("fusion_report"):
        fr = result["fusion_report"]
        print(f"  融合策略:    {fr.get('strategy', '?')}")
        print(f"  融合权重:    {fr.get('weights_used', '?')}")

    per_mod = result.get("per_modality_results", {})
    for mod, r in per_mod.items():
        tools = len(r.get("execution_log", []))
        has_pred = r.get("prediction") is not None
        print(f"  [{mod}] 工具数={tools}, 有预测={has_pred}")

    tl = result["timeline"]
    print(f"\n  各阶段耗时:")
    for stage_name, info in tl.items():
        ms = info.get("elapsed_ms", "?")
        print(f"    {stage_name:12s}  {ms}ms")

    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
