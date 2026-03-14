"""TimeClaw quickstart example -- cloud API version.

Usage:
    export OPENAI_API_KEY=sk-...
    export TIMECLAW_CONFIG=configs/cloud.yaml
    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

import numpy as np

from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner


async def main():
    config = TimeClawConfig.load()
    runner = PipelineRunner(config)

    np.random.seed(42)
    t = np.arange(200)
    data = 0.3 * t + 10 * np.sin(t * 0.1) + np.random.randn(200) * 3

    print("=" * 60)
    print("TimeClaw Quickstart -- Cloud API")
    print("=" * 60)
    print(f"Data shape: {data.shape}")
    print(f"LLM assignments: {runner.router.list_assignments()}")
    print()

    result = await runner.run(
        data=data,
        task_type="forecast",
        description="Forecast the next 20 steps of this time series with trend and seasonality",
        horizon=20,
        constraints=[{"type": "bound", "lower": -50, "upper": 150}],
    )

    print(f"Trace ID: {result['trace_id']}")
    print(f"Total time: {result['total_ms']}ms")
    print(f"Quality gate: {'PASS' if result['gate_accepted'] else 'FAIL'}")
    print()

    for step, info in result["timeline"].items():
        elapsed = info.get("elapsed_ms", 0)
        print(f"  [{step:>10}] {elapsed:>5}ms  {info}")

    if result["prediction"] is not None:
        pred = np.asarray(result["prediction"])
        print(f"\nPrediction shape: {pred.shape}")
        print(f"Prediction range: [{pred.min():.2f}, {pred.max():.2f}]")


if __name__ == "__main__":
    asyncio.run(main())
