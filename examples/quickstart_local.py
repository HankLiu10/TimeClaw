"""TimeClaw quickstart example -- local Ollama models only.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull a model: ollama pull qwen3:1.7b
    3. Run: python examples/quickstart_local.py

No API keys needed!
"""

from __future__ import annotations

import asyncio

import numpy as np

from timeclaw.core.config import LLMConfig, LLMModelConfig, TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner


async def main():
    config = TimeClawConfig(
        llm=LLMConfig(
            models=LLMModelConfig(
                manager="ollama/qwen3:1.7b",
                analyzer="ollama/qwen3:1.7b",
                executor="ollama/qwen3:1.7b",
                reasoner="ollama/qwen3:1.7b",
            ),
            fallback=None,  # no cloud fallback
            timeout=180.0,  # longer timeout for local models
        ),
    )

    runner = PipelineRunner(config)

    np.random.seed(42)
    t = np.arange(100)
    data = 0.5 * t + np.random.randn(100) * 2

    print("=" * 60)
    print("TimeClaw Quickstart -- Local Ollama Models")
    print("=" * 60)
    print(f"Data shape: {data.shape}")
    print(f"LLM assignments: {runner.router.list_assignments()}")
    print()

    result = await runner.run(
        data=data,
        task_type="forecast",
        description="Forecast the next 10 steps",
        horizon=10,
    )

    print(f"Trace ID: {result['trace_id']}")
    print(f"Total time: {result['total_ms']}ms")
    print(f"Quality gate: {'PASS' if result['gate_accepted'] else 'FAIL'}")
    print()

    for step, info in result["timeline"].items():
        elapsed = info.get("elapsed_ms", 0)
        print(f"  [{step:>10}] {elapsed:>5}ms")

    if result["prediction"] is not None:
        pred = np.asarray(result["prediction"])
        print(f"\nPrediction: {pred[:5]}... (shape: {pred.shape})")


if __name__ == "__main__":
    asyncio.run(main())
