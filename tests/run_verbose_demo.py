"""Run a full pipeline sample with verbose=True to demonstrate trace logging."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

sys.path.insert(0, ".")

from timeclaw.core.config import LLMConfig, LLMModelConfig, PipelineConfig, TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner


_CALL_COUNT = 0

def _make_mock_litellm_response(*args, **kwargs):
    """Return varied mock responses to exercise different pipeline stages."""
    global _CALL_COUNT
    _CALL_COUNT += 1

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = None
    response.usage = MagicMock()
    response.usage.prompt_tokens = 80 + _CALL_COUNT * 10
    response.usage.completion_tokens = 40 + _CALL_COUNT * 5
    response.usage.total_tokens = 120 + _CALL_COUNT * 15

    if _CALL_COUNT == 1:
        response.choices[0].message.content = (
            '{"task_type": "forecast", '
            '"sub_goals": ["analyze trend and seasonality", "forecast next 10 steps"], '
            '"horizon": 10, '
            '"target_variable": "temperature"}'
        )
    elif _CALL_COUNT == 2:
        response.choices[0].message.content = (
            '{"fingerprint": {"trend": "upward", "seasonality": "weekly", '
            '"stationarity": "non-stationary", "noise_level": "moderate"}, '
            '"data_quality": {"missing_ratio": 0.0, "outlier_ratio": 0.02}, '
            '"recommendations": ["Apply differencing before ARIMA", "Consider seasonal decomposition"]}'
        )
    elif _CALL_COUNT == 3:
        response.choices[0].message.content = (
            '{"tool_chain": ['
            '{"tool": "normalize", "args": {"method": "zscore"}}, '
            '{"tool": "arima_forecast", "args": {"horizon": 10}}'
            '], '
            '"reasoning": "Data shows upward trend with moderate noise. '
            'Normalize first for numerical stability, then apply ARIMA forecasting."}'
        )
    elif _CALL_COUNT <= 8:
        response.choices[0].message.content = (
            '{"done": true, "reasoning": "Evidence sufficient for forecast task", '
            '"covers_predicates": ["trend_analyzed", "forecast_generated"]}'
        )
    else:
        response.choices[0].message.content = (
            '{"accepted": true, "reasoning": "Prediction is within expected range '
            'and consistent with historical data patterns.", "consistent": true, '
            '"confidence": 0.85, "answer": null}'
        )

    return response


async def main():
    config = TimeClawConfig(
        llm=LLMConfig(
            models=LLMModelConfig(
                manager="mock-model",
                analyzer="mock-model",
                executor="mock-model",
                reasoner="mock-model",
            ),
            fallback=None,
        ),
        pipeline=PipelineConfig(verbose=True),
        workspace_dir="/tmp/timeclaw_verbose_test",
    )

    runner = PipelineRunner(config)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = _make_mock_litellm_response

        np.random.seed(42)
        data = 20.0 + np.sin(np.linspace(0, 4 * np.pi, 100)) * 3 + np.random.randn(100) * 0.5

        print("=" * 80)
        print("  TimeClaw Pipeline — Verbose Trace Demo")
        print("=" * 80)

        result = await runner.run(
            data=data,
            task_type="forecast",
            description="Forecast daily temperature for the next 10 days based on historical data",
            horizon=10,
            verbose=True,
        )

        print("\n" + "=" * 80)
        print("  FINAL RESULT SUMMARY")
        print("=" * 80)
        print(f"  trace_id:    {result['trace_id']}")
        print(f"  total_ms:    {result['total_ms']}ms")
        print(f"  modalities:  {result['modalities']}")
        print(f"  gate_pass:   {result['gate_accepted']}")
        pred = result.get("prediction")
        if pred is not None:
            pred_str = str(pred)[:200]
            print(f"  prediction:  {pred_str}")
        else:
            print("  prediction:  None")

        tl = result["timeline"]
        print(f"\n  Timeline:")
        for stage, info in tl.items():
            ms = info.get("elapsed_ms", "?")
            print(f"    {stage:12s}  {ms}ms")

        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
