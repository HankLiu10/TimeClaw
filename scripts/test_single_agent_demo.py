"""Quick demo to test the single-agent pipeline on a few MTBench samples."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TIMECLAW_CONFIG", "configs/gpt51.yaml")

from timeclaw.benchmark.mtbench_adapter import load_dataset, MTBenchSample
from timeclaw.pipeline.runner import PipelineRunner


async def run_one(pipeline: PipelineRunner, sample: MTBenchSample, task: str):
    """Run a single sample and print results."""
    print(f"\n{'='*70}")
    print(f"Dataset: {sample.dataset} | Task: {task} | Domain: {sample.domain}")
    print(f"Input shape: {len(sample.input_window)} points")
    if sample.text:
        print(f"Text: {sample.text[:100]}...")
    print(f"{'='*70}")

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    kwargs = {
        "data": sample.input_window,
        "text_data": text_data,
        "verbose": True,
        "domain": sample.domain,
    }

    if task == "forecast":
        horizon = len(sample.output_window) if sample.output_window is not None else 24
        kwargs.update({
            "task_type": "forecast",
            "description": (
                f"Temperature time-series from a US location. "
                f"Every 24 readings = one day. "
                f"Predict the next {horizon // 24} day(s) of hourly readings ({horizon} values). "
                f"Return a JSON array of {horizon} numbers."
            ) if sample.domain == "weather" else (
                f"Forecast the next {horizon} values. Return a JSON array of {horizon} numbers."
            ),
            "horizon": horizon,
        })
    elif task == "indicator":
        n_hours = len(sample.output_window) if sample.output_window is not None else 24
        n_days = max(1, n_hours // 24)
        kwargs.update({
            "task_type": "indicator",
            "description": (
                f"Temperature readings from a US location (hourly, 24/day). "
                f"Predict the next {n_days} day(s) max, min, and diff temperatures. "
                f'Return JSON: {{"max": number, "min": number, "diff": number}}'
            ),
            "horizon": n_hours,
        })
    elif task == "trend":
        kwargs.update({
            "task_type": "classification",
            "description": (
                "Predict the temperature trend for the next day. "
                "If mean difference >= 0.5: increasing; <= -0.5: decreasing; else stable. "
                'Return JSON: {"trend": "increasing|decreasing|stable"}'
            ),
        })

    try:
        result = await pipeline.run(**kwargs)
        print(f"\n--- RESULT ---")
        print(f"  prediction: {str(result.get('prediction'))[:300]}")
        print(f"  answer_type: {result.get('answer_type')}")
        print(f"  confidence: {result.get('confidence')}")
        print(f"  reasoning: {str(result.get('reasoning', ''))[:200]}")
        print(f"  tool_chain: {result.get('tool_chain')}")
        print(f"  gate_accepted: {result.get('gate_accepted')}")
        print(f"  total_ms: {result.get('total_ms')}")

        if sample.output_window is not None and task == "forecast":
            import numpy as np
            pred = result.get("prediction")
            if isinstance(pred, (list, np.ndarray)):
                pred_arr = np.asarray(pred, dtype=float)[:len(sample.output_window)]
                gt_arr = np.asarray(sample.output_window, dtype=float)[:len(pred_arr)]
                mae = float(np.mean(np.abs(pred_arr - gt_arr)))
                print(f"  MAE vs ground truth: {mae:.4f}")
                print(f"  Predicted len: {len(pred)}, Expected len: {len(sample.output_window)}")

        return result
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    print("Loading pipeline (single-agent, GPT-5.1)...")
    pipeline = PipelineRunner(disable_memory=True)

    demos = [
        ("weather_aligned_short", "forecast"),
        ("weather_aligned_short", "indicator"),
        ("weather_aligned_short", "trend"),
    ]

    for ds_name, task in demos:
        try:
            samples = load_dataset(ds_name)
            if not samples:
                print(f"No samples for {ds_name}, skipping")
                continue
            sample = samples[0]
            await run_one(pipeline, sample, task)
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    await pipeline.shutdown()
    print("\n\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
