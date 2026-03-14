"""SBM Model Comparison Demo: compare forecasting across available SBM models."""

from __future__ import annotations

import asyncio

import numpy as np

from timeclaw.benchmark.datasets import DatasetLoader
from timeclaw.benchmark.metrics import compute_forecast_metrics
from timeclaw.sbm.base import SBMConfig
from timeclaw.sbm.registry import get_global_registry, auto_register


async def main() -> None:
    print("=" * 60)
    print("TimeClaw Demo: SBM Model Comparison")
    print("=" * 60)

    # Load dataset
    loader = DatasetLoader()
    dataset = loader.load("synthetic:sine", length=2000)
    train = dataset["train"]
    test = dataset["test"]

    context = train[-512:]
    horizon = 96
    y_true = test[:horizon]

    print(f"\n[Dataset] {dataset['name']}")
    print(f"  Train: {len(train)} points")
    print(f"  Test: {len(test)} points")
    print(f"  Context: {len(context)} points")
    print(f"  Horizon: {horizon} steps")

    # Auto-register available SBMs
    n_registered = auto_register()
    registry = get_global_registry()
    available = registry.list_available()
    print(f"\n[SBM Registry] {n_registered} models registered: {available}")

    if not available:
        print("\nNo SBM models available. Running with statistical fallback.")
        _run_fallback(context, y_true, horizon)
        return

    # Compare models
    results = {}
    for model_name in available:
        print(f"\n[{model_name}] Loading and forecasting...")
        try:
            adapter = registry.get(model_name)
            adapter.load()
            output = adapter.forecast(context, prediction_length=horizon)
            y_pred = output.to_numpy()

            if y_pred is not None and len(y_pred) >= len(y_true):
                y_pred = y_pred[:len(y_true)]
                metrics = compute_forecast_metrics(y_true, y_pred, y_train=train)
                results[model_name] = metrics
                print(f"  MSE={metrics['mse']:.4f}  MAE={metrics['mae']:.4f}  "
                      f"RMSE={metrics['rmse']:.4f}  SMAPE={metrics['smape']:.2f}%")
            else:
                print(f"  Forecast length mismatch")
        except Exception as e:
            print(f"  Error: {e}")

    if results:
        print("\n" + "=" * 60)
        print("Model Comparison Summary:")
        print(f"{'Model':<15} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'SMAPE':>10}")
        print("-" * 55)
        for name, m in sorted(results.items(), key=lambda x: x[1].get("mse", float("inf"))):
            print(f"{name:<15} {m['mse']:>10.4f} {m['mae']:>10.4f} "
                  f"{m['rmse']:>10.4f} {m['smape']:>9.2f}%")

    print("\n" + "=" * 60)
    print("Demo complete.")


def _run_fallback(context: np.ndarray, y_true: np.ndarray, horizon: int) -> None:
    """Naive fallback comparison."""
    last_val = context[-1]
    trend = (context[-1] - context[-10]) / 10

    naive_pred = np.full(horizon, last_val)
    trend_pred = last_val + trend * np.arange(1, horizon + 1)

    for name, pred in [("naive_repeat", naive_pred), ("naive_trend", trend_pred)]:
        pred = pred[:len(y_true)]
        metrics = compute_forecast_metrics(y_true, pred, y_train=context)
        print(f"\n[{name}]")
        print(f"  MSE={metrics['mse']:.4f}  MAE={metrics['mae']:.4f}  SMAPE={metrics['smape']:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
