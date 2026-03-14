"""Forecasting benchmark task: evaluate point and probabilistic forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.benchmark.metrics import compute_forecast_metrics
from timeclaw.benchmark.tasks.base import BenchmarkTask, TaskResult


class ForecastingTask(BenchmarkTask):
    """Evaluate time-series forecasting on standard datasets (ETTh1, Weather, ECL, etc.)."""

    name: str = "forecasting"
    description: str = "Point and probabilistic time-series forecasting evaluation."

    def __init__(self, horizons: list[int] | None = None):
        self.horizons = horizons or [96, 192, 336, 720]

    async def evaluate(
        self,
        pipeline: Any,
        dataset: dict[str, Any],
        model_name: str = "default",
        **kwargs: Any,
    ) -> TaskResult:
        train = np.asarray(dataset["train"], dtype=np.float32)
        test = np.asarray(dataset["test"], dtype=np.float32)
        horizon = kwargs.get("horizon", self.horizons[0])

        context = train[-512:] if len(train) > 512 else train
        y_true = test[:horizon]

        if hasattr(pipeline, "forecast"):
            output = pipeline.forecast(context, prediction_length=horizon)
            y_pred = np.asarray(
                output.point_forecast if hasattr(output, "point_forecast") else output
            )
            samples = getattr(output, "samples", None)
        elif hasattr(pipeline, "run"):
            from timeclaw.core.tea_protocol import TaskSpec, EnvironmentSnapshot, TaskEnvelope

            task_spec = TaskSpec(
                task_id="bench_forecast",
                task_type="forecast",
                horizon=horizon,
            )
            env = EnvironmentSnapshot()
            env.set_data(context)
            envelope = TaskEnvelope(task=task_spec, environment=env)
            result = await pipeline.run(envelope)
            y_pred = np.asarray(result.get("prediction", np.zeros(horizon)))
            samples = None
        else:
            raise ValueError("Pipeline must have forecast() or run() method")

        y_pred = y_pred.ravel()[:horizon]
        y_true = y_true.ravel()[:horizon]
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        metrics = compute_forecast_metrics(
            y_true, y_pred, y_train=train, samples=samples
        )

        return TaskResult(
            task_name=self.name,
            dataset_name=dataset.get("name", "unknown"),
            model_name=model_name,
            metrics=metrics,
            predictions=y_pred.tolist(),
            metadata={"horizon": horizon, "context_length": len(context)},
        )
