"""Time-series imputation benchmark task."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.benchmark.metrics import mse, mae
from timeclaw.benchmark.tasks.base import BenchmarkTask, TaskResult


class ImputationTask(BenchmarkTask):
    """Evaluate time-series imputation with random masking."""

    name: str = "imputation"
    description: str = "Time-series missing value imputation evaluation."

    def __init__(self, mask_ratio: float = 0.25):
        self.mask_ratio = mask_ratio

    async def evaluate(
        self,
        pipeline: Any,
        dataset: dict[str, Any],
        model_name: str = "default",
        **kwargs: Any,
    ) -> TaskResult:
        data = np.asarray(dataset["data"], dtype=np.float32)
        mask_ratio = kwargs.get("mask_ratio", self.mask_ratio)

        rng = np.random.RandomState(42)
        mask = rng.random(data.shape) < mask_ratio

        masked_data = data.copy()
        masked_data[mask] = np.nan

        if hasattr(pipeline, "impute"):
            output = pipeline.impute(masked_data, mask)
            imputed = np.asarray(
                output.imputed if hasattr(output, "imputed") else output
            )
        elif hasattr(pipeline, "run"):
            from timeclaw.core.tea_protocol import TaskSpec, EnvironmentSnapshot, TaskEnvelope

            task_spec = TaskSpec(
                task_id="bench_impute",
                task_type="imputation",
                metadata={"mask_ratio": mask_ratio},
            )
            env = EnvironmentSnapshot()
            env.set_data(masked_data)
            envelope = TaskEnvelope(task=task_spec, environment=env)
            result = await pipeline.run(envelope)
            imputed = np.asarray(result.get("imputed", masked_data))
        else:
            imputed = masked_data.copy()
            imputed[mask] = np.nanmean(data)

        y_true = data[mask]
        y_pred = imputed[mask]

        valid = ~np.isnan(y_pred)
        if valid.sum() > 0:
            metrics = {
                "mse": mse(y_true[valid], y_pred[valid]),
                "mae": mae(y_true[valid], y_pred[valid]),
            }
        else:
            metrics = {"mse": float("inf"), "mae": float("inf")}

        metrics["mask_ratio"] = mask_ratio
        metrics["n_masked"] = int(mask.sum())

        return TaskResult(
            task_name=self.name,
            dataset_name=dataset.get("name", "unknown"),
            model_name=model_name,
            metrics=metrics,
            metadata={"mask_ratio": mask_ratio, "data_shape": list(data.shape)},
        )
