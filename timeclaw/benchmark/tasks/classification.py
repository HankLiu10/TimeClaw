"""Time-series classification benchmark task."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.benchmark.metrics import compute_classification_metrics
from timeclaw.benchmark.tasks.base import BenchmarkTask, TaskResult


class ClassificationTask(BenchmarkTask):
    """Evaluate time-series classification on UCR/UEA-style datasets."""

    name: str = "classification"
    description: str = "Time-series classification evaluation."

    async def evaluate(
        self,
        pipeline: Any,
        dataset: dict[str, Any],
        model_name: str = "default",
        **kwargs: Any,
    ) -> TaskResult:
        X_test = dataset["X_test"]  # list of arrays or 3D array
        y_true = np.asarray(dataset["y_test"], dtype=int)

        predictions = []
        scores_list = []

        for i, sample in enumerate(X_test):
            sample_arr = np.asarray(sample, dtype=np.float32)

            if hasattr(pipeline, "classify"):
                output = pipeline.classify(sample_arr)
                if hasattr(output, "class_label") and output.class_label is not None:
                    predictions.append(int(output.class_label))
                elif hasattr(output, "class_logits") and output.class_logits is not None:
                    logits = np.asarray(output.class_logits)
                    predictions.append(int(np.argmax(logits)))
                    scores_list.append(logits)
                else:
                    predictions.append(0)
            elif hasattr(pipeline, "run"):
                from timeclaw.core.tea_protocol import (
                    TaskSpec, EnvironmentSnapshot, TaskEnvelope,
                )

                task_spec = TaskSpec(
                    task_id=f"bench_cls_{i}", task_type="classification"
                )
                env = EnvironmentSnapshot()
                env.set_data(sample_arr)
                envelope = TaskEnvelope(task=task_spec, environment=env)
                result = await pipeline.run(envelope)
                predictions.append(int(result.get("class_label", 0)))
            else:
                predictions.append(0)

        y_pred = np.array(predictions)
        y_scores = np.array(scores_list) if scores_list else None

        metrics = compute_classification_metrics(y_true, y_pred, y_scores)

        return TaskResult(
            task_name=self.name,
            dataset_name=dataset.get("name", "unknown"),
            model_name=model_name,
            metrics=metrics,
            predictions=y_pred.tolist(),
            metadata={"n_classes": len(np.unique(y_true)), "n_samples": len(y_true)},
        )
