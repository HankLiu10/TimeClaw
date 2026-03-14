"""Anomaly detection benchmark task."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.benchmark.metrics import compute_anomaly_metrics
from timeclaw.benchmark.tasks.base import BenchmarkTask, TaskResult


class AnomalyDetectionTask(BenchmarkTask):
    """Evaluate anomaly detection with point-wise and range-based F1."""

    name: str = "anomaly_detection"
    description: str = "Time-series anomaly detection evaluation."

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    async def evaluate(
        self,
        pipeline: Any,
        dataset: dict[str, Any],
        model_name: str = "default",
        **kwargs: Any,
    ) -> TaskResult:
        data = np.asarray(dataset["data"], dtype=np.float32)
        labels = np.asarray(dataset["labels"], dtype=int)

        threshold = kwargs.get("threshold", self.threshold)

        if hasattr(pipeline, "detect_anomaly"):
            output = pipeline.detect_anomaly(data)
            if hasattr(output, "anomaly_scores") and output.anomaly_scores is not None:
                scores = np.asarray(output.anomaly_scores)
                y_pred = (scores > threshold).astype(int)
            else:
                y_pred = np.zeros_like(labels)
        elif hasattr(pipeline, "run"):
            from timeclaw.core.tea_protocol import TaskSpec, EnvironmentSnapshot, TaskEnvelope

            task_spec = TaskSpec(task_id="bench_anomaly", task_type="anomaly")
            env = EnvironmentSnapshot()
            env.set_data(data)
            envelope = TaskEnvelope(task=task_spec, environment=env)
            result = await pipeline.run(envelope)
            scores = np.asarray(result.get("anomaly_scores", np.zeros(len(data))))
            y_pred = (scores > threshold).astype(int)
        else:
            y_pred = np.zeros_like(labels)

        y_pred = y_pred.ravel()[: len(labels)]
        metrics = compute_anomaly_metrics(labels, y_pred)

        return TaskResult(
            task_name=self.name,
            dataset_name=dataset.get("name", "unknown"),
            model_name=model_name,
            metrics=metrics,
            predictions=y_pred.tolist(),
            metadata={"threshold": threshold, "n_anomalies_true": int(labels.sum())},
        )
