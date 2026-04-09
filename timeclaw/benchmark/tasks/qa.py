"""Multimodal QA benchmark task: time-series + text joint question answering."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.benchmark.metrics import compute_qa_metrics
from timeclaw.benchmark.tasks.base import BenchmarkTask, TaskResult


class MultimodalQATask(BenchmarkTask):
    """Evaluate multimodal QA: given time-series + text context, answer questions.

    Example: "Based on recent news and price trends, predict next week's direction."
    """

    name: str = "multimodal_qa"
    description: str = "Multimodal time-series + text question answering evaluation."

    async def evaluate(
        self,
        pipeline: Any,
        dataset: dict[str, Any],
        model_name: str = "default",
        **kwargs: Any,
    ) -> TaskResult:
        questions = dataset.get("questions", [])
        if not questions:
            return TaskResult(
                task_name=self.name,
                dataset_name=dataset.get("name", "unknown"),
                model_name=model_name,
                metrics={"bleu": 0.0, "rouge_l_f1": 0.0},
                metadata={"n_questions": 0},
            )

        all_metrics: list[dict[str, float]] = []
        predictions: list[str] = []

        for q in questions:
            question_text = q.get("question", "")
            reference_answer = q.get("answer", "")
            ts_context = q.get("timeseries")
            text_context = q.get("text_context", "")

            if hasattr(pipeline, "run"):
                from timeclaw.core.tea_protocol import (
                    TaskSpec, EnvironmentSnapshot, TaskEnvelope,
                )

                task_spec = TaskSpec(
                    task_id=f"bench_qa_{len(predictions)}",
                    task_type="qa",
                    description=question_text,
                )
                env = EnvironmentSnapshot()
                if ts_context is not None:
                    env.set_data(np.asarray(ts_context, dtype=np.float32))
                if text_context:
                    env.set_text_data([{"text": text_context, "timestamp": 0}])

                envelope = TaskEnvelope(task=task_spec, environment=env)
                result = await pipeline.run(envelope)
                answer = str(result.get("answer", ""))
            elif hasattr(pipeline, "complete"):
                prompt = (
                    f"Context: {text_context}\n"
                    f"Question: {question_text}\n"
                    f"Answer:"
                )
                resp = await pipeline.complete(
                    [{"role": "user", "content": prompt}]
                )
                answer = resp.text or ""
            else:
                answer = ""

            predictions.append(answer)

            if reference_answer:
                qa_metrics = compute_qa_metrics(reference_answer, answer)
                all_metrics.append(qa_metrics)

        if all_metrics:
            avg_metrics = {
                k: float(np.mean([m[k] for m in all_metrics]))
                for k in all_metrics[0]
            }
        else:
            avg_metrics = {"bleu": 0.0, "rouge_l_f1": 0.0}

        return TaskResult(
            task_name=self.name,
            dataset_name=dataset.get("name", "unknown"),
            model_name=model_name,
            metrics=avg_metrics,
            predictions=predictions,
            metadata={"n_questions": len(questions)},
        )
