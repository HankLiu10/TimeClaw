"""MTBench: multi-task time-series benchmark runner and report generator."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.benchmark.datasets import DatasetLoader
from timeclaw.benchmark.tasks.base import BenchmarkTask, TaskResult
from timeclaw.benchmark.tasks.forecasting import ForecastingTask
from timeclaw.benchmark.tasks.anomaly import AnomalyDetectionTask
from timeclaw.benchmark.tasks.classification import ClassificationTask
from timeclaw.benchmark.tasks.imputation import ImputationTask
from timeclaw.benchmark.tasks.qa import MultimodalQATask


class MTBenchRunner:
    """Orchestrate multi-task benchmark evaluation across models and datasets."""

    def __init__(
        self,
        tasks: list[str] | None = None,
        output_dir: str = "results/mtbench",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DatasetLoader()

        all_tasks: dict[str, BenchmarkTask] = {
            "forecasting": ForecastingTask(),
            "anomaly_detection": AnomalyDetectionTask(),
            "classification": ClassificationTask(),
            "imputation": ImputationTask(),
            "multimodal_qa": MultimodalQATask(),
        }

        if tasks:
            self.tasks = {k: v for k, v in all_tasks.items() if k in tasks}
        else:
            self.tasks = all_tasks

    async def run(
        self,
        pipelines: dict[str, Any],
        datasets: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[TaskResult]]:
        """Run all configured tasks across all pipelines.

        Args:
            pipelines: Dict mapping model_name -> pipeline/adapter instance.
            datasets: Optional pre-loaded datasets per task. If None, uses defaults.

        Returns:
            Dict mapping task_name -> list of TaskResult.
        """
        results: dict[str, list[TaskResult]] = {}

        for task_name, task in self.tasks.items():
            logger.info(f"Running task: {task_name}")
            task_results: list[TaskResult] = []

            dataset = self._get_dataset(task_name, datasets)

            for model_name, pipeline in pipelines.items():
                logger.info(f"  Model: {model_name}")
                start = time.time()
                try:
                    result = await task.evaluate(
                        pipeline, dataset, model_name=model_name, **kwargs
                    )
                    result.metadata["elapsed_seconds"] = round(time.time() - start, 3)
                    task_results.append(result)
                    logger.info(f"    Metrics: {result.metrics}")
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    task_results.append(
                        TaskResult(
                            task_name=task_name,
                            dataset_name=dataset.get("name", "unknown"),
                            model_name=model_name,
                            metrics={},
                            metadata={"error": str(e)},
                        )
                    )

            results[task_name] = task_results

        return results

    def _get_dataset(
        self,
        task_name: str,
        datasets: dict[str, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if datasets and task_name in datasets:
            return datasets[task_name]

        if task_name == "forecasting":
            return self.loader.load("synthetic:sine", length=2000)
        elif task_name == "anomaly_detection":
            return self.loader.load("synthetic:anomaly", length=1000)
        elif task_name == "classification":
            return self.loader.load_classification()
        elif task_name == "imputation":
            return self.loader.load("synthetic:random_walk", length=1000)
        elif task_name == "multimodal_qa":
            return self.loader.load_qa()
        else:
            return self.loader.load("synthetic:sine")

    def generate_report(
        self,
        results: dict[str, list[TaskResult]],
        format: str = "markdown",
    ) -> str:
        """Generate a formatted report from benchmark results."""
        if format == "markdown":
            return self._markdown_report(results)
        elif format == "latex":
            return self._latex_report(results)
        elif format == "json":
            return self._json_report(results)
        return self._markdown_report(results)

    def _markdown_report(self, results: dict[str, list[TaskResult]]) -> str:
        lines = ["# MTBench Evaluation Report\n"]

        for task_name, task_results in results.items():
            lines.append(f"\n## {task_name}\n")

            if not task_results:
                lines.append("No results.\n")
                continue

            all_metrics = set()
            for r in task_results:
                all_metrics.update(r.metrics.keys())
            metric_cols = sorted(all_metrics)

            header = "| Model | " + " | ".join(metric_cols) + " |"
            sep = "|---" * (len(metric_cols) + 1) + "|"
            lines.append(header)
            lines.append(sep)

            for r in task_results:
                vals = [f"{r.metrics.get(m, '-'):.3f}" if isinstance(r.metrics.get(m), float) else str(r.metrics.get(m, "-")) for m in metric_cols]
                lines.append(f"| {r.model_name} | " + " | ".join(vals) + " |")

            lines.append("")

        report = "\n".join(lines)
        report_path = self.output_dir / "report.md"
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")
        return report

    def _latex_report(self, results: dict[str, list[TaskResult]]) -> str:
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{MTBench Results}",
        ]

        for task_name, task_results in results.items():
            if not task_results:
                continue

            all_metrics = set()
            for r in task_results:
                all_metrics.update(r.metrics.keys())
            metric_cols = sorted(all_metrics)

            cols = "l" + "c" * len(metric_cols)
            lines.append(f"\\begin{{tabular}}{{{cols}}}")
            lines.append("\\hline")
            header = "Model & " + " & ".join(m.upper() for m in metric_cols) + " \\\\"
            lines.append(header)
            lines.append("\\hline")

            for r in task_results:
                vals = [
                    f"{r.metrics.get(m, 0):.3f}" if isinstance(r.metrics.get(m), float) else str(r.metrics.get(m, "-"))
                    for m in metric_cols
                ]
                lines.append(f"{r.model_name} & " + " & ".join(vals) + " \\\\")

            lines.append("\\hline")
            lines.append("\\end{tabular}")
            lines.append("")

        lines.append("\\end{table}")

        report = "\n".join(lines)
        report_path = self.output_dir / "report.tex"
        report_path.write_text(report)
        return report

    def _json_report(self, results: dict[str, list[TaskResult]]) -> str:
        data = {}
        for task_name, task_results in results.items():
            data[task_name] = [
                {
                    "model": r.model_name,
                    "dataset": r.dataset_name,
                    "metrics": r.metrics,
                    "metadata": {
                        k: v for k, v in r.metadata.items()
                        if not isinstance(v, (np.ndarray,))
                    },
                }
                for r in task_results
            ]

        report = json.dumps(data, indent=2, default=str)
        report_path = self.output_dir / "report.json"
        report_path.write_text(report)
        return report
