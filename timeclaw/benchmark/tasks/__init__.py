"""MTBench task definitions for 5 evaluation tasks."""

from timeclaw.benchmark.tasks.base import BenchmarkTask, TaskResult
from timeclaw.benchmark.tasks.forecasting import ForecastingTask
from timeclaw.benchmark.tasks.anomaly import AnomalyDetectionTask
from timeclaw.benchmark.tasks.classification import ClassificationTask
from timeclaw.benchmark.tasks.imputation import ImputationTask
from timeclaw.benchmark.tasks.qa import MultimodalQATask

__all__ = [
    "BenchmarkTask",
    "TaskResult",
    "ForecastingTask",
    "AnomalyDetectionTask",
    "ClassificationTask",
    "ImputationTask",
    "MultimodalQATask",
]
