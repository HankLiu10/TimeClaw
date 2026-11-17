"""Versioned benchmark inventories and their primary paper metric cells.

paper30_original is the paper's 17 executions / 30 reported metrics.
timeclaw18 retains the earlier software inventory as an explicit extension.
"""
from dataclasses import dataclass, asdict
import hashlib
import json

TASK_SET = "paper30_original"
TASK_SETS = (TASK_SET, "timeclaw18")


@dataclass(frozen=True)
class BenchmarkTask:
    key: str
    dataset_name: str
    task_name: str


_TASK_KINDS = (
    ("finance_pred_time", "finance_aligned", "forecast"),
    ("finance_pred_macd", "finance_aligned", "indicator_macd"),
    ("finance_trend", "finance_aligned", "trend"),
    ("finance_correlation", "finance_qa", "correlation"),
    ("finance_mcqa", "finance_qa", "mcqa"),
    ("weather_forecast", "weather_aligned", "forecast"),
    ("weather_trend_future", "weather_aligned", "trend"),
    ("weather_trend_past", "weather_aligned", "trend_past"),
    ("weather_mcqa", "weather_qa", "mcqa"),
)
EXTENDED_TASKS = tuple(BenchmarkTask(f"{key}_{length}", f"{dataset}_{length}", task)
                        for key, dataset, task in _TASK_KINDS for length in ("short", "long"))
BENCHMARK_TASKS = tuple(s for s in EXTENDED_TASKS if s.key not in {
    "weather_trend_future_long", "weather_trend_past_long",
}) + (BenchmarkTask("weather_indicator_short", "weather_aligned_short", "indicator"),)


def benchmark_tasks(task_set: str = TASK_SET) -> tuple[BenchmarkTask, ...]:
    if task_set == TASK_SET:
        return BENCHMARK_TASKS
    if task_set == "timeclaw18":
        return EXTENDED_TASKS
    raise ValueError(f"Unknown task set {task_set!r}; choose one of {TASK_SETS}")


def primary_metrics(spec: BenchmarkTask) -> tuple[str, ...]:
    """One metric cell is one task's reported metric, not another task."""
    domain = spec.dataset_name.split("_")[0]
    return {
        ("finance", "forecast"): ("mae", "mape"),
        ("finance", "indicator_macd"): ("macd_mse",),
        ("finance", "trend"): ("accuracy", "accuracy_3way"),
        ("finance", "correlation"): ("exact_accuracy", "brief_accuracy"),
        ("finance", "mcqa"): ("accuracy",),
        ("weather", "forecast"): ("mse", "mae"),
        ("weather", "indicator"): ("max_mse", "max_mae", "min_mse", "min_mae", "diff_mse", "diff_mae"),
        ("weather", "trend"): ("accuracy",),
        ("weather", "trend_past"): ("accuracy",),
        ("weather", "mcqa"): ("accuracy",),
    }[domain, spec.task_name]


def tasks_for_dataset(name: str, task_set: str = TASK_SET) -> list[str]:
    return [spec.task_name for spec in benchmark_tasks(task_set) if spec.dataset_name == name]


def task_registry_digest(task_set: str = TASK_SET) -> str:
    return hashlib.sha256(json.dumps([asdict(s) for s in benchmark_tasks(task_set)],sort_keys=True).encode()).hexdigest()


def require_paper_task(dataset: str, task: str, task_set: str = TASK_SET) -> None:
    if task not in tasks_for_dataset(dataset, task_set):
        raise ValueError(f"Unsupported {task_set} dataset/task pair: {dataset}/{task}")
