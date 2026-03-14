"""Base class for all MTBench tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class TaskResult(BaseModel):
    """Result from running a benchmark task."""

    task_name: str
    dataset_name: str
    model_name: str
    metrics: dict[str, float] = Field(default_factory=dict)
    predictions: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class BenchmarkTask(ABC):
    """Abstract base for MTBench evaluation tasks."""

    name: str = ""
    description: str = ""

    @abstractmethod
    async def evaluate(
        self,
        pipeline: Any,
        dataset: dict[str, Any],
        model_name: str = "default",
        **kwargs: Any,
    ) -> TaskResult:
        """Run the task on a dataset and return metrics.

        Args:
            pipeline: TimeClaw pipeline runner or model adapter.
            dataset: Dict with keys like 'train', 'test', 'labels', etc.
            model_name: Name of the model being evaluated.
        """
        ...

    def __repr__(self) -> str:
        return f"<BenchmarkTask {self.name}>"
