"""TEA Protocol: Task-Environment-Agent decoupling for inter-agent communication."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """Specification of what needs to be done."""

    task_id: str
    task_type: str  # forecast, anomaly, classification, imputation, qa
    description: str = ""
    target_variable: str | None = None
    horizon: int | None = None
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSnapshot(BaseModel):
    """Snapshot of the current data environment, supporting multimodal inputs."""

    model_config = {"arbitrary_types_allowed": True}

    # Time-series modality
    data: Any = None  # numpy array or pandas DataFrame
    data_shape: tuple[int, ...] | None = None
    column_names: list[str] = Field(default_factory=list)
    time_index: Any = None
    fingerprint: dict[str, Any] | None = None
    drift_detected: bool = False
    heterogeneity_score: float | None = None

    # Text modality: list of dicts like {"timestamp": ..., "text": ..., "source": ...}
    text_data: list[dict[str, Any]] = Field(default_factory=list)

    # Image modality: list of dicts like {"timestamp": ..., "path": ..., "data": ...}
    images: list[dict[str, Any]] = Field(default_factory=list)

    # Active modalities in this snapshot
    modalities: list[str] = Field(default_factory=list)

    def set_data(self, data: Any) -> None:
        self.data = data
        if hasattr(data, "shape"):
            self.data_shape = tuple(data.shape)
        if hasattr(data, "columns"):
            self.column_names = list(data.columns)
        if "timeseries" not in self.modalities:
            self.modalities.append("timeseries")

    def set_text_data(self, text_data: list[dict[str, Any]]) -> None:
        self.text_data = text_data
        if "text" not in self.modalities:
            self.modalities.append("text")

    def set_images(self, images: list[dict[str, Any]]) -> None:
        self.images = images
        if "image" not in self.modalities:
            self.modalities.append("image")

    @property
    def is_multimodal(self) -> bool:
        return len(self.modalities) > 1

    @property
    def has_text(self) -> bool:
        return "text" in self.modalities and len(self.text_data) > 0

    @property
    def has_images(self) -> bool:
        return "image" in self.modalities and len(self.images) > 0


class AgentState(BaseModel):
    """Current state of an agent within a TEA envelope."""

    agent_role: str
    step_count: int = 0
    current_sub_goal: str | None = None
    evidence_summary: str = ""
    confidence: float | None = None
    errors: list[str] = Field(default_factory=list)


class TaskEnvelope(BaseModel):
    """The TEA triple: bundles task, environment, and agent state for decoupled handoff."""

    model_config = {"arbitrary_types_allowed": True}

    task: TaskSpec
    environment: EnvironmentSnapshot = Field(default_factory=EnvironmentSnapshot)
    agent_state: AgentState = Field(default_factory=lambda: AgentState(agent_role="system"))
    created_at: float = Field(default_factory=time.time)
    trace_id: str = ""

    def with_agent(self, role: str) -> TaskEnvelope:
        """Return a copy with a fresh agent state for the given role."""
        return self.model_copy(
            update={"agent_state": AgentState(agent_role=role)},
            deep=True,
        )

    def update_environment(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self.environment, k, v)
