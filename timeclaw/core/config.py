"""Global configuration management with Pydantic Settings and YAML support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMModelConfig(BaseModel):
    """Per-agent model assignment."""

    manager: str = "ollama/qwen3:8b"
    analyzer: str = "ollama/qwen3:1.7b"
    executor: str = "ollama/qwen3:1.7b"
    reasoner: str = "ollama/qwen3:8b"
    text_light: str | None = None
    text_heavy: str | None = None


class LLMConfig(BaseModel):
    """LLM routing configuration."""

    models: LLMModelConfig = Field(default_factory=LLMModelConfig)
    fallback: str | None = "gpt-4o-mini"
    max_retries: int = 3
    timeout: float = 120.0
    temperature: float = 0.1
    max_tokens: int = 4096
    api_base: str | None = None
    api_key: str | None = None


class ReasoningConfig(BaseModel):
    """Three-layer reasoning engine parameters."""

    max_evidence_steps: int = 15
    max_reflact_iterations: int = 5
    max_quality_gate_retries: int = 3
    critic_enabled: bool = True


class PipelineConfig(BaseModel):
    """End-to-end pipeline parameters."""

    enable_fingerprint: bool = True
    enable_safeauto: bool = True
    parallel_execution: bool = False
    verbose: bool = False
    multi_agent_mode: bool = True


class MemoryConfig(BaseModel):
    """Experience pool and vector store configuration."""

    enable_vector: bool = True
    vector_dim: int = 256
    hybrid_vector_weight: float = 0.7
    hybrid_text_weight: float = 0.3
    max_experience_entries: int = 10000


class ContextConfig(BaseModel):
    """Context window management parameters."""

    max_tokens: int = 8000
    reserve_tokens: int = 2000
    compaction_threshold: float = 0.8
    prune_observation_chars: int = 200
    enable_lazy_tool_descriptions: bool = True


class SchedulerConfig(BaseModel):
    """Proactive behavior / heartbeat configuration."""

    enabled: bool = False
    heartbeat_interval_minutes: int = 30
    drift_check_enabled: bool = True
    experience_cleanup_enabled: bool = True


class FusionConfig(BaseModel):
    """Multimodal fusion strategy configuration."""

    strategy: str = "confidence_weighted"
    task_weights: dict[str, dict[str, float]] = Field(default_factory=lambda: {
        "forecast": {"timeseries": 0.7, "text": 0.2, "image": 0.1},
        "classification": {"timeseries": 0.5, "text": 0.3, "image": 0.2},
        "anomaly_detection": {"timeseries": 0.6, "text": 0.2, "image": 0.2},
        "qa": {"timeseries": 0.2, "text": 0.6, "image": 0.2},
        "imputation": {"timeseries": 0.8, "text": 0.1, "image": 0.1},
    })
    llm_arbitration_enabled: bool = True
    confidence_floor: float = 0.1


class TimeClawConfig(BaseModel):
    """Root configuration for the TimeClaw platform."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    log_level: str = "INFO"
    workspace_dir: str = ".timeclaw"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TimeClawConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path | None = None) -> TimeClawConfig:
        """Load config from explicit path, env var, or defaults."""
        if path is None:
            path = os.environ.get("TIMECLAW_CONFIG")
        if path is not None:
            return cls.from_yaml(path)
        return cls()
