"""Global configuration management with Pydantic Settings and YAML support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMModelConfig(BaseModel):
    """Per-agent model assignment."""

    manager: str = "azure/gpt-5.1"
    analyzer: str = "azure/gpt-5.1"
    executor: str = "azure/gpt-5.1"
    reasoner: str = "azure/gpt-5.1"
    text_light: str | None = "azure/gpt-5.1"
    text_heavy: str | None = "azure/gpt-5.1"


class LLMConfig(BaseModel):
    """LLM routing configuration."""

    models: LLMModelConfig = Field(default_factory=LLMModelConfig)
    fallback: str | None = None
    allow_fallback: bool = False
    max_retries: int = 3
    timeout: float = 180.0
    temperature: float = 0.1
    max_tokens: int = 4096
    api_base: str | None = "https://asure-test-openai-004.openai.azure.com/"
    api_key: str | None = "5jDLlQ5ccmal2OAOchrqcatcKMr4FfqQbOD1n0YUwkIFCwdxcI5aJQQJ99ALACYeBjFXJ3w3AAABACOGZLN4"
    azure_api_base: str | None = "https://asure-test-openai-004.openai.azure.com/"
    azure_api_key: str | None = "5jDLlQ5ccmal2OAOchrqcatcKMr4FfqQbOD1n0YUwkIFCwdxcI5aJQQJ99ALACYeBjFXJ3w3AAABACOGZLN4"
    azure_api_version: str | None = "2024-12-01-preview"


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


class MCPServerEntry(BaseModel):
    """Configuration for a single MCP tool server."""

    name: str
    command: str = "python"
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class BenchmarkConfig(BaseModel):
    """MTBench on-disk layout. Paths in ``DATASET_REGISTRY`` are relative to ``repo_root``."""

    repo_root: str = "."
    splits_dir: str = "data/splits"


class SBMRuntimeConfig(BaseModel):
    """Runtime placement for local time-series foundation models.

    New definition:
    - ``device`` is the default device for all local SBM adapters.
    - ``devices`` optionally overrides the device per model name, e.g.
      ``chronos2: cuda:2`` or ``moirai1_1_r_base: cuda:3``.
    """

    device: str = "cuda"
    devices: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) tool server configuration.

    ``servers`` lists the built-in category servers.  Additional external
    MCP servers (e.g. third-party models) can be added via
    ``extra_servers``.
    """

    enabled: bool = True
    servers: list[MCPServerEntry] = Field(default_factory=lambda: [
        MCPServerEntry(name="timeclaw-processing", args=["-m", "timeclaw.mcp.servers.processing"]),
        MCPServerEntry(name="timeclaw-profiling", args=["-m", "timeclaw.mcp.servers.profiling"]),
        MCPServerEntry(name="timeclaw-detection", args=["-m", "timeclaw.mcp.servers.detection"]),
        MCPServerEntry(name="timeclaw-decomposition", args=["-m", "timeclaw.mcp.servers.decomposition"]),
        MCPServerEntry(name="timeclaw-relation", args=["-m", "timeclaw.mcp.servers.relation"]),
        MCPServerEntry(name="timeclaw-prediction", args=["-m", "timeclaw.mcp.servers.prediction"]),
        MCPServerEntry(name="timeclaw-validation", args=["-m", "timeclaw.mcp.servers.validation"]),
        MCPServerEntry(name="timeclaw-text", args=["-m", "timeclaw.mcp.servers.text"]),
        MCPServerEntry(name="timeclaw-image", args=["-m", "timeclaw.mcp.servers.image"]),
        MCPServerEntry(name="timeclaw-generic-llm", args=["-m", "timeclaw.mcp.servers.generic_llm"]),
    ])
    extra_servers: list[MCPServerEntry] = Field(default_factory=list)


class TimeClawConfig(BaseModel):
    """Root configuration for the TimeClaw platform."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    sbm: SBMRuntimeConfig = Field(default_factory=SBMRuntimeConfig)
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
        """Load config from explicit path, env var, or auto-detected yaml."""
        if path is None:
            path = os.environ.get("TIMECLAW_CONFIG")
        if path is None:
            for candidate in ("configs/claude_proxy.yaml", "configs/gpt51.yaml", "configs/gpt5.yaml", "configs/cloud.yaml", "configs/default.yaml"):
                if Path(candidate).exists():
                    path = candidate
                    break
        if path is not None:
            return cls.from_yaml(path)
        return cls()
