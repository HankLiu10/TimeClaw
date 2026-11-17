"""Global configuration management with YAML and TOML support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

try:  # Python 3.11+
    import tomllib as toml_loader
except ModuleNotFoundError:  # Python 3.10
    import tomli as toml_loader


class LLMModelConfig(BaseModel):
    """Per-agent model assignment."""

    manager: str = "openai/gpt-6-sol"
    analyzer: str = "openai/gpt-6-sol"
    executor: str = "openai/gpt-6-sol"
    reasoner: str = "openai/gpt-6-sol"
    text_light: str | None = "openai/gpt-6-sol"
    text_heavy: str | None = "openai/gpt-6-sol"


class LLMConfig(BaseModel):
    """LLM routing configuration."""

    models: LLMModelConfig = Field(default_factory=LLMModelConfig)
    fallback: str | None = None
    allow_fallback: bool = False
    max_retries: int = 5
    # Reconnects after temporary network/API failures, including 429 and 5xx.
    # None (YAML null) means unlimited; 0 disables retries; positive values
    # count reconnects after the initial attempt. Permanent refusals still fail.
    network_retry_attempts: int | None = Field(default=None, ge=0)
    network_retry_delay_s: float = 30.0
    timeout: float = 3600.0
    # New definition:
    # - ``first_token_timeout`` is the maximum wait for the first streamed chunk,
    #   and thereafter the maximum idle gap between streamed chunks, on the native
    #   OpenAI-compatible SSE path.
    # - It does not replace ``timeout``; ``timeout`` remains the total request
    #   timeout budget.
    first_token_timeout: float | None = None
    # New definition:
    # - ``max_inflight_requests`` limits the number of real outbound LLM requests
    #   that may be in flight at the same time across the whole TimeClaw process.
    # - Local queue wait and retry backoff do not consume this budget; only the
    #   period after a request is actually dispatched to the provider counts.
    max_inflight_requests: int | None = None
    temperature: float = 0.1
    # False omits temperature from every wire request, including helper-level
    # overrides. This is an explicit provider capability, not error recovery.
    send_temperature: bool = True
    max_tokens: int | None = None
    api_base: str | None = None
    api_key: str | None = None
    azure_api_base: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str | None = "2024-12-01-preview"
    # New definition:
    # - ``disable_response_storage=True`` maps to Responses API `store=false`.
    # - This only affects providers on the native Responses transport path.
    disable_response_storage: bool = False
    # Supplier-native metadata kept in config without forcing TimeClaw to use
    # them as internal prompt budgets.
    model_context_window: int | None = None
    model_auto_compact_token_limit: int | None = None
    # New definition:
    # - ``supports_websockets`` declares whether the configured OpenAI-compatible
    #   provider supports Responses WebSocket transport.
    # - ``responses_websockets_v2`` is the matching feature gate from the
    #   provider-side config. TimeClaw only uses websocket transport when both
    #   flags are true.
    supports_websockets: bool = False
    responses_websockets_v2: bool = False
    # OpenAI Responses API: reasoning.effort (none/minimal/low/medium/high/xhigh).
    # Native openai/ + proxy path only. Does not request reasoning.summary, so optional
    # human-readable reasoning summaries are not added to the response output (see OpenAI reasoning guide).
    #
    # New definition:
    # - Current experiments request gpt-6-sol at medium effort.
    # - Explicit GLM profiles retain their separately configured max effort.
    reasoning_effort: str | None = "medium"
    # api_transport selects the wire protocol, independent of the model prefix.
    # GLM uses chat_completions; archived Responses configs must select responses.
    api_transport: Literal["responses", "chat_completions"] = "chat_completions"

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
    # New definition:
    # - ``disable_tool_dropout`` only affects training-time sub-agent tool
    #   exposure. Inference remains unchanged because tool dropout is a
    #   training-only exploration mechanism.
    disable_tool_dropout: bool = False


class MemoryConfig(BaseModel):
    """Experience pool and vector store configuration."""

    enable_vector: bool = True
    vector_dim: int = 256
    hybrid_vector_weight: float = 0.7
    hybrid_text_weight: float = 0.3
    max_experience_entries: int = 10000








class MCPServerEntry(BaseModel):
    """Configuration for a single MCP tool server."""

    name: str
    command: str = "python"
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class BenchmarkConfig(BaseModel):
    """Benchmark and independent-training data locations.

    ``repo_root``/``splits_dir`` describe the official MTBench evaluation
    layout. ``training_root`` is deliberately separate: training must consume
    the disjoint corpus under ``data_preparation/independent`` and must never
    silently fall back to the official benchmark pool.
    """

    task_set: Literal["paper30_original", "timeclaw18"] = "paper30_original"
    repo_root: str = "."
    splits_dir: str = "data/splits"
    training_root: str = "data_preparation/independent"
    # None declares every paper dataset; a demo can explicitly declare a subset.
    # Missing/empty declared evaluation pools block learning, never skip a check.
    evaluation_datasets: list[str] | None = None
    # Explicit audited source-time attachment, relative to repo_root or absolute.
    # None leaves the original data untouched; strict source admission still applies.
    observation_time_evidence: str | None = None


class SBMRuntimeConfig(BaseModel):
    """Runtime placement for local time-series foundation models.

    New definition:
    - ``device`` is the default device for all local SBM adapters.
    - ``devices`` optionally overrides the device per model name, e.g.
      ``chronos2: cuda:2`` or ``moirai1_1_r_base: cuda:3``.
    - ``managed_server_enabled`` turns on the benchmark-side managed SBM
      services: the script starts one HTTP server per model once, all
      concurrent samples reuse those services, and the processes stop on
      script exit.
    - ``managed_server_base_port`` is the default starting port for the
      per-model services.
    - ``managed_server_ports`` optionally pins exact ports per model name.
      Example: ``chronos2: 8100`` and ``moirai1_1_r_base: 8105``.
    - ``managed_server_replicas`` is the number of service replicas to start
      per configured model. When it is greater than 1, each model is exposed
      through round-robin HTTP clients and ``managed_server_ports`` is treated
      as the starting port of that model's replica block.
    - ``request_timeout_s`` is the client-side full HTTP request timeout for
      SBM server calls. ``None`` means no total timeout, so requests may wait
      in the model queue indefinitely.
    - ``first_byte_timeout_s`` is the client-side read timeout for SBM server
      calls. ``None`` means no read timeout, so queued or slow-running model
      requests do not fail just because they wait a long time for a response.
    """

    device: str = "cuda"
    devices: dict[str, str] = Field(default_factory=dict)
    managed_server_enabled: bool = True
    managed_server_host: str = "127.0.0.1"
    managed_server_base_port: int = 8100
    managed_server_ports: dict[str, int] = Field(default_factory=dict)
    managed_server_replicas: int = 1
    managed_server_models: list[str] = Field(
        default_factory=lambda: [
            "chronos2",
            "moirai1_1_r_base",
            "timesfm2",
            "sundial_base_128m",
            "aurora",
        ]
    )
    # Explicit interpreter placement for incompatible model dependency sets.
    # An absent entry retains the calling interpreter; launch failure never
    # selects a different environment or model.
    managed_server_python: dict[str, str] = Field(default_factory=dict)
    managed_server_health_timeout_s: float = 180.0
    request_timeout_s: float | None = None
    first_byte_timeout_s: float | None = None


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
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    sbm: SBMRuntimeConfig = Field(default_factory=SBMRuntimeConfig)
    # Supplier/client-side declarative fields. They are intentionally kept at
    # the root level because they are not TimeClaw's own LLM routing knobs.
    network_access: str | None = None
    windows_wsl_setup_acknowledged: bool | None = None
    log_level: str = "INFO"
    workspace_dir: str = ".timeclaw"

    @staticmethod
    def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge config dicts with `override` taking precedence."""
        merged = dict(base)
        for key, value in override.items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = TimeClawConfig._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _qualify_provider_model(provider_name: str | None, model_name: Any) -> str | None:
        """Map provider-local model names like `gpt-5.4` to `openai/gpt-5.4`."""
        raw = str(model_name or "").strip()
        if not raw:
            return None
        if "/" in raw:
            return raw
        provider = str(provider_name or "").strip().lower()
        if not provider:
            return raw
        return f"{provider}/{raw}"

    @classmethod
    def _normalize_provider_toml_payload(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize provider-style TOML config into TimeClaw's internal schema.

        New definition:
        - Supplier TOML may describe models through top-level `model` /
          `review_model` plus `[model_providers.<name>]`.
        - TimeClaw internally needs `llm.models.<role>` and `llm.*` runtime
          settings, so this method derives those fields and then lets any
          explicit `[llm]` section override the derived defaults.
        """
        normalized = dict(data)
        provider_name = str(normalized.get("model_provider") or "").strip()
        provider_cfg = None
        providers = normalized.get("model_providers")
        if isinstance(providers, dict) and provider_name:
            candidate = providers.get(provider_name)
            if isinstance(candidate, dict):
                provider_cfg = candidate

        derived_llm: dict[str, Any] = {}
        model_name = cls._qualify_provider_model(provider_name, normalized.get("model"))
        review_model_name = cls._qualify_provider_model(
            provider_name,
            normalized.get("review_model") or normalized.get("model"),
        )
        if model_name:
            derived_llm["models"] = {
                "manager": model_name,
                "executor": model_name,
                "reasoner": model_name,
                "text_light": model_name,
            }
        if review_model_name:
            derived_llm.setdefault("models", {})
            derived_llm["models"].update({
                "analyzer": review_model_name,
                "text_heavy": review_model_name,
            })

        reasoning_effort = str(normalized.get("model_reasoning_effort") or "").strip()
        if reasoning_effort:
            derived_llm["reasoning_effort"] = reasoning_effort
        if isinstance(normalized.get("disable_response_storage"), bool):
            derived_llm["disable_response_storage"] = normalized["disable_response_storage"]
        model_context_window = normalized.get("model_context_window")
        if isinstance(model_context_window, int):
            derived_llm["model_context_window"] = model_context_window
        auto_compact_limit = normalized.get("model_auto_compact_token_limit")
        if isinstance(auto_compact_limit, int):
            derived_llm["model_auto_compact_token_limit"] = auto_compact_limit

        if isinstance(provider_cfg, dict):
            base_url = str(provider_cfg.get("base_url") or "").strip()
            api_key = str(provider_cfg.get("api_key") or "").strip()
            supports_websockets = provider_cfg.get("supports_websockets")
            if base_url:
                derived_llm["api_base"] = base_url
            if api_key:
                derived_llm["api_key"] = api_key
            if isinstance(supports_websockets, bool):
                derived_llm["supports_websockets"] = supports_websockets

        features_cfg = normalized.get("features")
        if isinstance(features_cfg, dict):
            ws_v2 = features_cfg.get("responses_websockets_v2")
            if isinstance(ws_v2, bool):
                derived_llm["responses_websockets_v2"] = ws_v2

        explicit_llm = normalized.get("llm")
        if isinstance(explicit_llm, dict):
            normalized["llm"] = cls._deep_merge_dicts(derived_llm, explicit_llm)
        elif derived_llm:
            normalized["llm"] = derived_llm
        return normalized

    @classmethod
    def from_yaml(cls, path: str | Path) -> TimeClawConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def from_toml(cls, path: str | Path) -> TimeClawConfig:
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data: dict[str, Any] = toml_loader.load(f) or {}
        return cls.model_validate(cls._normalize_provider_toml_payload(data))

    @classmethod
    def from_file(cls, path: str | Path) -> TimeClawConfig:
        """Load configuration from YAML or TOML based on file suffix."""
        suffix = Path(path).suffix.lower()
        if suffix == ".toml":
            return cls.from_toml(path)
        if suffix in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        raise ValueError(f"Unsupported config format: {path}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> TimeClawConfig:
        """Load config from explicit path, env var, or auto-detected YAML/TOML."""
        workspace_override = os.environ.get("TIMECLAW_WORKSPACE_DIR")
        if path is None:
            path = os.environ.get("TIMECLAW_CONFIG")
        if path is None:
            for candidate in ("configs/default.yaml", "configs/default.toml"):
                if Path(candidate).exists():
                    path = candidate
                    break
        if path is not None:
            cfg = cls.from_file(path)
        else:
            cfg = cls()
        # Resolve secret-free configs against process credentials.  Native
        # Responses transport needs these values on LLMConfig itself; relying
        # only on LiteLLM's implicit environment lookup would incorrectly
        # select the chat-completions path for codex-channel proxies.
        if cfg.llm.api_base is None:
            cfg.llm.api_base = os.environ.get("OPENAI_API_BASE") or None
        if cfg.llm.api_key is None:
            cfg.llm.api_key = os.environ.get("OPENAI_API_KEY") or None
        if workspace_override:
            cfg.workspace_dir = workspace_override
        return cfg
