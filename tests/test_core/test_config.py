from __future__ import annotations

from pathlib import Path

from timeclaw.core.config import TimeClawConfig


def test_from_toml_normalizes_provider_style_config(tmp_path: Path):
    config_path = tmp_path / "provider.toml"
    config_path.write_text(
        """
model_provider = "OpenAI"
model = "gpt-5.4"
review_model = "gpt-5.4"
model_reasoning_effort = "medium"
disable_response_storage = true
network_access = "enabled"
windows_wsl_setup_acknowledged = true
model_context_window = 1000000
model_auto_compact_token_limit = 900000

[model_providers.OpenAI]
name = "OpenAI"
base_url = "https://example.test/v1"
api_key = "sk-test"
supports_websockets = true
requires_openai_auth = true

[features]
responses_websockets_v2 = true

[llm]
max_retries = 7
timeout = 99.0
max_inflight_requests = 12
        """.strip(),
        encoding="utf-8",
    )

    cfg = TimeClawConfig.from_toml(config_path)

    assert cfg.llm.models.manager == "openai/gpt-5.4"
    assert cfg.llm.models.executor == "openai/gpt-5.4"
    assert cfg.llm.models.reasoner == "openai/gpt-5.4"
    assert cfg.llm.models.text_light == "openai/gpt-5.4"
    assert cfg.llm.models.analyzer == "openai/gpt-5.4"
    assert cfg.llm.models.text_heavy == "openai/gpt-5.4"
    assert cfg.llm.reasoning_effort == "medium"
    assert cfg.llm.api_base == "https://example.test/v1"
    assert cfg.llm.api_key == "sk-test"
    assert cfg.llm.supports_websockets is True
    assert cfg.llm.responses_websockets_v2 is True
    assert cfg.llm.disable_response_storage is True
    assert cfg.llm.model_context_window == 1000000
    assert cfg.llm.model_auto_compact_token_limit == 900000
    assert cfg.llm.max_retries == 7
    assert cfg.llm.timeout == 99.0
    assert cfg.llm.max_inflight_requests == 12
    assert cfg.network_access == "enabled"
    assert cfg.windows_wsl_setup_acknowledged is True
