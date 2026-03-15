"""Unified LLM backend wrapping LiteLLM for both cloud and local model access.

Supports multimodal messages (text + image) via OpenAI-compatible content arrays.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from timeclaw.core import trace_logger as trace

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient API error worth retrying."""
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "rate limit", "too many requests",
                                   "server error", "502", "503", "timeout",
                                   "connection", "overloaded"))


class LLMResponse(BaseModel):
    """Standardized LLM response."""

    text: str | None = None
    parsed: dict[str, Any] | list | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    raw: Any = None


class LLMBackend:
    """Thin wrapper around litellm providing unified async interface for all providers."""

    def __init__(
        self,
        default_model: str = "ollama/qwen3:8b",
        fallback_model: str | None = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3,
        timeout: float = 120.0,
        api_base: str | None = None,
        api_key: str | None = None,
        azure_api_base: str | None = None,
        azure_api_key: str | None = None,
        azure_api_version: str | None = None,
    ):
        import os

        self.default_model = default_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_base = api_base
        self.api_key = api_key
        self.azure_api_base = azure_api_base
        self.azure_api_key = azure_api_key
        self.azure_api_version = azure_api_version or "2024-12-01-preview"

        if api_key and api_base:
            base_no_v1 = api_base.rstrip("/")
            if base_no_v1.endswith("/v1"):
                base_no_v1 = base_no_v1[:-3]
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            os.environ.setdefault("OPENAI_API_BASE", api_base)
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            os.environ.setdefault("ANTHROPIC_API_BASE", base_no_v1)

        if azure_api_key and azure_api_base:
            os.environ.setdefault("AZURE_API_KEY", azure_api_key)
            os.environ.setdefault("AZURE_API_BASE", azure_api_base)
            os.environ.setdefault("AZURE_API_VERSION", self.azure_api_version)

    async def _call_with_retry(
        self, call_kwargs: dict[str, Any], model_label: str,
        response_format: str | None, messages: list[dict[str, Any]],
    ) -> LLMResponse | None:
        """Execute a litellm call with automatic retry on transient errors."""
        import litellm

        for attempt in range(_MAX_RETRIES):
            try:
                response = await litellm.acompletion(**call_kwargs)
                parsed = self._parse_response(response, model_label)
                trace.llm_call(model_label, messages, parsed,
                               response_format=response_format)
                return parsed
            except Exception as e:
                if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"LLM {model_label} attempt {attempt+1} failed "
                        f"(retryable): {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        return None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request. Retries on transient errors, then falls back."""
        target_model = model or self.default_model
        call_kwargs = self._build_kwargs(messages, target_model, response_format, **kwargs)

        try:
            result = await self._call_with_retry(
                call_kwargs, target_model, response_format, messages)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"LLM call to {target_model} failed: {e}")
            if self.fallback_model and target_model != self.fallback_model:
                logger.info(f"Falling back to {self.fallback_model}")
                call_kwargs["model"] = self.fallback_model
                try:
                    result = await self._call_with_retry(
                        call_kwargs, self.fallback_model + " (fallback)",
                        response_format, messages)
                    if result is not None:
                        return result
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
        return LLMResponse(text=None, model=target_model)

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request with function calling / tool use."""
        import litellm

        target_model = model or self.default_model
        call_kwargs = self._build_kwargs(messages, target_model, **kwargs)
        call_kwargs["tools"] = tools

        for attempt in range(_MAX_RETRIES):
            try:
                response = await litellm.acompletion(**call_kwargs)
                return self._parse_response(response, target_model)
            except Exception as e:
                if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Tool call {target_model} attempt {attempt+1} "
                        f"(retryable): {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"Tool call to {target_model} failed: {e}")
                if self.fallback_model and target_model != self.fallback_model:
                    call_kwargs["model"] = self.fallback_model
                    try:
                        response = await litellm.acompletion(**call_kwargs)
                        return self._parse_response(response, self.fallback_model)
                    except Exception as e2:
                        logger.error(f"Fallback tool call also failed: {e2}")
                return LLMResponse(text=None, model=target_model)
        return LLMResponse(text=None, model=target_model)

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "num_retries": self.max_retries,
            "timeout": self.timeout,
        }
        is_anthropic = model.startswith("anthropic/")
        is_azure = model.startswith("azure/")

        if is_azure:
            if self.azure_api_base:
                call_kwargs["api_base"] = self.azure_api_base
            if self.azure_api_key:
                call_kwargs["api_key"] = self.azure_api_key
            call_kwargs["api_version"] = self.azure_api_version
        elif not is_anthropic:
            if self.api_base:
                call_kwargs["api_base"] = self.api_base
            if self.api_key:
                call_kwargs["api_key"] = self.api_key

        if response_format == "json":
            call_kwargs["response_format"] = {"type": "json_object"}
        call_kwargs.update(kwargs)
        return call_kwargs

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | list | None:
        """Best-effort JSON extraction: raw text, markdown fences, or first {/[ block."""
        import re

        text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        for start, end in [("{", "}"), ("[", "]")]:
            idx = text.find(start)
            if idx == -1:
                continue
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == start:
                    depth += 1
                elif text[i] == end:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx : i + 1])
                    except (json.JSONDecodeError, TypeError):
                        break
        return None

    @staticmethod
    def _parse_response(response: Any, model: str) -> LLMResponse:
        if not getattr(response, "choices", None):
            logger.warning(f"Empty choices from {model}")
            return LLMResponse(text=None, model=model)
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""

        parsed = LLMBackend._extract_json(text) if text else None

        tool_calls_raw = getattr(message, "tool_calls", None) or []
        tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.function
            args = fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({"name": fn.name, "arguments": args})

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return LLMResponse(
            text=text,
            parsed=parsed,
            tool_calls=tool_calls,
            model=model,
            usage=usage,
            raw=response,
        )

    # ── Multimodal helpers ──────────────────────────────────────────

    async def complete_multimodal(
        self,
        text: str,
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
        image_b64: list[str] | None = None,
        system: str | None = None,
        model: str | None = None,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Convenience method for multimodal (text + images) completion.

        Builds an OpenAI-compatible multimodal message with content array
        containing text and image_url parts.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]

        for path_str in image_paths or []:
            b64 = self._encode_image_file(path_str)
            if b64:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        for url in image_urls or []:
            content.append({"type": "image_url", "image_url": {"url": url}})

        for b64 in image_b64 or []:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        return await self.complete(messages, model=model, response_format=response_format, **kwargs)

    @staticmethod
    def _encode_image_file(path_str: str) -> str | None:
        """Read an image file and return its base64 encoding."""
        p = Path(path_str)
        if not p.exists():
            logger.warning(f"Image file not found: {path_str}")
            return None
        return base64.b64encode(p.read_bytes()).decode("utf-8")

    @staticmethod
    def build_multimodal_content(
        text: str,
        image_urls: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build an OpenAI-compatible content array for multimodal messages."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for url in image_urls or []:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content
