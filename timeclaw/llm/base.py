"""Unified LLM backend wrapping LiteLLM for both cloud and local model access.

Supports multimodal messages (text + image) via OpenAI-compatible content arrays.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
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


def _is_empty(resp: "LLMResponse | None") -> bool:
    """True if response has no usable content (text or tool calls)."""
    if resp is None:
        return True
    return not resp.text and not resp.tool_calls


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
        fallback_model: str | None = None,
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

    def _use_native_openai(self, model: str) -> bool:
        """Whether to use the native OpenAI SDK instead of litellm.

        litellm's ``openai/`` provider can corrupt responses from some
        third-party OpenAI-compatible proxies (empty content).  When we
        detect this pattern we use the openai SDK directly.
        """
        return (
            model.startswith("openai/")
            and self.api_base is not None
            and self.api_key is not None
        )

    async def shutdown(self) -> None:
        """Release transport resources held by this backend."""
        client = getattr(self, "_oai_client", None)
        if client is None:
            return
        await client.close()
        self._oai_client = None

    @staticmethod
    def _content_to_responses_input_parts(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]

        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    parts.append({"type": "input_text", "text": item})
                    continue
                if not isinstance(item, dict):
                    parts.append({"type": "input_text", "text": str(item)})
                    continue
                item_type = str(item.get("type") or "").strip()
                if item_type == "text":
                    parts.append({"type": "input_text", "text": str(item.get("text") or "")})
                    continue
                if item_type == "image_url":
                    raw_url = item.get("image_url")
                    if isinstance(raw_url, dict):
                        raw_url = raw_url.get("url")
                    if raw_url:
                        parts.append({
                            "type": "input_image",
                            "image_url": str(raw_url),
                            "detail": "auto",
                        })
                    continue
                parts.append({"type": "input_text", "text": json.dumps(item, ensure_ascii=False, default=str)})
            return parts or [{"type": "input_text", "text": ""}]

        return [{"type": "input_text", "text": str(content)}]

    @staticmethod
    def _assistant_content_to_responses_output_parts(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if not isinstance(item, dict):
                    text_parts.append(str(item))
                    continue
                item_type = str(item.get("type") or "").strip()
                if item_type == "text":
                    text_parts.append(str(item.get("text") or ""))
                    continue
                if item_type == "image_url":
                    raw_url = item.get("image_url")
                    if isinstance(raw_url, dict):
                        raw_url = raw_url.get("url")
                    text_parts.append(f"[image] {raw_url}")
                    continue
                text_parts.append(json.dumps(item, ensure_ascii=False, default=str))
            text = "\n".join(part for part in text_parts if part).strip()
        else:
            text = str(content)
        return [{
            "type": "output_text",
            "text": text,
            "annotations": [],
        }]

    @classmethod
    def _messages_to_responses_input(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _instructions, input_items = cls._messages_to_responses_request(messages)
        return input_items

    @classmethod
    def _messages_to_responses_request(
        cls,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instruction_blocks: list[str] = []
        input_items: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages or []):
            if not isinstance(msg, dict):
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": str(msg)}],
                })
                continue
            role = str(msg.get("role") or "user").strip().lower() or "user"
            content = msg.get("content", "")
            if role in {"system", "developer", "user"}:
                if role in {"system", "developer"}:
                    parts = cls._content_to_responses_input_parts(content)
                    text = "\n".join(
                        str(part.get("text") or "")
                        for part in parts
                        if isinstance(part, dict) and part.get("type") == "input_text"
                    ).strip()
                    if text:
                        instruction_blocks.append(text)
                    continue
                input_items.append({
                    "type": "message",
                    "role": role,
                    "content": cls._content_to_responses_input_parts(content),
                })
                continue
            if role == "assistant":
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": cls._assistant_content_to_responses_output_parts(content),
                })
                continue
            input_items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": str(content)}],
            })
        instructions = "\n\n".join(block for block in instruction_blocks if block).strip() or None
        return instructions, input_items

    @staticmethod
    def _responses_text_config(response_format: str | None) -> dict[str, Any] | None:
        if response_format == "json":
            return {"format": {"type": "json_object"}}
        return None

    @staticmethod
    def _ensure_json_keyword_in_responses_input(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure JSON mode mentions `json` in input text for API compatibility.

        New definition:
            Some OpenAI-compatible Responses backends reject
            ``text.format={"type":"json_object"}`` unless the request input
            explicitly contains the word ``json``. When migrating system prompts
            to ``instructions``, that keyword may disappear from the actual
            ``input`` items. We fix that at the transport layer only for JSON
            mode to preserve existing caller prompts unchanged.
        """
        normalized = json.dumps(input_items, ensure_ascii=False).lower()
        if "json" in normalized:
            return input_items
        for item in input_items:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text":
                    part["text"] = f"Return valid JSON.\n\n{str(part.get('text') or '')}".strip()
                    return input_items
        input_items.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Return valid JSON."}],
        })
        return input_items

    @staticmethod
    def _chat_tools_to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for spec in tools or []:
            if not isinstance(spec, dict):
                continue
            if spec.get("type") == "function" and isinstance(spec.get("function"), dict):
                fn = spec["function"]
                converted.append({
                    "type": "function",
                    "name": str(fn.get("name") or ""),
                    "description": str(fn.get("description") or ""),
                    "parameters": fn.get("parameters") or {"type": "object"},
                    "strict": bool(fn.get("strict", False)),
                })
                continue
            converted.append(spec)
        return converted

    def _get_openai_client(self):
        """Lazily create a shared AsyncOpenAI client for proxy calls."""
        if not hasattr(self, "_oai_client") or self._oai_client is None:
            from openai import AsyncOpenAI
            self._oai_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
            )
        return self._oai_client

    async def _call_native_openai(
        self, call_kwargs: dict[str, Any], model_label: str,
        response_format: str | None, messages: list[dict[str, Any]],
    ) -> LLMResponse | None:
        """Call via the native ``openai`` SDK Responses API."""
        client = self._get_openai_client()

        raw_model = call_kwargs["model"]
        if raw_model.startswith("openai/"):
            raw_model = raw_model[len("openai/"):]

        oai_kwargs: dict[str, Any] = {
            "model": raw_model,
            "max_output_tokens": call_kwargs.get("max_tokens", self.max_tokens),
        }
        instructions, input_items = self._messages_to_responses_request(messages)
        if response_format == "json":
            input_items = self._ensure_json_keyword_in_responses_input(input_items)
        oai_kwargs["input"] = input_items
        if instructions:
            oai_kwargs["instructions"] = instructions
        text_cfg = self._responses_text_config(response_format)
        if text_cfg:
            oai_kwargs["text"] = text_cfg

        for attempt in range(_MAX_RETRIES):
            try:
                response = await client.responses.create(**oai_kwargs)
                parsed = self._parse_response(response, model_label)
                if _is_empty(parsed) and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        f"LLM {model_label} attempt {attempt+1}: empty response, "
                        f"retrying in {_RETRY_BASE_DELAY * (2 ** attempt):.1f}s..."
                    )
                    await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    continue
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

    @staticmethod
    async def _stream_to_completion(client, oai_kwargs: dict, model: str):
        """Collect a streaming response into a fake ChatCompletion object."""
        oai_kwargs_stream = {**oai_kwargs, "stream": True}
        stream = await client.chat.completions.create(**oai_kwargs_stream)
        content_parts: list[str] = []
        tool_calls_agg: list[Any] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
            if getattr(delta, "tool_calls", None):
                tool_calls_agg.extend(delta.tool_calls)

        class _Msg:
            def __init__(self):
                self.content = "".join(content_parts)
                self.role = "assistant"
                self.tool_calls = tool_calls_agg or None

        class _Choice:
            def __init__(self):
                self.message = _Msg()
                self.finish_reason = "stop"

        class _FakeResp:
            def __init__(self):
                self.choices = [_Choice()]
                self.model = model
                self.usage = None

        return _FakeResp()

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
                if _is_empty(parsed) and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        f"LLM {model_label} attempt {attempt+1}: empty response, "
                        f"retrying in {_RETRY_BASE_DELAY * (2 ** attempt):.1f}s..."
                    )
                    await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    continue
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

        caller = (
            self._call_native_openai
            if self._use_native_openai(target_model)
            else self._call_with_retry
        )

        try:
            result = await caller(
                call_kwargs, target_model, response_format, messages)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"LLM call to {target_model} failed: {e}")
            if self.fallback_model and target_model != self.fallback_model:
                logger.info(f"Falling back to {self.fallback_model}")
                call_kwargs["model"] = self.fallback_model
                try:
                    result = await caller(
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
        target_model = model or self.default_model

        if self._use_native_openai(target_model):
            return await self._call_native_openai_tools(
                messages, tools, target_model, **kwargs)

        import litellm

        call_kwargs = self._build_kwargs(messages, target_model, **kwargs)
        call_kwargs["tools"] = tools

        for attempt in range(_MAX_RETRIES):
            try:
                response = await litellm.acompletion(**call_kwargs)
                parsed = self._parse_response(response, target_model)
                self._hydrate_textual_tool_calls(parsed, tools)
                trace.llm_call(target_model, messages, parsed, response_format="tool_call")
                return parsed
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
                        parsed = self._parse_response(response, self.fallback_model)
                        self._hydrate_textual_tool_calls(parsed, tools)
                        trace.llm_call(self.fallback_model, messages, parsed, response_format="tool_call")
                        return parsed
                    except Exception as e2:
                        logger.error(f"Fallback tool call also failed: {e2}")
                return LLMResponse(text=None, model=target_model)
        return LLMResponse(text=None, model=target_model)

    async def _call_native_openai_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Tool-calling via the native OpenAI SDK Responses API."""
        client = self._get_openai_client()
        raw_model = model[len("openai/"):] if model.startswith("openai/") else model

        oai_kwargs: dict[str, Any] = {
            "model": raw_model,
            "tools": self._chat_tools_to_responses_tools(tools),
            "tool_choice": kwargs.get("tool_choice"),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        instructions, input_items = self._messages_to_responses_request(messages)
        oai_kwargs["input"] = input_items
        if instructions:
            oai_kwargs["instructions"] = instructions
        if oai_kwargs["tool_choice"] is None:
            oai_kwargs.pop("tool_choice")

        for attempt in range(_MAX_RETRIES):
            try:
                response = await client.responses.create(**oai_kwargs)
                parsed = self._parse_response(response, model)
                self._hydrate_textual_tool_calls(parsed, tools)
                if _is_empty(parsed) and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        f"Tool call {model} attempt {attempt+1}: empty response, "
                        f"retrying in {_RETRY_BASE_DELAY * (2 ** attempt):.1f}s..."
                    )
                    await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                trace.llm_call(model, messages, parsed, response_format="tool_call")
                return parsed
            except Exception as e:
                if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Tool call {model} attempt {attempt+1} "
                        f"(retryable): {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"Tool call to {model} failed: {e}")
                return LLMResponse(text=None, model=model)
        return LLMResponse(text=None, model=model)

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
    def _tool_names_from_specs(tools: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for spec in tools or []:
            if not isinstance(spec, dict):
                continue
            fn = spec.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _recover_textual_tool_call(
        text: str | None,
        *,
        allowed_tool_names: list[str],
    ) -> list[dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return []

        for name in sorted(allowed_tool_names, key=len, reverse=True):
            if not raw.startswith(name):
                continue
            suffix = raw[len(name):].lstrip()
            if not suffix or suffix[0] not in "{[":
                continue
            try:
                decoder = json.JSONDecoder()
                arguments, end = decoder.raw_decode(suffix)
            except Exception:
                continue
            trailing = suffix[end:].strip()
            if trailing:
                return []
            if not isinstance(arguments, dict):
                return []
            return [{"name": name, "arguments": arguments}]
        return []

    @classmethod
    def _hydrate_textual_tool_calls(
        cls,
        response: LLMResponse,
        tools: list[dict[str, Any]],
    ) -> None:
        if response.tool_calls:
            return
        recovered = cls._recover_textual_tool_call(
            response.text,
            allowed_tool_names=cls._tool_names_from_specs(tools),
        )
        if not recovered:
            return
        response.tool_calls = recovered
        if response.parsed is None:
            response.parsed = recovered[0]["arguments"]

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
    def _usage_to_dict(usage_obj: Any) -> dict[str, int]:
        if not usage_obj:
            return {}
        raw: Any = usage_obj
        if hasattr(usage_obj, "model_dump"):
            try:
                raw = usage_obj.model_dump(mode="python")
            except TypeError:
                raw = usage_obj.model_dump()
        elif hasattr(usage_obj, "dict"):
            try:
                raw = usage_obj.dict()
            except Exception:
                raw = usage_obj
        elif hasattr(usage_obj, "__dict__"):
            raw = {
                key: value
                for key, value in vars(usage_obj).items()
                if not str(key).startswith("_")
            }

        def _coerce(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                return {str(k): _coerce(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_coerce(v) for v in value]
            if hasattr(value, "model_dump"):
                try:
                    return _coerce(value.model_dump(mode="python"))
                except TypeError:
                    return _coerce(value.model_dump())
            if hasattr(value, "dict"):
                try:
                    return _coerce(value.dict())
                except Exception:
                    return str(value)
            if hasattr(value, "__dict__"):
                return _coerce({
                    key: val
                    for key, val in vars(value).items()
                    if not str(key).startswith("_")
                })
            return str(value)

        usage = _coerce(raw)
        if not isinstance(usage, dict):
            return {}

        def _as_int(value: Any) -> int | None:
            if isinstance(value, bool) or value is None:
                return None
            if isinstance(value, (int, float)):
                return int(value)
            return None

        summary: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"):
            value = _as_int(usage.get(key))
            if value is not None:
                summary[key] = value

        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict):
            cached_tokens = _as_int(input_details.get("cached_tokens"))
            if cached_tokens is not None:
                summary["cached_tokens"] = cached_tokens

        output_details = usage.get("output_tokens_details")
        if isinstance(output_details, dict):
            reasoning_tokens = _as_int(output_details.get("reasoning_tokens"))
            if reasoning_tokens is not None:
                summary["reasoning_tokens"] = reasoning_tokens

        if "prompt_tokens" not in summary and "input_tokens" in summary:
            summary["prompt_tokens"] = summary["input_tokens"]
        if "completion_tokens" not in summary and "output_tokens" in summary:
            summary["completion_tokens"] = summary["output_tokens"]
        if "total_tokens" not in summary:
            prompt = summary.get("prompt_tokens", 0)
            completion = summary.get("completion_tokens", 0)
            summary["total_tokens"] = prompt + completion
        return summary

    @staticmethod
    def _parse_response(response: Any, model: str) -> LLMResponse:
        if isinstance(response, str):
            return LLMBackend._parse_responses_sse_string(response, model)

        if not getattr(response, "choices", None) and hasattr(response, "output"):
            output_items = getattr(response, "output", None)
            if output_items is None:
                logger.warning(f"Empty output from {model}: {getattr(response, 'error', None) or getattr(response, 'detail', None)}")
                return LLMResponse(text=None, model=model, raw=response)
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for item in output_items or []:
                item_type = getattr(item, "type", None)
                if item_type == "message":
                    for content in getattr(item, "content", []) or []:
                        content_type = getattr(content, "type", None)
                        if content_type == "output_text":
                            text_value = getattr(content, "text", None)
                            if text_value:
                                text_parts.append(str(text_value))
                    continue
                if item_type == "function_call":
                    args = getattr(item, "arguments", None)
                    parsed_args: Any = args
                    if isinstance(args, str):
                        try:
                            parsed_args = json.loads(args)
                        except json.JSONDecodeError:
                            parsed_args = args
                    tool_calls.append({
                        "name": getattr(item, "name", ""),
                        "arguments": parsed_args,
                    })

            text = "\n".join(part for part in text_parts if part).strip()
            if not text and output_items is not None and hasattr(response, "output_text"):
                text = str(getattr(response, "output_text") or "").strip()
            parsed = LLMBackend._extract_json(text) if text else None
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = LLMBackend._usage_to_dict(response.usage)
            return LLMResponse(
                text=text or None,
                parsed=parsed,
                tool_calls=tool_calls,
                model=model,
                usage=usage,
                raw=response,
            )

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
            usage = LLMBackend._usage_to_dict(response.usage)

        return LLMResponse(
            text=text,
            parsed=parsed,
            tool_calls=tool_calls,
            model=model,
            usage=usage,
            raw=response,
        )

    @staticmethod
    def _parse_responses_sse_string(raw: str, model: str) -> LLMResponse:
        """Parse a proxy-degraded Responses SSE transcript into LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        completed_response: dict[str, Any] | None = None

        blocks = [block.strip() for block in re.split(r"\n\s*\n", str(raw or "").strip()) if block.strip()]
        for block in blocks:
            event_name = ""
            data_json = None
            for line in block.splitlines():
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    try:
                        data_json = json.loads(payload)
                    except json.JSONDecodeError:
                        data_json = None
            if not isinstance(data_json, dict):
                continue

            if event_name == "response.output_item.done":
                item = data_json.get("item") or {}
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip()
                if item_type == "message":
                    for content in item.get("content") or []:
                        if not isinstance(content, dict):
                            continue
                        if str(content.get("type") or "").strip() == "output_text":
                            text_value = str(content.get("text") or "")
                            if text_value:
                                text_parts.append(text_value)
                    continue
                if item_type == "function_call":
                    args = item.get("arguments")
                    parsed_args: Any = args
                    if isinstance(args, str):
                        try:
                            parsed_args = json.loads(args)
                        except json.JSONDecodeError:
                            parsed_args = args
                    tool_calls.append({
                        "name": str(item.get("name") or ""),
                        "arguments": parsed_args,
                    })
                    continue

            if event_name == "response.completed":
                response_payload = data_json.get("response")
                if isinstance(response_payload, dict):
                    completed_response = response_payload

        if completed_response and completed_response.get("usage"):
            usage = LLMBackend._usage_to_dict(completed_response["usage"])

        text = "\n".join(part for part in text_parts if part).strip() or None
        parsed = LLMBackend._extract_json(text) if text else None
        if text is None and not tool_calls:
            logger.warning(
                f"Empty SSE output from {model}: "
                f"{(completed_response or {}).get('error') or (completed_response or {}).get('detail')}"
            )
        return LLMResponse(
            text=text,
            parsed=parsed,
            tool_calls=tool_calls,
            model=model,
            usage=usage,
            raw=raw,
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
