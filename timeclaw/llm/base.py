"""Unified LLM backend wrapping LiteLLM for both cloud and local model access.

Supports multimodal messages (text + image) via OpenAI-compatible content arrays.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import httpx
import json
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse, urlunparse

from loguru import logger
from pydantic import BaseModel, Field
import websockets
from websockets.exceptions import ConnectionClosed

from timeclaw.core import trace_logger as trace
from timeclaw.core.accounting import accounting_active, logical_call, physical_attempt
from timeclaw.core.retry_control import (
    RetryLater,
    current_work_item_id,
    should_defer_retry_to_scheduler,
)

_RETRY_DELAYS_SECONDS = (2.0, 8.0, 32.0, 128.0)
_DEFAULT_REASONING_EFFORT = object()
_DEFAULT_WS_USER_AGENT = "timeclaw/llm-ws"
_DEFAULT_WS_ORIGINATOR = "timeclaw"
_WS_PARSER_DIAGNOSTICS_DIR = Path(".timeclaw/ws_parser_diagnostics")
_SUPPLIER_POOL_RETRY_PATTERNS = (
    "usage_limit_reached",
    "the usage limit has been reached",
    "account is busy",
    "too many concurrent requests",
    "too many pending requests",
    "did not receive a valid http response",
)


class ProviderAccountQuotaError(RuntimeError):
    """The provider account has no paid quota; reconnecting cannot repair it."""


class ProviderContentPolicyError(RuntimeError):
    """The provider rejected content (1301); this is not a transport failure."""


class ProviderRateLimitError(RuntimeError):
    """The provider reports temporary rate limiting inside a streamed response."""


class ProviderStreamTransportError(RuntimeError):
    """An explicit provider error code reports an interrupted upstream stream."""


def _is_temporary_provider_payload(error: Any) -> bool:
    """Recognize the relay's explicit SSE temporary-unavailability envelope."""
    return (
        isinstance(error, dict)
        and str(error.get("type") or error.get("code") or "").lower() in {"upstream_error", "api_error"}
        and str(error.get("message", "")).strip().lower() in {
            "upstream service temporarily unavailable",
            "service temporarily unavailable",
            "service is temporarily unavailable",
        }
        and str(error.get("code")) not in {"1301", "1113"}
    )


def _check_provider_stream_error(line: bytes) -> None:
    payload = line.strip()
    if payload.startswith(b"data:"):
        payload = payload[5:].strip()
    if not payload.startswith(b"{"):
        return
    try:
        event = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return
    error = event.get("error")
    if not isinstance(error, dict):
        return
    code = str(error.get("code", "unknown"))
    message = str(error.get("message", "Provider rejected the request"))
    if code == "1301":
        raise ProviderContentPolicyError(f"Provider content policy rejection (code 1301): {message}")
    if code == "1113":
        raise ProviderAccountQuotaError("Provider account has no available balance/resource package (code 1113)")
    if code == "1302":
        raise ProviderRateLimitError(f"Provider rate limit (code 1302): {message}")
    if _is_temporary_provider_payload(error):
        raise ProviderStreamTransportError(f"Provider stream error: {message}")
    if code in {"upstream_http2_stream_error", "upstream_stream_read_error"}:
        raise ProviderStreamTransportError(f"Provider stream error (code {code}): {message}")
    raise RuntimeError(f"Provider stream error (code {code}): {message}")


class _ProviderErrorAwareStream(httpx.AsyncByteStream):
    """Expose JSON error tails that an SSE decoder would otherwise discard.

    GLM can end an HTTP 200 event stream with a bare JSON error line. Inspect
    complete wire lines before forwarding unchanged bytes to the SDK. Buffering
    bytes, rather than decoded text, also handles split UTF-8 characters.
    """

    def __init__(self, stream: httpx.AsyncByteStream):
        self.stream = stream

    async def __aiter__(self):
        pending = b""
        async for chunk in self.stream:
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                _check_provider_stream_error(line)
            yield chunk
        if pending:
            _check_provider_stream_error(pending)

    async def aclose(self):
        await self.stream.aclose()


async def _inspect_chat_stream_errors(response: httpx.Response) -> None:
    if response.status_code == 200 and "text/event-stream" in response.headers.get("content-type", ""):
        response.stream = _ProviderErrorAwareStream(response.stream)


def _retry_delay_seconds(attempt: int) -> float:
    """Return the configured retry delay for a 0-based retry attempt."""
    idx = max(0, min(int(attempt), len(_RETRY_DELAYS_SECONDS) - 1))
    return float(_RETRY_DELAYS_SECONDS[idx])


def retry_delay_seconds_for_attempt(attempt: int) -> float:
    """Public helper so non-LLM paths can reuse the same retry cadence."""
    return _retry_delay_seconds(attempt)


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient API error worth retrying."""
    if isinstance(exc, (ProviderContentPolicyError, ProviderAccountQuotaError)):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in {408, 429}:
        return False
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "429",
            "rate limit",
            "too many requests",
            "server error",
            "an error occurred while processing your request",
            "502",
            "503",
            "504",
            "timeout",
            "connection",
            "disconnect",
            "overloaded",
            "1013",
            "1011",
            "upstream websocket proxy failed",
            "busy",
            "incomplete",
            "retry later",
        )
    )


def is_retryable_error_message(message: str | None) -> bool:
    """Public helper for reusing the transport retry taxonomy on caller-side wrappers."""
    return _is_retryable(RuntimeError(str(message or "")))


def _is_supplier_pool_retry(exc: Exception) -> bool:
    """True for supplier-side account/concurrency rotation failures.

    New definition:
        These errors mean the request reached the supplier, but the supplier
        rejected it because its backend account pool or concurrency pool was
        temporarily unavailable. We retry them silently and they do not consume
        TimeClaw's own retry budget.
    """
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _SUPPLIER_POOL_RETRY_PATTERNS)


def is_supplier_pool_retry_message(message: str | None) -> bool:
    """Public helper mirroring `_is_supplier_pool_retry` for raw string errors."""
    return _is_supplier_pool_retry(RuntimeError(str(message or "")))


def _is_unbounded_supplier_backpressure(exc: Exception) -> bool:
    """True for supplier-side queue saturation that should retry without a global cap.

    New definition:
        This is narrower than `_is_supplier_pool_retry`. It is only for cases
        where the upstream explicitly says its pending request queue is full and
        asks the caller to retry later. Those retries should not consume
        TimeClaw's retry budget or the current request's accumulated deadline.
    """
    msg = str(exc).lower()
    return (
        "too many pending requests" in msg
        or "concurrency limit exceeded for user" in msg
        or "upstream rate limit exceeded" in msg
    )


def _is_unbounded_network_retry(exc: Exception) -> bool:
    """Classify transport failures for the configurable reconnect policy.

    New definition:
        Network-path failures such as 502/503/504, proxy disconnects,
        connection errors, and timeouts are treated as infrastructure noise
        instead of model-quality retries. The backend's network_retry_attempts
        controls their limit; None means unlimited reconnects.
    """
    # httpx and asyncio timeout classes commonly stringify to an empty string.
    # Classify transport failures by type first so they cannot bypass the
    # unlimited network retry policy merely because their message is blank.
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway",
            "timeout",
            "timed out",
            "connection",
            "disconnect",
            "upstream websocket proxy failed",
            "sse stream ended before response.completed",
            "the request could not be processed",
            "cloudflare",
            "host error",
        )
    )


def _is_transient_infrastructure_error(exc: Exception) -> bool:
    """Temporary API/network failures share one retry budget and delay.

    HTTP status and typed provider errors take precedence over message text.
    Content refusal, exhausted paid quota, authentication and invalid requests
    cannot be repaired by repeating the same request.
    """
    if isinstance(exc, (ProviderContentPolicyError, ProviderAccountQuotaError)):
        return False
    if isinstance(exc, (ProviderRateLimitError, ProviderStreamTransportError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in {408, 429} or 500 <= status < 600:
            return True
        if 400 <= status < 500:
            return False
    # The SDK may turn an SSE error into APIError before our stream wrapper.
    # Read its structured body as well as typed stream exceptions.
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and _is_temporary_provider_payload(body.get("error", body)):
        return True
    return (
        _is_unbounded_network_retry(exc)
        or _is_unbounded_supplier_backpressure(exc)
        or _is_supplier_pool_retry(exc)
        or any(token in str(exc).lower() for token in (
            "429", "rate limit", "too many requests", "500", "server error",
            "an error occurred while processing your request", "overloaded",
            # A relay may return HTTP 200, then report unavailability in SSE
            # without a status code. Use the same policy as an HTTP 503.
            "service is temporarily unavailable",
        ))
    )


def _exception_diagnostic(exc: BaseException) -> str:
    """Return a non-empty exception description suitable for logs/artifacts."""
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__module__}.{type(exc).__name__}"


def _reasoning_effort_for_model(model: str, effort: str | None) -> str | None:
    """Translate caller intent to the reasoning values supported by a provider."""
    normalized_model = str(model or "").strip().lower()
    if normalized_model in {"openai/glm-5.3-flash", "openai/glm-5.3-flashx"}:
        # This experiment uses the model card's recommended maximum effort.
        # Existing helper-level OpenAI `none`/`medium` overrides must not turn
        # off or alter GLM's mandatory thinking mode.
        return "max"
    normalized_effort = str(effort or "").strip().lower() or None
    # The Grok proxy returns HTTP 502 for `none`; an otherwise identical real
    # training request completes when sent with `high`.
    if normalized_effort == "none" and normalized_model.startswith("openai/grok-"):
        return "high"
    return normalized_effort


async def _sleep_with_deadline(delay: float, deadline: float) -> bool:
    """Sleep up to ``delay`` seconds without crossing the absolute deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    sleep_for = min(float(delay), remaining)
    if sleep_for <= 0:
        return False
    await asyncio.sleep(sleep_for)
    return True


def _is_empty(resp: "LLMResponse | None") -> bool:
    """True if response has no usable content."""
    if resp is None:
        return True
    return not resp.text


class LLMResponse(BaseModel):
    """Standardized LLM response."""

    text: str | None = None
    parsed: dict[str, Any] | list | None = None
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    raw: Any = None


class LLMRequestGate:
    """Process-wide gate for real outbound LLM requests.

    New definition:
        ``max_inflight_requests`` counts only requests that have already been
        dispatched to the external provider and are still awaiting completion.
        Local queue wait before dispatch and retry sleep after failure do not
        consume this budget.
    """

    def __init__(self, max_inflight_requests: int | None = None):
        limit = int(max_inflight_requests) if max_inflight_requests is not None else None
        self.max_inflight_requests = limit if limit and limit > 0 else None
        self._semaphore = (
            asyncio.Semaphore(self.max_inflight_requests)
            if self.max_inflight_requests is not None
            else None
        )
        self._active = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, model_label: str) -> dict[str, Any]:
        queued_at = time.monotonic()
        if self._semaphore is not None:
            await self._semaphore.acquire()
        queue_wait_s = time.monotonic() - queued_at
        request_id = f"llmreq-{uuid.uuid4().hex[:12]}"
        async with self._lock:
            self._active += 1
            inflight_after_acquire = self._active
        audit = {
            "request_id": request_id,
            "model_label": model_label,
            "queue_wait_ms": round(queue_wait_s * 1000.0, 1),
            "inflight_after_acquire": inflight_after_acquire,
            "max_inflight_requests": self.max_inflight_requests,
        }
        try:
            yield audit
        finally:
            async with self._lock:
                self._active = max(0, self._active - 1)
                inflight_after_release = self._active
            audit["inflight_after_release"] = inflight_after_release
            if self._semaphore is not None:
                self._semaphore.release()


class LLMBackend:
    """Thin wrapper around litellm providing unified async interface for all providers."""

    def __init__(
        self,
        default_model: str = "ollama/qwen3:8b",
        fallback_model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        max_retries: int = 5,
        network_retry_attempts: int | None = None,
        network_retry_delay_s: float = 30.0,
        timeout: float = 3600.0,
        first_token_timeout: float | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        azure_api_base: str | None = None,
        azure_api_key: str | None = None,
        azure_api_version: str | None = None,
        disable_response_storage: bool = False,
        supports_websockets: bool = False,
        responses_websockets_v2: bool = False,
        reasoning_effort: str | None = None,
        request_gate: LLMRequestGate | None = None,
        api_transport: str = "responses",
        send_temperature: bool = True,
    ):
        import os

        self.default_model = default_model
        if api_transport not in {"responses", "chat_completions"}:
            raise ValueError(f"Unsupported LLM API transport: {api_transport}")
        self.api_transport = api_transport
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.send_temperature = send_temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.network_retry_attempts = (
            None if network_retry_attempts is None else int(network_retry_attempts)
        )
        if self.network_retry_attempts is not None and self.network_retry_attempts < 0:
            raise ValueError("network_retry_attempts must be non-negative or None")
        self.network_retry_delay_s = max(0.0, float(network_retry_delay_s))
        self.timeout = timeout
        self.first_token_timeout = first_token_timeout
        self.api_base = api_base
        self.api_key = api_key
        self.azure_api_base = azure_api_base
        self.azure_api_key = azure_api_key
        self.azure_api_version = azure_api_version or "2024-12-01-preview"
        self.disable_response_storage = bool(disable_response_storage)
        self.supports_websockets = bool(supports_websockets)
        self.responses_websockets_v2 = bool(responses_websockets_v2)
        self.reasoning_effort = (reasoning_effort.strip() if reasoning_effort else None) or None
        self.request_gate = request_gate or LLMRequestGate()
        self._deferred_retry_state: dict[str, dict[str, float | int]] = {}

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

    def _use_responses_websocket(self) -> bool:
        """Whether native OpenAI proxy calls should use websocket transport."""
        return self.supports_websockets and self.responses_websockets_v2

    def _take_network_retry(self, supplier_retry_count: int) -> bool:
        """None permits unlimited transient retries; 0 disables reconnects."""
        return (
            self.network_retry_attempts is None
            or int(supplier_retry_count) < self.network_retry_attempts
        )

    @staticmethod
    def _request_audit_suffix(audit: dict[str, Any] | None) -> str:
        if not isinstance(audit, dict) or not audit:
            return ""
        limit = audit.get("max_inflight_requests")
        limit_text = str(limit) if limit is not None else "unbounded"
        return (
            " "
            f"[request_id={audit.get('request_id')}, "
            f"queue_wait_ms={audit.get('queue_wait_ms')}, "
            f"inflight={audit.get('inflight_after_acquire')}/{limit_text}]"
        )

    @staticmethod
    def _effective_max_attempts(call_kwargs: dict[str, Any], default: int) -> int:
        value = call_kwargs.get("num_retries", default)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(default))

    @staticmethod
    def _effective_timeout(call_kwargs: dict[str, Any], default: float) -> float:
        value = call_kwargs.get("timeout", default)
        try:
            return max(1.0, float(value))
        except (TypeError, ValueError):
            return max(1.0, float(default))

    @staticmethod
    def _effective_first_token_timeout(
        call_kwargs: dict[str, Any],
        default: float | None,
    ) -> float | None:
        value = call_kwargs.get("first_token_timeout", default)
        if value is None:
            return None
        try:
            return max(1.0, float(value))
        except (TypeError, ValueError):
            return default

    def _deferred_retry_key(
        self,
        *,
        model_label: str,
        messages: list[dict[str, Any]],
        response_format: str | None,
        reasoning_effort: str | None,
    ) -> str | None:
        work_item_id = current_work_item_id()
        if not should_defer_retry_to_scheduler() or not work_item_id:
            return None
        payload = {
            "model": model_label,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            "messages": messages,
        }
        signature = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"{work_item_id}:{signature}"

    def _deferred_retry_load_state(self, retry_key: str | None) -> dict[str, float | int]:
        if not retry_key:
            return {}
        state = self._deferred_retry_state.get(retry_key)
        if state is None:
            state = {}
            self._deferred_retry_state[retry_key] = state
        return state

    def _deferred_retry_clear_state(self, retry_key: str | None) -> None:
        if retry_key:
            self._deferred_retry_state.pop(retry_key, None)

    def _deferred_retry_raise(
        self,
        *,
        retry_key: str | None,
        delay: float,
        reason: str,
        layer: str,
        state: dict[str, float | int] | None = None,
    ) -> None:
        if retry_key and state is not None:
            self._deferred_retry_state[retry_key] = dict(state)
        raise RetryLater(delay_s=delay, reason=reason, layer=layer)

    @classmethod
    def _shadow_extract_text_from_ws_payload(cls, value: Any) -> list[str]:
        """Best-effort diagnostic text extraction for parser-empty WS responses.

        New definition:
            This is a verification-only extractor used when the current parser
            yields an empty response. It is intentionally more permissive than
            the production parser so we can tell whether the provider returned
            usable text that our strict parser failed to recover.
        """
        parts: list[str] = []
        if value is None:
            return parts
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text)
            return parts
        if isinstance(value, list):
            for item in value:
                parts.extend(cls._shadow_extract_text_from_ws_payload(item))
            return parts
        if not isinstance(value, dict):
            return parts

        for key in ("text", "output_text", "value", "delta"):
            if key in value:
                parts.extend(cls._extract_text_fragments(value.get(key)))

        for key in ("content", "part", "content_part", "item", "response", "output"):
            if key in value:
                parts.extend(cls._shadow_extract_text_from_ws_payload(value.get(key)))

        return parts

    @classmethod
    def _shadow_extract_text_from_ws_events(cls, events: list[dict[str, Any]]) -> str | None:
        parts: list[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            parts.extend(cls._shadow_extract_text_from_ws_payload(event))
        text = "".join(part for part in parts if part).strip()
        return text or None

    @classmethod
    def _record_empty_ws_parser_diagnostic(
        cls,
        *,
        events: list[dict[str, Any]],
        model_label: str,
        response_format: str | None,
        request_audit: dict[str, Any] | None,
        parsed: "LLMResponse | None",
    ) -> None:
        request_id = str((request_audit or {}).get("request_id") or f"wsdiag-{uuid.uuid4().hex[:12]}")
        event_types = [
            str(event.get("type") or "").strip()
            for event in events
            if isinstance(event, dict)
        ]
        completed_response = None
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            if str(event.get("type") or "").strip() == "response.completed":
                response_payload = event.get("response")
                if isinstance(response_payload, dict):
                    completed_response = response_payload
                break

        shadow_text = cls._shadow_extract_text_from_ws_events(events)
        payload = {
            "request_id": request_id,
            "model": model_label,
            "response_format": response_format,
            "event_count": len(events),
            "event_types": event_types,
            "had_response_completed": any(t == "response.completed" for t in event_types),
            "had_error_event": any(t == "error" for t in event_types),
            "current_parser_text": str(parsed.text or "") if parsed is not None else "",
            "shadow_parser_text": shadow_text or "",
            "queue_wait_ms": (request_audit or {}).get("queue_wait_ms"),
            "inflight_after_acquire": (request_audit or {}).get("inflight_after_acquire"),
            "max_inflight_requests": (request_audit or {}).get("max_inflight_requests"),
            "completed_response": completed_response,
            "events": events,
        }
        try:
            _WS_PARSER_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
            ts_ms = int(time.time() * 1000)
            path = _WS_PARSER_DIAGNOSTICS_DIR / f"{ts_ms}_{request_id}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            if shadow_text:
                logger.warning(
                    f"WebSocket parser-empty diagnostic recovered shadow text for {model_label} "
                    f"(request_id={request_id}, chars={len(shadow_text)}): {path}"
                )
            else:
                logger.warning(
                    f"WebSocket parser-empty diagnostic recorded raw events for {model_label} "
                    f"(request_id={request_id}): {path}"
                )
        except Exception as exc:
            logger.warning(f"Failed to write WS parser diagnostic for {model_label}: {exc}")

    async def shutdown(self) -> None:
        """Release transport resources held by this backend."""
        client = getattr(self, "_oai_client", None)
        if client is None:
            return
        if hasattr(client, "aclose"):
            await client.aclose()
        else:
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
                assistant_item = {
                    "type": "message",
                    "role": "assistant",
                    "content": cls._assistant_content_to_responses_output_parts(content),
                }
                if msg.get("phase") is not None:
                    assistant_item["phase"] = msg["phase"]
                input_items.append(assistant_item)
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

    def _responses_ws_endpoint(self) -> str:
        """Build the Responses WebSocket endpoint from the configured API base.

        New definition:
            For a configured base like ``https://host/v1`` we connect to
            ``wss://host/v1/responses``. For ``https://host`` we connect to
            ``wss://host/responses``.
        """
        parsed = urlparse(str(self.api_base or "").strip())
        if parsed.scheme not in {"http", "https", "ws", "wss"}:
            raise ValueError(
                f"Unsupported API base scheme for websocket transport: {parsed.scheme or '<empty>'}"
            )
        ws_scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
        path = parsed.path.rstrip("/")
        if not path:
            path = "/responses"
        elif path.endswith("/responses"):
            pass
        else:
            path = f"{path}/responses"
        return urlunparse((ws_scheme, parsed.netloc, path, "", parsed.query, ""))

    def _responses_ws_headers(self) -> dict[str, str]:
        """Return headers required by the Responses WebSocket transport.

        New definition:
            This supplier's ``/responses`` websocket ingress requires a
            ``session_id`` header even for stateless single-shot requests.
            We generate a fresh random value per request so the field satisfies
            the gateway requirement without turning separate requests into a
            shared long-lived conversation.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "responses=v1",
            "originator": _DEFAULT_WS_ORIGINATOR,
            "X-Request-ID": f"timeclaw-ws-{uuid.uuid4()}",
            "session_id": f"timeclaw-session-{uuid.uuid4()}",
        }

    async def _call_native_openai(
        self, call_kwargs: dict[str, Any], model_label: str,
        response_format: str | None, messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> LLMResponse | None:
        if self.api_transport == "chat_completions":
            return await self._call_with_retry(
                call_kwargs, model_label, response_format, messages,
                reasoning_effort, native_chat=True,
            )
        if self._use_responses_websocket():
            return await self._call_native_openai_ws(
                call_kwargs, model_label, response_format, messages, reasoning_effort
            )
        return await self._call_native_openai_http(
            call_kwargs, model_label, response_format, messages, reasoning_effort
        )

    async def _call_native_openai_ws(
        self, call_kwargs: dict[str, Any], model_label: str,
        response_format: str | None, messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> LLMResponse | None:
        """Call the proxy's Responses API over WebSocket.

        New definition:
            Native ``openai/`` proxy calls now open a WebSocket connection to
            the provider's ``/responses`` endpoint, send one
            ``response.create`` request, and collect JSON event messages until a
            terminal event. Upper layers still receive the same ``LLMResponse``
            contract as before.
        """
        raw_model = call_kwargs["model"]
        if raw_model.startswith("openai/"):
            raw_model = raw_model[len("openai/"):]

        oai_kwargs: dict[str, Any] = {
            "type": "response.create",
            "model": raw_model,
            "stream": True,
        }
        # Real cause:
        # the supplier's websocket endpoint has already rejected both
        # `max_output_tokens` and, in a real training run, `max_tokens`.
        # Until the supplier documents a supported limit field, do not send any
        # output-token cap on the websocket path.
        if self.disable_response_storage:
            oai_kwargs["store"] = False
        instructions, input_items = self._messages_to_responses_request(messages)
        if response_format == "json":
            input_items = self._ensure_json_keyword_in_responses_input(input_items)
        oai_kwargs["input"] = input_items
        # Real cause:
        # this provider's WebSocket Responses endpoint rejects requests that omit
        # the `instructions` field entirely, even when the caller has no
        # system/developer prompt. An empty string preserves caller semantics
        # better than injecting a synthetic assistant policy sentence.
        oai_kwargs["instructions"] = instructions or ""
        text_cfg = self._responses_text_config(response_format)
        if text_cfg:
            oai_kwargs["text"] = text_cfg
        # Only pass effort: OpenAI does not expose raw reasoning tokens; reasoning
        # *summaries* in output require an explicit reasoning.summary request — we omit it.
        if reasoning_effort:
            oai_kwargs["reasoning"] = {"effort": reasoning_effort}

        timeout_s = self._effective_timeout(call_kwargs, self.timeout)
        first_token_timeout_s = self._effective_first_token_timeout(
            call_kwargs,
            self.first_token_timeout,
        )

        async def _run_ws_request() -> list[dict[str, Any]]:
            terminal_error: str | None = None
            events: list[dict[str, Any]] = []
            read_timeout = (
                first_token_timeout_s
                if first_token_timeout_s is not None
                else timeout_s
            )
            async with websockets.connect(
                self._responses_ws_endpoint(),
                additional_headers=self._responses_ws_headers(),
                user_agent_header=_DEFAULT_WS_USER_AGENT,
                open_timeout=timeout_s,
                close_timeout=5,
                ping_interval=None,
                max_size=8 * 1024 * 1024,
            ) as ws:
                # Align with the supplier-provided websocket probe script:
                # serialize the request body with ASCII escapes so non-ASCII
                # prompt content does not depend on intermediary UTF-8 handling
                # inside the provider's websocket gateway.
                await ws.send(json.dumps(oai_kwargs, ensure_ascii=True))
                while True:
                    try:
                        raw_message = await asyncio.wait_for(ws.recv(), timeout=read_timeout)
                    except ConnectionClosed as exc:
                        if not events:
                            raise RuntimeError(
                                f"websocket closed before any event: code={exc.code} reason={exc.reason}"
                            ) from exc
                        break
                    event = self._parse_responses_ws_message(raw_message)
                    if event is None:
                        continue
                    events.append(event)
                    event_type = str(event.get("type") or "").strip()
                    if event_type == "error":
                        error_payload = event.get("error")
                        if isinstance(error_payload, dict):
                            terminal_error = str(
                                error_payload.get("message")
                                or error_payload.get("type")
                                or error_payload.get("code")
                                or "websocket error"
                            )
                        else:
                            terminal_error = str(event.get("message") or "websocket error")
                        break
                    if event_type in {"response.failed", "response.incomplete"}:
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            err = response_payload.get("error")
                            if isinstance(err, dict):
                                terminal_error = str(
                                    err.get("message")
                                    or err.get("type")
                                    or err.get("code")
                                    or event_type
                                )
                            else:
                                terminal_error = str(
                                    response_payload.get("status_details")
                                    or response_payload.get("status")
                                    or event_type
                                )
                        else:
                            terminal_error = event_type
                        break
                    if event_type == "response.completed":
                        break

            if terminal_error:
                raise RuntimeError(terminal_error)
            return events

        max_attempts = self._effective_max_attempts(call_kwargs, self.max_retries)
        retry_key = self._deferred_retry_key(
            model_label=model_label,
            messages=messages,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        deferred_state = self._deferred_retry_load_state(retry_key)
        deadline_value = deferred_state.get("deadline")
        deadline = float(deadline_value) if isinstance(deadline_value, (int, float)) else None
        billable_attempts_used = int(deferred_state.get("billable_attempts_used", 0) or 0)
        supplier_retry_count = int(deferred_state.get("supplier_retry_count", 0) or 0)
        while billable_attempts_used < max_attempts:
            request_audit: dict[str, Any] | None = None
            try:
                async with self.request_gate.acquire(model_label) as request_audit:
                    queue_wait_s = float(request_audit.get("queue_wait_ms", 0.0)) / 1000.0
                    if deadline is None:
                        deadline = time.monotonic() + timeout_s
                    else:
                        deadline += queue_wait_s
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("timeout")
                    with physical_attempt(model_label) as accounting_attempt:
                        events = await asyncio.wait_for(_run_ws_request(), timeout=remaining)
                        parsed = self._parse_responses_ws_events(events, model_label)
                        accounting_attempt["usage"] = parsed.usage
                if _is_empty(parsed) and billable_attempts_used < max_attempts - 1:
                    self._record_empty_ws_parser_diagnostic(
                        events=events,
                        model_label=model_label,
                        response_format=response_format,
                        request_audit=request_audit,
                        parsed=parsed,
                    )
                    delay = _retry_delay_seconds(supplier_retry_count)
                    supplier_retry_count += 1
                    if not await _sleep_with_deadline(delay, deadline or time.monotonic()):
                        raise RuntimeError("timeout")
                    continue
                trace.llm_call(
                    model_label,
                    messages,
                    parsed,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    transport=request_audit,
                )
                self._deferred_retry_clear_state(retry_key)
                return parsed
            except RetryLater:
                raise
            except Exception as e:
                if _is_transient_infrastructure_error(e):
                    if not self._take_network_retry(supplier_retry_count):
                        self._deferred_retry_clear_state(retry_key)
                        raise
                    supplier_retry_count += 1
                    delay = self.network_retry_delay_s
                    # Each new physical attempt receives the full request timeout.
                    deadline = None
                    retry_limit = self.network_retry_attempts if self.network_retry_attempts is not None else "unlimited"
                    logger.warning(
                        f"LLM {model_label} transient API/network failure: {_exception_diagnostic(e)}. "
                        f"Retry {supplier_retry_count}/{retry_limit} in {delay:.1f}s"
                        f"{self._request_audit_suffix(request_audit)}"
                    )
                    await asyncio.sleep(delay)
                    continue
                if _is_retryable(e) and billable_attempts_used < max_attempts - 1:
                    delay = _retry_delay_seconds(billable_attempts_used)
                    billable_attempts_used += 1
                    deferred_state = {
                        "deadline": float(deadline or time.monotonic()),
                        "billable_attempts_used": int(billable_attempts_used),
                        "supplier_retry_count": int(supplier_retry_count),
                    }
                    if retry_key:
                        self._deferred_retry_raise(
                            retry_key=retry_key,
                            delay=delay,
                            reason=str(e),
                            layer="llm",
                            state=deferred_state,
                        )
                    logger.warning(
                        f"LLM {model_label} attempt {billable_attempts_used} failed "
                        f"(retryable): {e}. Retrying in {delay:.1f}s..."
                        f"{self._request_audit_suffix(request_audit)}"
                    )
                    if not await _sleep_with_deadline(delay, deadline or time.monotonic()):
                        self._deferred_retry_clear_state(retry_key)
                        raise
                    continue
                self._deferred_retry_clear_state(retry_key)
                raise
        self._deferred_retry_clear_state(retry_key)
        return None

    def _responses_http_endpoint(self) -> str:
        """Build the Responses HTTP endpoint from the configured API base."""
        parsed = urlparse(str(self.api_base or "").strip())
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Unsupported API base scheme for HTTP transport: {parsed.scheme or '<empty>'}"
            )
        path = parsed.path.rstrip("/")
        if not path:
            path = "/responses"
        elif path.endswith("/responses"):
            pass
        else:
            path = f"{path}/responses"
        return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))

    def _responses_http_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    async def _call_native_openai_http(
        self, call_kwargs: dict[str, Any], model_label: str,
        response_format: str | None, messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> LLMResponse | None:
        """Call the proxy's Responses API over plain HTTP."""
        raw_model = call_kwargs["model"]
        if raw_model.startswith("openai/"):
            raw_model = raw_model[len("openai/"):]

        instructions, input_items = self._messages_to_responses_request(messages)
        if response_format == "json":
            input_items = self._ensure_json_keyword_in_responses_input(input_items)

        oai_kwargs: dict[str, Any] = {
            "model": raw_model,
            "input": input_items,
            "instructions": instructions or "",
            "stream": True,
        }
        if self.disable_response_storage:
            oai_kwargs["store"] = False
        text_cfg = self._responses_text_config(response_format)
        if text_cfg:
            oai_kwargs["text"] = text_cfg
        if reasoning_effort:
            oai_kwargs["reasoning"] = {"effort": reasoning_effort}

        timeout_s = self._effective_timeout(call_kwargs, self.timeout)
        first_token_timeout_s = self._effective_first_token_timeout(
            call_kwargs,
            self.first_token_timeout,
        )

        async def _run_http_request(remaining: float) -> str:
            read_timeout = min(
                float(first_token_timeout_s),
                remaining,
            ) if first_token_timeout_s is not None else remaining
            timeout = httpx.Timeout(
                connect=min(timeout_s, remaining),
                read=read_timeout,
                write=min(timeout_s, remaining),
                pool=min(timeout_s, remaining),
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                sse_blocks: list[str] = []
                current_block: list[str] = []
                response_completed = False
                async with client.stream(
                    "POST",
                    self._responses_http_endpoint(),
                    headers=self._responses_http_headers(),
                    json=oai_kwargs,
                ) as response:
                    # New definition:
                    # - On the HTTP Responses path, `first_token_timeout` maps to
                    #   the socket read timeout for the first response bytes and
                    #   subsequent read gaps while the body is being downloaded.
                    # - The overall request is still bounded by the outer
                    #   `asyncio.wait_for(..., timeout=remaining)` deadline.
                    async for line in response.aiter_lines():
                        if line == "":
                            if current_block:
                                block_text = "\n".join(current_block)
                                sse_blocks.append(block_text)
                                current_block = []
                                event_name, data_json = self._parse_responses_sse_block(block_text)
                                if event_name == "error":
                                    error_payload = data_json.get("error")
                                    if isinstance(error_payload, dict):
                                        _check_provider_stream_error(json.dumps({"error": error_payload}).encode())
                                    raise RuntimeError(
                                        str(data_json.get("message") or "responses http error")
                                    )
                                if event_name in {"response.failed", "response.incomplete"}:
                                    response_payload = data_json.get("response")
                                    if isinstance(response_payload, dict):
                                        err = response_payload.get("error")
                                        if isinstance(err, dict):
                                            _check_provider_stream_error(json.dumps({"error": err}).encode())
                                        raise RuntimeError(
                                            str(
                                                response_payload.get("status_details")
                                                or response_payload.get("status")
                                                or event_name
                                            )
                                        )
                                    raise RuntimeError(event_name)
                                # New definition:
                                # - On the HTTP/SSE path we must stop as soon as
                                #   the provider emits the terminal
                                #   `response.completed` event.
                                # - Some OpenAI-compatible proxies keep the SSE
                                #   socket open briefly after completion instead
                                #   of immediately closing the stream. Waiting
                                #   for EOF here can stall shutdown/distillation
                                #   even though the full model response has
                                #   already arrived.
                                if event_name == "response.completed":
                                    response_completed = True
                                    break
                            continue
                        current_block.append(line)
                if current_block:
                    sse_blocks.append("\n".join(current_block))
                body_text = "\n\n".join(block for block in sse_blocks if block.strip())
            if response.status_code >= 400:
                detail = body_text.strip() or response.reason_phrase
                raise RuntimeError(f"HTTP {response.status_code}: {detail}")
            if not response_completed:
                raise RuntimeError("SSE stream ended before response.completed")
            if not body_text.strip():
                raise RuntimeError("empty SSE response from Responses HTTP endpoint")
            return body_text

        max_attempts = self._effective_max_attempts(call_kwargs, self.max_retries)
        retry_key = self._deferred_retry_key(
            model_label=model_label,
            messages=messages,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        deferred_state = self._deferred_retry_load_state(retry_key)
        deadline_value = deferred_state.get("deadline")
        deadline = float(deadline_value) if isinstance(deadline_value, (int, float)) else None
        billable_attempts_used = int(deferred_state.get("billable_attempts_used", 0) or 0)
        supplier_retry_count = int(deferred_state.get("supplier_retry_count", 0) or 0)
        while billable_attempts_used < max_attempts:
            request_audit: dict[str, Any] | None = None
            try:
                async with self.request_gate.acquire(model_label) as request_audit:
                    queue_wait_s = float(request_audit.get("queue_wait_ms", 0.0)) / 1000.0
                    if deadline is None:
                        deadline = time.monotonic() + timeout_s
                    else:
                        deadline += queue_wait_s
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("timeout")
                    with physical_attempt(model_label) as accounting_attempt:
                        payload = await asyncio.wait_for(_run_http_request(remaining), timeout=remaining)
                        parsed = self._parse_response(payload, model_label)
                        accounting_attempt["usage"] = parsed.usage
                if _is_empty(parsed) and billable_attempts_used < max_attempts - 1:
                    delay = _retry_delay_seconds(supplier_retry_count)
                    supplier_retry_count += 1
                    if not await _sleep_with_deadline(delay, deadline or time.monotonic()):
                        raise RuntimeError("timeout")
                    continue
                trace.llm_call(
                    model_label,
                    messages,
                    parsed,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    transport=request_audit,
                )
                self._deferred_retry_clear_state(retry_key)
                return parsed
            except RetryLater:
                raise
            except Exception as e:
                if _is_transient_infrastructure_error(e):
                    if not self._take_network_retry(supplier_retry_count):
                        self._deferred_retry_clear_state(retry_key)
                        raise
                    supplier_retry_count += 1
                    delay = self.network_retry_delay_s
                    # Each new physical attempt receives the full request timeout.
                    deadline = None
                    retry_limit = self.network_retry_attempts if self.network_retry_attempts is not None else "unlimited"
                    logger.warning(
                        f"LLM {model_label} transient API/network failure: {_exception_diagnostic(e)}. "
                        f"Retry {supplier_retry_count}/{retry_limit} in {delay:.1f}s"
                        f"{self._request_audit_suffix(request_audit)}"
                    )
                    await asyncio.sleep(delay)
                    continue
                if _is_retryable(e) and billable_attempts_used < max_attempts - 1:
                    delay = _retry_delay_seconds(billable_attempts_used)
                    billable_attempts_used += 1
                    deferred_state = {
                        "deadline": float(deadline or time.monotonic()),
                        "billable_attempts_used": int(billable_attempts_used),
                        "supplier_retry_count": int(supplier_retry_count),
                    }
                    if retry_key:
                        self._deferred_retry_raise(
                            retry_key=retry_key,
                            delay=delay,
                            reason=str(e),
                            layer="llm",
                            state=deferred_state,
                        )
                    logger.warning(
                        f"LLM {model_label} attempt {billable_attempts_used} failed "
                        f"(retryable): {e}. Retrying in {delay:.1f}s..."
                        f"{self._request_audit_suffix(request_audit)}"
                    )
                    if not await _sleep_with_deadline(delay, deadline or time.monotonic()):
                        self._deferred_retry_clear_state(retry_key)
                        raise
                    continue
                self._deferred_retry_clear_state(retry_key)
                raise
        self._deferred_retry_clear_state(retry_key)
        return None

    @staticmethod
    def _parse_responses_ws_message(raw_message: str | bytes) -> dict[str, Any] | None:
        """Parse one raw WebSocket message into a JSON event payload."""
        text = (
            raw_message.decode("utf-8", "replace")
            if isinstance(raw_message, bytes)
            else str(raw_message)
        )
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"Ignoring non-JSON websocket frame: {text[:200]}")
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def _extract_text_fragments(cls, value: Any) -> list[str]:
        parts: list[str] = []
        if value is None:
            return parts
        if isinstance(value, str):
            # Preserve provider-emitted whitespace inside streamed text
            # fragments. Stripping each chunk turns token/word-level deltas like
            # ["Predict", " the", " future"] into "Predictthefuture".
            if value != "":
                parts.append(value)
            return parts
        if isinstance(value, list):
            for item in value:
                parts.extend(cls._extract_text_fragments(item))
            return parts
        if not isinstance(value, dict):
            return parts

        for key in ("text", "output_text", "value"):
            if key in value:
                parts.extend(cls._extract_text_fragments(value.get(key)))
        if parts:
            return parts

        if isinstance(value.get("content"), list):
            for item in value.get("content") or []:
                parts.extend(cls._extract_text_fragments(item))
        return parts

    @classmethod
    def _extract_text_from_responses_content_part(cls, part: Any) -> list[str]:
        if not isinstance(part, dict):
            return cls._extract_text_fragments(part)

        part_type = str(part.get("type") or "").strip()
        if part_type in {"input_text"}:
            return []

        parts: list[str] = []
        for key in ("text", "output_text", "value"):
            if key in part:
                parts.extend(cls._extract_text_fragments(part.get(key)))
        if parts:
            return parts

        return cls._extract_text_fragments(part)

    @classmethod
    def _extract_text_from_responses_output_item(cls, item: Any) -> list[str]:
        if not isinstance(item, dict):
            return cls._extract_text_fragments(item)

        # Commentary is a progress update, never the host's final tool decision.
        if item.get("phase") == "commentary":
            return []

        parts: list[str] = []
        content = item.get("content")
        if isinstance(content, list):
            for content_part in content:
                parts.extend(cls._extract_text_from_responses_content_part(content_part))
        if parts:
            return parts

        for key in ("text", "output_text", "value"):
            if key in item:
                parts.extend(cls._extract_text_fragments(item.get(key)))
        return parts

    @staticmethod
    def _responses_output_has_phase(output_items: Any) -> bool:
        return any(
            (item.get("phase") if isinstance(item, dict) else getattr(item, "phase", None))
            is not None
            for item in output_items or []
        )

    @classmethod
    def _parse_responses_ws_events(cls, events: list[dict[str, Any]], model: str) -> LLMResponse:
        """Parse collected Responses WebSocket events into LLMResponse."""
        delta_parts: list[str] = []
        item_done_parts: list[str] = []
        completed_response: dict[str, Any] | None = None

        for event in events:
            event_type = str(event.get("type") or "").strip()
            if event_type == "response.output_text.delta":
                delta_parts.extend(cls._extract_text_fragments(event.get("delta")))
                continue
            if event_type == "response.output_text.done":
                item_done_parts.extend(
                    cls._extract_text_fragments(
                        event.get("text") or event.get("output_text") or event.get("delta")
                    )
                )
                continue
            if event_type in {"response.content_part.added", "response.content_part.done"}:
                item_done_parts.extend(
                    cls._extract_text_from_responses_content_part(
                        event.get("part") or event.get("content_part") or event.get("delta")
                    )
                )
                continue
            if event_type in {"response.output_item.added", "response.output_item.done"}:
                item = event.get("item")
                item_done_parts.extend(cls._extract_text_from_responses_output_item(item))
                continue
            if event_type == "response.completed":
                response_payload = event.get("response")
                if isinstance(response_payload, dict):
                    completed_response = response_payload

        text: str | None = None
        if delta_parts:
            text = "".join(delta_parts).strip() or None
        elif item_done_parts:
            text = "".join(part for part in item_done_parts if part).strip() or None

        phase_aware = isinstance(completed_response, dict) and cls._responses_output_has_phase(
            completed_response.get("output")
        )
        if (text is None or phase_aware) and isinstance(completed_response, dict):
            output_items = completed_response.get("output")
            if isinstance(output_items, list):
                completed_parts: list[str] = []
                for item in output_items:
                    completed_parts.extend(cls._extract_text_from_responses_output_item(item))
                text = "".join(part for part in completed_parts if part).strip() or None
            if text is None and not phase_aware:
                output_text_parts = cls._extract_text_fragments(completed_response.get("output_text"))
                text = "".join(part for part in output_text_parts if part).strip() or None

        usage = {}
        if isinstance(completed_response, dict) and completed_response.get("usage"):
            usage = cls._usage_to_dict(completed_response.get("usage"))
        parsed = cls._extract_json(text) if text else None
        if text is None:
            logger.warning(
                f"Empty WebSocket output from {model}: "
                f"{(completed_response or {}).get('error') or (completed_response or {}).get('detail')}"
            )
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=model,
            usage=usage,
            raw=events,
        )

    @staticmethod
    async def _collect_chat_completion_stream(stream: Any, model: str):
        """Collect a chat-completions stream into a fake completion object.

        New definition:
            This helper normalizes an async iterable of chunk objects into the
            same shape as a non-streaming chat-completions response, so upper
            layers keep seeing the same ``LLMResponse`` contract.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_obj: Any = None
        finish_reason: str | None = None
        async for chunk in stream:
            usage_obj = getattr(chunk, "usage", None) or usage_obj
            if not chunk.choices:
                continue
            finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(str(reasoning))
            content = getattr(delta, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                content_parts.append(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        content_parts.append(item)
                        continue
                    if isinstance(item, dict):
                        text_value = item.get("text")
                        if text_value:
                            content_parts.append(str(text_value))
                        continue
                    text_value = getattr(item, "text", None)
                    if text_value:
                        content_parts.append(str(text_value))
                continue
            content_parts.append(str(content))

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="".join(content_parts),
                        role="assistant",
                        reasoning_content="".join(reasoning_parts) or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            model=model,
            usage=usage_obj,
        )

    async def _native_chat_completion(self, call_kwargs: dict[str, Any]):
        """Open one Chat Completions stream with retries owned by TimeClaw.

        api_transport selects a wire protocol, not a fallback. GLM retains its
        native thinking mode; Responses storage fields and output caps are
        deliberately absent from this request.
        """
        from openai import APIStatusError, AsyncOpenAI

        if getattr(self, "_oai_client", None) is None:
            self._oai_client = AsyncOpenAI(
                api_key=self.api_key, base_url=self.api_base, max_retries=0,
                http_client=httpx.AsyncClient(
                    event_hooks={"response": [_inspect_chat_stream_errors]},
                ),
            )
        request = dict(call_kwargs)
        for key in (
            "api_key", "api_base", "api_version", "num_retries",
            "first_token_timeout", "max_tokens", "max_output_tokens",
            "max_completion_tokens", "reasoning", "store",
        ):
            request.pop(key, None)
        request["model"] = str(request["model"]).removeprefix("openai/")
        if request["model"] in {"glm-5.3-flash", "glm-5.3-flashx"}:
            extra_body = dict(request.get("extra_body") or {})
            extra_body["thinking"] = {"type": "enabled"}
            request["extra_body"] = extra_body
        timeout_s = self._effective_timeout(call_kwargs, self.timeout)
        read_timeout = self._effective_first_token_timeout(
            call_kwargs, self.first_token_timeout,
        ) or timeout_s
        request["timeout"] = httpx.Timeout(timeout_s, read=read_timeout)
        try:
            return await self._oai_client.chat.completions.create(**request)
        except APIStatusError as exc:
            body = exc.body if isinstance(exc.body, dict) else {}
            error = body.get("error", body)
            if isinstance(error, dict) and str(error.get("code")) == "1113":
                raise ProviderAccountQuotaError(
                    "Provider account has no available balance/resource package (code 1113)"
                ) from exc
            if isinstance(error, dict) and str(error.get("code")) == "1301":
                raise ProviderContentPolicyError(
                    f"Provider content policy rejection (code 1301): {error.get('message', '')}"
                ) from exc
            raise

    async def _call_with_retry(
        self, call_kwargs: dict[str, Any], model_label: str,
        response_format: str | None, messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        *, native_chat: bool = False,
    ) -> LLMResponse | None:
        """Execute a litellm call with automatic retry using stream mode.

        New definition:
            We request chat-completions style streaming for completion calls and
            reassemble the chunks locally. This makes the transport incremental
            while preserving the existing non-streaming ``LLMResponse`` contract
            for the rest of the codebase.
        """
        if not native_chat:
            import litellm

        effective_kwargs = dict(call_kwargs)
        if reasoning_effort and str(call_kwargs.get("model", "")).startswith(("openai/", "azure/")):
            effective_kwargs["reasoning_effort"] = reasoning_effort
        effective_kwargs["stream"] = True
        effective_kwargs["stream_options"] = {"include_usage": True}
        if accounting_active():
            # Keep retry ownership here so hidden SDK retries are not counted
            # as a single physical attempt in the v2 cost ledger.
            effective_kwargs["num_retries"] = 0

        timeout_s = self._effective_timeout(call_kwargs, self.timeout)
        max_attempts = self._effective_max_attempts(call_kwargs, self.max_retries)
        retry_key = self._deferred_retry_key(
            model_label=model_label,
            messages=messages,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        deferred_state = self._deferred_retry_load_state(retry_key)
        deadline_value = deferred_state.get("deadline")
        deadline = float(deadline_value) if isinstance(deadline_value, (int, float)) else None
        billable_attempts_used = int(deferred_state.get("billable_attempts_used", 0) or 0)
        supplier_retry_count = int(deferred_state.get("supplier_retry_count", 0) or 0)
        while billable_attempts_used < max_attempts:
            request_audit: dict[str, Any] | None = None
            try:
                async with self.request_gate.acquire(model_label) as request_audit:
                    queue_wait_s = float(request_audit.get("queue_wait_ms", 0.0)) / 1000.0
                    if deadline is None:
                        deadline = time.monotonic() + timeout_s
                    else:
                        deadline += queue_wait_s
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("timeout")
                    with physical_attempt(model_label) as accounting_attempt:
                        response = await asyncio.wait_for(
                            self._native_chat_completion(effective_kwargs) if native_chat
                            else litellm.acompletion(**effective_kwargs),
                            timeout=remaining,
                        )
                        if hasattr(response, "__aiter__"):
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise RuntimeError("timeout")
                            stream = response
                            try:
                                response = await asyncio.wait_for(
                                    self._collect_chat_completion_stream(stream, model_label),
                                    timeout=remaining,
                                )
                            finally:
                                if native_chat:
                                    await stream.close()
                        parsed = self._parse_response(response, model_label)
                        accounting_attempt["usage"] = parsed.usage
                        if native_chat:
                            finish_reason = response.choices[0].finish_reason
                            if finish_reason is None:
                                raise RuntimeError("Chat completion connection ended before finish_reason")
                            if finish_reason != "stop":
                                raise ValueError(f"Chat completion did not finish normally: {finish_reason}")
                if _is_empty(parsed) and billable_attempts_used < max_attempts - 1:
                    delay = _retry_delay_seconds(billable_attempts_used)
                    billable_attempts_used += 1
                    logger.warning(
                        f"LLM {model_label} attempt {billable_attempts_used}: empty response, "
                        f"retrying in {delay:.1f}s..."
                        f"{self._request_audit_suffix(request_audit)}"
                    )
                    if not await _sleep_with_deadline(delay, deadline or time.monotonic()):
                        raise RuntimeError("timeout")
                    continue
                trace.llm_call(
                    model_label,
                    messages,
                    parsed,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    transport=request_audit,
                )
                self._deferred_retry_clear_state(retry_key)
                return parsed
            except RetryLater:
                raise
            except (ProviderContentPolicyError, ProviderAccountQuotaError):
                self._deferred_retry_clear_state(retry_key)
                raise
            except Exception as e:
                if _is_transient_infrastructure_error(e):
                    if not self._take_network_retry(supplier_retry_count):
                        self._deferred_retry_clear_state(retry_key)
                        raise
                    supplier_retry_count += 1
                    delay = self.network_retry_delay_s
                    # Each new physical attempt receives the full request timeout.
                    deadline = None
                    retry_limit = self.network_retry_attempts if self.network_retry_attempts is not None else "unlimited"
                    logger.warning(
                        f"LLM {model_label} transient API/network failure: {_exception_diagnostic(e)}. "
                        f"Retry {supplier_retry_count}/{retry_limit} in {delay:.1f}s"
                        f"{self._request_audit_suffix(request_audit)}"
                    )
                    await asyncio.sleep(delay)
                    continue
                if _is_retryable(e) and billable_attempts_used < max_attempts - 1:
                    delay = _retry_delay_seconds(billable_attempts_used)
                    billable_attempts_used += 1
                    deferred_state = {
                        "deadline": float(deadline or time.monotonic()),
                        "billable_attempts_used": int(billable_attempts_used),
                        "supplier_retry_count": int(supplier_retry_count),
                    }
                    if retry_key:
                        self._deferred_retry_raise(
                            retry_key=retry_key,
                            delay=delay,
                            reason=str(e),
                            layer="llm",
                            state=deferred_state,
                        )
                    logger.warning(
                        f"LLM {model_label} attempt {billable_attempts_used} failed "
                        f"(retryable): {e}. Retrying in {delay:.1f}s..."
                        f"{self._request_audit_suffix(request_audit)}"
                    )
                    if not await _sleep_with_deadline(delay, deadline or time.monotonic()):
                        self._deferred_retry_clear_state(retry_key)
                        raise
                    continue
                self._deferred_retry_clear_state(retry_key)
                raise
        self._deferred_retry_clear_state(retry_key)
        return None

    @logical_call
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        response_format: str | None = None,
        reasoning_effort: str | None | object = _DEFAULT_REASONING_EFFORT,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request. Retries on transient errors, then falls back."""
        target_model = model or self.default_model
        if reasoning_effort is _DEFAULT_REASONING_EFFORT:
            effective_reasoning_effort = self.reasoning_effort
        else:
            effective_reasoning_effort = (
                str(reasoning_effort).strip() if reasoning_effort is not None else None
            ) or None
        effective_reasoning_effort = _reasoning_effort_for_model(
            target_model,
            effective_reasoning_effort,
        )
        call_kwargs = self._build_kwargs(messages, target_model, response_format, **kwargs)

        caller = (
            self._call_native_openai
            if self._use_native_openai(target_model)
            else self._call_with_retry
        )

        try:
            result = await caller(
                call_kwargs, target_model, response_format, messages, effective_reasoning_effort)
            if result is not None and not _is_empty(result):
                return result
            raise RuntimeError(f"LLM provider returned an empty response for {target_model}")
        except RetryLater:
            raise
        except (ProviderContentPolicyError, ProviderAccountQuotaError):
            raise
        except Exception as e:
            if not _is_supplier_pool_retry(e):
                logger.warning(f"LLM call to {target_model} failed: {_exception_diagnostic(e)}")
            if self.fallback_model and target_model != self.fallback_model:
                logger.info(f"Falling back to {self.fallback_model}")
                call_kwargs["model"] = self.fallback_model
                try:
                    result = await caller(
                        call_kwargs, self.fallback_model + " (fallback)",
                        response_format, messages, effective_reasoning_effort)
                    if result is not None and not _is_empty(result):
                        return result
                    raise RuntimeError(
                        f"LLM fallback returned an empty response for {self.fallback_model}"
                    )
                except RetryLater:
                    raise
                except Exception as e2:
                    if not _is_supplier_pool_retry(e2):
                        logger.error(f"Fallback also failed: {e2}")
            raise
        raise RuntimeError(f"LLM provider returned no response for {target_model}")

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        temperature = kwargs.pop("temperature", self.temperature)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "num_retries": kwargs.pop("num_retries", kwargs.pop("max_retries", self.max_retries)),
            "timeout": kwargs.pop("timeout", self.timeout),
            "first_token_timeout": kwargs.pop("first_token_timeout", self.first_token_timeout),
        }
        is_anthropic = model.startswith("anthropic/")
        if self.send_temperature:
            call_kwargs["temperature"] = temperature
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

        input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
        if isinstance(input_details, dict):
            cached_tokens = _as_int(input_details.get("cached_tokens"))
            if cached_tokens is not None:
                summary["cached_tokens"] = cached_tokens

        output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
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

        if isinstance(response, dict) and "output" in response and "choices" not in response:
            output_items = response.get("output")
            if output_items is None:
                logger.warning(
                    f"Empty output from {model}: "
                    f"{response.get('error') or response.get('detail')}"
                )
                return LLMResponse(text=None, model=model, raw=response)

            text_parts: list[str] = []
            for item in output_items or []:
                text_parts.extend(LLMBackend._extract_text_from_responses_output_item(item))

            text = "".join(part for part in text_parts if part).strip()
            if not text and not LLMBackend._responses_output_has_phase(output_items):
                output_text_parts = LLMBackend._extract_text_fragments(response.get("output_text"))
                text = "".join(part for part in output_text_parts if part).strip()
            parsed = LLMBackend._extract_json(text) if text else None
            usage = LLMBackend._usage_to_dict(response.get("usage"))
            return LLMResponse(
                text=text or None,
                parsed=parsed,
                model=model,
                usage=usage,
                raw=response,
            )

        if not getattr(response, "choices", None) and hasattr(response, "output"):
            output_items = getattr(response, "output", None)
            if output_items is None:
                logger.warning(f"Empty output from {model}: {getattr(response, 'error', None) or getattr(response, 'detail', None)}")
                return LLMResponse(text=None, model=model, raw=response)
            text_parts: list[str] = []
            for item in output_items or []:
                if hasattr(item, "model_dump"):
                    try:
                        item = item.model_dump(mode="python")
                    except TypeError:
                        item = item.model_dump()
                elif hasattr(item, "dict"):
                    try:
                        item = item.dict()
                    except Exception:
                        item = item
                text_parts.extend(LLMBackend._extract_text_from_responses_output_item(item))

            text = "".join(part for part in text_parts if part).strip()
            if (not text and output_items is not None and hasattr(response, "output_text")
                    and not LLMBackend._responses_output_has_phase(output_items)):
                output_text_parts = LLMBackend._extract_text_fragments(getattr(response, "output_text"))
                text = "".join(part for part in output_text_parts if part).strip()
            parsed = LLMBackend._extract_json(text) if text else None
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = LLMBackend._usage_to_dict(response.usage)
            return LLMResponse(
                text=text or None,
                parsed=parsed,
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

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = LLMBackend._usage_to_dict(response.usage)

        return LLMResponse(
            text=text,
            parsed=parsed,
            model=model,
            usage=usage,
            raw=response,
        )

    @staticmethod
    def _parse_responses_sse_block(block: str) -> tuple[str, dict[str, Any] | None]:
        event_name = ""
        data_lines: list[str] = []
        for line in str(block or "").splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if not data_lines:
            return event_name, None
        payload = "\n".join(data_lines).strip()
        try:
            data_json = json.loads(payload)
        except json.JSONDecodeError:
            return event_name, None
        return event_name, data_json if isinstance(data_json, dict) else None

    @staticmethod
    def _parse_responses_sse_string(raw: str, model: str) -> LLMResponse:
        """Parse a proxy-degraded Responses SSE transcript into LLMResponse."""
        delta_parts: list[str] = []
        item_done_parts: list[str] = []
        usage: dict[str, Any] = {}
        completed_response: dict[str, Any] | None = None

        blocks = [block.strip() for block in re.split(r"\n\s*\n", str(raw or "").strip()) if block.strip()]
        for block in blocks:
            event_name, data_json = LLMBackend._parse_responses_sse_block(block)
            if not isinstance(data_json, dict):
                continue

            if event_name == "response.output_text.delta":
                delta_parts.extend(LLMBackend._extract_text_fragments(data_json.get("delta")))
                continue

            if event_name == "response.output_text.done":
                item_done_parts.extend(
                    LLMBackend._extract_text_fragments(
                        data_json.get("text") or data_json.get("output_text") or data_json.get("delta")
                    )
                )
                continue

            if event_name in {"response.content_part.added", "response.content_part.done"}:
                item_done_parts.extend(
                    LLMBackend._extract_text_from_responses_content_part(
                        data_json.get("part") or data_json.get("content_part") or data_json.get("delta")
                    )
                )
                continue

            if event_name in {"response.output_item.added", "response.output_item.done"}:
                item_done_parts.extend(
                    LLMBackend._extract_text_from_responses_output_item(data_json.get("item") or {})
                )
                continue

            if event_name == "response.completed":
                response_payload = data_json.get("response")
                if isinstance(response_payload, dict):
                    completed_response = response_payload

        if completed_response and completed_response.get("usage"):
            usage = LLMBackend._usage_to_dict(completed_response["usage"])

        text: str | None = None
        if delta_parts:
            text = "".join(delta_parts).strip() or None
        elif item_done_parts:
            text = "".join(part for part in item_done_parts if part).strip() or None
        phase_aware = isinstance(completed_response, dict) and LLMBackend._responses_output_has_phase(
            completed_response.get("output")
        )
        if (text is None or phase_aware) and isinstance(completed_response, dict):
            output_items = completed_response.get("output")
            if isinstance(output_items, list):
                completed_parts: list[str] = []
                for item in output_items:
                    completed_parts.extend(LLMBackend._extract_text_from_responses_output_item(item))
                text = "".join(part for part in completed_parts if part).strip() or None
            if text is None and not phase_aware:
                output_text_parts = LLMBackend._extract_text_fragments(completed_response.get("output_text"))
                text = "".join(part for part in output_text_parts if part).strip() or None
        parsed = LLMBackend._extract_json(text) if text else None
        if text is None:
            logger.warning(
                f"Empty SSE output from {model}: "
                f"{(completed_response or {}).get('error') or (completed_response or {}).get('detail')}"
            )
        return LLMResponse(
            text=text,
            parsed=parsed,
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
