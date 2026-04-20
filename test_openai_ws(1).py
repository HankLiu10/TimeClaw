#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import sys
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import websockets
from websockets.client import ClientConnection
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import InvalidStatus


DEFAULT_USER_AGENT = "codex_cli_rs/0.104.0"
DEFAULT_ORIGINATOR = "codex_cli_rs"
DEFAULT_MODEL = "gpt-5.3-codex"
DEFAULT_INSTRUCTIONS = "You are Codex. Reply briefly and follow the user's request."


@dataclass
class ProbeResult:
    ws_url: str
    handshake_ok: bool
    sent_first_message: bool
    received_events: list[dict[str, Any]]
    http_status: int | None = None
    response_headers: dict[str, str] | None = None
    close_code: int | None = None
    close_reason: str | None = None
    error: str | None = None


def build_ws_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError(f"unsupported base URL scheme: {parsed.scheme or '<empty>'}")

    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    path = parsed.path.rstrip("/")
    if not path:
        path = "/responses"
    elif path.endswith("/responses"):
        pass
    elif path.endswith("/v1"):
        path = f"{path}/responses"
    else:
        path = f"{path}/responses"

    return urlunparse((scheme, parsed.netloc, path, "", parsed.query, ""))


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()

    for env_name in ("OPENAI_API_KEY", "SUB2API_WS_TEST_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    raise ValueError("API key is required via --api-key or OPENAI_API_KEY / SUB2API_WS_TEST_KEY")


def build_first_message(args: argparse.Namespace) -> str:
    payload: dict[str, Any] = {
        "type": "response.create",
        "model": args.model,
        "stream": not args.non_stream,
        "store": args.store,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": args.prompt,
                    }
                ],
            }
        ],
    }
    if args.instructions is not None:
        payload["instructions"] = args.instructions
    if args.prompt_cache_key:
        payload["prompt_cache_key"] = args.prompt_cache_key
    if args.metadata_json:
        payload["metadata"] = json.loads(args.metadata_json)
    return json.dumps(payload, ensure_ascii=True)


def summarize_message(raw_message: str | bytes) -> dict[str, Any]:
    text = raw_message.decode("utf-8", "replace") if isinstance(raw_message, bytes) else raw_message
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        preview = text[:300] + ("..." if len(text) > 300 else "")
        return {
            "kind": "text",
            "preview": preview,
        }

    event_type = payload.get("type")
    summary: dict[str, Any] = {
        "kind": "json",
        "type": event_type,
    }

    if isinstance(payload.get("response"), dict):
        response = payload["response"]
        if response.get("id"):
            summary["response_id"] = response.get("id")
        if response.get("status"):
            summary["response_status"] = response.get("status")

    if isinstance(payload.get("error"), dict):
        error = payload["error"]
        summary["error_type"] = error.get("type") or error.get("code")
        summary["error_message"] = error.get("message")

    if "message" in payload and "error_message" not in summary:
        summary["message"] = payload.get("message")

    return summary


async def probe_ws(args: argparse.Namespace) -> ProbeResult:
    ws_url = build_ws_url(args.base_url)
    api_key = load_api_key(args)
    first_message = build_first_message(args)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "originator": args.originator,
        "OpenAI-Beta": "responses=v1",
        "X-Request-ID": f"wsprobe-{uuid.uuid4()}",
    }
    if args.session_id:
        headers["session_id"] = args.session_id
    if args.conversation_id:
        headers["conversation_id"] = args.conversation_id

    ssl_context = None
    if ws_url.startswith("wss://") and args.insecure:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    result = ProbeResult(
        ws_url=ws_url,
        handshake_ok=False,
        sent_first_message=False,
        received_events=[],
    )

    connect_kwargs: dict[str, Any] = {
        "additional_headers": headers,
        "user_agent_header": args.user_agent,
        "open_timeout": args.connect_timeout,
        "close_timeout": 5,
        "ping_interval": None,
        "max_size": 8 * 1024 * 1024,
    }
    if args.proxy is not None:
        connect_kwargs["proxy"] = args.proxy
    if ssl_context is not None:
        connect_kwargs["ssl"] = ssl_context

    try:
        async with websockets.connect(ws_url, **connect_kwargs) as ws:
            ws = ws  # type: ClientConnection
            result.handshake_ok = True

            await ws.send(first_message)
            result.sent_first_message = True

            for _ in range(args.max_events):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=args.read_timeout)
                except asyncio.TimeoutError:
                    result.error = f"timed out waiting for event after {args.read_timeout:.1f}s"
                    break
                except ConnectionClosed as exc:
                    result.close_code = exc.code
                    result.close_reason = exc.reason
                    if not result.received_events:
                        result.error = "connection closed before any event was received"
                    break

                summary = summarize_message(raw)
                result.received_events.append(summary)

                event_type = summary.get("type")
                if event_type in {"response.completed", "response.failed", "response.incomplete", "error"}:
                    break

            await ws.close()
    except InvalidStatus as exc:
        response = getattr(exc, "response", None)
        if response is not None:
            result.http_status = getattr(response, "status_code", None) or getattr(response, "status", None)
            headers = getattr(response, "headers", None)
            if headers is not None:
                try:
                    result.response_headers = {str(k): str(v) for k, v in headers.raw_items()}
                except Exception:
                    try:
                        result.response_headers = dict(headers)
                    except Exception:
                        result.response_headers = None
        result.error = f"{type(exc).__name__}: {exc}"
    except ConnectionClosed as exc:
        result.close_code = exc.code
        result.close_reason = exc.reason
        if not result.error:
            result.error = "connection closed during handshake or send"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def print_result(args: argparse.Namespace, result: ProbeResult) -> int:
    print(f"Base URL: {args.base_url}")
    print(f"WS URL:   {result.ws_url}")
    print(f"Model:    {args.model}")
    print(f"Prompt:   {args.prompt!r}")
    print(f"Headers:  User-Agent={args.user_agent!r}, originator={args.originator!r}")
    if args.session_id:
        print(f"session_id:        {args.session_id}")
    if args.conversation_id:
        print(f"conversation_id:   {args.conversation_id}")
    if args.prompt_cache_key:
        print(f"prompt_cache_key:  {args.prompt_cache_key}")
    print()
    print(f"Handshake OK:        {result.handshake_ok}")
    print(f"First message sent:  {result.sent_first_message}")
    print(f"Events received:     {len(result.received_events)}")
    if result.http_status is not None:
        print(f"HTTP status:         {result.http_status}")

    if result.received_events:
        print("Event summaries:")
        for idx, event in enumerate(result.received_events, start=1):
            print(f"  {idx}. {json.dumps(event, ensure_ascii=False)}")

    if result.response_headers:
        print("Response headers:")
        for key, value in sorted(result.response_headers.items()):
            print(f"  {key}: {value}")

    if result.close_code is not None:
        print(f"Close code:          {result.close_code}")
    if result.close_reason:
        print(f"Close reason:        {result.close_reason}")
    if result.error:
        print(f"Error:               {result.error}")

    if result.received_events:
        return 0
    if result.handshake_ok and result.sent_first_message:
        return 2
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Sub2API /v1/responses WebSocket ingress with a Codex-like client payload.",
    )
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. https://example.com/v1")
    parser.add_argument("--api-key", help="API key. If omitted, OPENAI_API_KEY or SUB2API_WS_TEST_KEY is used.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to request. Default: {DEFAULT_MODEL}")
    parser.add_argument("--prompt", default="Reply with a single word: pong", help="Probe prompt text.")
    parser.add_argument(
        "--instructions",
        default=None,
        help=(
            "Top-level instructions field. If omitted, a minimal default is auto-added for codex models "
            f"({DEFAULT_INSTRUCTIONS!r}). Use --instructions '' to send an empty string on purpose."
        ),
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent header to send.")
    parser.add_argument("--originator", default=DEFAULT_ORIGINATOR, help="originator header to send.")
    parser.add_argument(
        "--session-id",
        default=None,
        help="session_id header. If omitted, a random one is auto-generated unless --no-random-session is set.",
    )
    parser.add_argument("--conversation-id", default=None, help="conversation_id header.")
    parser.add_argument("--prompt-cache-key", default=None, help="prompt_cache_key field in request body.")
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional metadata JSON string to include in the request body. Omitted by default because some upstreams reject metadata.",
    )
    parser.add_argument(
        "--no-random-session",
        action="store_true",
        help="Do not auto-generate a random session_id when one isn't provided.",
    )
    parser.add_argument("--connect-timeout", type=float, default=15.0, help="WebSocket connect timeout in seconds.")
    parser.add_argument("--read-timeout", type=float, default=20.0, help="Per-event read timeout in seconds.")
    parser.add_argument("--max-events", type=int, default=6, help="Maximum number of events to read before stopping.")
    parser.add_argument("--non-stream", action="store_true", help="Use stream=false instead of stream=true.")
    parser.add_argument("--store", action="store_true", help="Set store=true in response.create payload.")
    parser.add_argument("--proxy", default=None, help="Proxy URL for websockets.connect. Pass '' to disable env proxy.")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification for wss:// endpoints.")
    args = parser.parse_args()
    if args.proxy == "":
        args.proxy = None
    if args.instructions is None and "codex" in args.model.lower():
        args.instructions = DEFAULT_INSTRUCTIONS
    if args.session_id is None and not args.no_random_session:
        args.session_id = f"wsprobe-{uuid.uuid4()}"
    if args.prompt_cache_key is None and args.no_random_session:
        args.prompt_cache_key = None
    return args


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(probe_ws(args))
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 1

    return print_result(args, result)


if __name__ == "__main__":
    raise SystemExit(main())
