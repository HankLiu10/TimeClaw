"""Replay a real relay error through the native transports and retry loop.

The error fields were observed on 2026-10-10. HTTP is mocked; no paid calls
or research-directory fixtures are needed.
"""
import json

import httpx
from openai import AsyncOpenAI
import pytest

import timeclaw.llm.base as base
from timeclaw.core.accounting import RequestLedger


def frames(transport, failed):
    error = {"code": "upstream_error", "message": "Upstream service temporarily unavailable"}
    if transport == "chat_completions":
        if failed:
            event = {"error": {"type": error["code"], "message": error["message"]}}
        else:
            event = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                     "model": "gpt-6-sol", "choices": [{"index": 0,
                     "delta": {"content": '{"ready":true}'}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
        return 'data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n'
    if failed:
        event = {"type": "response.failed", "sequence_number": 0, "response": {
            "id": "fixture", "object": "response", "model": "gpt-6-sol",
            "status": "failed", "output": [], "error": error}}
    else:
        event = {"type": "response.completed", "response": {"status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text",
                        "text": '{"ready":true}'}]}],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}}}
    return 'event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n'


@pytest.mark.asyncio
@pytest.mark.parametrize('transport', ['chat_completions', 'responses'])
async def test_temporary_sse_uses_unlimited_network_budget(monkeypatch, tmp_path, transport):
    attempts, delays = [], []
    original_client = httpx.AsyncClient

    async def handler(request):
        attempts.append(request)
        return httpx.Response(200, content=frames(transport, len(attempts) <= 3),
                              headers={'content-type': 'text/event-stream'})

    mock_transport = httpx.MockTransport(handler)
    if transport == 'responses':
        monkeypatch.setattr(base.httpx, 'AsyncClient',
                            lambda **kwargs: original_client(transport=mock_transport, **kwargs))

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(base.asyncio, 'sleep', record_sleep)
    monkeypatch.chdir(tmp_path)
    backend = base.LLMBackend(default_model='openai/gpt-6-sol', api_base='https://probe.invalid/v1',
        api_key='fixture-key', api_transport=transport, max_retries=1,
        network_retry_attempts=None, network_retry_delay_s=30, timeout=5)
    if transport == 'chat_completions':
        backend._oai_client = AsyncOpenAI(api_key='fixture-key', base_url=backend.api_base,
            max_retries=0, http_client=original_client(transport=mock_transport,
            event_hooks={'response': [base._inspect_chat_stream_errors]}))
    ledger = RequestLedger('relay-retry', tmp_path)
    try:
        with ledger.activate():
            response = await backend.complete([{'role': 'user', 'content': 'Return JSON readiness.'}],
                                              response_format='json')
    finally:
        await backend.shutdown()
    assert json.loads(response.text) == {'ready': True}
    assert len(attempts) == 4
    assert delays == [30, 30, 30]
    summary = ledger.summary()
    assert summary['logical_calls'] == 1
    assert summary['physical_attempts'] == 4
    assert summary['usage_covers_all_attempts'] is False
    assert summary['billed_usd'] is None


@pytest.mark.parametrize('error,transient', [
    ({'type': 'upstream_error', 'message': 'Upstream service temporarily unavailable'}, True),
    ({'code': 'upstream_error', 'message': 'Upstream service temporarily unavailable'}, True),
    ({'type': 'api_error', 'message': 'Service temporarily unavailable'}, True),
    ({'code': '1301', 'type': 'api_error', 'message': 'Service temporarily unavailable'}, False),
    ({'code': '1113', 'type': 'api_error', 'message': 'Service temporarily unavailable'}, False),
    ({'type': 'invalid_request_error', 'message': 'Service temporarily unavailable'}, False),
])
def test_permanent_errors_are_not_reclassified_as_transient(error, transient):
    with pytest.raises(Exception) as info:
        base._check_provider_stream_error(('data: ' + json.dumps({'error': error})).encode())
    assert base._is_transient_infrastructure_error(info.value) is transient
