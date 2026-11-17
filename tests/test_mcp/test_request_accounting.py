"""Real stdio transport with deterministic, non-network LM test doubles."""
import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

from timeclaw.mcp.manager import MCPManager, MCPServerConfig
from timeclaw.core.accounting import AccountingIntegrityError, RequestLedger, logical_call, physical_attempt

SERVER = '''
from timeclaw.mcp.server_factory import AccountedFastMCP, register_tool
from timeclaw.clawhub.tool_base import ClawTool
from timeclaw.clawhub.validation.consistency import ConsistencyCheckTool
from timeclaw.core.accounting import logical_call, physical_attempt

class Backend:
    @logical_call
    async def complete(self, retry):
        if retry:
            try:
                with physical_attempt('test-transport'):
                    raise ConnectionError('test-only connection failure')
            except ConnectionError:
                pass
        with physical_attempt('test-transport') as result:
            result['usage'] = {'prompt_tokens': 7, 'completion_tokens': 3}

class Tool(ClawTool):
    name = 'counted_tool'
    description = 'test tool'
    parameters_schema = {'type':'object', 'properties': {
        'retry': {'type':'boolean'}, 'fail': {'type':'boolean'}}, 'required': []}
    async def _run(self, ts_data=None, **params):
        await Backend().complete(params.get('retry', False))
        if params.get('fail', False):
            raise ValueError('test-only tool failure')
        return {'ok': True}

mcp = AccountedFastMCP('accounting-test')
register_tool(mcp, Tool())
register_tool(mcp, ConsistencyCheckTool())
@mcp.tool()
async def untracked_tool() -> str:
    return 'untracked'
if __name__ == '__main__':
    mcp.run(transport='stdio')
'''


class Backend:
    @logical_call
    async def complete(self):
        with physical_attempt('test-transport') as result:
            result['usage'] = {'prompt_tokens': 5, 'completion_tokens': 2}


@pytest.fixture
async def manager(tmp_path):
    script = tmp_path / 'server.py'
    script.write_text(SERVER)
    repo = Path(__file__).resolve().parents[2]
    mgr = MCPManager([MCPServerConfig(name='test', command=sys.executable,
        args=[str(script)], env={'PYTHONPATH': str(repo)})])
    await mgr.start()
    try:
        yield mgr
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_parent_and_child_LM_costs_reconcile_through_real_stdio(manager, tmp_path):
    assert 'timeclaw_context' not in manager.get_tool('counted_tool').input_schema['properties']
    ledger = RequestLedger('parent-plus-child', tmp_path / 'ledgers')
    with ledger.activate():
        await Backend().complete()
        result = await manager.call_tool('counted_tool', {'ts_data': [1, 2], 'retry': False})
        assert result['success'] is True
    summary = ledger.summary()
    assert summary['logical_calls'] == summary['physical_attempts'] == 2
    assert summary['observed_prompt_tokens'] == 12
    assert summary['observed_completion_tokens'] == 5
    assert summary['usage_covers_all_attempts'] is True
    assert summary['mcp_tool_dispatches'] == 1
    events = ledger.read_events()
    assert len({e['pid'] for e in events if e['kind'] == 'logical_start'}) == 2
    assert sum(e['kind'] == 'request_start' for e in events) == 1
    assert sum(e['kind'] == 'request_end' for e in events) == 1
    assert summary['mcp_tool_logical_calls'] == 1
    for event in events:
        assert not any(k in event for k in ('messages', 'prompt', 'api_key', 'arguments', 'error_message'))


@pytest.mark.asyncio
async def test_child_retry_and_failure_preserve_usage_unknown(manager, tmp_path):
    ledger = RequestLedger('failure', tmp_path / 'ledgers')
    with ledger.activate():
        result = await manager.call_tool('counted_tool', {'ts_data':[1], 'retry':True, 'fail':True})
        assert result['success'] is False
    summary = ledger.summary()
    assert summary['logical_calls'] == 1
    assert summary['physical_attempts'] == 2
    assert summary['observed_prompt_tokens'] == 7
    assert summary['usage_covers_all_attempts'] is False
    assert summary['billed_usd'] is None
    ends = [e for e in ledger.read_events() if e['kind'] == 'mcp_tool_end']
    assert ends[0]['status'] == 'failed'


@pytest.mark.asyncio
async def test_uninstrumented_tool_fails_immediately_instead_of_zero_cost(manager, tmp_path):
    # Existing no-ledger clients retain their original tool transport.
    assert (await manager.call_tool('untracked_tool', {}))['success'] is True
    ledger = RequestLedger('missing-receipt', tmp_path / 'ledgers')
    with pytest.raises(AccountingIntegrityError, match='receipt'):
        with ledger.activate():
            await manager.call_tool('untracked_tool', {})
            pytest.fail('An uninstrumented tool must abort the request immediately')
    with pytest.raises(AccountingIntegrityError, match='receipt'):
        ledger.summary()


@pytest.mark.asyncio
@pytest.mark.parametrize('tool,args,field', [
    ('counted_tool', {'ts_data': [1], 'retry': []}, 'retry'),
    ('consistency_check', {'prediction': [64.625, 64.624],
                           'ts_data': [64.71, 64.66, 64.62]}, 'historical'),
])
async def test_schema_rejection_has_server_receipt_without_executing_tool(manager, tmp_path, tool, args, field):
    """An invalid call is a normal MCP error, not a missing-cost integrity error."""
    ledger = RequestLedger('schema-rejection', tmp_path / 'ledgers')
    with ledger.activate():
        result = await manager.call_tool(tool, args)
    assert result['success'] is False
    assert field in result['error']
    summary = ledger.summary()
    assert summary['logical_calls'] == summary['physical_attempts'] == 0
    assert summary['mcp_tool_dispatches'] == 1
    events = ledger.read_events()
    assert sum(e['kind'] == 'mcp_tool_start' for e in events) == 1
    assert sum(e['kind'] == 'mcp_tool_end' and e['status'] == 'failed' for e in events) == 1


@pytest.mark.asyncio
async def test_original_parameter_error_can_be_corrected_within_same_request(manager, tmp_path):
    args = {'prediction': [64.625, 64.624], 'ts_data': [64.71, 64.66, 64.62]}
    original_error = await manager.call_tool('consistency_check', args)
    ledger = RequestLedger('corrected-call', tmp_path / 'ledgers')
    with ledger.activate():
        rejected = await manager.call_tool('consistency_check', args)
        assert rejected == original_error
        corrected = await manager.call_tool('consistency_check',
            dict(args, historical=args['ts_data']))
    assert corrected['success'] is True
    assert corrected['data']['n_total'] == 3
    summary = ledger.summary()
    assert summary['mcp_tool_dispatches'] == 2
    assert summary['physical_attempts'] == 0


@pytest.mark.asyncio
async def test_five_concurrent_requests_keep_child_receipts_separate(manager, tmp_path):
    async def one(index):
        ledger = RequestLedger(f'parallel-{index}', tmp_path/'parallel')
        with ledger.activate():
            result = await manager.call_tool('counted_tool', {'ts_data': [index], 'retry': False})
            assert result['success'] is True
        summary = ledger.summary()
        assert summary['logical_calls'] == summary['physical_attempts'] == 1
        assert summary['observed_prompt_tokens'] == 7
        assert summary['mcp_tool_dispatches'] == 1
        assert {e['request_id'] for e in ledger.read_events()} == {f'parallel-{index}'}
        return summary['run_id']
    assert len(set(await asyncio.gather(*(one(i) for i in range(5))))) == 5
