"""Request-scoped append-only accounting, independent of discardable Agent traces.

logical_call = one backend.complete invocation; physical_attempt = one transport
attempt, including connection failures. Provider-internal attempts are unknown.
Usage is summed over observed transport responses, not over logical calls.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import fcntl
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from .errors import ExecutionContractError


_ledger = ContextVar("timeclaw_ledger", default=None)
_call_id = ContextVar("timeclaw_call_id", default=None)
MCP_ACCOUNTING_META = 'timeclaw_request_accounting'


class AccountingIntegrityError(ExecutionContractError):
    """A tool's execution cannot be reconciled with its request cost ledger."""


def current_ledger():
    return _ledger.get()


def accounting_active():
    return _ledger.get() is not None


class RequestLedger:
    def __init__(self, request_id: str, directory: Path):
        self.request_id = request_id
        self.run_id = uuid4().hex
        directory.mkdir(parents=True, exist_ok=True)
        self.path = (directory / f"{self.run_id}.jsonl").resolve()
        self.path.touch(exist_ok=False)
        self.events = []
        self.dispatch_id = None

    def read_events(self):
        with self.path.open(encoding='utf-8') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            try:
                events = [json.loads(line) for line in stream if line.strip()]
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        if any(e['run_id'] != self.run_id or e['request_id'] != self.request_id for e in events):
            raise AccountingIntegrityError('Mixed request identities in accounting ledger')
        return events

    def begin_dispatch(self, tool_name, server_name):
        dispatch_id = uuid4().hex
        self.record('mcp_dispatch_start', dispatch_id=dispatch_id,
                    tool_name=tool_name, server_name=server_name)
        return {'version': 1, 'ledger_path': str(self.path), 'run_id': self.run_id,
                'request_id': self.request_id, 'dispatch_id': dispatch_id, 'tool_name': tool_name}

    def validate_dispatch(self, dispatch_id):
        events = [e for e in self.read_events() if e.get('dispatch_id') == dispatch_id]
        starts = [e for e in events if e['kind'] == 'mcp_tool_start']
        ends = [e for e in events if e['kind'] == 'mcp_tool_end']
        sent = [e for e in events if e['kind'] == 'mcp_dispatch_start']
        if len(sent) != 1 or len(starts) != 1 or len(ends) != 1:
            raise AccountingIntegrityError('MCP tool execution lacks exactly one start/end receipt')
        if starts[0]['tool_name'] != sent[0]['tool_name'] or ends[0]['tool_name'] != sent[0]['tool_name']:
            raise AccountingIntegrityError('MCP tool accounting receipt names differ')
        if ends[0]['monotonic_ns'] < starts[0]['monotonic_ns']:
            raise AccountingIntegrityError('MCP tool accounting receipt has invalid timing')

    @classmethod
    @contextmanager
    def attach_tool(cls, metadata, tool_name):
        """Attach a child process to a parent's ledger, without request boundaries.

        Metadata travels in MCP protocol `_meta`, never in model-visible args.
        The child acknowledges even failed execution; missing acknowledgements
        invalidate accounting instead of being reported as zero LM calls.
        """
        if _ledger.get() is not None:
            raise AccountingIntegrityError('Nested MCP ledger attachment')
        if not isinstance(metadata, dict) or metadata.get('version') != 1:
            raise AccountingIntegrityError('Invalid MCP accounting metadata version')
        if metadata.get('tool_name') != tool_name:
            raise AccountingIntegrityError('MCP accounting metadata identifies another tool')
        ledger = cls.__new__(cls)
        ledger.path = Path(metadata['ledger_path'])
        if not ledger.path.is_absolute() or not ledger.path.is_file():
            raise AccountingIntegrityError('MCP accounting requires an existing absolute ledger path')
        ledger.run_id, ledger.request_id = metadata['run_id'], metadata['request_id']
        ledger.dispatch_id, ledger.events = metadata['dispatch_id'], []
        events = ledger.read_events()
        sent = [e for e in events if e['kind'] == 'mcp_dispatch_start'
                and e['dispatch_id'] == ledger.dispatch_id and e['tool_name'] == tool_name]
        if len(sent) != 1:
            raise AccountingIntegrityError('MCP child cannot find its parent dispatch')
        token = _ledger.set(ledger)
        status = 'completed'
        try:
            ledger.record('mcp_tool_start', tool_name=tool_name)
            yield ledger
        except BaseException:
            status = 'failed'
            raise
        finally:
            try:
                ledger.record('mcp_tool_end', tool_name=tool_name, status=status)
            finally:
                _ledger.reset(token)

    def record(self, kind, **fields):
        event = {"kind": kind, "run_id": self.run_id, "request_id": self.request_id,
                 "monotonic_ns": time.perf_counter_ns(), 'pid': os.getpid(), **fields}
        if self.dispatch_id is not None:
            event['dispatch_id'] = self.dispatch_id
        # No prompts, credentials, or raw exception messages in the cost ledger.
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        self.events.append(event)

    @contextmanager
    def activate(self):
        if _ledger.get() is not None:
            raise RuntimeError("nested request accounting is not supported")
        token = _ledger.set(self)
        self.record("request_start")
        status = "completed"
        try:
            yield self
        except BaseException as exc:
            status = "cancelled" if type(exc).__name__ == "CancelledError" else "failed"
            raise
        finally:
            try:
                self.record("request_end", status=status)
            finally:
                _ledger.reset(token)

    def summary(self):
        self.events = self.read_events()
        dispatches = [e for e in self.events if e['kind'] == 'mcp_dispatch_start']
        for event in dispatches:
            self.validate_dispatch(event['dispatch_id'])
        starts = [e for e in self.events if e['kind'] == 'attempt_start']
        ends = [e for e in self.events if e['kind'] == 'attempt_end']
        calls = [e for e in self.events if e['kind'] == 'logical_start']
        call_ends = [e for e in self.events if e['kind'] == 'logical_end']
        used = [e for e in ends if e.get('usage_complete')]
        # Missing any attempt's usage cannot be represented as a zero bill.
        complete = len(used) == len(starts) and len(call_ends) == len(calls)
        if any(not any(a['call_id'] == c['call_id'] for a in starts) for c in calls):
            complete = False
        request_start = next(e for e in self.events if e['kind'] == 'request_start')
        request_end = next(e for e in reversed(self.events) if e['kind'] == 'request_end')
        protocols = [e for e in self.events if e['kind'] == 'input_protocol']
        node_starts, node_counts, node_seconds = {}, {}, {}
        for event in self.events:
            if event['kind'] == 'node_start':
                node_starts[event['node_id']] = event
                kind = event['node_kind']
                node_counts[kind] = node_counts.get(kind, 0) + 1
            elif event['kind'] == 'node_end':
                start = node_starts[event['node_id']]
                kind = start['node_kind']
                node_seconds[kind] = node_seconds.get(kind, 0.0) + (
                    event['monotonic_ns'] - start['monotonic_ns']) / 1e9
        return {"request_id": self.request_id, "run_id": self.run_id,
                "information_protocol": protocols[0]['protocol'] if protocols else None,
                "information_protocol_ref": protocols[0]['protocol_ref'] if protocols else None,
                "status": request_end['status'], "ledger_path": str(self.path),
                "online_seconds": (request_end['monotonic_ns']-request_start['monotonic_ns'])/1e9,
                "logical_calls": len(calls), "physical_attempts": len(starts),
                'mcp_tool_dispatches': len(dispatches),
                'mcp_tool_logical_calls': sum('dispatch_id' in e for e in calls),
                "observed_prompt_tokens": sum(e['usage']['prompt_tokens'] for e in used),
                "observed_completion_tokens": sum(e['usage']['completion_tokens'] for e in used),
                "usage_covers_all_attempts": complete,
                "node_counts_by_kind": node_counts, "node_seconds_by_kind": node_seconds,
                "billed_usd": None, "provider_internal_attempts": None}


def logical_call(function):
    @wraps(function)
    async def wrapped(self, *args, **kwargs):
        ledger = _ledger.get()
        if ledger is None:
            return await function(self, *args, **kwargs)
        call_id = uuid4().hex
        token = _call_id.set(call_id)
        ledger.record("logical_start", call_id=call_id)
        status = "completed"
        try:
            return await function(self, *args, **kwargs)
        except BaseException:
            status = "failed"
            raise
        finally:
            ledger.record("logical_end", call_id=call_id, status=status)
            _call_id.reset(token)
    return wrapped


@contextmanager
def physical_attempt(model):
    ledger = _ledger.get()
    result = {}
    if ledger is None:
        yield result
        return
    attempt_id = uuid4().hex
    fields = {"call_id": _call_id.get(), "attempt_id": attempt_id, "model": model}
    ledger.record("attempt_start", **fields)
    status = "completed"
    error_type = None
    try:
        yield result
    except BaseException as exc:
        status, error_type = "failed", type(exc).__name__
        raise
    finally:
        usage = result.get('usage') or {}
        complete = all(isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)
                       and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens'))
        ledger.record("attempt_end", **fields, status=status, error_type=error_type,
                      usage=usage, usage_complete=complete)
