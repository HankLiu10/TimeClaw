import asyncio
import json

from timeclaw.core import trace_logger
from timeclaw.memory import hierarchical as hm


async def _run_memory_distill_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_logger, "_trace_dir", tmp_path / "traces")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hm, "SOUL_FILE", tmp_path / "missing_SOUL.md")

    memory = hm.HierarchicalMemory(enabled=True)

    async def fake_on_commit(scope_keys) -> None:
        scope = scope_keys[0]
        await asyncio.sleep(0)
        trace_logger.decision("Distiller", f"finished {scope}")

    memory.distiller.on_commit = fake_on_commit

    trace_logger.start_run("trace-test", {"task_type": "forecast"})
    await memory.record(
        domain="weather",
        task="forecast",
        scope_keys=["weather_aligned_short_forecast"],
        description="forecast sample",
        predicted="[1, 2, 3]",
        correct=True,
        tool_chain=["seasonal_baseline"],
    )
    trace_logger.end_run("completed", {"trace_id": "trace-test"})

    return trace_logger._trace_dir / "trace-test.jsonl"


def test_trace_run_end_is_last_after_waiting_memory_tasks(tmp_path, monkeypatch):
    trace_path = asyncio.run(_run_memory_distill_trace(tmp_path, monkeypatch))

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert events == ["run_start", "decision", "run_end"]
    assert records[1]["payload"]["text"] == "finished weather_aligned_short_forecast"


async def _run_background_distill_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_logger, "_trace_dir", tmp_path / "traces")
    monkeypatch.chdir(tmp_path)

    async def background_emit() -> None:
        await asyncio.sleep(0)
        trace_logger.decision("Distiller", "background distill wrote")

    trace_logger.start_run("trace-bg", {"task_type": "forecast"})
    task = hm.DistillationAgent._create_background_task(background_emit(), "trace-bg-test")
    trace_logger.end_run("completed", {"trace_id": "trace-bg"})
    await task

    return trace_logger._trace_dir / "trace-bg.jsonl"


def test_background_distill_task_does_not_inherit_sample_trace(tmp_path, monkeypatch):
    trace_path = asyncio.run(_run_background_distill_trace(tmp_path, monkeypatch))

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert events == ["run_start", "run_end"]


def test_llm_call_trace_keeps_summary_not_full_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_logger, "_trace_dir", tmp_path / "traces")
    trace_logger.configure(verbose=False, full_dump=False)

    trace_logger.start_run(
        "trace-llm-summary",
        {
            "task_type": "forecast",
            "description": "A" * 800,
        },
    )
    trace_logger.llm_call(
        "unit-test",
        [
            {"role": "system", "content": "System prompt " + ("S" * 500)},
            {"role": "user", "content": "User prompt " + ("U" * 700)},
        ],
        type(
            "Resp",
            (),
            {
                "text": '{"done": true}',
                "parsed": {"done": True, "reasoning": "short answer"},
                "usage": {"prompt_tokens": 123, "completion_tokens": 45},
            },
        )(),
        transport={
            "request_id": "llmreq-123",
            "queue_wait_ms": 12.5,
            "inflight_after_acquire": 4,
            "max_inflight_requests": 32,
        },
    )
    trace_logger.end_run("completed", {"trace_id": "trace-llm-summary"})

    trace_path = trace_logger._trace_dir / "trace-llm-summary.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]

    assert records[0]["event"] == "content_block"
    assert records[0]["payload"]["kind"] == "run_description"
    assert records[1]["payload"]["description"]["chars"] == 800

    llm_payload = next(record["payload"] for record in records if record["event"] == "llm_call")
    assert "messages" not in llm_payload
    assert "response_text" not in llm_payload
    assert llm_payload["prompt"]["message_count"] == 2
    assert llm_payload["prompt"]["messages"][0]["role"] == "system"
    assert llm_payload["prompt"]["context_blocks"][0]["role"] == "system"
    assert llm_payload["prompt"]["context_blocks"][0]["ref"].startswith("prompt_message/system:")
    assert "decision" in llm_payload["response"]
    assert llm_payload["transport"]["request_id"] == "llmreq-123"
    assert llm_payload["transport"]["queue_wait_ms"] == 12.5


def test_llm_call_full_dump_writes_sidecar_file(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_logger, "_trace_dir", tmp_path / "traces")
    trace_logger.configure(verbose=False, full_dump=True)

    trace_logger.start_run("trace-llm-full", {"task_type": "forecast"})
    trace_logger.llm_call(
        "unit-test",
        [{"role": "user", "content": "full prompt body"}],
        type(
            "Resp",
            (),
            {
                "text": '{"done": true}',
                "parsed": {"done": True},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )(),
    )
    trace_logger.end_run("completed", {"trace_id": "trace-llm-full"})

    full_path = (trace_logger._trace_dir / "trace-llm-full.jsonl").with_suffix(".full.jsonl")
    assert full_path.exists()
    records = [json.loads(line) for line in full_path.read_text().splitlines()]
    assert records[0]["event"] == "llm_call_full"
    assert records[0]["payload"]["messages"][0]["content"] == "full prompt body"


def test_policy_transition_records_compact_and_full_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_logger, "_trace_dir", tmp_path / "traces")
    trace_logger.configure(verbose=False, full_dump=True)

    trace_logger.start_run("trace-policy-transition", {"task_type": "forecast"})
    trace_logger.policy_transition(
        state_before={"source_ref": "raw", "length": 12},
        action="seasonal_baseline",
        args={"horizon": 3},
        result={"forecast": [1.0, 2.0, 3.0]},
        state_after={"source_ref": "seasonal_baseline:1", "length": 3},
        llm_called=False,
    )
    trace_logger.end_run("completed", {"trace_id": "trace-policy-transition"})

    trace_path = trace_logger._trace_dir / "trace-policy-transition.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    transition = next(record for record in records if record["event"] == "policy_transition")
    assert set(transition["payload"]) == {
        "state_before", "action", "args", "result", "state_after", "llm_called",
    }
    assert transition["payload"]["action"] == "seasonal_baseline"
    assert transition["payload"]["llm_called"] is False

    full_path = trace_path.with_suffix(".full.jsonl")
    full_records = [json.loads(line) for line in full_path.read_text().splitlines()]
    full_transition = next(
        record for record in full_records if record["event"] == "policy_transition_full"
    )
    assert full_transition["payload"]["result"] == {"forecast": [1.0, 2.0, 3.0]}


def test_retrylater_run_does_not_persist_trace_files(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_logger, "_trace_dir", tmp_path / "traces")
    trace_logger.configure(verbose=False, full_dump=True)

    trace_logger.start_run("trace-retrylater", {"task_type": "forecast"})
    trace_logger.decision("Pipeline", "transient upstream rejection")
    trace_logger.end_run(
        "failed",
        {
            "trace_id": "trace-retrylater",
            "error_type": "RetryLater",
            "failure_reason": "RetryLater",
            "error": "llm_supplier_pool: websocket closed before any event: code=1013 reason=too many concurrent requests, please retry later",
        },
    )

    trace_path = trace_logger._trace_dir / "trace-retrylater.jsonl"
    full_path = trace_path.with_suffix(".full.jsonl")
    assert not trace_path.exists()
    assert not full_path.exists()
