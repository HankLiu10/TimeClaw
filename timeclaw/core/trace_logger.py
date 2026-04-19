"""Centralized verbose trace logger for the TimeClaw pipeline.

Produces human-readable, narrative-style logs that show how the system
thinks, decides, and executes — not raw JSON dumps.

Usage:
    from timeclaw.core.trace_logger import trace as tl

    tl.configure(verbose=True)
    tl.stage("T+0 Plan")
    tl.thinking("Manager", "Parsing user intent...")
    tl.llm_call("Manager", messages, response)
    tl.decision("Manager", "Task type: forecast, horizon: 10")
    tl.tool_exec("Executor", "normalize", args, result)
"""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

_verbose: bool = False
_full_dump: bool = False
_trace_dir: Path = Path(".timeclaw/traces")
_active_trace_path_var: ContextVar[Path | None] = ContextVar(
    "timeclaw_active_trace_path",
    default=None,
)
_seen_content_refs_var: ContextVar[set[str] | None] = ContextVar(
    "timeclaw_seen_content_refs",
    default=None,
)

_SEPARATOR = "═" * 72
_THIN_SEP = "─" * 60


def configure(verbose: bool, full_dump: bool = False) -> None:
    global _verbose, _full_dump
    _verbose = verbose
    _full_dump = full_dump


def start_run(trace_id: str, metadata: dict[str, Any] | None = None) -> None:
    _trace_dir.mkdir(parents=True, exist_ok=True)
    active_trace_path = _trace_dir / f"{trace_id}.jsonl"
    _active_trace_path_var.set(active_trace_path)
    _seen_content_refs_var.set(set())
    if active_trace_path.exists():
        active_trace_path.unlink()
    full_path = active_trace_path.with_suffix(".full.jsonl")
    if full_path.exists():
        full_path.unlink()
    payload = dict(metadata or {})
    if "description" in payload:
        description = str(payload.pop("description") or "")
        block = _store_content_block("run_description", description)
        payload["description"] = block
    _record_event("run_start", payload)


def end_run(status: str, metadata: dict[str, Any] | None = None) -> None:
    _record_event("run_end", {"status": status, **(metadata or {})})
    _active_trace_path_var.set(None)
    _seen_content_refs_var.set(None)


def detach_context() -> None:
    """Clear the active trace context for the current async context.

    This is used when spawning background work that must not inherit the
    sample-level trace sink, such as post-commit memory distillation.
    """
    _active_trace_path_var.set(None)
    _seen_content_refs_var.set(None)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _record_event(event_type: str, payload: dict[str, Any] | Any) -> None:
    active_trace_path = _active_trace_path_var.get()
    if active_trace_path is None:
        return
    record = {
        "ts": time.time(),
        "event": event_type,
        "payload": _json_safe(payload),
    }
    try:
        with active_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug(f"[trace_logger] failed to persist trace event: {exc}")


def _record_full_dump(event_type: str, payload: dict[str, Any] | Any) -> None:
    active_trace_path = _active_trace_path_var.get()
    if active_trace_path is None:
        return
    full_path = active_trace_path.with_suffix(".full.jsonl")
    record = {
        "ts": time.time(),
        "event": event_type,
        "payload": _json_safe(payload),
    }
    try:
        with full_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug(f"[trace_logger] failed to persist full trace event: {exc}")


# ── Formatting helpers ──────────────────────────────────────────

def _smart_trunc(obj: Any, limit: int = 2000) -> str:
    """Truncate intelligently: arrays get summarized, dicts get key-listed.

    Default limit is generous (2000 chars) — logs are important for debugging
    and post-hoc analysis. Use smaller limits only for dense array previews.
    """
    import numpy as np

    if obj is None:
        return "None"

    if isinstance(obj, np.ndarray):
        if obj.size <= 50:
            return str(obj.tolist())
        return (
            f"ndarray shape={obj.shape} "
            f"range=[{float(obj.min()):.4f}, {float(obj.max()):.4f}] "
            f"mean={float(obj.mean()):.4f}"
        )

    if isinstance(obj, (list, tuple)) and len(obj) > 20:
        try:
            arr = [float(x) for x in obj[:10]]
            return f"[{', '.join(f'{v:.4f}' for v in arr)}, ... ({len(obj)} items)]"
        except (TypeError, ValueError):
            pass

    if isinstance(obj, dict):
        if "forecast" in obj:
            fc = obj["forecast"]
            if isinstance(fc, (list, tuple)) and len(fc) > 10:
                preview = [f"{float(v):.4f}" for v in fc[:10]]
                return f"forecast=[{', '.join(preview)}, ...] ({len(fc)} steps)"
            return f"forecast={fc}"
        if "acf" in obj or "pacf" in obj:
            return f"ACF/PACF computed (lags={len(obj.get('acf', []))})"

    s = str(obj)
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [{len(s)} chars total]"


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _preview_text(text: Any, limit: int = 220) -> str:
    normalized = _normalize_whitespace(str(text))
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + f"… [{len(normalized)} chars]"


def _text_fingerprint(text: Any) -> str:
    raw = str(text).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:10]


def _flatten_message_content(content: Any) -> tuple[str, bool]:
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return str(content), False

    text_parts: list[str] = []
    has_image = False
    for item in content:
        if not isinstance(item, dict):
            text_parts.append(str(item))
            continue
        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(str(item.get("text", "")))
        elif item_type == "image_url":
            has_image = True
            text_parts.append("[IMAGE]")
        else:
            text_parts.append(str(item))
    return "\n".join(text_parts), has_image


def _summarize_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    outlined: list[dict[str, Any]] = []
    total_chars = 0
    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "?"))
        content, has_image = _flatten_message_content(msg.get("content", ""))
        total_chars += len(content)
        outlined.append({
            "index": idx,
            "role": role,
            "chars": len(content),
            "has_image": has_image,
            "hash": _text_fingerprint(content),
            "preview": _preview_text(content, 180),
        })
    return {
        "message_count": len(messages),
        "total_chars": total_chars,
        "messages": outlined,
    }


def _store_content_block(kind: str, text: Any) -> dict[str, Any]:
    content = str(text)
    ref = f"{kind}:{_text_fingerprint(content)}"
    ref_payload = {
        "ref": ref,
        "kind": kind,
        "chars": len(content),
    }
    seen = _seen_content_refs_var.get()
    if seen is None:
        return ref_payload
    if ref not in seen:
        seen.add(ref)
        _record_event("content_block", {
            **ref_payload,
            "preview": _preview_text(content, 220),
            "content": content,
        })
    return ref_payload


def _summarize_messages_with_refs(messages: list[dict[str, Any]]) -> dict[str, Any]:
    outlined: list[dict[str, Any]] = []
    context_blocks: list[dict[str, Any]] = []
    total_chars = 0
    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "?"))
        content, has_image = _flatten_message_content(msg.get("content", ""))
        total_chars += len(content)
        ref_payload = _store_content_block(f"prompt_message/{role}", content)
        outlined.append({
            "index": idx,
            "role": role,
            "has_image": has_image,
            **ref_payload,
        })
        context_blocks.append({
            "index": idx,
            "role": role,
            "ref": ref_payload["ref"],
            "kind": ref_payload["kind"],
            "chars": ref_payload["chars"],
        })
    return {
        "message_count": len(messages),
        "total_chars": total_chars,
        "messages": outlined,
        "context_blocks": context_blocks,
    }


def _summarize_mapping(mapping: dict[str, Any], *, value_limit: int = 180) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None or isinstance(value, (bool, int, float)):
            summarized[str(key)] = value
            continue
        if isinstance(value, str):
            summarized[str(key)] = _preview_text(value, value_limit)
            continue
        if isinstance(value, (list, tuple)):
            items = list(value)
            preview = [_preview_text(item, 80) for item in items[:4]]
            summarized[str(key)] = {"len": len(items), "preview": preview}
            continue
        if isinstance(value, dict):
            summarized[str(key)] = {
                "keys": list(value.keys())[:8],
                "preview": _preview_text(_smart_trunc(value, 300), value_limit),
            }
            continue
        summarized[str(key)] = _preview_text(value, value_limit)
    return summarized


def _summarize_trace_value(value: Any, *, limit: int = 220) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _preview_text(value, limit)
    if isinstance(value, dict):
        return {
            "keys": list(value.keys())[:8],
            "preview": _preview_text(_smart_trunc(value, min(400, limit * 2)), limit),
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        return {
            "len": len(items),
            "preview": [_summarize_trace_value(item, limit=80) for item in items[:4]],
        }
    return _preview_text(value, limit)


def _summarize_response(parsed: Any, text: str) -> dict[str, Any]:
    if parsed is not None:
        summary = {
            "kind": "parsed",
            "decision": _extract_llm_decision(parsed),
        }
        if isinstance(parsed, dict):
            summary["parsed"] = _json_safe(parsed)
        return summary
    if text:
        return {
            "kind": "text",
            **_store_content_block("response_text", text),
        }
    return {"kind": "empty"}


def _format_result_data(data: Any) -> str:
    """Format tool result data for display."""
    import numpy as np

    if data is None:
        return "None"
    if isinstance(data, np.ndarray):
        if data.size <= 100:
            return f"ndarray shape={data.shape} values={data.tolist()}"
        return (
            f"ndarray shape={data.shape} "
            f"range=[{float(data.min()):.4f}, {float(data.max()):.4f}] "
            f"mean={float(data.mean()):.4f}"
        )
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            lines.append(f"    {k}: {_smart_trunc(v, 2000)}")
        return "\n".join(lines)
    return _smart_trunc(data, 5000)


def _extract_llm_decision(parsed: Any) -> str:
    """Extract the key decision from an LLM response."""
    if not isinstance(parsed, dict):
        return _smart_trunc(parsed, 2000)

    parts = []
    if "task_type" in parsed:
        parts.append(f"任务类型: {parsed['task_type']}")
    if "sub_goals" in parsed:
        parts.append(f"子目标: {parsed['sub_goals']}")
    if "horizon" in parsed:
        parts.append(f"预测步长: {parsed['horizon']}")
    if "tool_chain" in parsed:
        chain = []
        for item in parsed["tool_chain"]:
            if isinstance(item, dict):
                name = item.get("tool", "?")
                args = item.get("args", {})
                chain.append(f"{name}({args})" if args else name)
            else:
                chain.append(str(item))
        parts.append(f"工具链: {' → '.join(chain)}")
    if "reasoning" in parsed:
        parts.append(f"推理: {_smart_trunc(parsed['reasoning'], 2000)}")
    if "tool" in parsed and "tool_chain" not in parsed:
        parts.append(f"选择工具: {parsed['tool']}")
        if parsed.get("args"):
            parts.append(f"参数: {parsed['args']}")
    if "done" in parsed:
        parts.append(f"完成: {parsed['done']}")
    if "fingerprint" in parsed:
        fp = parsed["fingerprint"]
        if isinstance(fp, dict):
            parts.append(f"指纹: {', '.join(f'{k}={v}' for k, v in fp.items())}")
    if "recommendations" in parsed:
        parts.append(f"建议: {parsed['recommendations']}")
    if "accepted" in parsed:
        parts.append(f"接受: {parsed['accepted']}")
    if "confidence" in parsed:
        parts.append(f"置信度: {parsed['confidence']}")
    if "answer" in parsed and parsed["answer"] is not None:
        parts.append(f"答案: {_smart_trunc(parsed['answer'], 2000)}")
    if "consistent" in parsed:
        parts.append(f"一致性: {parsed['consistent']}")
    if "should_reanalyze" in parsed:
        parts.append(f"需要重分析: {parsed['should_reanalyze']}")
    if "covers_predicates" in parsed:
        parts.append(f"覆盖谓词: {parsed['covers_predicates']}")

    if not parts:
        return _smart_trunc(parsed, 2000)
    return "\n    ".join(parts)


# ── Public API ──────────────────────────────────────────────────

def stage(name: str, extra: str = "") -> None:
    """Mark a major pipeline stage."""
    _record_event("stage", {"name": name, "extra": _preview_text(extra, 220)})
    if not _verbose:
        return
    msg = f"\n{_SEPARATOR}\n  {name}\n{_SEPARATOR}"
    if extra:
        msg += f"\n  {extra}"
    logger.info(msg)


def thinking(agent: str, description: str) -> None:
    """Show that an agent is about to think (LLM call)."""
    _record_event("thinking", {"agent": agent, "description": _preview_text(description, 220)})
    if not _verbose:
        return
    logger.info(f"[{agent}] 思考中… {description}")


def llm_call(
    caller: str,
    messages: list[dict[str, Any]],
    response: Any,
    *,
    response_format: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    """Log an LLM call — focus on the decision, not the raw prompt."""
    resp_parsed = None
    resp_usage = {}
    resp_text = ""
    if hasattr(response, "parsed"):
        resp_parsed = response.parsed
    if hasattr(response, "usage"):
        resp_usage = response.usage or {}
    if hasattr(response, "text"):
        resp_text = response.text or ""

    tokens_in = resp_usage.get("prompt_tokens", "?")
    tokens_out = resp_usage.get("completion_tokens", "?")

    _record_event("llm_call", {
        "caller": caller,
        "response_format": response_format,
        "reasoning_effort": reasoning_effort,
        "usage": resp_usage,
        "prompt": _summarize_messages_with_refs(messages),
        "response": _summarize_response(resp_parsed, resp_text),
    })
    if _full_dump:
        _record_full_dump("llm_call_full", {
            "caller": caller,
            "messages": messages,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            "response_text": resp_text,
            "response_parsed": resp_parsed,
            "usage": resp_usage,
        })

    if not _verbose:
        return

    if _full_dump:
        parts = [f"\n{_THIN_SEP}", f"[{caller}] LLM 调用详情 (tokens: {tokens_in} in / {tokens_out} out)"]
        parts.append(f"{_THIN_SEP}")
        parts.append("【发送的 Prompt】:")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        text_parts.append("[IMAGE]")
                content = "\n".join(text_parts)
            parts.append(f"  --- [{role}] (msg {i+1}/{len(messages)}) ---")
            parts.append(f"  {content}")
        parts.append(f"  --- end of prompt ---")
        parts.append("")
        parts.append("【LLM 原始返回】:")
        parts.append(f"  {resp_text}")
        parts.append(f"{_THIN_SEP}")
        logger.info("\n".join(parts))
    else:
        decision_source = resp_parsed if resp_parsed is not None else resp_text
        decision = _extract_llm_decision(decision_source)
        effort_text = f", effort={reasoning_effort}" if reasoning_effort else ""
        logger.info(
            f"[{caller}] LLM 返回 (tokens: {tokens_in} in / {tokens_out} out{effort_text}):\n"
            f"    {decision}"
        )


def llm_prompt_detail(
    caller: str,
    messages: list[dict[str, Any]],
) -> None:
    """Log full LLM prompt — only when extra detail is needed."""
    _record_event("llm_prompt_detail", {
        "caller": caller,
        "prompt": _summarize_messages_with_refs(messages),
    })
    if _full_dump:
        _record_full_dump("llm_prompt_detail_full", {"caller": caller, "messages": messages})
    if not _verbose:
        return
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = str(content)
        if len(content) > 5000:
            content = content[:5000] + f"… [{len(content)} chars]"
        parts.append(f"    [{role}]: {content}")
    logger.debug(f"[{caller}] LLM Prompt 详情:\n" + "\n".join(parts))


def decision(agent: str, text: str) -> None:
    """Log a key decision point."""
    _record_event("decision", {"agent": agent, "text": _preview_text(text, 260)})
    if not _verbose:
        return
    logger.info(f"[{agent}] → {text}")


def tool_exec(
    caller: str,
    tool_name: str,
    args: dict[str, Any],
    result: Any,
) -> None:
    """Log a tool execution with concise result summary."""
    success = getattr(result, "success", None)
    data = getattr(result, "data", result)
    error = getattr(result, "error", None)
    metadata = getattr(result, "metadata", {})

    _record_event("tool_exec", {
        "caller": caller,
        "tool_name": tool_name,
        "args": _summarize_mapping(args, value_limit=120),
        "success": success,
        "result": _summarize_trace_value(data, limit=220),
        "error": error,
        "metadata": _summarize_mapping(metadata, value_limit=120) if isinstance(metadata, dict) else _summarize_trace_value(metadata, limit=120),
    })

    if not _verbose:
        return

    status = "✓" if success else f"✗ {error}"
    args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "默认参数"

    lines = [f"[{caller}] 工具 {tool_name}({args_str}): {status}"]
    if success and data is not None:
        lines.append(f"    结果: {_format_result_data(data)}")
    if metadata:
        interesting = {k: v for k, v in metadata.items() if k != "method"}
        if interesting:
            lines.append(f"    元数据: {_smart_trunc(interesting, 2000)}")

    logger.info("\n".join(lines))


def agent_step(
    agent: str,
    phase: str,
    data: dict[str, Any] | Any,
) -> None:
    """Log an agent lifecycle phase — extract key info only."""
    event_payload: dict[str, Any] = {"agent": agent, "phase": phase}
    if isinstance(data, dict):
        event_payload["summary"] = _summarize_mapping(data, value_limit=160)
    else:
        event_payload["summary"] = _summarize_trace_value(data, limit=200)
    _record_event("agent_step", event_payload)
    if not _verbose:
        return

    if not isinstance(data, dict):
        logger.info(f"[{agent}] {phase}: {_smart_trunc(data, 2000)}")
        return

    if phase == "perceive":
        parts = []
        if "task_description" in data:
            parts.append(f"任务: {data['task_description']}")
        if "data_shape" in data:
            parts.append(f"数据: shape={data['data_shape']}")
        if "modalities" in data:
            parts.append(f"模态: {data['modalities']}")
        if "task_type" in data:
            parts.append(f"类型: {data['task_type']}")
        if parts:
            logger.info(f"[{agent}] 感知:\n    " + "\n    ".join(parts))
            return

    if phase == "reflect":
        summary_parts = []
        for k, v in data.items():
            summary_parts.append(f"{k}={v}")
        logger.info(f"[{agent}] 反思: {', '.join(summary_parts)}")
        return

    if phase == "plan":
        if "tool_chain" in data:
            chain = []
            for item in data["tool_chain"]:
                if isinstance(item, dict):
                    chain.append(item.get("tool", "?"))
                else:
                    chain.append(str(item))
            logger.info(
                f"[{agent}] 规划工具链: {' → '.join(chain)}\n"
                f"    理由: {data.get('reasoning', '?')}"
            )
            return
        if "intent" in data:
            intent = data["intent"]
            logger.info(
                f"[{agent}] 规划:\n"
                f"    意图: {_smart_trunc(intent, 2000)}\n"
                f"    DAG: {' → '.join(data.get('dag_nodes', []))}"
            )
            return
        if "checks_to_run" in data:
            logger.info(f"[{agent}] 验证计划: {data['checks_to_run']}")
            return
        if "diagnosis" in data:
            diag = data["diagnosis"]
            logger.info(f"[{agent}] 诊断结果:\n    {_smart_trunc(diag, 5000)}")
            return

    if phase == "act":
        if "gate_accepted" in data:
            checks = data.get("checks", [])
            check_strs = [f"{'✓' if c['passed'] else '✗'} {c['name']}" for c in checks]
            logger.info(
                f"[{agent}] 验证结果: {'通过' if data['gate_accepted'] else '未通过'}\n"
                f"    检查项: {', '.join(check_strs)}\n"
                f"    预测: {_smart_trunc(data.get('prediction_preview'), 2000)}"
            )
            return
        if "prediction_type" in data:
            logger.info(
                f"[{agent}] 执行完成:\n"
                f"    工具: {data.get('tools_passed', 0)}/{data.get('tools_executed', 0)} 通过\n"
                f"    反归一化: {data.get('denormalized', False)}\n"
                f"    预测: {_smart_trunc(data.get('prediction_preview'), 2000)}"
            )
            return

    if phase in ("fusion", "cross_modal_check", "synthesis"):
        logger.info(f"[{agent}] {phase}:\n    {_smart_trunc(data, 5000)}")
        return

    logger.info(f"[{agent}] {phase}: {_smart_trunc(data, 3000)}")


def pipeline_stage(
    stage_name: str,
    data: dict[str, Any] | Any,
) -> None:
    """Log a pipeline stage — minimal, key facts only."""
    payload = {"stage_name": stage_name}
    if isinstance(data, dict):
        payload["summary"] = _summarize_mapping(data, value_limit=160)
    else:
        payload["summary"] = _summarize_trace_value(data, limit=200)
    _record_event("pipeline_stage", payload)
    if not _verbose:
        return
    # Pipeline stages are already covered by stage() + agent logs.
    # Only log if there's something the agent logs didn't cover.
    pass


def reasoning_step(
    caller: str,
    step: int,
    thought: dict[str, Any] | None = None,
    observation: Any = None,
    critic: str | None = None,
) -> None:
    """Log a reasoning iteration step."""
    _record_event("reasoning_step", {
        "caller": caller,
        "step": step,
        "thought": _summarize_mapping(thought, value_limit=160) if isinstance(thought, dict) else _summarize_trace_value(thought, limit=180),
        "observation": _summarize_trace_value(observation, limit=180),
        "critic": _preview_text(critic, 200) if critic else None,
    })
    if not _verbose:
        return

    lines = [f"[{caller}] 推理 step {step}:"]
    if thought:
        if thought.get("done"):
            lines.append(f"    结论: {thought.get('reasoning', '完成')}")
        else:
            if thought.get("tool"):
                lines.append(f"    思考: {thought.get('reasoning', '?')}")
                lines.append(f"    → 调用工具: {thought['tool']}({thought.get('args', {})})")
            else:
                lines.append(f"    思考: {_smart_trunc(thought, 2000)}")
    if observation is not None:
        lines.append(f"    观察: {_smart_trunc(observation, 2000)}")
    if critic:
        lines.append(f"    评审: {_smart_trunc(critic, 2000)}")

    logger.info("\n".join(lines))
