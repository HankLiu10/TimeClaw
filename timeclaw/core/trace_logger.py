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

import json
from typing import Any

from loguru import logger

_verbose: bool = False

_SEPARATOR = "═" * 72
_THIN_SEP = "─" * 60


def configure(verbose: bool) -> None:
    global _verbose
    _verbose = verbose


# ── Formatting helpers ──────────────────────────────────────────

def _smart_trunc(obj: Any, limit: int = 300) -> str:
    """Truncate intelligently: arrays get summarized, dicts get key-listed."""
    import numpy as np

    if obj is None:
        return "None"

    if isinstance(obj, np.ndarray):
        if obj.size <= 8:
            return str(obj.tolist())
        return (
            f"ndarray shape={obj.shape} "
            f"range=[{float(obj.min()):.4f}, {float(obj.max()):.4f}] "
            f"mean={float(obj.mean()):.4f}"
        )

    if isinstance(obj, (list, tuple)) and len(obj) > 10:
        try:
            arr = [float(x) for x in obj[:5]]
            return f"[{', '.join(f'{v:.4f}' for v in arr)}, ... ({len(obj)} items)]"
        except (TypeError, ValueError):
            pass

    if isinstance(obj, dict):
        if "forecast" in obj:
            fc = obj["forecast"]
            if isinstance(fc, (list, tuple)) and len(fc) > 5:
                preview = [f"{float(v):.4f}" for v in fc[:5]]
                return f"forecast=[{', '.join(preview)}, ...] ({len(fc)} steps)"
            return f"forecast={fc}"
        if "acf" in obj or "pacf" in obj:
            return f"ACF/PACF computed (lags={len(obj.get('acf', []))})"

    s = str(obj)
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [{len(s)} chars total]"


def _format_result_data(data: Any) -> str:
    """Format tool result data for display."""
    import numpy as np

    if data is None:
        return "None"
    if isinstance(data, np.ndarray):
        return (
            f"ndarray shape={data.shape} "
            f"range=[{float(data.min()):.4f}, {float(data.max()):.4f}] "
            f"mean={float(data.mean()):.4f}"
        )
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            lines.append(f"    {k}: {_smart_trunc(v, 200)}")
        return "\n".join(lines)
    return _smart_trunc(data, 500)


def _extract_llm_decision(parsed: Any) -> str:
    """Extract the key decision from an LLM response."""
    if not isinstance(parsed, dict):
        return _smart_trunc(parsed, 300)

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
        parts.append(f"推理: {_smart_trunc(parsed['reasoning'], 300)}")
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
        parts.append(f"答案: {_smart_trunc(parsed['answer'], 200)}")
    if "consistent" in parsed:
        parts.append(f"一致性: {parsed['consistent']}")
    if "should_reanalyze" in parsed:
        parts.append(f"需要重分析: {parsed['should_reanalyze']}")
    if "covers_predicates" in parsed:
        parts.append(f"覆盖谓词: {parsed['covers_predicates']}")

    if not parts:
        return _smart_trunc(parsed, 300)
    return "\n    ".join(parts)


# ── Public API ──────────────────────────────────────────────────

def stage(name: str, extra: str = "") -> None:
    """Mark a major pipeline stage."""
    if not _verbose:
        return
    msg = f"\n{_SEPARATOR}\n  {name}\n{_SEPARATOR}"
    if extra:
        msg += f"\n  {extra}"
    logger.info(msg)


def thinking(agent: str, description: str) -> None:
    """Show that an agent is about to think (LLM call)."""
    if not _verbose:
        return
    logger.info(f"[{agent}] 思考中… {description}")


def llm_call(
    caller: str,
    messages: list[dict[str, Any]],
    response: Any,
    *,
    response_format: str | None = None,
) -> None:
    """Log an LLM call — focus on the decision, not the raw prompt."""
    if not _verbose:
        return

    user_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_msg = content[:500]
            break

    resp_parsed = None
    resp_usage = {}
    if hasattr(response, "parsed"):
        resp_parsed = response.parsed
    if hasattr(response, "usage"):
        resp_usage = response.usage or {}

    tokens_in = resp_usage.get("prompt_tokens", "?")
    tokens_out = resp_usage.get("completion_tokens", "?")

    decision = _extract_llm_decision(resp_parsed)

    logger.info(
        f"[{caller}] LLM 返回 (tokens: {tokens_in} in / {tokens_out} out):\n"
        f"    {decision}"
    )


def llm_prompt_detail(
    caller: str,
    messages: list[dict[str, Any]],
) -> None:
    """Log full LLM prompt — only when extra detail is needed."""
    if not _verbose:
        return
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = str(content)
        if len(content) > 500:
            content = content[:500] + f"… [{len(content)} chars]"
        parts.append(f"    [{role}]: {content}")
    logger.debug(f"[{caller}] LLM Prompt 详情:\n" + "\n".join(parts))


def decision(agent: str, text: str) -> None:
    """Log a key decision point."""
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
    if not _verbose:
        return

    success = getattr(result, "success", None)
    data = getattr(result, "data", result)
    error = getattr(result, "error", None)
    metadata = getattr(result, "metadata", {})

    status = "✓" if success else f"✗ {error}"
    args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "默认参数"

    lines = [f"[{caller}] 工具 {tool_name}({args_str}): {status}"]
    if success and data is not None:
        lines.append(f"    结果: {_format_result_data(data)}")
    if metadata:
        interesting = {k: v for k, v in metadata.items() if k != "method"}
        if interesting:
            lines.append(f"    元数据: {_smart_trunc(interesting, 300)}")

    logger.info("\n".join(lines))


def agent_step(
    agent: str,
    phase: str,
    data: dict[str, Any] | Any,
) -> None:
    """Log an agent lifecycle phase — extract key info only."""
    if not _verbose:
        return

    if not isinstance(data, dict):
        logger.info(f"[{agent}] {phase}: {_smart_trunc(data, 300)}")
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
                f"    意图: {_smart_trunc(intent, 300)}\n"
                f"    DAG: {' → '.join(data.get('dag_nodes', []))}"
            )
            return
        if "checks_to_run" in data:
            logger.info(f"[{agent}] 验证计划: {data['checks_to_run']}")
            return
        if "diagnosis" in data:
            diag = data["diagnosis"]
            logger.info(f"[{agent}] 诊断结果:\n    {_smart_trunc(diag, 500)}")
            return

    if phase == "act":
        if "gate_accepted" in data:
            checks = data.get("checks", [])
            check_strs = [f"{'✓' if c['passed'] else '✗'} {c['name']}" for c in checks]
            logger.info(
                f"[{agent}] 验证结果: {'通过' if data['gate_accepted'] else '未通过'}\n"
                f"    检查项: {', '.join(check_strs)}\n"
                f"    预测: {_smart_trunc(data.get('prediction_preview'), 200)}"
            )
            return
        if "prediction_type" in data:
            logger.info(
                f"[{agent}] 执行完成:\n"
                f"    工具: {data.get('tools_passed', 0)}/{data.get('tools_executed', 0)} 通过\n"
                f"    反归一化: {data.get('denormalized', False)}\n"
                f"    预测: {_smart_trunc(data.get('prediction_preview'), 200)}"
            )
            return

    if phase in ("fusion", "cross_modal_check", "synthesis"):
        logger.info(f"[{agent}] {phase}:\n    {_smart_trunc(data, 500)}")
        return

    logger.info(f"[{agent}] {phase}: {_smart_trunc(data, 400)}")


def pipeline_stage(
    stage_name: str,
    data: dict[str, Any] | Any,
) -> None:
    """Log a pipeline stage — minimal, key facts only."""
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
                lines.append(f"    思考: {_smart_trunc(thought, 300)}")
    if observation is not None:
        lines.append(f"    观察: {_smart_trunc(observation, 300)}")
    if critic:
        lines.append(f"    评审: {_smart_trunc(critic, 200)}")

    logger.info("\n".join(lines))
