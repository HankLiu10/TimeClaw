"""ExecutorAgent: mini-ReAct tool orchestration layer.

Instead of planning the entire tool chain upfront, the Executor runs a
step-by-step Think→Act→Observe loop. Each step, the LLM sees the results
of all previous tools and decides what to do next — enabling dynamic
parameter passing (e.g., detect_seasonality → stl_decompose(period=N)).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.agents.prompts import executor_prompt
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.clawhub.tool_registry import get_tool_instance
from timeclaw.core import trace_logger as trace
from timeclaw.core.agent_base import AgentBase
from timeclaw.core.evidence import EvidenceEntry
from timeclaw.core.message import AgentRole
from timeclaw.core.tea_protocol import TaskEnvelope

MAX_REACT_STEPS = 10


class ExecutorAgent(AgentBase):
    role = AgentRole.EXECUTOR

    def __init__(
        self,
        *args: Any,
        text_backends: dict[str, Any] | None = None,
        experience_pool: Any | None = None,
        mcp_manager: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._text_backends = text_backends
        self._experience_pool = experience_pool
        self._mcp_manager = mcp_manager

    # ── Agent lifecycle ───────────────────────────────────────────────

    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        fingerprint = envelope.environment.fingerprint or {}
        modality = envelope.task.metadata.get("modality") or "timeseries"

        text_summary = ""
        if envelope.environment.text_data:
            texts = envelope.environment.text_data
            parts = [f"{len(texts)} text documents:\n"]
            for t in texts[:10]:
                src = t.get("source", "?")
                ts = t.get("timestamp", "?")
                content = str(t.get("content", t.get("text", "")))[:5000]
                parts.append(f"### [{src} @ {ts}]\n{content}\n")
            text_summary = "\n".join(parts)

        image_summary = ""
        if envelope.environment.images:
            imgs = envelope.environment.images
            img_parts = [f"{len(imgs)} images:"]
            for img in imgs[:10]:
                if isinstance(img, dict):
                    path = img.get("path", img.get("url", "unknown"))
                    desc = img.get("description", "")
                    img_parts.append(f"- {path}" + (f": {desc}" if desc else ""))
                else:
                    img_parts.append(f"- {str(img)[:200]}")
            image_summary = "\n".join(img_parts)

        return {
            "task_type": envelope.task.task_type,
            "task_description": envelope.task.description or "",
            "fingerprint": fingerprint,
            "data_shape": envelope.environment.data_shape,
            "horizon": envelope.task.horizon,
            "constraints": envelope.task.constraints,
            "modality": modality,
            "modalities": envelope.environment.modalities or ["timeseries"],
            "has_text": envelope.environment.has_text,
            "has_images": envelope.environment.has_images,
            "text_summary": text_summary,
            "image_summary": image_summary,
            "suggested_strategy": envelope.task.metadata.get("suggested_strategy"),
            "soul_prompt": envelope.task.metadata.get("soul_prompt", ""),
            "principles_prompt": envelope.task.metadata.get("principles_prompt", ""),
            "memory_l2": envelope.task.metadata.get("memory_l2", ""),
            "memory_l1": envelope.task.metadata.get("memory_l1", ""),
            "tool_notes": envelope.task.metadata.get("tool_notes", {}),
            "playbook_prompt": envelope.task.metadata.get("playbook_prompt", ""),
            "_warmup_filter": envelope.task.metadata.get("_warmup_filter"),
            "domain": envelope.task.metadata.get("domain", ""),
            "reanalysis_context": envelope.task.metadata.get("reanalysis_context"),
        }

    async def plan(self, perception: dict[str, Any]) -> dict[str, Any]:
        """In mini-ReAct, plan just prepares context — actual execution is in act()."""
        modality = perception["modality"]
        task_type = perception["task_type"]

        # Expose ALL registered tools — no modality/task_type filtering.
        # The agent reads tool descriptions and selects autonomously.
        if self._mcp_manager:
            candidates = list(self._mcp_manager.tool_names())
        else:
            candidates = executor_prompt.get_all_tool_names()

        warmup_filter = perception.get("_warmup_filter")
        if warmup_filter:
            candidates = warmup_filter(candidates)

        tool_notes = perception.get("tool_notes", {})
        mcp_tool_map = None
        if self._mcp_manager:
            mcp_tool_map = {t.name: t for t in self._mcp_manager.list_tools()}
        tool_guide = executor_prompt.build_tool_guide_text(
            candidates, experience_insights=tool_notes, mcp_tools=mcp_tool_map,
        )

        strategy = perception.get("suggested_strategy") or ""
        strategy_section = ""
        if strategy:
            strategy_section = (
                f"## Recommended Strategy\n"
                f"Recommendation: **{strategy}**\n\n"
            )

        # SOUL (fixed) + PRINCIPLES (learned) + MEMORY → system prompt
        system_memory = ""
        soul = perception.get("soul_prompt", "")
        principles = perception.get("principles_prompt", "")
        mem = perception.get("memory_l2", "")
        mem_parts = [p for p in [soul, principles, mem] if p]
        if mem_parts:
            system_memory = "\n".join(mem_parts)

        # Daily notes → first-turn user message
        daily_section = perception.get("memory_l1", "")

        # PLAYBOOK → first-turn user message (scene-specific SOP)
        skills_section = perception.get("playbook_prompt", "")

        logger.debug(
            f"[Executor.plan] Memory injection: SOUL={len(soul)}, "
            f"PRINCIPLES={len(principles)}, MEMORY={len(mem)}, "
            f"Daily={len(daily_section)}, PLAYBOOK={len(skills_section)}, "
            f"ToolNotes={len(perception.get('tool_notes', {}))} tools"
        )

        modality_hint = ""
        available_data = []
        if perception.get("has_text") or modality == "text":
            available_data.append("text")
        if perception.get("has_images") or modality == "image":
            available_data.append("image")
        modalities_detected = perception.get("modalities", [])
        if modalities_detected and len(modalities_detected) > 1:
            modality_hint = f"\nDetected modalities: {', '.join(modalities_detected)}. You can use tools for any modality."

        data_context = ""
        if perception.get("text_summary"):
            data_context += f"\n## Text Data\n{perception['text_summary']}\n"
        if perception.get("image_summary"):
            data_context += f"\n## Image Data\n{perception['image_summary']}\n"

        reanalysis_section = ""
        reanalysis_ctx = perception.get("reanalysis_context")
        if reanalysis_ctx:
            prev_logs = reanalysis_ctx.get("previous_execution_logs", [])
            conflict_detail = reanalysis_ctx.get("conflict_details", "")
            conflict_reason = reanalysis_ctx.get("conflict_reasoning", "")
            suggested = reanalysis_ctx.get("suggested_tools", [])
            parts = ["## Previous Analysis (failed cross-modal check)\n"]
            if conflict_detail:
                parts.append(f"Conflict: {conflict_detail}\n")
            if conflict_reason:
                parts.append(f"Reasoning: {conflict_reason}\n")
            if suggested:
                parts.append(f"Suggested alternative tools: {', '.join(suggested)}\n")
            for plog in prev_logs:
                tool = plog.get("tool", "?")
                ok = plog.get("success", False)
                summ = plog.get("result_summary", "")[:10000]
                parts.append(f"- {tool}: {'OK' if ok else 'FAIL'} → {summ}")
            parts.append(
                "\nPlease try a DIFFERENT approach from the previous round.\n"
            )
            reanalysis_section = "\n".join(parts) + "\n"

        return {
            "modality": modality,
            "task_type": task_type,
            "task_description": perception.get("task_description", ""),
            "candidates": candidates,
            "tool_guide": tool_guide,
            "strategy_section": strategy_section,
            "skills_section": skills_section,
            "system_memory": system_memory,
            "l1_section": daily_section,
            "modality_hint": modality_hint,
            "data_context": data_context,
            "fingerprint": perception["fingerprint"],
            "horizon": perception.get("horizon"),
            "domain": perception.get("domain", ""),
            "reanalysis_section": reanalysis_section,
        }

    async def act(self, plan: dict[str, Any], envelope: TaskEnvelope) -> dict[str, Any]:
        """Mini-ReAct loop: Think → Act → Observe, repeat until done or max steps."""
        data = envelope.environment.data
        text_data = envelope.environment.text_data or None
        images = envelope.environment.images or None
        candidates = plan["candidates"]
        horizon = plan.get("horizon")

        execution_log: list[dict[str, Any]] = []
        history_lines: list[str] = []
        current_data = data
        final_result: Any = None
        norm_inverse: dict[str, Any] | None = None
        agent_answer_type: str = ""
        agent_confidence: float | None = None
        agent_reasoning: str = ""

        system_memory = plan.get("system_memory", "")
        system_prompt = executor_prompt.REACT_SYSTEM_PROMPT.format(
            max_steps=MAX_REACT_STEPS,
        ) + plan["modality_hint"]
        if system_memory:
            system_prompt += f"\n\n{system_memory}"

        first_user = executor_prompt.REACT_FIRST_TURN.format(
            task_type=plan["task_type"],
            task_description=plan.get("task_description", ""),
            fingerprint=str(plan["fingerprint"])[:5000],
            reanalysis_section=plan.get("reanalysis_section", ""),
            strategy_section=plan["strategy_section"],
            skills_section=plan.get("skills_section", ""),
            l1_section=plan.get("l1_section", ""),
            tool_guide=plan["tool_guide"],
        ) + plan.get("data_context", "")

        for step in range(1, MAX_REACT_STEPS + 1):
            if step == 1:
                user_msg = first_user
            else:
                user_msg = executor_prompt.REACT_NEXT_TURN.format(
                    step=step,
                    max_steps=MAX_REACT_STEPS,
                    history="\n".join(history_lines),
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]

            response = await self.llm.complete(messages, response_format="json")
            decision = response.parsed or {"done": True, "thought": "LLM parse failed"}

            thought = decision.get("thought", "")
            is_done = decision.get("done", False)

            if is_done:
                agent_answer = decision.get("answer")
                agent_answer_type = decision.get("answer_type", "")
                agent_confidence = decision.get("confidence")
                agent_reasoning = decision.get("reasoning", thought)
                if agent_answer is not None:
                    final_result = agent_answer
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought, "done": True,
                    "answer_type": agent_answer_type,
                    "confidence": agent_confidence,
                    "has_answer": agent_answer is not None,
                })
                break

            tool_name = decision.get("tool", "")
            tool_args = decision.get("args", {})

            if tool_name not in candidates:
                history_lines.append(
                    f"[Step {step}] ✗ Tool '{tool_name}' not in candidate list. "
                    f"Available: {candidates}"
                )
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought, "tool": tool_name, "error": "not in candidates",
                })
                continue

            if horizon and tool_name.endswith("_forecast"):
                tool_args.setdefault("horizon", horizon)

            if plan.get("domain"):
                tool_args.setdefault("domain", plan["domain"])

            result = await self._execute_tool(
                tool_name, tool_args, current_data, text_data, images,
            )

            success = result is not None and result.success
            status_icon = "✓" if success else "✗"
            result_summary = ""

            if success:
                result_summary = self._summarize_result(result.data)
                final_result = result.data

                if tool_name == "normalize" and result.metadata.get("inverse"):
                    norm_inverse = result.metadata["inverse"]
                if self._is_passthrough_transform(tool_name, result.data):
                    current_data = result.data
            else:
                error_msg = result.error if result else "tool not found"
                result_summary = f"ERROR: {error_msg}"

            history_lines.append(
                f"[Step {step}] {status_icon} {tool_name}({self._fmt_args(tool_args)})\n"
                f"  → {result_summary}"
            )

            log_entry = {
                "tool": tool_name, "args": tool_args, "success": success,
                "result_summary": result_summary[:50000],
                "output": result.data if success else None,
            }
            execution_log.append(log_entry)

            self.evidence_log.append(EvidenceEntry(
                step=self.evidence_log.num_steps,
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                observation=result.data if success else None,
                critic_feedback="ok" if success else f"error: {result_summary}",
            ))

            trace.agent_step("executor", f"react_step_{step}", {
                "thought": thought, "tool": tool_name, "args": tool_args,
                "success": success, "result_preview": result_summary[:5000],
            })

        if norm_inverse and final_result is not None:
            final_result = self._denormalize_prediction(final_result, norm_inverse)
            logger.info(f"[Executor] Auto-denormalized with {norm_inverse['method']}")

        act_result = {
            "prediction": final_result,
            "answer": final_result,
            "answer_type": agent_answer_type,
            "confidence": agent_confidence,
            "reasoning": agent_reasoning,
            "execution_log": execution_log,
            "method_used": f"mini-react ({len(execution_log)} steps)",
            "modality": plan["modality"],
        }
        trace.agent_step("executor", "act_complete", {
            "tools_executed": len(execution_log),
            "tools_passed": sum(1 for e in execution_log if e.get("success")),
            "denormalized": norm_inverse is not None,
            "prediction_type": type(final_result).__name__,
            "prediction_preview": str(final_result)[:5000] if final_result else None,
            "answer_type": act_result.get("answer_type"),
            "confidence": act_result.get("confidence"),
        })
        return act_result

    @staticmethod
    def _is_passthrough_transform(tool_name: str, result_data: Any) -> bool:
        """Check if a tool's output should replace current_data for downstream tools.

        Only true transforms (normalize, handle_missing, etc.) that output an
        array of the same shape should replace current_data. Decomposition tools
        output dicts and must NOT replace the working array.
        """
        if isinstance(result_data, dict):
            return False
        if hasattr(result_data, "shape") or isinstance(result_data, (list, np.ndarray)):
            return True
        return False

    async def reflect(self, action_result: dict[str, Any]) -> dict[str, Any]:
        log = action_result.get("execution_log", [])
        return {
            "has_prediction": action_result.get("prediction") is not None,
            "tools_succeeded": sum(1 for e in log if e.get("success", False)),
            "tools_total": len(log),
            "modality": action_result.get("modality"),
        }

    # ── Tool execution ────────────────────────────────────────────────

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        data: Any,
        text_data: Any,
        images: Any,
    ) -> ToolResult | None:
        """Execute a single tool — routes through MCP when available, falls back to in-process."""
        # ── MCP path ──
        if self._mcp_manager and self._mcp_manager.get_tool(tool_name):
            mcp_args: dict[str, Any] = dict(tool_args)
            if data is not None and "ts_data" not in mcp_args:
                mcp_args["ts_data"] = data.tolist() if hasattr(data, "tolist") else list(data) if data is not None else None
            if text_data is not None and "text" not in mcp_args:
                if isinstance(text_data, list):
                    import json as _json
                    mcp_args["text"] = _json.dumps(text_data, ensure_ascii=False, default=str)
                else:
                    mcp_args["text"] = str(text_data)
            if images is not None and "images" not in mcp_args:
                mcp_args["images"] = images

            r = await self._mcp_manager.call_tool(tool_name, mcp_args)
            status = "PASS" if r["success"] else f"FAIL | {r['error']}"
            logger.info(f"[Executor/MCP] {tool_name}: {status}")
            return ToolResult(
                success=r["success"],
                data=r["data"],
                error=r.get("error"),
            )

        # ── In-process fallback ──
        try:
            tool = get_tool_instance(tool_name)
        except KeyError:
            logger.warning(f"[Executor] Tool '{tool_name}' not found")
            return None

        safe_args = {
            k: v for k, v in tool_args.items()
            if k not in ("text", "images", "llm", "ts_data")
        }
        llm_for_tool = (
            self._text_backends
            if self._text_backends and tool.category.value == "text"
            else self.llm
        )

        try:
            result: ToolResult = await tool.execute(
                data, text=text_data, images=images, llm=llm_for_tool, **safe_args,
            )
            status = "PASS" if result.success else f"FAIL | {result.error}"
            logger.info(f"[Executor] {tool_name}: {status}")
            trace.tool_exec("executor", tool_name, tool_args, result)
            return result
        except Exception as e:
            logger.error(f"[Executor] Tool '{tool_name}' crashed: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _summarize_result(data: Any) -> str:
        """Produce a summary of tool output for the LLM's history.

        Preserves full data when feasible (GPT-5.1 has 400K token context).
        Arrays up to 2000 elements are kept in full; larger ones get
        head-200 + tail-200.  Dict values up to 10K chars each.
        """
        if data is None:
            return "None"
        if isinstance(data, np.ndarray):
            flat = data.ravel()
            if flat.size <= 2000:
                vals = ", ".join(f"{v:.4f}" for v in flat)
                return f"ndarray shape={data.shape} values=[{vals}]"
            head = ", ".join(f"{v:.4f}" for v in flat[:200])
            tail = ", ".join(f"{v:.4f}" for v in flat[-200:])
            return (
                f"ndarray shape={data.shape} "
                f"range=[{float(data.min()):.4f}, {float(data.max()):.4f}] "
                f"values=[{head}, ..., {tail}]"
            )
        if isinstance(data, dict):
            parts = []
            for k, v in list(data.items())[:20]:
                v_str = str(v)
                if len(v_str) > 10000:
                    v_str = v_str[:10000] + "..."
                parts.append(f"{k}={v_str}")
            return ", ".join(parts)
        s = str(data)
        return s[:50000] if len(s) > 50000 else s

    @staticmethod
    def _fmt_args(args: dict[str, Any]) -> str:
        """Format tool args for the history display."""
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 500:
                v_str = v_str[:500] + "..."
            parts.append(f"{k}={v_str}")
        return ", ".join(parts)

    @staticmethod
    def _denormalize_prediction(prediction: Any, inv: dict[str, Any]) -> Any:
        """Reverse normalization on forecast output."""
        if isinstance(prediction, dict) and "forecast" in prediction:
            arr = np.asarray(prediction["forecast"], dtype=float)
            if inv["method"] == "zscore":
                arr = arr * inv["std"] + inv["mean"]
            elif inv["method"] == "minmax":
                arr = arr * inv["span"] + inv["min"]
            prediction["forecast"] = arr.tolist()
            prediction["denormalized"] = True
            return prediction
        if isinstance(prediction, (list, np.ndarray)):
            arr = np.asarray(prediction, dtype=float)
            if inv["method"] == "zscore":
                return (arr * inv["std"] + inv["mean"]).tolist()
            if inv["method"] == "minmax":
                return (arr * inv["span"] + inv["min"]).tolist()
        return prediction
