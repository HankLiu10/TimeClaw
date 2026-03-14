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

MAX_REACT_STEPS = 6


class ExecutorAgent(AgentBase):
    role = AgentRole.EXECUTOR

    def __init__(
        self,
        *args: Any,
        text_backends: dict[str, Any] | None = None,
        experience_pool: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._text_backends = text_backends
        self._experience_pool = experience_pool
        self._tool_insights: dict[str, str] = {}
        if experience_pool is not None:
            try:
                self._tool_insights = experience_pool.load_tool_insights()
            except Exception:
                pass

    # ── Agent lifecycle ───────────────────────────────────────────────

    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        fingerprint = envelope.environment.fingerprint or {}
        modality = envelope.task.metadata.get("modality") or "timeseries"

        text_summary = ""
        if envelope.environment.text_data:
            texts = envelope.environment.text_data
            text_summary = f"{len(texts)} text documents: "
            previews = []
            for t in texts[:3]:
                src = t.get("source", "?")
                ts = t.get("timestamp", "?")
                content = str(t.get("content", ""))[:80]
                previews.append(f"[{src}@{ts}] {content}")
            text_summary += "; ".join(previews)

        image_summary = ""
        if envelope.environment.images:
            image_summary = f"{len(envelope.environment.images)} images available"

        return {
            "task_type": envelope.task.task_type,
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
            "similar_experiences": envelope.task.metadata.get("similar_experiences", []),
        }

    async def plan(self, perception: dict[str, Any]) -> dict[str, Any]:
        """In mini-ReAct, plan just prepares context — actual execution is in act()."""
        modality = perception["modality"]
        task_type = perception["task_type"]
        candidates = executor_prompt.get_candidate_tools(modality, task_type)
        tool_guide = executor_prompt.build_tool_guide_text(
            candidates, experience_insights=self._tool_insights,
        )

        strategy = perception.get("suggested_strategy") or ""
        strategy_section = ""
        if strategy:
            strategy_section = (
                f"## Analyzer's Suggested Strategy\n"
                f"Recommendation: **{strategy}**\n\n"
            )

        experiences = perception.get("similar_experiences") or []
        experience_section = ""
        if experiences:
            experience_section = "## Similar Past Experiences\n"
            for exp in experiences[:3]:
                passed = "PASSED" if exp.get("quality_gate_passed") else "FAILED"
                chain = exp.get("tool_chain", [])
                experience_section += (
                    f"- tools={chain}, quality_gate={passed}\n"
                )
            experience_section += "\n"

        if modality in ("text", "image"):
            modality_hint = (
                f"\nYou are the '{modality}' modality branch. "
                f"Extract insights from {modality} data using {modality} tools only. "
                f"The fingerprint describes the time-series — ignore it for tool selection."
            )
        else:
            modality_hint = f"\nYou are specialized for the '{modality}' modality."

        data_context = ""
        if modality == "text" and perception.get("text_summary"):
            data_context = (
                f"\n## Your Data\n{perception['text_summary']}\n"
                f"You MUST use text tools to extract insights. Do NOT declare done without running tools.\n"
            )
        elif modality == "image" and perception.get("image_summary"):
            data_context = f"\n## Your Data\n{perception['image_summary']}\n"

        return {
            "modality": modality,
            "task_type": task_type,
            "candidates": candidates,
            "tool_guide": tool_guide,
            "strategy_section": strategy_section,
            "experience_section": experience_section,
            "modality_hint": modality_hint,
            "data_context": data_context,
            "fingerprint": perception["fingerprint"],
            "horizon": perception.get("horizon"),
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

        system_prompt = executor_prompt.REACT_SYSTEM_PROMPT.format(
            max_steps=MAX_REACT_STEPS,
        ) + plan["modality_hint"]

        first_user = executor_prompt.REACT_FIRST_TURN.format(
            task_type=plan["task_type"],
            fingerprint=str(plan["fingerprint"])[:2000],
            strategy_section=plan["strategy_section"],
            experience_section=plan["experience_section"],
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
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought, "done": True,
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
                if result.success and getattr(
                    get_tool_instance(tool_name), "io_type", None
                ) and get_tool_instance(tool_name).io_type.value == "X->X":
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
                "result_summary": result_summary[:300],
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
                "success": success, "result_preview": result_summary[:200],
            })

        if norm_inverse and final_result is not None:
            final_result = self._denormalize_prediction(final_result, norm_inverse)
            logger.info(f"[Executor] Auto-denormalized with {norm_inverse['method']}")

        act_result = {
            "prediction": final_result,
            "execution_log": execution_log,
            "method_used": f"mini-react ({len(execution_log)} steps)",
            "modality": plan["modality"],
        }
        trace.agent_step("executor", "act_complete", {
            "tools_executed": len(execution_log),
            "tools_passed": sum(1 for e in execution_log if e.get("success")),
            "denormalized": norm_inverse is not None,
            "prediction_type": type(final_result).__name__,
            "prediction_preview": str(final_result)[:500] if final_result else None,
        })
        return act_result

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
        """Execute a single tool with proper multimodal data injection."""
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
        """Produce a concise summary of tool output for the LLM's history."""
        if data is None:
            return "None"
        if isinstance(data, dict):
            parts = []
            for k, v in list(data.items())[:8]:
                v_str = str(v)
                if len(v_str) > 120:
                    v_str = v_str[:120] + "..."
                parts.append(f"{k}={v_str}")
            return ", ".join(parts)
        if isinstance(data, np.ndarray):
            return f"ndarray shape={data.shape} range=[{data.min():.4f}, {data.max():.4f}]"
        s = str(data)
        return s[:300] if len(s) > 300 else s

    @staticmethod
    def _fmt_args(args: dict[str, Any]) -> str:
        """Format tool args for the history display."""
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 50:
                v_str = v_str[:50] + "..."
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
