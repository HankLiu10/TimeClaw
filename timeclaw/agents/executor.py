"""ExecutorAgent: mini-ReAct tool orchestration layer.

Instead of planning the entire tool chain upfront, the Executor runs a
step-by-step Think→Act→Observe loop. Each step, the LLM sees the results
of all previous tools and decides what to do next — enabling dynamic
parameter passing (e.g., detect_seasonality → stl_decompose(period=N)).
"""

from __future__ import annotations

import asyncio
import json
import re
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
MAX_SUB_REACT_STEPS = 5
MAX_CONCURRENT_SUBAGENTS = 4
SUBAGENT_TIMEOUT_SECONDS = 120


class ExecutorAgent(AgentBase):
    role = AgentRole.EXECUTOR

    def __init__(
        self,
        *args: Any,
        text_backends: dict[str, Any] | None = None,
        mcp_manager: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._text_backends = text_backends
        self._mcp_manager = mcp_manager

    # ── Agent lifecycle ───────────────────────────────────────────────

    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        fingerprint = envelope.environment.fingerprint or {}
        modality = envelope.task.metadata.get("modality") or "timeseries"
        profiling_results = envelope.task.metadata.get("profiling_results", {})

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
            "profiling_results": profiling_results,
            "soul_prompt": envelope.task.metadata.get("soul_prompt", ""),
            "skill_prompt": envelope.task.metadata.get("skill_prompt", ""),
            "memory_l2": envelope.task.metadata.get("memory_l2", ""),
            "tool_notes": envelope.task.metadata.get("tool_notes", {}),
            "domain": envelope.task.metadata.get("domain", ""),
            "reanalysis_context": envelope.task.metadata.get("reanalysis_context"),
            "_mode": envelope.task.metadata.get("_mode", "test"),
            "_defer_learning_record": envelope.task.metadata.get("_defer_learning_record", False),
            "_ground_truth": envelope.task.metadata.get("_ground_truth"),
        }

    async def plan(self, perception: dict[str, Any]) -> dict[str, Any]:
        """In mini-ReAct, plan just prepares context — actual execution is in act()."""
        modality = perception["modality"]
        task_type = perception["task_type"]
        modalities_detected = perception.get("modalities", []) or ["timeseries"]

        # Expose only tools whose required modalities are available.
        if self._mcp_manager:
            from timeclaw.clawhub.tool_registry import get_tool_instance

            candidates = []
            available = set(modalities_detected)
            for name in self._mcp_manager.tool_names():
                try:
                    tool = get_tool_instance(name)
                except KeyError:
                    candidates.append(name)
                    continue
                if tool.required_modalities().issubset(available):
                    candidates.append(name)
        else:
            candidates = executor_prompt.get_all_tool_names(modalities_detected)

        run_mode = perception.get("_mode", "test")
        eval_tools = {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"}
        if run_mode == "train":
            for eval_tool in eval_tools:
                if eval_tool not in candidates:
                    candidates.append(eval_tool)
        else:
            candidates = [c for c in candidates if c not in eval_tools]

        tool_names_text = executor_prompt.build_tool_name_text(candidates)
        tool_notes_section = self._build_tool_notes_section(
            perception.get("tool_notes", {}),
            candidates,
            task_type,
        )

        # SOUL (fixed) + MEMORY (per-scope + cross_domain) → system prompt
        system_memory = ""
        soul = perception.get("soul_prompt", "")
        mem = perception.get("memory_l2", "")
        mem_parts = [p for p in [soul, mem] if p]
        if mem_parts:
            system_memory = "\n".join(mem_parts)

        # SKILL → first-turn user message (scene-specific SOP)
        skills_section = perception.get("skill_prompt", "")

        logger.debug(
            f"[Executor.plan] Memory injection: SOUL={len(soul)}, "
            f"MEMORY={len(mem)}, "
            f"SKILL={len(skills_section)}, "
            f"ToolNotes={len(perception.get('tool_notes', {}))} tools"
        )

        modality_hint = ""
        available_data = []
        if perception.get("has_text") or modality == "text":
            available_data.append("text")
        if perception.get("has_images") or modality == "image":
            available_data.append("image")
        if modalities_detected and len(modalities_detected) > 1:
            modality_hint = (
                f"\nDetected modalities: {', '.join(modalities_detected)}. "
                "Only tools compatible with these modalities are available."
            )

        profiling_section = self._build_profiling_section(
            perception.get("profiling_results", {}),
        )

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
                obs = plog.get("observation")
                if obs is None:
                    obs = plog.get("output")
                obs_text = self._format_observation_for_prompt(obs)[:10000] if obs is not None else ""
                parts.append(f"- {tool}: {'OK' if ok else 'FAIL'} → {obs_text}")
            parts.append(
                "\nPlease try a DIFFERENT approach from the previous round.\n"
            )
            reanalysis_section = "\n".join(parts) + "\n"

        return {
            "modality": modality,
            "task_type": task_type,
            "task_description": perception.get("task_description", ""),
            "candidates": candidates,
            "tool_names": tool_names_text,
            "tool_notes": perception.get("tool_notes", {}),
            "tool_notes_section": tool_notes_section,
            "skills_section": skills_section,
            "system_memory": system_memory,
            "modality_hint": modality_hint,
            "profiling_section": profiling_section,
            "fingerprint": perception["fingerprint"],
            "horizon": perception.get("horizon"),
            "domain": perception.get("domain", ""),
            "reanalysis_section": reanalysis_section,
            "_mode": run_mode,
            "_defer_learning_record": bool(perception.get("_defer_learning_record", False)),
            "_ground_truth": perception.get("_ground_truth"),
        }

    async def act(self, plan: dict[str, Any], envelope: TaskEnvelope) -> dict[str, Any]:
        """Mini-ReAct loop: Think → Act → Observe, repeat until done or max steps."""
        data = envelope.environment.data
        text_data = envelope.environment.text_data or None
        images = envelope.environment.images or None
        candidates = plan["candidates"]
        horizon = plan.get("horizon")

        execution_log: list[dict[str, Any]] = []
        current_data = data
        final_result: Any = None
        norm_inverse: dict[str, Any] | None = None
        agent_answer_type: str = ""
        agent_confidence: float | None = None
        agent_reasoning: str = ""

        system_memory = plan.get("system_memory", "")
        system_prompt = executor_prompt.REACT_SYSTEM_PROMPT.format(
            max_steps=MAX_REACT_STEPS,
            max_subagents=MAX_CONCURRENT_SUBAGENTS,
        ) + plan["modality_hint"]
        if system_memory:
            system_prompt += f"\n\n{system_memory}"

        is_train = plan.get("_mode") == "train"
        if is_train:
            if plan.get("_defer_learning_record"):
                system_prompt += executor_prompt.DEFERRED_TRAIN_MODE_HINT
            else:
                system_prompt += executor_prompt.TRAIN_MODE_HINT

        first_user = executor_prompt.REACT_FIRST_TURN.format(
            task_type=plan["task_type"],
            task_description=plan.get("task_description", ""),
            fingerprint=str(plan["fingerprint"])[:5000],
            profiling_section=plan.get("profiling_section", ""),
            reanalysis_section=plan.get("reanalysis_section", ""),
            tool_notes_section=plan.get("tool_notes_section", ""),
            skills_section=plan.get("skills_section", ""),
            tool_names=plan["tool_names"],
        )

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]
        tool_notes_section = plan.get("tool_notes_section", "")
        if tool_notes_section:
            conversation.append({
                "role": "user",
                "content": tool_notes_section,
            })
        _gt_obs_idx: int | None = None

        for step in range(1, MAX_REACT_STEPS + 1):
            if step > 1:
                follow_up = executor_prompt.REACT_NEXT_TURN.format(
                    step=step,
                    max_steps=MAX_REACT_STEPS,
                )
                conversation.append({"role": "user", "content": follow_up})

            response = await self.llm.complete(conversation, response_format="json")
            decision = self._coerce_llm_decision(
                response.parsed, response.text, fallback_thought="LLM parse failed",
            )

            # Record assistant reply in conversation for multi-turn context
            conversation.append({"role": "assistant", "content": response.text or ""})

            # GT sanitisation: after the LLM has seen and analysed the
            # evaluate_against_gt result (this step), strip the ground_truth
            # values from the tool observation so subsequent steps can't see them.
            if _gt_obs_idx is not None:
                self._sanitise_gt_observation(conversation, _gt_obs_idx)
                _gt_obs_idx = None

            thought = decision.get("thought", "")
            is_done = decision.get("done", False)

            if not self._has_structured_thought(thought) and step < MAX_REACT_STEPS:
                trace.agent_step("executor", f"react_step_{step}", {
                    "self_correction": True,
                    "reason": "unstructured_or_unparsed_response",
                    "parsed_type": type(response.parsed).__name__,
                    "parsed_is_dict": isinstance(response.parsed, dict),
                    "thought": thought,
                    "response_preview": (response.text or "")[:4000],
                })
                conversation.append({
                    "role": "user",
                    "content": (
                        "Self-correction: your `thought` must explicitly include "
                        "`Reason:` and `Action:`. Please continue "
                        "with the required structure."
                    ),
                })
                continue

            if is_done:
                agent_answer = decision.get("answer")
                agent_answer_type = decision.get("answer_type", "")
                agent_confidence = decision.get("confidence")
                agent_reasoning = decision.get("reasoning", thought)
                is_train = plan.get("_mode") == "train"
                deferred_learning_record = bool(plan.get("_defer_learning_record", False))
                requested_type = plan["task_type"]

                if (
                    is_train
                    and not self._has_training_evaluation(execution_log)
                    and step < MAX_REACT_STEPS
                ):
                    logger.info(
                        "[Executor] Self-correction: training summary without GT evaluation, retrying"
                    )
                    correction_msg = (
                        "Self-correction: in training mode you must evaluate at least one "
                        "valid candidate with `evaluate_batch_against_gt`, "
                        "`batch_evaluate_against_gt`, or `evaluate_against_gt` before "
                        "returning answer_type='learning_summary'. Continue working."
                    )
                    conversation.append({"role": "user", "content": correction_msg})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "missing_training_evaluation": True,
                    })
                    continue

                if (
                    is_train
                    and self._count_training_candidates(
                        execution_log,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                    ) < 2
                    and step < MAX_REACT_STEPS
                ):
                    logger.info(
                        "[Executor] Self-correction: insufficient candidate exploration, retrying"
                    )
                    correction_msg = (
                        "Self-correction: in training mode you must explore at least 2 valid "
                        "candidate paths before finishing. Continue exploring and compare more "
                        "task-valid candidates."
                    )
                    conversation.append({"role": "user", "content": correction_msg})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "insufficient_candidates": True,
                    })
                    continue

                # Self-correction: if the declared answer_type doesn't match
                # the requested task_type and we have steps remaining, ask the
                # agent to fix it instead of accepting a wrong-typed answer.
                # In training mode, "learning_summary" is always accepted.
                type_mismatch = (
                    requested_type
                    and agent_answer_type
                    and agent_answer_type != requested_type
                    and not (
                        is_train
                        and not deferred_learning_record
                        and agent_answer_type == "learning_summary"
                    )
                    and step < MAX_REACT_STEPS
                )
                if type_mismatch:
                    logger.info(
                        f"[Executor] Self-correction: task requires '{requested_type}' "
                        f"but agent returned '{agent_answer_type}', retrying"
                    )
                    correction_msg = (
                        f"Self-correction: the task requires "
                        f"answer_type='{requested_type}' but you declared "
                        f"answer_type='{agent_answer_type}'. Please continue "
                        f"processing to produce a '{requested_type}' result."
                    )
                    conversation.append({"role": "user", "content": correction_msg})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought, "done": True,
                        "self_correction": True,
                        "mismatch": f"{agent_answer_type} != {requested_type}",
                    })
                    continue

                learning_summary_allowed = (
                    is_train
                    and not deferred_learning_record
                    and agent_answer_type == "learning_summary"
                )
                invalid_answer = (
                    agent_answer is not None
                    and not learning_summary_allowed
                    and not self._is_valid_final_answer(
                        requested_type, agent_answer, horizon=plan.get("horizon"),
                    )
                    and step < MAX_REACT_STEPS
                )
                if invalid_answer:
                    _, invalid_reason = self._validate_final_answer(
                        requested_type, agent_answer, horizon=plan.get("horizon"),
                    )
                    try:
                        answer_preview = json.dumps(agent_answer, ensure_ascii=False, default=str)
                    except TypeError:
                        answer_preview = str(agent_answer)
                    logger.info(
                        f"[Executor] Self-correction: invalid answer payload for '{requested_type}', retrying"
                    )
                    logger.warning(
                        "[Executor] Invalid final answer detail: "
                        f"answer_type={agent_answer_type!r}, "
                        f"python_type={type(agent_answer).__name__}, "
                        f"reason={invalid_reason}, "
                        f"answer_preview={answer_preview[:800]}"
                    )
                    correction_msg = (
                        f"Self-correction: the current answer is not a valid '{requested_type}' payload. "
                        f"For forecast, return a numeric array. For indicator, return a numeric JSON object. "
                        f"For classification/correlation, return a label or JSON containing the label. "
                        f"Please continue and produce a valid final answer."
                    )
                    conversation.append({"role": "user", "content": correction_msg})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought, "done": True,
                        "self_correction": True,
                        "invalid_payload": True,
                        "requested_type": requested_type,
                        "invalid_reason": invalid_reason,
                        "answer_type": agent_answer_type,
                        "answer_python_type": type(agent_answer).__name__,
                        "answer_preview": answer_preview[:2000],
                    })
                    continue

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

            # ── Sub-agent spawn (special built-in tool) ──
            if tool_name == "spawn_subagent":
                sub_results = await self._spawn_subagents(
                    tool_args, plan, envelope,
                )
                observation_msg = sub_results["observation"]
                conversation.append({"role": "user", "content": observation_msg})
                execution_log.append({
                    "tool": "spawn_subagent",
                    "args": {"n_tasks": sub_results["n_tasks"]},
                    "success": sub_results["n_success"] > 0,
                    "output": sub_results["results"],
                    "observation": sub_results["results"],
                    "thought": thought,
                })
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought, "tool": "spawn_subagent",
                    "n_tasks": sub_results["n_tasks"],
                    "n_success": sub_results["n_success"],
                })
                continue

            if tool_name not in candidates:
                err_msg = (
                    f"Tool '{tool_name}' not available. "
                    f"Choose from the tools listed in the first message."
                )
                conversation.append({"role": "user", "content": err_msg})
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought, "tool": tool_name, "error": "not in candidates",
                })
                continue

            if horizon and tool_name.endswith("_forecast"):
                tool_args["horizon"] = horizon

            if plan.get("domain"):
                tool_args.setdefault("domain", plan["domain"])

            if tool_name.startswith("llm_reasoning_"):
                tool_args["context"] = self._build_llm_reasoning_context(
                    plan=plan,
                    execution_log=execution_log,
                    existing_context=tool_args.get("context", ""),
                )

            if tool_name in {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"} and is_train:
                tool_args.setdefault("task_type", plan["task_type"])
                if tool_name == "evaluate_against_gt":
                    prediction = tool_args.get("prediction")
                    if not self._is_valid_final_answer(
                        plan["task_type"], prediction, horizon=plan.get("horizon"),
                    ):
                        conversation.append({
                            "role": "user",
                            "content": (
                                "Self-correction: `evaluate_against_gt` requires a valid "
                                "task-typed `prediction`. Produce a concrete candidate "
                                "prediction first, then evaluate it."
                            ),
                        })
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "tool": tool_name,
                            "self_correction": True,
                            "missing_evaluation_prediction": True,
                        })
                        continue
                if tool_name in {"evaluate_batch_against_gt", "batch_evaluate_against_gt"}:
                    predictions = tool_args.get("predictions")
                    if not isinstance(predictions, list) or not predictions:
                        autofill = self._collect_batch_eval_predictions(
                            execution_log,
                            task_type=plan["task_type"],
                            horizon=plan.get("horizon"),
                        )
                        if autofill:
                            tool_args["predictions"] = autofill
                    predictions = tool_args.get("predictions")
                    if not isinstance(predictions, list) or len(predictions) == 0:
                        conversation.append({
                            "role": "user",
                            "content": (
                                "Self-correction: `evaluate_batch_against_gt` requires a non-empty "
                                "list of valid candidate predictions. Explore valid candidates "
                                "first, then evaluate them together."
                            ),
                        })
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "tool": tool_name,
                            "self_correction": True,
                            "missing_batch_predictions": True,
                        })
                        continue
                tool_args["_ground_truth"] = plan.get("_ground_truth")

            result = await self._execute_tool(
                tool_name, tool_args, current_data, text_data, images,
            )

            success = result is not None and result.success
            status_icon = "✓" if success else "✗"

            if success:
                final_result = result.data

                if tool_name == "normalize" and result.metadata.get("inverse"):
                    norm_inverse = result.metadata["inverse"]
                if self._is_passthrough_transform(tool_name, result.data):
                    current_data = result.data
            else:
                error_msg = result.error if result else "tool not found"

            # Feed tool execution result back into conversation
            observation_payload = (
                result.data if success else {"error": error_msg}
            )
            observation_msg = (
                f"[Step {step}] {status_icon} {tool_name}({self._fmt_args(tool_args)})\n"
                f"→ {self._format_observation_for_prompt(observation_payload)}"
            )
            conversation.append({"role": "user", "content": observation_msg})

            if tool_name in {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"}:
                _gt_obs_idx = len(conversation) - 1

            log_entry = {
                "tool": tool_name, "args": tool_args, "success": success,
                "output": result.data if success else None,
                "observation": observation_payload,
                "thought": thought,
            }
            execution_log.append(log_entry)

            self.evidence_log.append(EvidenceEntry(
                step=self.evidence_log.num_steps,
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                observation=observation_payload,
                critic_feedback="ok" if success else f"error: {error_msg}",
            ))

            trace.agent_step("executor", f"react_step_{step}", {
                "thought": thought, "tool": tool_name, "args": tool_args,
                "success": success, "observation_preview": self._format_observation_for_prompt(observation_payload)[:5000],
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
    def _has_training_evaluation(execution_log: list[dict[str, Any]]) -> bool:
        """Return True iff training already produced at least one GT evaluation."""
        for entry in execution_log:
            if entry.get("tool") not in {
                "evaluate_against_gt",
                "evaluate_batch_against_gt",
                "batch_evaluate_against_gt",
            }:
                continue
            if not entry.get("success"):
                continue
            obs = entry.get("observation")
            if isinstance(obs, dict) and isinstance(obs.get("metrics"), dict) and obs.get("metrics"):
                return True
            if isinstance(obs, dict) and isinstance(obs.get("results"), list):
                for item in obs.get("results", []):
                    if isinstance(item, dict) and isinstance(item.get("metrics"), dict) and item.get("metrics"):
                        return True
        return False

    @classmethod
    def _count_training_candidates(
        cls,
        execution_log: list[dict[str, Any]],
        task_type: str,
        horizon: int | None = None,
    ) -> int:
        names: set[str] = set()
        for entry in execution_log:
            if entry.get("tool") != "spawn_subagent":
                continue
            output = entry.get("output") or []
            for idx, item in enumerate(output):
                if not isinstance(item, dict) or not item.get("success"):
                    continue
                result = item.get("result") or {}
                if not isinstance(result, dict):
                    continue
                answer = result.get("answer", result.get("prediction"))
                if not cls._is_valid_final_answer(task_type, answer, horizon=horizon):
                    continue
                sub_execution_log = result.get("execution_log") or []
                tool_chain = [
                    str(log_entry.get("tool") or "").strip()
                    for log_entry in sub_execution_log
                    if isinstance(log_entry, dict) and str(log_entry.get("tool") or "").strip()
                ]
                name = (
                    (tool_chain[0] if tool_chain else "")
                    or result.get("candidate_name")
                    or result.get("tools_hint")
                    or result.get("sub_task_description")
                    or f"candidate_{idx}"
                )
                names.add(str(name))
        return len(names)

    @staticmethod
    def _hinted_tools(tools_hint: str) -> list[str]:
        if not isinstance(tools_hint, str) or not tools_hint.strip():
            return []
        hinted: list[str] = []
        for token in re.split(r"[\s,;:/()\[\]\n]+", tools_hint):
            token = token.strip()
            if not token or "_" not in token:
                continue
            hinted.append(token)
        return hinted

    @staticmethod
    def _subagent_followed_tools_hint(
        execution_log: list[dict[str, Any]],
        tools_hint: str,
    ) -> bool:
        hinted = ExecutorAgent._hinted_tools(tools_hint)
        if not hinted:
            return True
        executed = {
            str(entry.get("tool") or "").strip()
            for entry in execution_log
            if isinstance(entry, dict) and str(entry.get("tool") or "").strip()
        }
        return any(tool in executed for tool in hinted)

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

    # ── Sub-agent spawning ─────────────────────────────────────────────

    async def _spawn_subagents(
        self,
        tool_args: dict[str, Any],
        plan: dict[str, Any],
        envelope: TaskEnvelope,
    ) -> dict[str, Any]:
        """Spawn up to MAX_CONCURRENT_SUBAGENTS parallel sub-agents.

        Each sub-agent is an independent mini-ReAct loop that inherits:
          - Full hierarchical memory (SOUL, MEMORY, SKILLS, tool_notes)
          - Same data and tool set as the main agent
          - Its own conversation context
        """
        tasks_spec = tool_args.get("tasks", [])
        if not tasks_spec:
            return {"n_tasks": 0, "n_success": 0, "results": [], "observation": "No tasks specified."}

        tasks_spec = tasks_spec[:MAX_CONCURRENT_SUBAGENTS]
        n_tasks = len(tasks_spec)
        logger.info(f"[Executor] Spawning {n_tasks} sub-agents")
        trace.agent_step("executor", "spawn_subagents", {
            "n_tasks": n_tasks,
            "tasks": tasks_spec,
        })

        coros = []
        for i, task in enumerate(tasks_spec):
            sub_desc = task.get("task", f"sub-task-{i}")
            tools_hint = task.get("tools_hint", "")
            coros.append(self._run_sub_executor(i, sub_desc, tools_hint, plan, envelope))

        done_results = await asyncio.gather(*coros, return_exceptions=True)

        results = []
        n_success = 0
        observation_parts = [f"Sub-agent results ({n_tasks} tasks):\n"]

        for i, r in enumerate(done_results):
            if isinstance(r, Exception):
                results.append({"sub_task": i, "success": False, "error": str(r)})
                observation_parts.append(f"  Sub-agent {i}: FAILED — {str(r)[:500]}")
            else:
                results.append({"sub_task": i, "success": True, "result": r})
                n_success += 1
                answer = r.get("answer", r.get("prediction"))
                reasoning = r.get("reasoning", "")
                observation_parts.append(
                    f"  Sub-agent {i}: SUCCESS — answer={str(answer)[:2000]}, "
                    f"reasoning={reasoning[:500]}"
                )

        return {
            "n_tasks": n_tasks,
            "n_success": n_success,
            "results": results,
            "observation": "\n".join(observation_parts),
        }

    async def _run_sub_executor(
        self,
        idx: int,
        task_description: str,
        tools_hint: str,
        plan: dict[str, Any],
        envelope: TaskEnvelope,
    ) -> dict[str, Any]:
        """Run a single sub-agent ReAct loop with full memory injection."""
        trace.agent_step("executor", "subagent_start", {
            "sub_agent_idx": idx,
            "sub_task_description": task_description,
            "tools_hint": tools_hint,
        })
        data = envelope.environment.data
        text_data = envelope.environment.text_data or None
        images = envelope.environment.images or None
        blocked_eval_tools = {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"}
        candidates = [c for c in plan["candidates"] if c not in blocked_eval_tools]

        system_memory = plan.get("system_memory", "")
        system_prompt = executor_prompt.SUBAGENT_SYSTEM_PROMPT.format(
            max_steps=MAX_SUB_REACT_STEPS,
        )
        if system_memory:
            system_prompt += f"\n\n{system_memory}"
        system_prompt += plan.get("modality_hint", "")

        memory_section = ""
        skills = plan.get("skills_section", "")
        if skills:
            memory_section = skills.strip() + "\n"

        hinted_tool_notes = self._build_tool_notes_section(
            plan.get("tool_notes", {}),
            [tool for tool in self._hinted_tools(tools_hint) if tool in candidates],
            plan["task_type"],
        )

        first_user = executor_prompt.SUBAGENT_FIRST_TURN.format(
            parent_task_type=plan["task_type"],
            parent_task_description=plan.get("task_description", ""),
            task_description=task_description,
            fingerprint=str(plan["fingerprint"])[:3000],
            profiling_section=plan.get("profiling_section", ""),
            tool_notes_section=hinted_tool_notes,
            memory_section=memory_section,
            tool_names=executor_prompt.build_tool_name_text(candidates),
        )

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]

        horizon = plan.get("horizon")
        current_data = data
        final_result: dict[str, Any] = {}
        execution_log: list[dict[str, Any]] = []

        for step in range(1, MAX_SUB_REACT_STEPS + 1):
            if step > 1:
                follow_up = executor_prompt.REACT_NEXT_TURN.format(
                    step=step, max_steps=MAX_SUB_REACT_STEPS,
                )
                conversation.append({"role": "user", "content": follow_up})

            response = await self.llm.complete(conversation, response_format="json")
            decision = self._coerce_llm_decision(
                response.parsed, response.text, fallback_thought="parse failed",
            )
            conversation.append({"role": "assistant", "content": response.text or ""})
            thought = decision.get("thought", "")
            trace.agent_step("executor", "subagent_react_step", {
                "sub_agent_idx": idx,
                "step": step,
                "decision": decision,
            })

            if not self._has_structured_thought(thought) and step < MAX_SUB_REACT_STEPS:
                conversation.append({
                    "role": "user",
                    "content": (
                        "Your `thought` must explicitly include `Reason:` and "
                        "`Action:`. Please continue with that structure."
                    ),
                })
                continue

            if decision.get("done", False):
                answer = decision.get("answer")
                answer_type = decision.get("answer_type", "")
                requested_type = plan["task_type"]
                if (
                    not self._subagent_followed_tools_hint(execution_log, tools_hint)
                    and step < MAX_SUB_REACT_STEPS
                ):
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": "tools_hint_not_followed",
                        "tools_hint": tools_hint,
                    })
                    hinted_text = ", ".join(self._hinted_tools(tools_hint)) or tools_hint
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Self-correction: this sub-task was assigned a specific tool path. "
                            f"You must actually call one of these tools before finishing: {hinted_text}."
                        ),
                    })
                    continue
                if (
                    answer is not None
                    and not self._is_valid_final_answer(
                        requested_type, answer, horizon=plan.get("horizon"),
                    )
                    and step < MAX_SUB_REACT_STEPS
                ):
                    _, invalid_reason = self._validate_final_answer(
                        requested_type, answer, horizon=plan.get("horizon"),
                    )
                    try:
                        answer_preview = json.dumps(answer, ensure_ascii=False, default=str)
                    except TypeError:
                        answer_preview = str(answer)
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": "invalid_final_answer",
                        "requested_type": requested_type,
                        "invalid_reason": invalid_reason,
                        "answer_preview": answer_preview[:2000],
                    })
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"Your current final answer is not a valid '{requested_type}' payload. "
                            f"Return a proper task-typed result instead of an explanation."
                        ),
                    })
                    continue
                final_result = {
                    "answer": answer,
                    "answer_type": answer_type,
                    "confidence": decision.get("confidence"),
                    "reasoning": decision.get("reasoning", decision.get("thought", "")),
                    "execution_log": execution_log,
                    "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                    "sub_task_description": task_description,
                    "tools_hint": tools_hint,
                }
                break

            tool_name = decision.get("tool", "")
            tool_args = decision.get("args", {})

            if tool_name == "spawn_subagent":
                conversation.append({"role": "user", "content": "Sub-agents cannot spawn further sub-agents."})
                continue

            if tool_name not in candidates:
                conversation.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' not available.",
                })
                continue

            if horizon and tool_name.endswith("_forecast"):
                tool_args["horizon"] = horizon

            if tool_name.startswith("llm_reasoning_"):
                tool_args["context"] = self._build_llm_reasoning_context(
                    plan=plan,
                    execution_log=execution_log,
                    existing_context=tool_args.get("context", ""),
                )

            result = await self._execute_tool(tool_name, tool_args, current_data, text_data, images)
            success = result is not None and result.success
            observation_payload = (
                result.data if success else {"error": result.error if result else "not found"}
            )
            execution_log.append({
                "tool": tool_name,
                "args": tool_args,
                "success": success,
                "output": result.data if success else None,
                "observation": observation_payload,
                "thought": thought,
            })
            trace.agent_step("executor", "subagent_tool_result", {
                "sub_agent_idx": idx,
                "step": step,
                "tool": tool_name,
                "args": tool_args,
                "success": success,
                "observation": observation_payload,
            })

            observation = (
                f"[Sub-{idx} Step {step}] {'OK' if success else 'FAIL'} {tool_name}\n"
                f"→ {self._format_observation_for_prompt(observation_payload)}"
            )
            conversation.append({"role": "user", "content": observation})

            if success and self._is_passthrough_transform(tool_name, result.data):
                current_data = result.data

        logger.info(f"[Executor] Sub-agent {idx} done: {final_result.get('answer_type', '?')}")
        trace.agent_step("executor", "subagent_complete", {
            "sub_agent_idx": idx,
            "sub_task_description": task_description,
            "tools_hint": tools_hint,
            "answer_type": final_result.get("answer_type", ""),
            "answer_preview": str(final_result.get("answer", ""))[:2000],
            "tool_chain": final_result.get("tool_chain", []),
            "steps": len(execution_log),
        })
        return final_result

    # ── Tool execution ────────────────────────────────────────────────

    _FORCE_INPROCESS = frozenset({
        "timesfm_forecast", "chronos_forecast", "timer_forecast", "moirai_forecast",
        "evaluate_against_gt",
    })

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        data: Any,
        text_data: Any,
        images: Any,
    ) -> ToolResult | None:
        """Execute a single tool — routes through MCP when available, falls back to in-process."""
        use_mcp = (
            self._mcp_manager
            and self._mcp_manager.get_tool(tool_name)
            and tool_name not in self._FORCE_INPROCESS
        )
        # ── MCP path ──
        if use_mcp:
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
    def _is_valid_final_answer(task_type: str, answer: Any, horizon: int | None = None) -> bool:
        valid, _ = ExecutorAgent._validate_final_answer(task_type, answer, horizon=horizon)
        return valid

    @staticmethod
    def _validate_final_answer(task_type: str, answer: Any, horizon: int | None = None) -> tuple[bool, str]:
        if answer is None:
            return False, "answer is None"

        if task_type == "forecast":
            candidate = answer
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    return False, "forecast answer string is not valid JSON"
            if isinstance(candidate, dict):
                for key in ("forecast", "point_forecast", "prediction", "values", "answer"):
                    if key in candidate:
                        candidate = candidate[key]
                        break
            try:
                arr = np.asarray(candidate, dtype=float).ravel()
            except (ValueError, TypeError):
                return False, f"forecast answer cannot be converted to numeric array from {type(candidate).__name__}"
            if arr.size == 0:
                return False, "forecast numeric array is empty"
            if horizon is not None and arr.size < horizon:
                return False, f"forecast array too short: got {arr.size}, need at least {horizon}"
            return True, ""

        if task_type == "indicator":
            if isinstance(answer, str):
                try:
                    answer = json.loads(answer)
                except (json.JSONDecodeError, TypeError):
                    return False, "indicator answer string is not valid JSON"
            if not isinstance(answer, dict) or not answer:
                return False, "indicator answer is not a non-empty JSON object"
            numeric_found = False
            for value in answer.values():
                try:
                    float(value)
                    numeric_found = True
                except (ValueError, TypeError):
                    return False, "indicator answer contains non-numeric value"
            return numeric_found, "" if numeric_found else "indicator answer has no numeric values"

        if task_type in {"classification", "correlation"}:
            if isinstance(answer, str):
                return bool(answer.strip()), "" if answer.strip() else "classification/correlation answer is empty string"
            if isinstance(answer, dict):
                for key in ("trend", "correlation", "answer", "label", "prediction"):
                    if key in answer and str(answer[key]).strip():
                        return True, ""
                return False, "classification/correlation JSON has no non-empty label field"
            return False, "classification/correlation answer is neither string nor dict"

        if task_type == "qa":
            if isinstance(answer, str):
                return bool(answer.strip()), "" if answer.strip() else "qa answer is empty string"
            if isinstance(answer, dict):
                ok = any(str(v).strip() for v in answer.values())
                return ok, "" if ok else "qa answer dict has only empty values"
            return False, "qa answer is neither string nor dict"

        return True, ""

    @staticmethod
    def _coerce_prediction_for_evaluation(task_type: str, answer: Any) -> Any:
        """Restore structured payloads from JSON-looking string answers."""
        if isinstance(answer, str) and task_type in {"forecast", "indicator"}:
            try:
                return json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                return answer
        return answer

    @classmethod
    def _collect_batch_eval_predictions(
        cls,
        execution_log: list[dict[str, Any]],
        task_type: str,
        horizon: int | None = None,
    ) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for entry in reversed(execution_log):
            if entry.get("tool") != "spawn_subagent":
                continue
            output = entry.get("output") or []
            for idx, item in enumerate(output):
                if not isinstance(item, dict) or not item.get("success"):
                    continue
                result = item.get("result") or {}
                if not isinstance(result, dict):
                    continue
                answer = result.get("answer", result.get("prediction"))
                if not cls._is_valid_final_answer(task_type, answer, horizon=horizon):
                    continue
                eval_prediction = cls._coerce_prediction_for_evaluation(task_type, answer)
                sub_execution_log = result.get("execution_log") or []
                tool_chain = [
                    str(log_entry.get("tool") or "").strip()
                    for log_entry in sub_execution_log
                    if isinstance(log_entry, dict) and str(log_entry.get("tool") or "").strip()
                ]
                name = (
                    (tool_chain[0] if tool_chain else "")
                    or result.get("candidate_name")
                    or result.get("tools_hint")
                    or result.get("sub_task_description")
                    or f"candidate_{idx}"
                )
                predictions.append({"name": str(name), "prediction": eval_prediction})
            if predictions:
                break
        return predictions

    @staticmethod
    def _build_profiling_section(profiling_results: dict[str, Any]) -> str:
        if not isinstance(profiling_results, dict) or not profiling_results:
            return ""
        keys = [
            "modality_report",
            "fingerprint_snapshot",
            "compute_statistics",
            "detect_seasonality",
            "autocorrelation",
            "stationarity_test",
            "detect_trend",
        ]
        lines: list[str] = []
        for key in keys:
            if key not in profiling_results:
                continue
            value = profiling_results.get(key)
            try:
                value_text = json.dumps(value, ensure_ascii=False, default=str)
            except TypeError:
                value_text = str(value)
            if value_text:
                lines.append(f"- {key}: {value_text[:1200]}")
        if not lines:
            return ""
        return "## Profiling Results\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _has_structured_thought(thought: str) -> bool:
        if not thought:
            return False
        return all(label in thought for label in ("Reason:", "Action:"))

    @staticmethod
    def _coerce_llm_decision(
        parsed: Any,
        text: str | None,
        *,
        fallback_thought: str,
    ) -> dict[str, Any]:
        if isinstance(parsed, dict):
            return parsed
        raw = (text or "").strip()
        if raw:
            try:
                recovered = json.loads(raw)
            except Exception:
                recovered = None
            if isinstance(recovered, dict):
                return recovered
            if isinstance(recovered, list) and recovered and isinstance(recovered[0], dict):
                return recovered[0]
        return {"done": True, "thought": fallback_thought}

    @staticmethod
    def _tool_note_candidates(tool_name: str, task_type: str) -> list[str]:
        candidates = [tool_name]
        if tool_name.startswith("llm_reasoning_"):
            candidates.append("llm_reasoning")
            candidates.append(f"llm_reasoning::{task_type}")
        return candidates

    def _build_tool_notes_section(
        self,
        tool_notes: dict[str, Any],
        candidates: list[str],
        task_type: str,
    ) -> str:
        if not isinstance(tool_notes, dict) or not tool_notes:
            return ""
        lines: list[str] = []
        for tool_name in candidates:
            matched_note = None
            for key in self._tool_note_candidates(tool_name, task_type):
                if key in tool_notes and str(tool_notes[key]).strip():
                    matched_note = str(tool_notes[key]).strip()
                    break
            if matched_note:
                lines.append(f"- {tool_name}: {matched_note[:800]}")
        if not lines:
            return ""
        return "## Tool Notes\n" + "\n".join(lines) + "\n"

    def _build_llm_reasoning_context(
        self,
        plan: dict[str, Any],
        execution_log: list[dict[str, Any]],
        existing_context: str = "",
    ) -> str:
        parts: list[str] = []
        desc = plan.get("task_description", "")
        if desc:
            parts.append(f"## Task Description\n{desc}")
        profiling = plan.get("profiling_section", "")
        if profiling:
            parts.append(profiling.strip())
        tool_notes = plan.get("tool_notes_section", "")
        if tool_notes:
            parts.append(tool_notes.strip())
        if execution_log:
            lines = ["## Prior Tool Observations"]
            for entry in execution_log[-5:]:
                tool = entry.get("tool", "?")
                thought = entry.get("thought", "")
                observation = entry.get("observation")
                obs_text = self._format_observation_for_prompt(observation) if observation is not None else ""
                line = f"- tool={tool}"
                if thought:
                    line += f" | thought={thought[:1000]}"
                if obs_text:
                    line += f" | observation={obs_text[:3000]}"
                lines.append(line)
            parts.append("\n".join(lines))
        if existing_context:
            parts.append(f"## Caller Context\n{existing_context}")
        return "\n\n".join(p for p in parts if p).strip()

    @staticmethod
    def _format_observation_for_prompt(observation: Any) -> str:
        try:
            text = json.dumps(observation, ensure_ascii=False, default=str)
        except TypeError:
            text = str(observation)
        return text[:50000]

    @staticmethod
    def _sanitise_gt_observation(
        conversation: list[dict[str, str]], obs_idx: int,
    ) -> None:
        """Remove ground_truth values from an evaluate_against_gt observation.

        After the LLM has analysed the full tool output (metrics + GT) in the
        current step, we strip the GT from the conversation so that subsequent
        steps only see the metrics and the agent's own analysis text.
        """
        import json as _json
        import re

        msg = conversation[obs_idx]
        content = msg.get("content", "")

        # The observation looks like:  "[Step N] ... → {json payload}"
        # Parse the JSON-ish payload and drop ground_truth recursively.
        try:
            arrow_idx = content.find("→ ")
            if arrow_idx < 0:
                return
            prefix = content[: arrow_idx + 2]
            payload = content[arrow_idx + 2 :]

            parsed = _json.loads(payload)

            def _strip_gt(value: Any) -> Any:
                if isinstance(value, dict):
                    return {
                        k: _strip_gt(v)
                        for k, v in value.items()
                        if k != "ground_truth"
                    }
                if isinstance(value, list):
                    return [_strip_gt(v) for v in value]
                return value

            cleaned = _json.dumps(
                _strip_gt(parsed), ensure_ascii=False, default=str,
            )
            conversation[obs_idx] = {
                "role": msg["role"],
                "content": prefix + cleaned,
            }
            logger.debug("[Executor] Sanitised GT from conversation")
        except Exception:
            pass

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
