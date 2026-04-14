"""ExecutorAgent: mini-ReAct tool orchestration layer.

Instead of planning the entire tool chain upfront, the Executor runs a
step-by-step Think→Act→Observe loop. Each step, the LLM sees the results
of all previous tools and decides what to do next — enabling dynamic
parameter passing (e.g., detect_seasonality → stl_decompose(period=N)).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.agents.prompts import executor_prompt
from timeclaw.agents.tool_dropout import generate_dropout_candidates
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.clawhub.tool_registry import get_tool_instance
from timeclaw.core import trace_logger as trace
from timeclaw.core.agent_base import AgentBase
from timeclaw.core.evidence import EvidenceEntry
from timeclaw.core.message import AgentRole
from timeclaw.core.tea_protocol import TaskEnvelope

MAX_REACT_STEPS = 10
MAX_SUB_REACT_STEPS = 6
SUBAGENT_SLOT_POOL_SIZE = 10
MAX_CONCURRENT_SUBAGENTS = SUBAGENT_SLOT_POOL_SIZE
MIN_TRAINING_CANDIDATES = 2
SUBAGENT_TIMEOUT_SECONDS = 120
MAX_JSON_JUDGE_ATTEMPTS = 2
MAX_TERMINAL_REPAIR_ATTEMPTS = 3
_ORCHESTRATION_ONLY_TOOLS = frozenset({
    "spawn_subagent",
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
})
_DECISION_TOOL_CALL = "emit_tool_call"
_DECISION_SPAWN = "emit_spawn_subagent"
_DECISION_EVALUATE = "emit_evaluate_call"
_DECISION_FINISH = "emit_final_answer"


class MaxStepsExhaustedError(RuntimeError):
    """Raised when an agent exhausts its allowed ReAct budget without a final answer."""

    def __init__(
        self,
        *,
        agent_kind: str,
        max_steps: int,
        sub_agent_idx: int | None = None,
        partial_result: dict[str, Any] | None = None,
    ) -> None:
        self.agent_kind = str(agent_kind)
        self.max_steps = int(max_steps)
        self.sub_agent_idx = sub_agent_idx
        self.partial_result = dict(partial_result or {})
        detail = (
            f"{self.agent_kind} exhausted max steps without producing a finalized answer"
        )
        if sub_agent_idx is not None:
            detail = f"{self.agent_kind} {sub_agent_idx} exhausted max steps without producing a finalized answer"
        super().__init__(detail)


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

    @staticmethod
    def _parse_memory_entry_fields(line: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        text = str(line).strip()
        if text.startswith("- "):
            text = text[2:].strip()
        for part in text.split(" | "):
            key, sep, value = part.partition(":")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            if key and value:
                fields[key] = value
        return fields

    @staticmethod
    def _parse_memory_tool_list(value: str) -> list[str]:
        return [
            token
            for token in (
                item.strip()
                for item in str(value or "").split(",")
            )
            if token and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token)
        ]

    @classmethod
    def _extract_train_path_priors(
        cls,
        memory_text: str,
    ) -> list[dict[str, Any]]:
        priors: list[dict[str, Any]] = []
        for raw_line in str(memory_text or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            fields = cls._parse_memory_entry_fields(line)
            preferred = cls._parse_memory_tool_list(fields.get("Prefer", ""))
            if not preferred:
                continue
            support_match = re.search(
                r"(\d+)\s+note\(s\)",
                str(fields.get("Evidence", "")).strip(),
            )
            support = int(support_match.group(1)) if support_match else 0
            priors.append({
                "path": preferred,
                "support": support,
                "summary": fields.get("Rule", ""),
            })
        return priors

    @classmethod
    def _strip_train_path_priors_from_memory(
        cls,
        memory_text: str,
    ) -> str:
        if not str(memory_text or "").strip():
            return ""
        lines: list[str] = []
        for raw_line in str(memory_text).splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped.startswith("- "):
                lines.append(line)
                continue
            fields = cls._parse_memory_entry_fields(stripped)
            if "Prefer" not in fields and "Avoid" not in fields:
                lines.append(line)
                continue
            kept_parts: list[str] = []
            for key in ("Rule", "When", "Why", "Evidence"):
                value = fields.get(key)
                if value:
                    kept_parts.append(f"{key}: {value}")
            if kept_parts:
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}- " + " | ".join(kept_parts))
        return "\n".join(lines).strip()

    @staticmethod
    def _slot_supports_path(slot_spec: dict[str, Any], path: list[str]) -> bool:
        visible = {
            str(name).strip()
            for name in (slot_spec.get("visible_candidates") or [])
            if str(name).strip()
        }
        if not path or not visible:
            return False
        prior = ExecutorAgent._train_spawn_prior_spec(path)
        anchor_tools = set(prior.get("anchor_tools") or [])
        support_tools = set(prior.get("support_tools") or [])
        reasoning_tool = str(prior.get("reasoning_tool") or "").strip()
        if anchor_tools:
            if not (anchor_tools & visible):
                return False
            return not reasoning_tool or reasoning_tool in visible
        if support_tools:
            return bool(support_tools & visible) and (
                not reasoning_tool or reasoning_tool in visible
            )
        return bool(reasoning_tool) and reasoning_tool in visible

    @staticmethod
    def _train_spawn_prior_spec(prior_path: list[str]) -> dict[str, Any]:
        tools = [
            str(tool).strip()
            for tool in prior_path
            if str(tool).strip()
        ]
        if not tools:
            return {
                "tools": [],
                "anchor_tools": [],
                "support_tools": [],
                "reasoning_tool": "",
            }
        support_tools = [
            tool for tool in tools
            if ExecutorAgent._is_train_prior_support_tool(tool)
        ]
        anchor_tools = [
            tool for tool in tools
            if not tool.startswith("llm_reasoning_") and tool not in support_tools
        ]
        reasoning_tool = next(
            (tool for tool in tools if tool.startswith("llm_reasoning_")),
            "",
        )
        return {
            "tools": tools,
            "anchor_tools": anchor_tools,
            "support_tools": support_tools,
            "reasoning_tool": reasoning_tool,
        }

    @staticmethod
    def _is_train_prior_support_tool(tool_name: str) -> bool:
        tool_name = str(tool_name or "").strip()
        if not tool_name:
            return False
        if tool_name.endswith("_check"):
            return True
        return tool_name in {
            "window_stats",
            "basic_stats",
            "autocorrelation",
            "stationarity_test",
            "detect_trend",
            "detect_seasonality",
            "summarize",
            "sentiment",
            "keyword_extract",
            "ner",
            "text_to_event",
        }

    @classmethod
    def _select_train_spawn_prior(
        cls,
        *,
        path_priors: list[dict[str, Any]],
        subagent_slots: list[dict[str, Any]],
        task_type: str,
    ) -> list[str]:
        compatible: list[dict[str, Any]] = []
        allowed_reasoning_tools = cls._allowed_llm_reasoning_tools_for_task(task_type)
        for prior in path_priors:
            path = [
                str(tool).strip()
                for tool in (prior.get("path") or [])
                if str(tool).strip()
            ]
            if not path:
                continue
            reasoning_tool = str(cls._train_spawn_prior_spec(path).get("reasoning_tool") or "").strip()
            if (
                reasoning_tool
                and allowed_reasoning_tools
                and reasoning_tool not in allowed_reasoning_tools
            ):
                continue
            if any(cls._slot_supports_path(slot, path) for slot in subagent_slots):
                compatible.append({
                    "path": path,
                    "support": int(prior.get("support", 0) or 0),
                })
        if not compatible:
            return []
        compatible.sort(
            key=lambda item: (-int(item.get("support", 0) or 0), len(item.get("path") or []))
        )
        return list(compatible[0]["path"])

    @staticmethod
    def _prior_anchor_tool(prior_path: list[str]) -> str:
        normalized = [
            str(tool).strip()
            for tool in prior_path
            if str(tool).strip()
        ]
        for tool in normalized:
            if not tool.startswith("llm_reasoning_"):
                return tool
        return normalized[0] if normalized else ""

    @staticmethod
    def _build_train_spawn_prior_section(path: list[str]) -> str:
        prior = ExecutorAgent._train_spawn_prior_spec(path)
        tools = prior["tools"]
        anchor_tools = prior["anchor_tools"]
        support_tools = prior["support_tools"]
        reasoning_tool = prior["reasoning_tool"]
        if not tools:
            return ""
        if anchor_tools and reasoning_tool:
            guide_line = (
                "- On the first `spawn_subagent` round, include at least one branch that "
                f"starts from one preferred anchor ({', '.join(anchor_tools)})"
                f" so the later branch can reach `{reasoning_tool}`"
                + (
                    f"; support tools such as {', '.join(support_tools)} are optional helpers"
                    if support_tools
                    else ""
                )
                + ", plus at least one distinct alternative branch.\n"
                + "- Make that prior-guided branch explicit by putting the preferred anchor "
                "as the FIRST tool in its `tools_hint`.\n"
            )
            title_line = (
                "Training First-Spawn Prior:\n"
                f"- Historical preferred branch family: {', '.join(tools)}.\n"
            )
        elif support_tools and not anchor_tools:
            guide_line = (
                "- On the first `spawn_subagent` round, include at least one branch that "
                f"uses one of these preferred support tools ({', '.join(support_tools)}) "
                "and at least one distinct alternative branch.\n"
                "- If you make such a branch, name that preferred support tool explicitly in "
                "its `tools_hint`.\n"
            )
            title_line = (
                "Training First-Spawn Prior:\n"
                f"- Historical preferred support family for branch design: {', '.join(tools)}.\n"
            )
        elif len(anchor_tools) > 1:
            guide_line = (
                "- On the first `spawn_subagent` round, include at least one branch anchored "
                f"by one of these preferred tools ({', '.join(anchor_tools)}) and at least "
                "one distinct alternative branch.\n"
                "- Make that prior-guided branch explicit by putting the chosen preferred "
                "anchor first in its `tools_hint`.\n"
            )
            title_line = (
                "Training First-Spawn Prior:\n"
                f"- Historical preferred tool family for branch design: {', '.join(tools)}.\n"
            )
        elif anchor_tools:
            guide_line = (
                "- On the first `spawn_subagent` round, include at least one branch anchored "
                f"by `{anchor_tools[0]}` and at least one distinct alternative branch.\n"
                f"- Make that prior-guided branch explicit by starting its `tools_hint` with `{anchor_tools[0]}`.\n"
            )
            title_line = (
                "Training First-Spawn Prior:\n"
                f"- Historical preferred branch anchor: {', '.join(tools)}.\n"
            )
        else:
            guide_line = (
                "- On the first `spawn_subagent` round, include at least one branch that "
                f"starts directly with `{reasoning_tool}` and at least one distinct "
                "alternative branch.\n"
                f"- Make that prior-guided branch explicit by starting its `tools_hint` with `{reasoning_tool}`.\n"
            )
            title_line = (
                "Training First-Spawn Prior:\n"
                f"- Historical preferred direct reasoning path: {', '.join(tools)}.\n"
            )
        return (
            title_line
            + guide_line
            + (
                "- If the preferred anchor tool is not visible in the chosen slot, keep "
                "the slot-local `tools_hint` fully supported by that slot and record the "
                "preferred anchor explicitly in `branch_metadata.anchor`.\n"
                if anchor_tools
                else ""
            )
            + "- Treat this only as a branch-construction prior, not as final evidence.\n"
        )

    @staticmethod
    def _fingerprint_missing_ratio(fingerprint: Any) -> float | None:
        if isinstance(fingerprint, dict):
            value = fingerprint.get("missing_ratio")
        else:
            value = getattr(fingerprint, "missing_ratio", None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _filter_task_specific_candidates(
        cls,
        candidates: list[str],
        *,
        task_type: str,
        fingerprint: Any,
    ) -> list[str]:
        allowed_reasoning_tools = cls._allowed_llm_reasoning_tools_for_task(task_type)
        missing_ratio = cls._fingerprint_missing_ratio(fingerprint)
        filtered: list[str] = []
        for raw_name in candidates:
            name = str(raw_name or "").strip()
            if not name:
                continue
            if (
                name.startswith("llm_reasoning_")
                and allowed_reasoning_tools
                and name not in allowed_reasoning_tools
            ):
                continue
            if name == "handle_missing" and missing_ratio is not None and missing_ratio <= 0.0:
                continue
            if name not in filtered:
                filtered.append(name)
        return filtered

    @staticmethod
    def _runtime_compatible_candidates(
        candidates: list[str],
        *,
        current_data: Any,
        current_series_state: dict[str, Any] | None,
    ) -> list[str]:
        filtered: list[str] = []
        for raw_name in candidates:
            name = str(raw_name or "").strip()
            if not name:
                continue
            try:
                tool = get_tool_instance(name)
            except KeyError:
                filtered.append(name)
                continue
            incompatibility = tool.runtime_compatibility_error(
                current_data,
                current_series_state=current_series_state,
            )
            if incompatibility:
                continue
            filtered.append(name)
        return filtered

    @staticmethod
    def _tool_runtime_incompatibility_message(
        tool_name: str,
        *,
        current_data: Any,
        current_series_state: dict[str, Any] | None,
    ) -> str | None:
        name = str(tool_name or "").strip()
        if not name:
            return None
        try:
            tool = get_tool_instance(name)
        except KeyError:
            return None
        incompatibility = tool.runtime_compatibility_error(
            current_data,
            current_series_state=current_series_state,
        )
        if not incompatibility:
            return None
        return "Self-correction: " + incompatibility

    @classmethod
    def _task_mismatched_reasoning_tool_message(
        cls,
        *,
        task_type: str,
        tool_name: str,
    ) -> str | None:
        name = str(tool_name or "").strip()
        if not name.startswith("llm_reasoning_"):
            return None
        allowed = cls._allowed_llm_reasoning_tools_for_task(task_type)
        if allowed and name not in allowed:
            return (
                "Self-correction: this is a task-mismatched reasoning tool. "
                f"For task_type={task_type}, use one of {sorted(allowed)} rather than `{name}`."
            )
        return None

    @classmethod
    def _task_uses_prior_path(
        cls,
        task: dict[str, Any],
        prior_path: list[str],
    ) -> bool:
        prior = cls._train_spawn_prior_spec(prior_path)
        normalized_prior = prior["tools"]
        anchor_tools = prior["anchor_tools"]
        support_tools = prior["support_tools"]
        reasoning_tool = prior["reasoning_tool"]
        if not normalized_prior:
            return False
        branch_metadata = task.get("branch_metadata")
        if isinstance(branch_metadata, dict):
            if branch_metadata.get("uses_prior") is True:
                return True
            if branch_metadata.get("uses_prior") is False:
                return False
        metadata_tools_hint = (
            branch_metadata.get("tools_hint")
            if isinstance(branch_metadata, dict)
            else None
        )
        hinted = cls._hinted_tools(
            metadata_tools_hint if metadata_tools_hint is not None else task.get("tools_hint")
        )
        if not hinted:
            return False
        if anchor_tools:
            if hinted[0] not in set(anchor_tools):
                return False
            if reasoning_tool:
                return reasoning_tool in hinted
            return True
        if support_tools:
            if reasoning_tool:
                return reasoning_tool in hinted and any(tool in hinted for tool in support_tools)
            return any(tool in hinted for tool in support_tools)
        return bool(reasoning_tool) and hinted[0] == reasoning_tool

    @classmethod
    def _task_is_prior_guided_branch(
        cls,
        task: dict[str, Any],
        prior_path: list[str],
    ) -> bool:
        """Return whether the branch design is guided by the preferred prior family.

        New definition:
            - This is intentionally looser than `_task_uses_prior_path(...)`.
            - It is used only for validating the first training spawn mix:
              we only need to know whether one branch was designed from the
              preferred family and one branch explored an alternative family.
            - For anchored workflows, matching the first hinted anchor tool is
              enough to count as prior-guided, because subagents may extend the
              branch later.
        """
        prior = cls._train_spawn_prior_spec(prior_path)
        normalized_prior = prior["tools"]
        anchor_tools = prior["anchor_tools"]
        support_tools = prior["support_tools"]
        reasoning_tool = prior["reasoning_tool"]
        if not normalized_prior:
            return False
        branch_metadata = task.get("branch_metadata")
        if isinstance(branch_metadata, dict):
            if branch_metadata.get("uses_prior") is True:
                return True
            if branch_metadata.get("uses_prior") is False:
                return False
            if anchor_tools:
                explicit_anchor = ""
                for key in ("anchor", "anchor_tool"):
                    value = branch_metadata.get(key)
                    if isinstance(value, str) and value.strip():
                        explicit_anchor = value.strip()
                        break
                if explicit_anchor and explicit_anchor in set(anchor_tools):
                    return True
        metadata_tools_hint = (
            branch_metadata.get("tools_hint")
            if isinstance(branch_metadata, dict)
            else None
        )
        hinted = cls._hinted_tools(
            metadata_tools_hint if metadata_tools_hint is not None else task.get("tools_hint")
        )
        if not hinted:
            return False
        if anchor_tools:
            return hinted[0] in set(anchor_tools)
        if support_tools:
            if reasoning_tool:
                return reasoning_tool in hinted and any(tool in hinted for tool in support_tools)
            return any(tool in hinted for tool in support_tools)
        return bool(reasoning_tool) and hinted[0] == reasoning_tool

    @classmethod
    def _validate_first_spawn_prior_mix(
        cls,
        tasks: list[dict[str, Any]],
        *,
        prior_path: list[str],
    ) -> str | None:
        normalized_prior = [str(tool).strip() for tool in prior_path if str(tool).strip()]
        if not normalized_prior:
            return None
        has_guided = any(cls._task_is_prior_guided_branch(task, normalized_prior) for task in tasks)
        has_alternative = any(
            not cls._task_is_prior_guided_branch(task, normalized_prior)
            for task in tasks
        )
        if has_guided and has_alternative:
            return None
        prior_text = ", ".join(normalized_prior)
        if not has_guided:
            return (
                "Self-correction: on the first training spawn round, include at least one "
                f"branch guided by the historical preferred branch family `{prior_text}`."
            )
        return (
            "Self-correction: on the first training spawn round, also include at least one "
            "distinct alternative branch that does NOT rely on the historical preferred "
            f"branch family `{prior_text}`."
        )

    # ── Agent lifecycle ───────────────────────────────────────────────

    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        fingerprint = envelope.environment.fingerprint or {}
        modality = envelope.task.metadata.get("modality") or "timeseries"
        profiling_results = envelope.task.metadata.get("profiling_results", {})

        return {
            "trace_id": envelope.trace_id,
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
            "indicator_output_keys": envelope.task.metadata.get("indicator_output_keys", []),
            "_tool_dropout_scope": envelope.task.metadata.get("_tool_dropout_scope", ""),
            "_tool_dropout_counts": envelope.task.metadata.get("_tool_dropout_counts", {}),
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
            candidates = executor_prompt.filter_agent_available_tool_names(candidates)
        else:
            candidates = executor_prompt.get_all_tool_names(modalities_detected)

        run_mode = perception.get("_mode", "test")
        eval_tools = {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"}
        if run_mode == "train":
            for eval_tool in eval_tools:
                if eval_tool not in candidates:
                    candidates.append(eval_tool)
            if "spawn_subagent" not in candidates:
                candidates.append("spawn_subagent")
        else:
            candidates = [c for c in candidates if c not in eval_tools and c != "spawn_subagent"]

        candidates = self._filter_task_specific_candidates(
            candidates,
            task_type=task_type,
            fingerprint=perception.get("fingerprint"),
        )

        subagent_slots = self._precompute_subagent_slots(
            trace_id=str(perception.get("trace_id") or ""),
            candidates=candidates,
            task_type=task_type,
            run_mode=run_mode,
            scope=str(perception.get("_tool_dropout_scope") or ""),
            scope_counts=perception.get("_tool_dropout_counts") or {},
        )
        subagent_slot_section = self._build_subagent_slot_section(
            subagent_slots,
            task_type=task_type,
        )

        tool_names_text = executor_prompt.build_tool_name_text(candidates)
        tool_notes_section = self._build_tool_notes_section(
            perception.get("tool_notes", {}),
            candidates,
            task_type,
        )

        # SOUL (fixed) + MEMORY (per-scope + cross_domain) → system prompt
        system_memory = ""
        soul = perception.get("soul_prompt", "")
        raw_memory = perception.get("memory_l2", "")
        mem = raw_memory
        train_path_priors: list[dict[str, Any]] = []
        if run_mode == "train":
            train_path_priors = self._extract_train_path_priors(raw_memory)
            mem = self._strip_train_path_priors_from_memory(raw_memory)
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
            parts = ["Previous Analysis (failed cross-modal check)\n"]
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
                obs_text = self._format_observation_for_prompt(obs) if obs is not None else ""
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
            "train_path_priors": train_path_priors,
            "modality_hint": modality_hint,
            "profiling_section": profiling_section,
            "subagent_slots": subagent_slots,
            "train_spawn_prior_path": self._select_train_spawn_prior(
                path_priors=train_path_priors,
                subagent_slots=subagent_slots,
                task_type=task_type,
            ),
            "subagent_slot_section": subagent_slot_section,
            "fingerprint": perception["fingerprint"],
            "horizon": perception.get("horizon"),
            "domain": perception.get("domain", ""),
            "reanalysis_section": reanalysis_section,
            "_mode": run_mode,
            "_defer_learning_record": bool(perception.get("_defer_learning_record", False)),
            "_ground_truth": perception.get("_ground_truth"),
            "indicator_output_keys": perception.get("indicator_output_keys", []),
            "_tool_dropout_scope": perception.get("_tool_dropout_scope", ""),
            "_tool_dropout_counts": perception.get("_tool_dropout_counts", {}),
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
        current_series_state = self._initial_series_state(current_data)
        last_tool_result: Any = None
        accepted_answer: Any = None
        norm_inverse: dict[str, Any] | None = None
        accepted_answer_type: str = ""
        accepted_confidence: float | None = None
        accepted_reasoning: str = ""
        recoverable_answer: dict[str, Any] | None = None
        recoverable_final: dict[str, Any] | None = None
        repair_delivery_mode = False
        accepted_execution_context: dict[str, Any] = self._normalize_execution_context(None)
        accepted_control_context: dict[str, Any] = self._normalize_control_context(None)

        system_memory = plan.get("system_memory", "")
        system_prompt = executor_prompt.REACT_SYSTEM_PROMPT.format(
            max_steps=MAX_REACT_STEPS,
            max_subagents=MAX_CONCURRENT_SUBAGENTS,
        ) + plan["modality_hint"]
        if system_memory:
            system_prompt += f"\n\n{system_memory}"

        is_train = plan.get("_mode") == "train"
        defer_learning_record = bool(plan.get("_defer_learning_record"))
        system_prompt += executor_prompt.build_visible_tool_guidance(
            candidates,
            defer_learning_record=defer_learning_record,
        )
        expected_final_type = (
            "learning_summary"
            if is_train and not defer_learning_record
            else plan["task_type"]
        )

        first_user = executor_prompt.REACT_FIRST_TURN.format(
            task_prompt=plan.get("task_description", ""),
            task_boundary_section=self._render_prompt_section(
                "Task Boundary",
                self._build_task_boundary_section(
                    plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ),
            ),
            fingerprint=self._format_prompt_payload(
                self._prepare_prompt_fingerprint(
                    plan["fingerprint"],
                    profiling_section=plan.get("profiling_section", ""),
                ),
                limit=5000,
            ),
            profiling_section=plan.get("profiling_section", ""),
            execution_context_section=self._render_prompt_section(
                "Execution Context",
                "",
            ),
            control_context_section=self._render_prompt_section(
                "Control Context",
                self._join_prompt_sections(
                    self._build_action_state_section(
                        state="need_evidence",
                        required_final_type=expected_final_type,
                        execution_log=execution_log,
                    ),
                    self._build_train_spawn_prior_section(
                        plan.get("train_spawn_prior_path", []),
                    ) if is_train else "",
                    plan.get("subagent_slot_section", ""),
                ),
            ),
            output_contract_section=self._render_prompt_section(
                "Output Contract",
                self._build_main_output_contract_section(
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    is_train=is_train,
                    defer_learning_record=defer_learning_record,
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                    allow_finish_current_turn=self._main_finish_allowed_this_turn(
                        execution_log=execution_log,
                        plan=plan,
                    ),
                ),
            ),
            support_section=self._render_prompt_section(
                "Support",
                self._join_prompt_sections(
                    plan.get("reanalysis_section", ""),
                    plan.get("tool_notes_section", ""),
                    plan.get("skills_section", ""),
                ),
            ),
            decision_section=self._build_decision_section(
                allow_finish=self._main_finish_allowed_this_turn(
                    execution_log=execution_log,
                    plan=plan,
                ),
                spawn_available="spawn_subagent" in candidates,
            ),
            tool_names=plan["tool_names"],
        )

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]
        _gt_obs_idx: int | None = None
        skip_generic_turn_prompt_once = False
        pending_spawn_requirement = False
        repair_delivery_feedback_messages: list[str] = []

        for step in range(1, MAX_REACT_STEPS + 1):
            turn_candidates = self._runtime_compatible_candidates(
                candidates,
                current_data=current_data,
                current_series_state=current_series_state,
            )
            is_train = (
                plan.get("_mode") == "train"
                and not bool(plan.get("_defer_learning_record", False))
            )
            candidate_count_now = self._count_training_candidates(
                execution_log,
                task_type=plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
            ) if is_train else 0
            spawned_candidate_count_now = self._count_training_spawn_candidates(
                execution_log,
                plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
            ) if is_train else 0
            has_eval_now = self._has_training_evaluation(execution_log) if is_train else False
            needs_more_candidates = bool(
                is_train
                and (
                    candidate_count_now < MIN_TRAINING_CANDIDATES
                    or spawned_candidate_count_now < MIN_TRAINING_CANDIDATES
                )
                and not has_eval_now
            )
            if not needs_more_candidates:
                pending_spawn_requirement = False
            if repair_delivery_mode:
                spawn_only_turn = False
                eval_only_turn = False
                allow_finish = True
                spawn_available = False
                ordinary_tool_names = []
                evaluate_tool_names = []
            else:
                spawn_only_turn = self._main_requires_spawn_only_turn(
                    step=step,
                    execution_log=execution_log,
                    plan=plan,
                    spawn_candidate_available="spawn_subagent" in turn_candidates,
                )
                if pending_spawn_requirement and "spawn_subagent" in turn_candidates and needs_more_candidates:
                    spawn_only_turn = True
                eval_only_turn = self._main_requires_eval_only_turn(
                    step=step,
                    execution_log=execution_log,
                    plan=plan,
                )
                allow_finish = self._main_finish_allowed_this_turn(
                    execution_log=execution_log,
                    plan=plan,
                )
                spawn_available = "spawn_subagent" in turn_candidates and not eval_only_turn
                ordinary_tool_names = [
                    name for name in turn_candidates
                    if name not in {
                        "spawn_subagent",
                        "evaluate_against_gt",
                        "evaluate_batch_against_gt",
                        "batch_evaluate_against_gt",
                    }
                ]
                if spawn_only_turn or eval_only_turn:
                    ordinary_tool_names = []
                evaluate_tool_names = [
                    name for name in turn_candidates
                    if name in {
                        "evaluate_against_gt",
                        "evaluate_batch_against_gt",
                        "batch_evaluate_against_gt",
                    }
                ]
                if spawn_only_turn:
                    evaluate_tool_names = []
            if step > 1 and not skip_generic_turn_prompt_once:
                follow_up = executor_prompt.REACT_NEXT_TURN.format(
                    step=step,
                    max_steps=MAX_REACT_STEPS,
                    decision_section=self._build_decision_section(
                        allow_finish=allow_finish,
                        spawn_available=spawn_available,
                        spawn_only=spawn_only_turn,
                        eval_only=eval_only_turn,
                        tool_call_allowed=not repair_delivery_mode,
                    ),
                )
                conversation.append({"role": "user", "content": follow_up})
            if not skip_generic_turn_prompt_once:
                budget_msg = self._build_main_budget_message(
                    step=step,
                    execution_log=execution_log,
                    plan=plan,
                )
                if budget_msg:
                    conversation.append({"role": "user", "content": budget_msg})
            skip_generic_turn_prompt_once = False
            conversation.append({
                "role": "user",
                "content": self._render_prompt_section(
                    "Decision State",
                    self._build_action_state_section(
                        state="repair_delivery" if repair_delivery_mode else "need_evidence",
                        required_final_type=expected_final_type,
                        execution_log=execution_log,
                        recoverable_final=recoverable_answer or recoverable_final,
                    ),
                ).strip(),
            })

            decision_conversation = conversation
            if repair_delivery_mode:
                decision_conversation = self._build_clean_repair_delivery_conversation(
                    expected_type=expected_final_type,
                    task_type=plan["task_type"],
                    task_description=plan.get("task_description", ""),
                    execution_log=execution_log,
                    recoverable_final=recoverable_answer or recoverable_final,
                    feedback_messages=repair_delivery_feedback_messages,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
            response, decision = await self._request_structured_decision(
                decision_conversation,
                allow_tool_call=not repair_delivery_mode,
                allow_finish=allow_finish,
                ordinary_tool_names=ordinary_tool_names,
                evaluate_tool_names=evaluate_tool_names,
                spawn_available=spawn_available,
                fallback_thought="LLM parse failed",
                finish_mode=(
                    "learning_summary"
                    if is_train
                    else "task"
                ),
            )
            # GT sanitisation: after the LLM has seen and analysed the
            # evaluate_against_gt result (this step), strip the ground_truth
            # values from the tool observation so subsequent steps can't see them.
            if _gt_obs_idx is not None:
                self._sanitise_gt_observation(conversation, _gt_obs_idx)
                _gt_obs_idx = None

            thought = decision.get("thought", "")
            is_done = decision.get("done", False)
            parse_failed = bool(decision.get("_parse_failed"))

            if parse_failed and step < MAX_REACT_STEPS:
                invalid_json_msg = self._invalid_json_self_correction_message(thought)
                if repair_delivery_mode:
                    repair_delivery_feedback_messages = [invalid_json_msg]
                trace.agent_step("executor", f"react_step_{step}", {
                    "self_correction": True,
                    "reason": "invalid_json_response",
                    "thought": thought,
                    "response_preview": self._decision_response_preview(response),
                })
                conversation.append({
                    "role": "user",
                    "content": invalid_json_msg,
                })
                skip_generic_turn_prompt_once = True
                continue

            conversation.append({
                "role": "assistant",
                "content": self._decision_to_conversation_text(decision),
            })

            if not self._has_structured_thought(thought) and step < MAX_REACT_STEPS:
                structured_thought_msg = (
                    "Self-correction: your `thought` must explicitly include "
                    "`Reason:` and `Action:`. Please continue "
                    "with the required structure."
                )
                if repair_delivery_mode:
                    repair_delivery_feedback_messages = [structured_thought_msg]
                trace.agent_step("executor", f"react_step_{step}", {
                    "self_correction": True,
                    "reason": "unstructured_or_unparsed_response",
                    "parsed_type": type(response.parsed).__name__,
                    "parsed_is_dict": isinstance(response.parsed, dict),
                    "thought": thought,
                    "response_preview": self._decision_response_preview(response),
                })
                conversation.append({
                    "role": "user",
                    "content": structured_thought_msg,
                })
                skip_generic_turn_prompt_once = True
                continue

            if is_done:
                agent_answer = decision.get("answer")
                agent_answer_type = decision.get("answer_type", "")
                agent_confidence = decision.get("confidence")
                agent_reasoning = decision.get("reasoning", thought)
                agent_execution_context = self._normalize_execution_context(
                    decision.get("execution_context"),
                    execution_log=execution_log,
                )
                agent_reasoning = self._normalize_reasoning_for_reuse(
                    agent_reasoning,
                    execution_log,
                    agent_execution_context,
                )
                agent_control_context = self._normalize_control_context(
                    decision.get("control_context")
                )
                is_train = plan.get("_mode") == "train"
                deferred_learning_record = bool(plan.get("_defer_learning_record", False))
                requested_type = plan["task_type"]
                learning_summary_required = is_train and not deferred_learning_record

                if agent_answer is None:
                    logger.info(
                        "[Executor] Self-correction: done=true without answer payload, retrying"
                    )
                    missing_answer_msg = (
                        "Self-correction: you set `done=true` but did not provide "
                        "a final `answer`. Continue until you can return a valid "
                        "task-typed answer payload."
                    )
                    if repair_delivery_mode:
                        repair_delivery_feedback_messages = [missing_answer_msg]
                    conversation.append({
                        "role": "user",
                        "content": missing_answer_msg,
                    })
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "missing_final_answer": True,
                        "answer_type": agent_answer_type,
                    })
                    continue

                if (
                    not is_train
                    and not self._subagent_made_any_tool_call(execution_log)
                    and step < MAX_REACT_STEPS
                ):
                    logger.info(
                        "[Executor] Self-correction: main agent tried to finish without any tool call, retrying"
                    )
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Self-correction: before finishing in test mode, you must actually call "
                            "at least one available tool and base the final answer on that evidence. "
                            "Direct zero-tool answers are not allowed."
                        ),
                    })
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "reason": "main_agent_no_tool_used",
                    })
                    continue

                if (
                    is_train
                    and self._count_training_candidates(
                        execution_log,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ) < MIN_TRAINING_CANDIDATES
                ):
                    logger.info(
                        "[Executor] Self-correction: insufficient candidate exploration, retrying"
                    )
                    correction_msg = (
                        "Self-correction: in training mode you must explore at least 2 valid "
                        "candidate paths before finishing. Continue exploring and compare more "
                        "task-valid candidates."
                    )
                    if (
                        "spawn_subagent" in candidates
                        and not any(
                            isinstance(entry, dict) and entry.get("tool") == "spawn_subagent"
                            for entry in execution_log
                        )
                    ):
                        correction_msg += (
                            " If distinct branches would help, use `spawn_subagent` now to create "
                            "at least 2 distinct candidate paths before finishing."
                        )
                    conversation.append({"role": "user", "content": correction_msg})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "insufficient_candidates": True,
                    })
                    continue

                if (
                    is_train
                    and not self._has_training_evaluation(execution_log)
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

                if learning_summary_required and agent_answer_type != "learning_summary":
                    correction_msg = (
                        "Self-correction: in training mode, the main agent must finish "
                        "with `answer_type='learning_summary'`, not a task-typed final answer. "
                        "Summarize the explored candidates, the winner, and the reusable lesson. "
                        "Use this exact shape: "
                        '`{"answer":{"insight":"...","recommendation":"..."},"answer_type":"learning_summary","reasoning":"1. ...","execution_context":{"tool_chain":["..."],"public_reasoning":["..."],"evidence_summary":"..."},"control_context":{"completion_notes":""}}`. '
                        "Do not put the whole summary into a top-level string `answer`."
                    )
                    if step < MAX_REACT_STEPS:
                        logger.info(
                            "[Executor] Self-correction: training main agent returned task-typed final answer, retrying"
                        )
                        conversation.append({"role": "user", "content": correction_msg})
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "done": True,
                            "self_correction": True,
                            "reason": "training_main_requires_learning_summary",
                            "answer_type": agent_answer_type,
                        })
                        continue
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "reason": "training_main_requires_learning_summary_at_last_step",
                        "answer_type": agent_answer_type,
                    })
                    continue

                if learning_summary_required and not self._is_valid_learning_summary_answer(agent_answer):
                    correction_msg = self._learning_summary_validation_message(agent_answer)
                    if step < MAX_REACT_STEPS:
                        logger.info(
                            "[Executor] Self-correction: malformed learning_summary payload, retrying"
                        )
                        conversation.append({"role": "user", "content": correction_msg})
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "done": True,
                            "self_correction": True,
                            "reason": "invalid_learning_summary_payload",
                            "answer_type": agent_answer_type,
                            "answer_python_type": type(agent_answer).__name__,
                        })
                        continue
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "reason": "invalid_learning_summary_payload_at_last_step",
                        "answer_type": agent_answer_type,
                        "answer_python_type": type(agent_answer).__name__,
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
                        requested_type,
                        agent_answer,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    and step < MAX_REACT_STEPS
                )
                if invalid_answer:
                    _, invalid_reason = self._validate_final_answer(
                        requested_type,
                        agent_answer,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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

                if (
                    agent_answer is not None
                    and not learning_summary_allowed
                    and not self._is_task_answer_backed_by_tool_output(
                        requested_type,
                        agent_answer,
                        execution_log,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                ):
                    correction_msg = (
                        "Self-correction: you may not invent a new task answer in the final step. "
                        "Finish only by returning the exact task answer that was already "
                        "produced by a real tool in this branch. Do not describe packaging, "
                        "JSON repair, or payload wrapping. If no tool has produced one yet, "
                        "call another tool."
                    )
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "reason": "final_answer_not_backed_by_tool_output",
                        "answer_type": agent_answer_type,
                        "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                    })
                    if step < MAX_REACT_STEPS:
                        conversation.append({"role": "user", "content": correction_msg})
                    continue

                solve_ready_current = False
                if learning_summary_allowed:
                    solve_ready_current = self._is_learning_summary_solve_ready_finish(
                        answer=agent_answer,
                        answer_type=agent_answer_type,
                        execution_log=execution_log,
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                else:
                    solve_ready_current = self._is_task_solve_ready_finish(
                        task_type=requested_type,
                        answer=agent_answer,
                        answer_type=agent_answer_type,
                        execution_log=execution_log,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        require_tool_use=(plan.get("_mode") != "train"),
                    )
                if solve_ready_current:
                    recoverable_answer = self._recoverable_final_record(
                        answer=agent_answer,
                        answer_type=agent_answer_type,
                        confidence=agent_confidence,
                        reasoning=agent_reasoning,
                        execution_context=agent_execution_context,
                        control_context=agent_control_context,
                    )
                execution_context_verdict = {"passed": True, "missing": [], "explanation": ""}
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                ):
                    execution_context_verdict = await self._execution_context_verdict(
                        agent_execution_context,
                        execution_log,
                    )
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                    and not execution_context_verdict.get("passed", False)
                ):
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "reason": "execution_context_not_reusable",
                        "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                        "execution_context_preview": json.dumps(
                            agent_execution_context,
                            ensure_ascii=False,
                        )[:1200],
                        "execution_context_gate_missing": list(
                            execution_context_verdict.get("missing", [])
                        ),
                        "execution_context_gate_explanation": str(
                            execution_context_verdict.get("explanation") or ""
                        ),
                    })
                    if step < MAX_REACT_STEPS:
                        repair_delivery_feedback_messages = [
                            self._execution_context_correction_message(
                                execution_log,
                                verdict=execution_context_verdict,
                            )
                        ]
                        conversation.append({
                            "role": "user",
                            "content": repair_delivery_feedback_messages[0],
                        })
                    if recoverable_answer is not None:
                        repair_delivery_mode = True
                    continue

                if solve_ready_current:
                    recoverable_final = recoverable_answer

                reasoning_verdict = {"passed": True, "missing": [], "explanation": ""}
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                ):
                    reasoning_verdict = await self._stepwise_reasoning_verdict(
                        agent_reasoning,
                        execution_log,
                    )
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                    and not reasoning_verdict.get("passed", False)
                ):
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "done": True,
                        "self_correction": True,
                        "reasoning_too_shallow": True,
                        "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                        "reasoning_preview": str(agent_reasoning)[:1200],
                        "reasoning_gate_missing": list(reasoning_verdict.get("missing", [])),
                        "reasoning_gate_explanation": str(reasoning_verdict.get("explanation") or ""),
                    })
                    if step < MAX_REACT_STEPS:
                        repair_delivery_feedback_messages = [
                            self._stepwise_reasoning_correction_message(
                                execution_log,
                                verdict=reasoning_verdict,
                            )
                        ]
                        conversation.append({
                            "role": "user",
                            "content": repair_delivery_feedback_messages[0],
                        })
                    if recoverable_answer is not None:
                        repair_delivery_mode = True
                    continue

                if agent_answer is not None:
                    accepted_answer = agent_answer
                accepted_answer_type = agent_answer_type
                accepted_confidence = agent_confidence
                accepted_reasoning = agent_reasoning
                accepted_execution_context = agent_execution_context
                accepted_control_context = agent_control_context
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought, "done": True,
                    "answer_type": agent_answer_type,
                    "confidence": agent_confidence,
                    "has_answer": agent_answer is not None,
                })
                break

            tool_name = decision.get("tool", "")
            tool_args = decision.get("args", {})

            if (
                is_train
                and step == MAX_REACT_STEPS - 1
                and self._training_completion_contract_satisfied(
                    execution_log,
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
            ):
                logger.info(
                    "[Executor] Self-correction: training contract already satisfied near step limit, forcing finish"
                )
                conversation.append({
                    "role": "user",
                    "content": (
                    "Self-correction: you already satisfy the training completion contract "
                    "(at least 2 valid candidate paths and GT evaluation). With only one "
                    "final turn remaining after this one, do not start another tool call. "
                    "Return `done=true` with answer_type='learning_summary' now."
                ),
                })
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought,
                    "tool": tool_name,
                    "self_correction": True,
                    "reason": "reserve_final_step_for_training_summary",
                })
                continue

            # ── Sub-agent spawn (special built-in tool) ──
            if tool_name == "spawn_subagent":
                if tool_name not in candidates:
                    err_msg = (
                        "`spawn_subagent` is unavailable in this run. "
                        "Continue with a single-path tool chain using the listed tools."
                    )
                    conversation.append({"role": "user", "content": err_msg})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "tool": "spawn_subagent",
                        "error": "not_available_in_mode",
                    })
                    continue
                used_slot_ids: set[int] = set()
                spawn_rounds = 0
                for entry in execution_log:
                    if not isinstance(entry, dict) or entry.get("tool") != "spawn_subagent":
                        continue
                    spawn_rounds += 1
                    output = entry.get("output") or []
                    if isinstance(output, dict):
                        output = output.get("results") or []
                    for item in output:
                        if not isinstance(item, dict):
                            continue
                        result = item.get("result") or {}
                        slot_id = result.get("slot_id")
                        if isinstance(slot_id, int):
                            used_slot_ids.add(slot_id)
                spawn_min_tasks = 0
                if is_train:
                    if needs_more_candidates:
                        pending_spawn_requirement = True
                    spawn_min_tasks = MIN_TRAINING_CANDIDATES
                slot_error = self._validate_spawn_subagent_tasks(
                    tool_args.get("tasks", []),
                    plan.get("subagent_slots", []),
                    min_tasks=spawn_min_tasks,
                    disallowed_slot_ids=used_slot_ids if is_train and spawn_rounds > 0 else None,
                )
                if slot_error:
                    conversation.append({"role": "user", "content": slot_error})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "tool": "spawn_subagent",
                        "self_correction": True,
                        "reason": "invalid_subagent_slots",
                        "detail": slot_error,
                    })
                    continue
                normalized_tasks, normalize_error = self._normalize_spawn_tasks(
                    tool_args.get("tasks", []),
                    plan=plan,
                )
                if normalize_error:
                    conversation.append({"role": "user", "content": normalize_error})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "tool": "spawn_subagent",
                        "self_correction": True,
                        "reason": "non_distinct_subagent_paths_after_dropout",
                        "detail": normalize_error,
                    })
                    continue
                if is_train and spawn_rounds == 0:
                    prior_mix_error = self._validate_first_spawn_prior_mix(
                        normalized_tasks,
                        prior_path=plan.get("train_spawn_prior_path", []),
                    )
                    if prior_mix_error:
                        conversation.append({"role": "user", "content": prior_mix_error})
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "tool": "spawn_subagent",
                            "self_correction": True,
                            "reason": "first_spawn_prior_mix_missing",
                            "detail": prior_mix_error,
                            "prior_path": plan.get("train_spawn_prior_path", []),
                        })
                        continue
                if len(normalized_tasks) < max(1, spawn_min_tasks):
                    detail = (
                        "Self-correction: after deriving advisory paths from the selected "
                        "slots' visible tools, only "
                        f"{len(normalized_tasks)} distinct branch(es) remain. Choose more "
                        "distinct `slot_id`s so this spawn round keeps enough different paths."
                    )
                    conversation.append({"role": "user", "content": detail})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "tool": "spawn_subagent",
                        "self_correction": True,
                        "reason": "insufficient_distinct_subagent_paths_after_dropout",
                        "requested_tasks": len(tool_args.get("tasks", [])),
                        "normalized_tasks": len(normalized_tasks),
                        "min_tasks": max(1, spawn_min_tasks),
                    })
                    continue
                sub_results = await self._spawn_subagents(
                    {"tasks": normalized_tasks}, plan, envelope,
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
                if is_train:
                    pending_spawn_requirement = (
                        self._count_training_spawn_candidates(
                            execution_log,
                            plan["task_type"],
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                        ) < MIN_TRAINING_CANDIDATES
                        and not self._has_training_evaluation(execution_log)
                    )
                continue

            if tool_name not in turn_candidates:
                err_msg = (
                    f"Tool '{tool_name}' not available. "
                    f"Choose from the tools listed in the first message."
                )
                conversation.append({"role": "user", "content": err_msg})
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought, "tool": tool_name, "error": "not in candidates",
                })
                continue

            runtime_incompatibility_msg = self._tool_runtime_incompatibility_message(
                tool_name,
                current_data=current_data,
                current_series_state=current_series_state,
            )
            if runtime_incompatibility_msg:
                conversation.append({"role": "user", "content": runtime_incompatibility_msg})
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought,
                    "tool": tool_name,
                    "self_correction": True,
                    "reason": "runtime_incompatible_tool",
                })
                continue

            mismatch_msg = self._task_mismatched_reasoning_tool_message(
                task_type=plan["task_type"],
                tool_name=tool_name,
            )
            if mismatch_msg:
                conversation.append({"role": "user", "content": mismatch_msg})
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought,
                    "tool": tool_name,
                    "self_correction": True,
                    "reason": "task_mismatched_reasoning_tool",
                })
                continue

            if horizon and tool_name.endswith("_forecast"):
                tool_args["horizon"] = horizon

            if plan.get("domain"):
                if tool_name.startswith("llm_reasoning_"):
                    tool_args["domain"] = plan["domain"]
                else:
                    tool_args.setdefault("domain", plan["domain"])

            if tool_name.startswith("llm_reasoning_"):
                tool_args["context"] = self._build_llm_reasoning_context(
                    plan=plan,
                    execution_log=execution_log,
                    existing_context=tool_args.get("context", ""),
                )

            if tool_name in {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"} and is_train:
                pending_predictions: list[Any] = []
                if "prediction" in tool_args:
                    pending_predictions.append(tool_args.get("prediction"))
                if isinstance(tool_args.get("predictions"), list):
                    pending_predictions.extend(tool_args.get("predictions") or [])
                candidate_count = self._count_training_candidates(
                    execution_log,
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                    pending_predictions=pending_predictions,
                )
                spawned_candidate_count = self._count_training_spawn_candidates(
                    execution_log,
                    plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
                if (
                    candidate_count < MIN_TRAINING_CANDIDATES
                    or spawned_candidate_count < MIN_TRAINING_CANDIDATES
                ):
                    pending_spawn_requirement = True
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Self-correction: in training mode, GT evaluation is only valid "
                            "after `spawn_subagent` has produced at least 2 distinct task-valid "
                            "branch candidates. Explore more spawned branches first, then "
                            "evaluate them."
                        ),
                    })
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "tool": tool_name,
                        "self_correction": True,
                        "insufficient_candidates_for_evaluation": True,
                        "candidate_count": candidate_count,
                    })
                    continue
                tool_args.setdefault("task_type", plan["task_type"])
                if tool_name == "evaluate_against_gt":
                    prediction = tool_args.get("prediction")
                    if not self._is_valid_final_answer(
                        plan["task_type"],
                        prediction,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                    normalized_predictions = self._normalize_batch_eval_predictions_arg(
                        predictions,
                        task_type=plan["task_type"],
                        execution_log=execution_log,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    if (
                        not isinstance(predictions, list)
                        or not predictions
                        or len(normalized_predictions) != len(predictions)
                    ):
                        autofill = self._collect_batch_eval_predictions(
                            execution_log,
                            task_type=plan["task_type"],
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                        )
                        if autofill:
                            tool_args["predictions"] = autofill
                        elif normalized_predictions:
                            tool_args["predictions"] = normalized_predictions
                    elif normalized_predictions:
                        tool_args["predictions"] = normalized_predictions
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

            pre_tool_series_state = dict(current_series_state)
            result = await self._execute_tool(
                tool_name, tool_args, current_data, text_data, images,
            )

            success = result is not None and result.success
            status_icon = "✓" if success else "✗"
            downstream_data = None

            if success:
                last_tool_result = result.data

                if tool_name == "normalize" and result.metadata.get("inverse"):
                    norm_inverse = result.metadata["inverse"]
                downstream_data = self._downstream_series_from_result(tool_name, result.data)
                if downstream_data is not None:
                    current_data = downstream_data
                    current_series_state = self._artifact_state_from_result(
                        tool_name=tool_name,
                        result_data=result.data,
                        downstream_data=downstream_data,
                        previous_series_state=pre_tool_series_state,
                        tool_args=tool_args,
                    )
            else:
                error_msg = result.error if result else "tool not found"

            # Feed tool execution result back into conversation
            observation_payload = (
                result.data if success else {"error": error_msg}
            )
            latest_artifact_state = self._artifact_state_from_result(
                tool_name=tool_name,
                result_data=observation_payload,
                downstream_data=downstream_data,
                previous_series_state=pre_tool_series_state,
                tool_args=tool_args,
            )
            observation_msg = self._build_tool_observation_message(
                step=step,
                tool_name=tool_name,
                tool_args=tool_args,
                success=success,
                observation_payload=observation_payload,
                task_type=plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
            )
            conversation.append({"role": "user", "content": observation_msg})
            conversation.append({
                "role": "user",
                "content": self._build_state_update_message(
                    current_series_state=current_series_state,
                    latest_artifact_state=latest_artifact_state,
                    task_answer_status=self._task_answer_status(
                        task_type=plan["task_type"],
                        candidate=observation_payload,
                        source_tool=tool_name,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ),
                    branch_task_answer_status=self._latest_branch_task_answer_status(
                        task_type=plan["task_type"],
                        execution_log=execution_log,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ),
                ),
            })

            if tool_name in {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"}:
                _gt_obs_idx = len(conversation) - 1

            log_entry = {
                "tool": tool_name, "args": tool_args, "success": success,
                "output": result.data if success else None,
                "observation": observation_payload,
                "thought": thought,
            }
            redundancy_status = self._latest_redundancy_status(
                task_type=plan["task_type"],
                execution_log=execution_log + [log_entry],
                latest_tool_name=tool_name,
                previous_series_state=pre_tool_series_state,
                current_series_state=current_series_state,
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
            )
            conversation[-1] = {
                "role": "user",
                "content": self._build_state_update_message(
                    current_series_state=current_series_state,
                    latest_artifact_state=latest_artifact_state,
                    task_answer_status=self._task_answer_status(
                        task_type=plan["task_type"],
                        candidate=observation_payload,
                        source_tool=tool_name,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ),
                    branch_task_answer_status=self._latest_branch_task_answer_status(
                        task_type=plan["task_type"],
                        execution_log=execution_log + [log_entry],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ),
                    latest_redundancy_status=redundancy_status,
                ),
            }
            if redundancy_status.get("redundant") and step < MAX_REACT_STEPS:
                trace.agent_step("executor", "self_correction", {
                    "step": step,
                    "reason": "repeated_task_valid_answer_without_new_evidence",
                    "tool": tool_name,
                    "repeat_count": redundancy_status.get("repeat_count"),
                    "answer_preview": redundancy_status.get("answer_preview", ""),
                })
                conversation.append({
                    "role": "user",
                    "content": (
                        "Self-correction: the latest tool call repeated the same task-valid answer "
                        "from the same tool without changing the working artifact. This did not "
                        "create a new candidate. If solving is already complete, finish with "
                        "the exact earlier answer. Otherwise, switch to a tool that "
                        "changes the evidence basis before asking for the same label again."
                    ),
                })
                continue

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

        if not accepted_answer_type:
            expected_final_type = (
                "learning_summary"
                if (
                    plan.get("_mode") == "train"
                    and not bool(plan.get("_defer_learning_record", False))
                )
                else plan["task_type"]
            )
            terminal_decision: dict[str, Any] = {}
            terminal_reuse_verdict = {
                "passed": True,
                "execution_context_verdict": {"passed": True, "missing": [], "explanation": ""},
                "reasoning_verdict": {"passed": True, "missing": [], "explanation": ""},
            }
            terminal_feedback_messages: list[str] = []
            learning_summary_required = (
                plan.get("_mode") == "train"
                and not bool(plan.get("_defer_learning_record", False))
            )
            terminal_valid = False
            for terminal_attempt in range(MAX_TERMINAL_REPAIR_ATTEMPTS):
                terminal_decision = await self._request_terminal_path_review(
                    expected_type=expected_final_type,
                    task_type=plan["task_type"],
                    task_description=plan.get("task_description", ""),
                    execution_log=execution_log,
                    trace_phase="act_terminal_self_correction",
                    feedback_messages=terminal_feedback_messages,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                    recoverable_final=recoverable_answer or recoverable_final,
                )
                terminal_answer = terminal_decision.get("answer")
                terminal_answer_type = terminal_decision.get("answer_type", "")
                terminal_reasoning = terminal_decision.get(
                    "reasoning",
                    terminal_decision.get("thought", ""),
                )
                terminal_failure_reason = ""
                terminal_valid = bool(terminal_decision.get("done", False))
                if terminal_decision.get("_parse_failed"):
                    terminal_valid = False
                    terminal_failure_reason = "invalid_json_or_missing_required_fields"
                elif not terminal_valid:
                    terminal_failure_reason = "not_finished"
                elif terminal_answer is None:
                    terminal_valid = False
                    terminal_failure_reason = "missing_answer"
                elif terminal_answer_type != expected_final_type:
                    terminal_valid = False
                    terminal_failure_reason = "wrong_answer_type"
                elif (
                    not learning_summary_required
                    and plan.get("_mode") != "train"
                    and not self._subagent_made_any_tool_call(execution_log)
                ):
                    terminal_valid = False
                    terminal_failure_reason = "no_tool_use"
                elif learning_summary_required:
                    if not self._subagent_made_any_tool_call(execution_log):
                        terminal_valid = False
                        terminal_failure_reason = "no_tool_use"
                    elif not self._is_valid_learning_summary_answer(terminal_answer):
                        terminal_valid = False
                        terminal_failure_reason = "invalid_learning_summary"
                    elif not self._training_completion_contract_satisfied(
                        execution_log,
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ):
                        terminal_valid = False
                        terminal_failure_reason = "training_contract_unsatisfied"
                else:
                    present_in_prior_assistant_history = self._is_task_answer_present_in_assistant_history(
                        plan["task_type"],
                        terminal_answer,
                        conversation[:-1],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    if not self._is_valid_final_answer(
                        plan["task_type"],
                        terminal_answer,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ):
                        terminal_valid = False
                        terminal_failure_reason = "invalid_final_answer"
                    elif not (
                        self._is_task_answer_backed_by_tool_output(
                            plan["task_type"],
                            terminal_answer,
                            execution_log,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                        )
                        or present_in_prior_assistant_history
                    ):
                        terminal_valid = False
                        terminal_failure_reason = "unsupported_answer"
                if not terminal_valid:
                    trace.agent_step("executor", "act_terminal_self_correction_invalid", {
                        "reason": terminal_failure_reason or "invalid_terminal_delivery",
                        "done": bool(terminal_decision.get("done", False)),
                        "answer_type": terminal_answer_type,
                        "parse_failed": bool(terminal_decision.get("_parse_failed")),
                    })
                    if terminal_attempt + 1 >= MAX_TERMINAL_REPAIR_ATTEMPTS:
                        break
                    terminal_feedback_messages = [
                        self._terminal_delivery_correction_message(
                            expected_type=expected_final_type,
                            failure_reason=terminal_failure_reason,
                        )
                    ]
                    continue

                terminal_execution_context = self._normalize_execution_context(
                    terminal_decision.get("execution_context"),
                    execution_log=execution_log,
                )
                terminal_reasoning = self._normalize_reasoning_for_reuse(
                    terminal_reasoning,
                    execution_log,
                    terminal_execution_context,
                )
                terminal_recoverable = self._recoverable_final_record(
                    answer=terminal_answer,
                    answer_type=terminal_answer_type,
                    confidence=terminal_decision.get("confidence"),
                    reasoning=terminal_reasoning,
                    execution_context=terminal_execution_context,
                    control_context=self._normalize_control_context(
                        terminal_decision.get("control_context")
                    ),
                )
                try:
                    terminal_reuse_verdict = await self._final_payload_reusability_verdict(
                        mode=plan.get("_mode"),
                        reasoning=terminal_reasoning,
                        execution_context=terminal_execution_context,
                        execution_log=execution_log,
                    )
                except Exception as e:
                    self._attach_partial_result_to_exception(
                        e,
                        answer=terminal_answer,
                        answer_type=terminal_answer_type,
                        confidence=terminal_decision.get("confidence"),
                        reasoning=terminal_reasoning,
                        execution_context=terminal_execution_context,
                        control_context=self._normalize_control_context(
                            terminal_decision.get("control_context")
                        ),
                        execution_log=execution_log,
                    )
                    raise
                if terminal_reuse_verdict.get("passed", False):
                    accepted_answer = terminal_answer
                    accepted_answer_type = terminal_answer_type
                    accepted_confidence = terminal_decision.get("confidence")
                    accepted_reasoning = terminal_reasoning
                    accepted_execution_context = terminal_execution_context
                    accepted_control_context = terminal_recoverable.get("control_context")
                    trace.agent_step("executor", "act_terminal_self_correction_accepted", {
                        "answer_type": terminal_answer_type,
                        "confidence": accepted_confidence,
                        "has_answer": True,
                    })
                    terminal_valid = True
                    break

                terminal_valid = False
                recoverable_final = terminal_recoverable
                trace.agent_step("executor", "act_terminal_self_correction_rejected", {
                    "reason": (
                        "execution_context_not_reusable"
                        if not terminal_reuse_verdict["execution_context_verdict"].get("passed", False)
                        else "reasoning_too_shallow"
                    ),
                    "execution_context_gate_missing": list(
                        terminal_reuse_verdict["execution_context_verdict"].get("missing", [])
                    ),
                    "execution_context_gate_explanation": str(
                        terminal_reuse_verdict["execution_context_verdict"].get("explanation") or ""
                    ),
                    "reasoning_gate_missing": list(
                        terminal_reuse_verdict["reasoning_verdict"].get("missing", [])
                    ),
                    "reasoning_gate_explanation": str(
                        terminal_reuse_verdict["reasoning_verdict"].get("explanation") or ""
                    ),
                })
                if terminal_attempt + 1 >= MAX_TERMINAL_REPAIR_ATTEMPTS:
                    break
                terminal_feedback_messages = self._terminal_payload_repair_messages(
                    execution_log,
                    terminal_reuse_verdict,
                )
            if not accepted_answer_type and recoverable_final is not None:
                accepted_answer = recoverable_final.get("answer")
                accepted_answer_type = str(recoverable_final.get("answer_type") or "")
                accepted_confidence = recoverable_final.get("confidence")
                accepted_reasoning = str(recoverable_final.get("reasoning") or "")
                accepted_execution_context = self._normalize_execution_context(
                    recoverable_final.get("execution_context"),
                    execution_log=execution_log,
                )
                accepted_reasoning = self._normalize_reasoning_for_reuse(
                    accepted_reasoning,
                    execution_log,
                    accepted_execution_context,
                )
                accepted_control_context = self._normalize_control_context(
                    recoverable_final.get("control_context")
                )
                trace.agent_step("executor", "act_recoverable_final_fallback", {
                    "answer_type": accepted_answer_type,
                    "confidence": accepted_confidence,
                    "has_answer": accepted_answer is not None,
                })

        if not accepted_answer_type:
            trace.agent_step("executor", "act_incomplete", {
                "reason": "max_steps_exhausted_without_final_answer",
                "steps": MAX_REACT_STEPS,
                "last_tool_result_preview": str(last_tool_result)[:5000] if last_tool_result is not None else None,
            })
            raise MaxStepsExhaustedError(
                agent_kind="executor",
                max_steps=MAX_REACT_STEPS,
                partial_result={
                    "prediction": accepted_answer,
                    "answer": accepted_answer,
                    "answer_type": accepted_answer_type,
                    "confidence": accepted_confidence,
                    "reasoning": accepted_reasoning,
                    "execution_context": accepted_execution_context,
                    "control_context": accepted_control_context,
                    "execution_log": list(execution_log),
                    "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                    "failure_reason": "max_steps_exhausted_without_final_answer",
                    "last_tool_result_preview": (
                        str(last_tool_result)[:5000] if last_tool_result is not None else ""
                    ),
                },
            )

        final_result = accepted_answer
        if norm_inverse and final_result is not None:
            final_result = self._denormalize_prediction(final_result, norm_inverse)
            logger.info(f"[Executor] Auto-denormalized with {norm_inverse['method']}")

        act_result = {
            "prediction": final_result,
            "answer": final_result,
            "answer_type": accepted_answer_type,
            "confidence": accepted_confidence,
            "reasoning": accepted_reasoning,
            "execution_context": accepted_execution_context,
            "control_context": accepted_control_context,
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
        indicator_output_keys: list[str] | None = None,
        pending_predictions: list[Any] | None = None,
    ) -> int:
        candidate_ids: set[str] = set()
        real_candidate_signatures: set[str] = set()

        def _signature(answer: Any) -> str | None:
            normalized = cls._normalize_answer_for_comparison(
                task_type,
                answer,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
            if normalized is None:
                return None

            def _to_jsonable(value: Any) -> Any:
                if isinstance(value, np.ndarray):
                    return [_to_jsonable(v) for v in value.tolist()]
                if isinstance(value, dict):
                    return {str(k): _to_jsonable(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_to_jsonable(v) for v in value]
                if isinstance(value, (np.floating, np.integer)):
                    return value.item()
                return value

            return json.dumps(
                _to_jsonable(normalized),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )

        def _record_real_candidate(candidate_id: str, answer: Any) -> None:
            signature = _signature(answer)
            if signature is None:
                return
            candidate_ids.add(candidate_id)
            real_candidate_signatures.add(signature)

        def _record_pending_candidate(candidate_id: str, answer: Any) -> None:
            signature = _signature(answer)
            if signature is None:
                return
            if signature in real_candidate_signatures:
                return
            candidate_ids.add(candidate_id)

        for entry_idx, entry in enumerate(execution_log):
            tool_name = str(entry.get("tool") or "").strip()
            if tool_name == "spawn_subagent":
                output = entry.get("output")
                if isinstance(output, dict):
                    sub_results = output.get("results") or []
                elif isinstance(output, list):
                    sub_results = output
                else:
                    continue
                for idx, item in enumerate(sub_results):
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    if not isinstance(result, dict):
                        continue
                    answer = result.get("answer", result.get("prediction"))
                    if not cls._is_valid_final_answer(
                        task_type,
                        answer,
                        horizon=horizon,
                        indicator_output_keys=indicator_output_keys,
                    ):
                        continue
                    _record_real_candidate(
                        cls._candidate_branch_id(result, fallback_idx=idx),
                        answer,
                    )
                continue
        for record in cls._top_level_candidate_records(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
            _record_real_candidate(
                str(record.get("name") or "").strip(),
                record.get("prediction"),
            )
        for idx, prediction in enumerate(pending_predictions or []):
            if not cls._is_valid_final_answer(
                task_type,
                prediction,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                continue
            _record_pending_candidate(f"pending_eval::{idx}", prediction)
        return len(candidate_ids)

    @staticmethod
    def _count_training_spawn_candidates(
        execution_log: list[dict[str, Any]],
        task_type: str,
        *,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> int:
        """Count distinct task-valid candidates that came from spawned branches.

        New definition:
            - Training is considered properly branched only when `spawn_subagent`
              has produced at least two distinct task-valid branch outputs.
            - Top-level tool calls can still be useful evidence, but they do not
              satisfy the mandatory multi-branch exploration requirement by
              themselves.
        """
        candidate_ids: set[str] = set()

        for entry in execution_log:
            if str(entry.get("tool") or "").strip() != "spawn_subagent":
                continue
            output = entry.get("output")
            if isinstance(output, dict):
                sub_results = output.get("results") or []
            elif isinstance(output, list):
                sub_results = output
            else:
                continue
            for idx, item in enumerate(sub_results):
                if not isinstance(item, dict) or not item.get("success"):
                    continue
                result = item.get("result") or {}
                if not isinstance(result, dict):
                    continue
                answer = result.get("answer", result.get("prediction"))
                if not ExecutorAgent._is_valid_final_answer(
                    task_type,
                    answer,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
                ):
                    continue
                candidate_ids.add(
                    ExecutorAgent._candidate_branch_id(result, fallback_idx=idx),
                    )
        return len(candidate_ids)

    @staticmethod
    def _top_level_candidate_path_name(tool_chain: list[str]) -> str:
        cleaned = [str(tool).strip() for tool in tool_chain if str(tool).strip()]
        if not cleaned:
            return "top_level"
        return f"top_level::{ ' -> '.join(cleaned) }"

    @classmethod
    def _top_level_candidate_records(
        cls,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Collect top-level task-valid candidates using their full executed path.

        New definition:
            A top-level candidate is identified by the substantive tool path
            executed up to the task-valid answer, not by the final tool alone.
            This keeps `value_at -> window_stats -> llm_reasoning_classification`
            distinct from `segment -> llm_reasoning_classification`, even though
            both end at the same classifier.
        """
        framework_tools = {
            "spawn_subagent",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
            "bound_check",
            "consistency_check",
        }
        records: list[dict[str, Any]] = []
        pending_chain: list[str] = []

        for entry in execution_log:
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name:
                continue
            if tool_name in framework_tools:
                pending_chain = []
                continue

            pending_chain.append(tool_name)
            answer = cls._candidate_answer_from_entry(entry, task_type=task_type)
            if task_type == "indicator":
                answer = cls._canonicalize_indicator_answer(
                    answer,
                    indicator_output_keys=indicator_output_keys,
                )
            if not cls._is_valid_final_answer(
                task_type,
                answer,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                continue

            records.append(
                {
                    "name": cls._top_level_candidate_path_name(pending_chain),
                    "tool_chain": list(pending_chain),
                    "prediction": answer,
                }
            )
            pending_chain = []
        return records

    @staticmethod
    def _candidate_answer_from_entry(
        entry: dict[str, Any],
        *,
        task_type: str,
    ) -> Any:
        def _extract(payload: Any) -> Any:
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    return None
            if isinstance(payload, dict):
                keys = ["answer", "prediction", "forecast", "point_forecast", "values"]
                if task_type in {"classification", "correlation"}:
                    keys.extend(["label", "trend", "correlation"])
                for key in keys:
                    value = payload.get(key)
                    if value is not None:
                        return value
                if task_type == "indicator":
                    return payload
            return None

        answer = _extract(entry.get("observation"))
        if answer is not None:
            return answer
        return _extract(entry.get("output"))

    @staticmethod
    def _canonical_indicator_key(key: str) -> str:
        normalized = str(key).strip().lower()
        alias_map = {
            "upper_bollinger_band": "upper_bb",
            "upper_bollinger_bands": "upper_bb",
            "bollinger_upper": "upper_bb",
            "bb_upper": "upper_bb",
            "macd_values": "macd",
            "macd_forecast": "macd",
            "requested_indicator_key": "indicator_name",
        }
        return alias_map.get(normalized, normalized)

    @classmethod
    def _canonicalize_indicator_answer(
        cls,
        answer: Any,
        *,
        indicator_output_keys: list[str] | None = None,
    ) -> Any:
        candidate = answer
        if isinstance(candidate, dict) and isinstance(candidate.get("answer"), dict):
            normalized_inner = cls._canonicalize_indicator_answer(
                candidate["answer"],
                indicator_output_keys=indicator_output_keys,
            )
            if normalized_inner is candidate["answer"]:
                return candidate
            wrapped = dict(candidate)
            wrapped["answer"] = normalized_inner
            return wrapped
        if not isinstance(candidate, dict) or not candidate:
            return answer

        expected_keys = [
            str(key).strip()
            for key in (indicator_output_keys or [])
            if str(key).strip()
        ]
        canonical_expected = [cls._canonical_indicator_key(key) for key in expected_keys]
        actual_items = list(candidate.items())
        canonical_actual = [
            (cls._canonical_indicator_key(str(key)), value) for key, value in actual_items
        ]

        if expected_keys and len(canonical_actual) == 1:
            canonical_key, value = canonical_actual[0]
            if canonical_key == "indicator_name" or canonical_key in canonical_expected:
                return {expected_keys[0]: value}

        normalized = {
            canonical_key: value for canonical_key, value in canonical_actual
        }
        return normalized

    @classmethod
    def _normalize_batch_eval_predictions_arg(
        cls,
        predictions: Any,
        *,
        task_type: str,
        execution_log: list[dict[str, Any]] | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(predictions, list):
            return []
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(predictions):
            candidate_name = f"approach_{idx}"
            candidate_prediction = item
            if isinstance(item, dict):
                candidate_name = str(item.get("name") or candidate_name)
                if "prediction" in item:
                    candidate_prediction = item.get("prediction")
                elif "answer" in item:
                    candidate_prediction = item.get("answer")
            if (
                isinstance(candidate_prediction, dict)
                and isinstance(candidate_prediction.get("answer"), (dict, list, str, int, float))
                and str(candidate_prediction.get("answer_type") or "").strip() == str(task_type).strip()
            ):
                candidate_prediction = candidate_prediction.get("answer")
            candidate_prediction = cls._coerce_prediction_for_evaluation(
                task_type,
                candidate_prediction,
            )
            if task_type == "indicator":
                candidate_prediction = cls._canonicalize_indicator_answer(
                    candidate_prediction,
                    indicator_output_keys=indicator_output_keys,
                )
            if not cls._is_valid_final_answer(
                task_type,
                candidate_prediction,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                continue
            normalized.append({
                "name": candidate_name,
                "prediction": candidate_prediction,
            })
        if execution_log:
            normalized = cls._align_batch_eval_prediction_names(
                normalized,
                execution_log=execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        return normalized

    @classmethod
    def _align_batch_eval_prediction_names(
        cls,
        predictions: list[dict[str, Any]],
        *,
        execution_log: list[dict[str, Any]],
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not predictions:
            return predictions

        stable_predictions = cls._collect_batch_eval_predictions(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if not stable_predictions:
            return predictions

        aligned: list[dict[str, Any]] = []
        used_stable_indices: set[int] = set()
        for idx, item in enumerate(predictions):
            if not isinstance(item, dict):
                continue
            candidate_name = str(item.get("name") or "").strip()
            candidate_prediction = item.get("prediction")
            aligned_name = candidate_name

            # New definition:
            #   Batch GT candidate names should carry stable branch identity when
            #   that identity can be recovered from executed paths. Generic
            #   placeholders like `approach_i` are evaluation-local only and must
            #   not leak into downstream winner binding if a real branch name is
            #   available.
            if re.fullmatch(r"approach_\d+", candidate_name):
                matched_idx = None
                approach_idx_match = re.fullmatch(r"approach_(\d+)", candidate_name)
                approach_idx = (
                    int(approach_idx_match.group(1))
                    if approach_idx_match is not None
                    else idx
                )
                if (
                    0 <= approach_idx < len(stable_predictions)
                    and approach_idx not in used_stable_indices
                ):
                    matched_idx = approach_idx
                for stable_idx, stable_item in enumerate(stable_predictions):
                    if matched_idx is not None:
                        break
                    if stable_idx in used_stable_indices:
                        continue
                    if cls._answers_match_semantically(
                        task_type,
                        candidate_prediction,
                        stable_item.get("prediction"),
                        horizon=horizon,
                        indicator_output_keys=indicator_output_keys,
                    ):
                        matched_idx = stable_idx
                        break
                if matched_idx is None and idx < len(stable_predictions) and idx not in used_stable_indices:
                    matched_idx = idx
                if matched_idx is not None:
                    used_stable_indices.add(matched_idx)
                    aligned_name = str(stable_predictions[matched_idx].get("name") or candidate_name).strip()

            aligned.append({
                "name": aligned_name or candidate_name,
                "prediction": candidate_prediction,
            })
        return aligned

    @classmethod
    def _training_completion_contract_satisfied(
        cls,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        return (
            cls._has_training_evaluation(execution_log)
            and cls._count_training_candidates(
                execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ) >= MIN_TRAINING_CANDIDATES
            and cls._count_training_spawn_candidates(
                execution_log,
                task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ) >= MIN_TRAINING_CANDIDATES
        )

    @classmethod
    def _build_main_budget_message(
        cls,
        *,
        step: int,
        execution_log: list[dict[str, Any]],
        plan: dict[str, Any],
    ) -> str:
        remaining_turns = MAX_REACT_STEPS - step + 1
        is_train = plan.get("_mode") == "train"
        if remaining_turns <= 0:
            return ""
        if is_train and not bool(plan.get("_defer_learning_record", False)):
            unmet_training_budget = cls._build_training_unmet_budget_message(
                remaining_turns=remaining_turns,
                execution_log=execution_log,
                plan=plan,
            )
            if unmet_training_budget:
                return unmet_training_budget
        if step == MAX_REACT_STEPS:
            return executor_prompt.FINAL_STEP_MUST_FINISH
        if step == MAX_REACT_STEPS - 1:
            if (
                is_train
                and cls._training_completion_contract_satisfied(
                    execution_log,
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
            ):
                return executor_prompt.TRAIN_COMPLETION_BUDGET_ALERT.format(
                    remaining_turns=remaining_turns,
                )
            return executor_prompt.REACT_BUDGET_ALERT.format(
                remaining_turns=remaining_turns,
                budget_guidance=(
                    "Reserve the final turn for `done=true` with the required "
                    "final answer. Do not spend the last turn on another tool call."
                ),
            )
        return ""

    @classmethod
    def _build_training_unmet_budget_message(
        cls,
        *,
        remaining_turns: int,
        execution_log: list[dict[str, Any]],
        plan: dict[str, Any],
    ) -> str:
        if remaining_turns <= 0:
            return ""

        candidate_count = cls._count_training_candidates(
            execution_log,
            task_type=plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
        )
        spawned_candidate_count = cls._count_training_spawn_candidates(
            execution_log,
            plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
        )
        has_eval = cls._has_training_evaluation(execution_log)

        missing_candidates = candidate_count < MIN_TRAINING_CANDIDATES
        missing_spawned_candidates = spawned_candidate_count < MIN_TRAINING_CANDIDATES
        missing_eval = not has_eval
        if not missing_candidates and not missing_spawned_candidates and not missing_eval:
            return ""

        gap_summary: list[str] = []
        if missing_candidates:
            gap_summary.append(
                f"you still have only {candidate_count} distinct task-valid candidate path(s) "
                f"but need at least {MIN_TRAINING_CANDIDATES}"
            )
        if missing_spawned_candidates:
            gap_summary.append(
                "you still have only "
                f"{spawned_candidate_count} distinct task-valid spawned branch(es) "
                f"but need at least {MIN_TRAINING_CANDIDATES} from `spawn_subagent`"
            )
        if missing_eval:
            gap_summary.append("you still have no GT evaluation")
        gap_text = "; ".join(gap_summary)

        if remaining_turns == 1:
            if (missing_candidates or missing_spawned_candidates) and missing_eval:
                next_action = (
                    "Use this turn for the highest-yield missing training action. Prefer "
                    "`spawn_subagent` now if it is available, because one spawn can still create "
                    "multiple distinct candidate paths in a single step; otherwise create the most "
                    "direct new task-valid candidate path you can. Repeating the same top-level "
                    "tool path does not create a new candidate. Do not spend this turn on "
                    "rephrasing, rewrapping, or packaging an already known answer."
                )
            elif missing_candidates or missing_spawned_candidates:
                next_action = (
                    "Use this turn to create the remaining distinct spawned task-valid branch(es). "
                    "Repeating the same top-level tool path does not create a new candidate. "
                    "Do not return `done=true` yet."
                )
            else:
                next_action = (
                    "Use this turn to run GT evaluation on the explored candidate(s). "
                    "Do not return `done=true` yet."
                )
            return executor_prompt.REACT_BUDGET_ALERT.format(
                remaining_turns=remaining_turns,
                budget_guidance=(
                    "Training completion contract is still unmet: "
                    f"{gap_text}. {next_action}"
                ),
            )

        if remaining_turns == 2:
            if (missing_candidates or missing_spawned_candidates) and missing_eval:
                next_action = (
                    "Do not reserve the next turn for `done=true` yet. Use this turn to create "
                    "the missing spawned candidate paths, ideally via `spawn_subagent`, so "
                    "the final normal turn can run GT evaluation instead of low-value packaging. "
                    "Repeating the same top-level tool path does not create a new candidate."
                )
            elif missing_candidates or missing_spawned_candidates:
                next_action = (
                    "Use this turn to create the remaining distinct spawned branch path(s), and use "
                    "the final normal turn for GT evaluation. Repeating the same top-level tool "
                    "path does not create a new candidate."
                )
            else:
                next_action = (
                    "Use this turn or the final normal turn for GT evaluation, then finish with "
                    "the learning summary immediately after."
                )
            return executor_prompt.REACT_BUDGET_ALERT.format(
                remaining_turns=remaining_turns,
                budget_guidance=(
                    "Training completion contract is not satisfied yet: "
                    f"{gap_text}. {next_action}"
                ),
            )

        return ""

    @staticmethod
    def _build_subagent_budget_message(*, step: int, max_steps: int) -> str:
        remaining_turns = max_steps - step + 1
        if remaining_turns <= 0:
            return ""
        if step == max_steps:
            return executor_prompt.FINAL_STEP_MUST_FINISH
        if step == max_steps - 1:
            return executor_prompt.REACT_BUDGET_ALERT.format(
                remaining_turns=remaining_turns,
                budget_guidance=(
                    "If a real tool in this branch has already produced a task-valid answer, "
                    "stop exploring and reserve the final turn for `done=true`. "
                    "If not, use this turn for the most direct remaining task-producing tool; "
                    "do not spend the final turn on another intermediate analysis."
                ),
            )
        return ""

    @classmethod
    def _main_finish_allowed_this_turn(
        cls,
        *,
        execution_log: list[dict[str, Any]],
        plan: dict[str, Any],
    ) -> bool:
        if plan.get("_mode") != "train" or bool(plan.get("_defer_learning_record", False)):
            return True
        return cls._training_completion_contract_satisfied(
            execution_log,
            task_type=plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
        )

    @classmethod
    def _main_requires_eval_only_turn(
        cls,
        *,
        step: int,
        execution_log: list[dict[str, Any]],
        plan: dict[str, Any],
    ) -> bool:
        if plan.get("_mode") != "train" or bool(plan.get("_defer_learning_record", False)):
            return False

        candidate_count = cls._count_training_candidates(
            execution_log,
            task_type=plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
        )
        spawned_candidate_count = cls._count_training_spawn_candidates(
            execution_log,
            plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
        )
        has_eval = cls._has_training_evaluation(execution_log)
        if (
            candidate_count >= MIN_TRAINING_CANDIDATES
            and spawned_candidate_count >= MIN_TRAINING_CANDIDATES
            and not has_eval
        ):
            return True
        if step != MAX_REACT_STEPS:
            return False
        return False

    @classmethod
    def _main_requires_spawn_only_turn(
        cls,
        *,
        step: int,
        execution_log: list[dict[str, Any]],
        plan: dict[str, Any],
        spawn_candidate_available: bool,
    ) -> bool:
        if plan.get("_mode") != "train" or bool(plan.get("_defer_learning_record", False)):
            return False
        if not spawn_candidate_available:
            return False

        candidate_count = cls._count_training_candidates(
            execution_log,
            task_type=plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
        )
        spawned_candidate_count = cls._count_training_spawn_candidates(
            execution_log,
            plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
        )
        has_eval = cls._has_training_evaluation(execution_log)
        # New definition:
        # once training already has any task-valid candidate but still lacks the
        # required spawned branch set, another ordinary top-level tool call does
        # not advance the contract. Force a spawn-only turn immediately instead
        # of waiting until the budget is nearly exhausted.
        if (
            candidate_count >= 1
            and spawned_candidate_count < MIN_TRAINING_CANDIDATES
            and not has_eval
        ):
            return True
        if step < MAX_REACT_STEPS - 2 or step >= MAX_REACT_STEPS:
            return False
        return (
            (
                candidate_count < MIN_TRAINING_CANDIDATES
                or spawned_candidate_count < MIN_TRAINING_CANDIDATES
            )
            and not has_eval
        )

    @staticmethod
    def _should_enforce_stepwise_reasoning(run_mode: str | None) -> bool:
        return str(run_mode or "").strip() == "train"

    @staticmethod
    def _build_decision_section(
        *,
        allow_finish: bool,
        spawn_available: bool,
        spawn_only: bool = False,
        eval_only: bool = False,
        tool_call_allowed: bool = True,
    ) -> str:
        lines = ["Choose exactly one action for this turn:"]
        idx = 1
        if spawn_available:
            lines.append(
                f"{idx}. If `spawn_subagent` is available and branch comparison is genuinely needed, call `spawn_subagent`."
            )
            idx += 1
        if not tool_call_allowed:
            lines.append(f"{idx}. Do not call any more tools in this turn.")
        elif spawn_only:
            lines.append(f"{idx}. Do not call ordinary tools in this turn.")
        elif eval_only:
            lines.append(f"{idx}. Call exactly one GT evaluation tool from the available list.")
        else:
            lines.append(f"{idx}. Call exactly one tool from the available list.")
        idx += 1
        if allow_finish:
            lines.append(f"{idx}. Finish with `done=true` and return the final answer.")
        else:
            lines.append("`done=true` is invalid in this turn because the required completion contract is not yet satisfied.")
        lines.append("")
        lines.append(
            "This reply is only for the current turn. Do not include a plan for later turns such as `first use tool A, then tool B, then finish`."
        )
        lines.append(
            "If this turn is a tool call, return only that tool-call JSON. Do not preview, simulate, or append any later final answer, winner analysis, or learning summary in the same reply."
        )
        lines.append(
            "Never mix tool-call fields (`tool`, `args`) with final-answer fields (`answer`, `answer_type`, `confidence`, `reasoning`) in the same JSON object."
        )
        if spawn_available:
            lines.append("If you call `spawn_subagent` in this turn, then `done=true` is invalid for this turn.")
        if spawn_only:
            lines.append(
                "This is a spawn-only turn: use `spawn_subagent` now so the final normal turn can perform GT evaluation."
            )
        if eval_only:
            lines.append(
                "This is an evaluation-only turn: do not spend it on ordinary tools, extra branching, or packaging."
            )
        if not tool_call_allowed:
            lines.append(
                "This is a finish-only turn: do not seek new evidence. Reuse the already-solved answer and repair only its final delivery now."
            )
        return "\n".join(lines)

    @staticmethod
    def _render_prompt_section(title: str, content: Any) -> str:
        body = str(content or "").strip()
        if not body:
            return ""
        return f"## {title}\n{body}\n\n"

    @staticmethod
    def _join_prompt_sections(*sections: Any) -> str:
        parts = [str(section).strip() for section in sections if str(section or "").strip()]
        if not parts:
            return ""
        return "\n\n".join(parts) + "\n\n"

    @classmethod
    def _build_action_state_section(
        cls,
        *,
        state: str,
        required_final_type: str,
        execution_log: list[dict[str, Any]],
        recoverable_final: dict[str, Any] | None = None,
    ) -> str:
        lines = [
            f"- Current state: {state}",
            f"- Required final type in this run: {required_final_type}",
        ]
        substantive_chain = [
            str(entry.get("tool") or "").strip()
            for entry in execution_log
            if (
                str(entry.get("tool") or "").strip()
                and str(entry.get("tool") or "").strip() not in _ORCHESTRATION_ONLY_TOOLS
            )
        ]
        if substantive_chain:
            lines.append(f"- Executed substantive tools so far: {' -> '.join(substantive_chain)}")
        if state == "repair_delivery":
            lines.append(
                "- Solving is already complete for this run. Do NOT call any tool; only repair the final JSON delivery."
            )
            if recoverable_final is not None:
                answer = recoverable_final.get("answer")
                try:
                    answer_preview = json.dumps(
                        cls._round_prompt_numbers(answer),
                        ensure_ascii=False,
                        default=str,
                    )
                except TypeError:
                    answer_preview = str(answer)
                lines.append(f"- Reuse this exact earlier answer unchanged: {answer_preview[:240]}")
        else:
            lines.append(
                "- If you do not yet have a supported final answer, continue gathering evidence with the visible tools."
            )
        return "\n".join(lines)

    @classmethod
    def _build_clean_repair_trace_section(
        cls,
        execution_log: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []

        for entry in execution_log:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name:
                continue

            if tool_name == "spawn_subagent":
                output = entry.get("output") or []
                for item in output:
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    if not isinstance(result, dict):
                        continue
                    sub_execution_log = result.get("execution_log") or []
                    candidate_chain = cls._reasoning_substantive_tool_chain(sub_execution_log)
                    candidate_label = (
                        " -> ".join(candidate_chain)
                        or str(result.get("candidate_name") or "").strip()
                        or " -> ".join(cls._hinted_tools(result.get("tools_hint")))
                        or str(result.get("sub_task_description") or "").strip()
                        or "candidate"
                    )
                    answer_preview = cls._format_prompt_payload(
                        result.get("answer"),
                        limit=400,
                    )
                    execution_context = (
                        result.get("execution_context")
                        if isinstance(result.get("execution_context"), dict)
                        else {}
                    )
                    public_reasoning = [
                        str(item).strip()
                        for item in execution_context.get("public_reasoning") or []
                        if str(item).strip()
                    ]
                    evidence_summary = str(
                        execution_context.get("evidence_summary") or ""
                    ).strip()
                    evidence_parts = public_reasoning[:2]
                    if evidence_summary:
                        evidence_parts.append(evidence_summary)
                    evidence_text = " | ".join(evidence_parts[:2])
                    line = f"- Candidate {candidate_label}: answer={answer_preview}"
                    if evidence_text:
                        line += f"; evidence={evidence_text[:500]}"
                    lines.append(line)
                continue

            if tool_name in _ORCHESTRATION_ONLY_TOOLS or not bool(entry.get("success")):
                continue
            args_text = cls._format_prompt_payload(entry.get("args", {}), limit=400)
            observation_text = cls._format_prompt_payload(
                entry.get("observation"),
                limit=1200,
            )
            lines.append(
                f"- {tool_name}: args={args_text}; result={observation_text}"
            )

        return "\n".join(lines) if lines else "- (no reusable execution-side trace available)"

    @classmethod
    def _build_clean_repair_prompt(
        cls,
        *,
        expected_type: str,
        task_type: str,
        task_description: str,
        execution_log: list[dict[str, Any]],
        recoverable_final: dict[str, Any] | None,
        feedback_messages: list[str] | None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
        sub_task_description: str = "",
        tools_hint: Any = None,
        terminal_json: bool = False,
    ) -> str:
        feedback_messages = [
            str(message).strip()
            for message in (feedback_messages or [])
            if str(message).strip()
        ]
        substantive_tools = cls._available_substantive_tool_names(execution_log)
        candidate_labels = cls._reasoning_candidate_labels(execution_log)

        repair_lines = [
            f"- Required final type: {expected_type}",
            f"- Underlying task type: {task_type}",
            "- Repair mode only: do not gather new evidence and do not introduce any new tool result.",
            "- `execution_context` must contain execution-side evidence only.",
            "- `control_context` may hold completion/repair notes, or be empty.",
            "- Never copy repair instructions, validator wording, GT/evaluation notes, or control-flow language into `execution_context`.",
        ]
        if sub_task_description:
            repair_lines.append(f"- Sub-task: {sub_task_description}")
        if tools_hint:
            hinted_tools = ", ".join(cls._hinted_tools(tools_hint)) or str(tools_hint)
            repair_lines.append(f"- Assigned visible tool path in this branch: {hinted_tools}")

        reuse_section = ""
        if recoverable_final is not None:
            answer = recoverable_final.get("answer")
            try:
                answer_preview = json.dumps(
                    cls._round_prompt_numbers(answer),
                    ensure_ascii=False,
                    default=str,
                )
            except TypeError:
                answer_preview = str(answer)
            reuse_section = cls._render_prompt_section(
                "Reuse Answer",
                (
                    "Reuse this exact answer payload unchanged unless it is structurally invalid "
                    "for the required final type.\n"
                    f"{answer_preview[:2000]}"
                ),
            )

        names_lines: list[str] = []
        if substantive_tools:
            names_lines.append("- Executed substantive tools: " + ", ".join(substantive_tools))
        if candidate_labels:
            names_lines.append("- Candidate path labels: " + ", ".join(candidate_labels))
        if not names_lines:
            names_lines.append("- (no substantive tool names recorded)")

        feedback_section = ""
        if feedback_messages:
            feedback_section = cls._render_prompt_section(
                "Repair Feedback",
                (
                    "Controller-side repair constraints only. Use them to repair the delivery, "
                    "but do not quote or paraphrase them inside `execution_context`.\n"
                    + "\n".join(f"- {message}" for message in feedback_messages)
                ),
            )

        output_lines = [
            "- Keep the answer grounded only in the execution trace below.",
            "- Rewrite `reasoning` as numbered steps that expand the same execution-side evidence.",
        ]
        if terminal_json:
            output_lines.append(
                "- Return exactly one final JSON object with `thought`, `done=true`, `answer`, "
                f"`answer_type='{expected_type}'`, `confidence`, `reasoning`, `execution_context`, and `control_context`."
            )
        else:
            output_lines.append(
                "- Use the finish action now. Do not choose any ordinary tool call."
            )
        if expected_type == "learning_summary":
            output_lines.append(
                "- For `learning_summary`, `answer` must be a JSON object with short non-empty string fields `insight` and `recommendation`."
            )

        return cls._join_prompt_sections(
            cls._render_prompt_section(
                "Repair Mode",
                "\n".join(repair_lines),
            ),
            cls._render_prompt_section(
                "Task Description",
                task_description or "",
            ),
            cls._render_prompt_section(
                "Task Boundary",
                cls._build_task_boundary_section(
                    task_type,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
                ),
            ),
            reuse_section,
            cls._render_prompt_section(
                "Execution Names",
                "\n".join(names_lines),
            ),
            cls._render_prompt_section(
                "Execution Trace",
                cls._build_clean_repair_trace_section(execution_log),
            ),
            feedback_section,
            cls._render_prompt_section(
                "Output Requirements",
                "\n".join(output_lines),
            ),
        ).strip()

    @classmethod
    def _build_clean_repair_delivery_conversation(
        cls,
        *,
        expected_type: str,
        task_type: str,
        task_description: str,
        execution_log: list[dict[str, Any]],
        recoverable_final: dict[str, Any] | None,
        feedback_messages: list[str] | None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
        sub_task_description: str = "",
        tools_hint: Any = None,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are repairing the final JSON delivery for a TimeClaw run. "
                    "Use only the execution-side evidence provided below. "
                    "Do not call any new analysis tool. "
                    "Do not copy controller-side repair wording into `execution_context`."
                ),
            },
            {
                "role": "user",
                "content": cls._build_clean_repair_prompt(
                    expected_type=expected_type,
                    task_type=task_type,
                    task_description=task_description,
                    execution_log=execution_log,
                    recoverable_final=recoverable_final,
                    feedback_messages=feedback_messages,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
                    sub_task_description=sub_task_description,
                    tools_hint=tools_hint,
                    terminal_json=False,
                ),
            },
        ]

    @staticmethod
    def _recoverable_final_record(
        *,
        answer: Any,
        answer_type: str,
        confidence: Any,
        reasoning: str,
        execution_context: dict[str, Any],
        control_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "answer_type": answer_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "execution_context": execution_context,
            "control_context": control_context,
        }

    @classmethod
    def _is_task_solve_ready_finish(
        cls,
        *,
        task_type: str,
        answer: Any,
        answer_type: str,
        execution_log: list[dict[str, Any]],
        horizon: int | None,
        indicator_output_keys: list[str] | None,
        require_tool_use: bool,
        tools_hint: Any = None,
    ) -> bool:
        if answer is None or answer_type != str(task_type).strip():
            return False
        if require_tool_use and not cls._subagent_made_any_tool_call(execution_log):
            return False
        if tools_hint is not None and not cls._subagent_followed_tools_hint(execution_log, tools_hint):
            return False
        return (
            cls._is_valid_final_answer(
                task_type,
                answer,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
            and cls._is_task_answer_backed_by_tool_output(
                task_type,
                answer,
                execution_log,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        )

    @classmethod
    def _is_learning_summary_solve_ready_finish(
        cls,
        *,
        answer: Any,
        answer_type: str,
        execution_log: list[dict[str, Any]],
        task_type: str,
        horizon: int | None,
        indicator_output_keys: list[str] | None,
    ) -> bool:
        return (
            answer_type == "learning_summary"
            and cls._is_valid_learning_summary_answer(answer)
            and cls._training_completion_contract_satisfied(
                execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        )

    @staticmethod
    def _build_tool_observation_message(
        *,
        step: int,
        tool_name: str,
        tool_args: dict[str, Any],
        success: bool,
        observation_payload: Any,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
        sub_agent_idx: int | None = None,
    ) -> str:
        step_label = f"Sub-agent {sub_agent_idx} Step {step}" if sub_agent_idx is not None else f"Step {step}"
        status = "success" if success else "failure"
        message = (
            "## New Observation\n"
            f"- Step: {step_label}\n"
            f"- Tool: {tool_name}\n"
            f"- Status: {status}\n"
            f"- Args: {ExecutorAgent._fmt_args(tool_args)}\n"
            "### Tool Result\n"
            f"{ExecutorAgent._format_observation_for_prompt(observation_payload)}"
        )
        answer_status = ExecutorAgent._task_answer_status(
            task_type=task_type,
            candidate=observation_payload,
            source_tool=tool_name,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if answer_status.get("available"):
            message += (
                "\n\n## Answer Status\n"
                "- This tool result already contains a task-valid answer for the current task.\n"
                "- On a later turn, you may finish by returning this exact tool-produced answer unchanged.\n"
            )
        return message

    @staticmethod
    def _build_parent_branch_metadata_section(branch_metadata: Any) -> str:
        if not isinstance(branch_metadata, dict) or not branch_metadata:
            return ""
        lines = ["Parent Branch Metadata:"]
        for key, value in branch_metadata.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                value_text = str(value)
            else:
                try:
                    value_text = json.dumps(value, ensure_ascii=False, default=str)
                except TypeError:
                    value_text = str(value)
            value_text = value_text.strip()
            if not value_text:
                continue
            lines.append(f"- {key_text}: {value_text[:240]}")
        return "\n".join(lines) + ("\n" if len(lines) > 1 else "")

    @classmethod
    def _precompute_subagent_slots(
        cls,
        *,
        trace_id: str,
        candidates: list[str],
        task_type: str,
        run_mode: str,
        scope: str,
        scope_counts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        blocked_tools = {
            "spawn_subagent",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
        }
        base_candidates = [c for c in candidates if c not in blocked_tools]
        if (
            run_mode != "train"
            or not scope
            or not isinstance(scope_counts, dict)
            or not base_candidates
        ):
            return []

        never_drop = {
            tool_name
            for tool_name in base_candidates
            if tool_name == cls._task_llm_reasoning_tool(task_type)
        }
        slots: list[dict[str, Any]] = []
        for slot_id in range(SUBAGENT_SLOT_POOL_SIZE):
            rng = random.Random(f"{trace_id}:{scope}:slot:{slot_id}")
            visible, keep_probs = generate_dropout_candidates(
                base_candidates,
                scope_counts,
                never_drop=never_drop,
                rng=rng,
            )
            slots.append({
                "slot_id": slot_id,
                "visible_candidates": visible,
                "keep_probabilities": keep_probs,
                "never_drop": sorted(never_drop),
            })
        return slots

    @classmethod
    def _slot_recommended_paths(
        cls,
        slot_spec: dict[str, Any],
        *,
        task_type: str,
        max_paths: int = 2,
    ) -> list[list[str]]:
        visible = [
            str(name).strip()
            for name in (slot_spec.get("visible_candidates") or [])
            if str(name).strip()
        ]
        if not visible:
            return []

        reasoning_tool = cls._task_llm_reasoning_tool(task_type)
        paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        def _append(path: list[str]) -> None:
            normalized = tuple(
                str(tool).strip()
                for tool in path
                if str(tool).strip()
            )
            if not normalized or normalized in seen:
                return
            if any(tool not in visible for tool in normalized):
                return
            seen.add(normalized)
            paths.append(list(normalized))

        if task_type == "classification":
            if "segment" in visible and reasoning_tool in visible:
                _append(["segment", reasoning_tool])
            if "detect_trend" in visible and reasoning_tool in visible:
                _append(["detect_trend", reasoning_tool])
            if reasoning_tool in visible:
                _append([reasoning_tool])
            for tool in ["basic_stats", "window_stats", "slice", "value_at", "resample"]:
                if tool in visible and reasoning_tool in visible:
                    _append([tool, reasoning_tool])
                elif tool in visible:
                    _append([tool])
        elif task_type in {"forecast", "indicator"}:
            for tool in [
                "timesfm2_forecast",
                "chronos2_forecast",
                "aurora_forecast",
                "moirai1_1_r_base_forecast",
                "prophet_forecast",
                "sundial_base_128m_forecast",
                "ets_forecast",
                "arima_forecast",
            ]:
                if tool in visible and reasoning_tool in visible:
                    _append([tool, reasoning_tool])
                elif tool in visible:
                    _append([tool])
            if reasoning_tool in visible:
                _append([reasoning_tool])
        else:
            for tool in [
                "slice",
                "value_at",
                "window_stats",
                "correlation_matrix",
                "lagged_correlation",
                "mutual_information",
                "summarize",
                "keyword_extract",
                "ner",
                "text_to_event",
                "temporal_align_text",
            ]:
                if tool in visible and reasoning_tool in visible:
                    _append([tool, reasoning_tool])
                elif tool in visible:
                    _append([tool])
            if reasoning_tool in visible:
                _append([reasoning_tool])

        fallback = cls._derive_tools_hint_from_slot(
            "",
            slot_spec=slot_spec,
            task_type=task_type,
            used_signatures=seen,
        )
        if fallback:
            _append(fallback)
        return paths[:max_paths]

    @staticmethod
    def _format_slot_recommended_paths(paths: list[list[str]]) -> str:
        formatted = [
            " -> ".join(
                str(tool).strip()
                for tool in path
                if str(tool).strip()
            )
            for path in paths
            if path
        ]
        if not formatted:
            return ""
        return "; ".join(formatted)

    @classmethod
    def _build_subagent_slot_section(
        cls,
        subagent_slots: list[dict[str, Any]],
        *,
        task_type: str,
    ) -> str:
        if not subagent_slots:
            return ""
        lines = [
            "Candidate Sub-Agent Slots:",
            "- Spawn rule: choose `slot_id` first, then make sure EVERY tool named in that branch's `tools_hint` appears in that slot's `visible_tools`.",
            "- If one hinted tool is missing, that branch/slot pairing is invalid; pick another slot or simplify the path.",
            "- Prefer using the listed `recommended_slot_local_path_tools` for that slot instead of inventing a path from tools seen only in the top-level main trace.",
        ]
        for slot in subagent_slots:
            if not isinstance(slot, dict):
                continue
            slot_id = slot.get("slot_id")
            if not isinstance(slot_id, int):
                continue
            visible = [
                str(name).strip()
                for name in (slot.get("visible_candidates") or [])
                if str(name).strip()
            ]
            visible_text = ", ".join(visible) if visible else "(none)"
            recommended_paths = cls._slot_recommended_paths(
                slot,
                task_type=task_type,
            )
            recommended_text = cls._format_slot_recommended_paths(recommended_paths)
            lines.append(
                f"- slot_id={slot_id}: visible_tools={visible_text}"
                + (
                    f" | recommended_slot_local_path_tools={recommended_text}"
                    if recommended_text
                    else ""
                )
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _validate_spawn_subagent_tasks(
        tasks_spec: Any,
        subagent_slots: list[dict[str, Any]],
        *,
        min_tasks: int = 0,
        disallowed_slot_ids: set[int] | None = None,
    ) -> str | None:
        if not isinstance(tasks_spec, list) or not tasks_spec:
            return "Self-correction: `spawn_subagent` requires a non-empty `tasks` list."
        if min_tasks > 0 and len(tasks_spec) < min_tasks:
            return (
                "Self-correction: in training mode, `spawn_subagent` must create at least "
                f"{min_tasks} distinct branches in this exploration round."
            )
        if not subagent_slots:
            return None
        valid_slot_ids = {
            int(slot.get("slot_id"))
            for slot in subagent_slots
            if isinstance(slot, dict) and isinstance(slot.get("slot_id"), int)
        }
        blocked_slot_ids = {
            int(slot_id)
            for slot_id in (disallowed_slot_ids or set())
            if isinstance(slot_id, int)
        }
        seen_slot_ids: set[int] = set()
        for idx, task in enumerate(tasks_spec):
            if not isinstance(task, dict):
                return (
                    "Self-correction: each sub-agent task spec must be a JSON object "
                    "with at least `slot_id` and `task`."
                )
            slot_id = task.get("slot_id")
            if not isinstance(slot_id, int):
                return (
                    "Self-correction: each sub-agent task must include an integer "
                    "`slot_id` chosen from the provided candidate slots."
                )
            if slot_id not in valid_slot_ids:
                return (
                    f"Self-correction: slot_id={slot_id} is invalid. Choose only from "
                    f"the provided candidate slots: {sorted(valid_slot_ids)}."
                )
            if slot_id in blocked_slot_ids:
                return (
                    f"Self-correction: slot_id={slot_id} was already used in a previous "
                    "spawn round. Choose a fresh slot_id so the new branch is actually distinct."
                )
            if slot_id in seen_slot_ids:
                return (
                    "Self-correction: each spawned sub-agent must use a different "
                    "`slot_id`. Do not reuse the same candidate slot twice in one round."
                )
            seen_slot_ids.add(slot_id)
            task_desc = str(task.get("task") or "").strip()
            if not task_desc:
                return (
                    f"Self-correction: sub-agent task {idx} is missing a non-empty `task` description."
                )
            branch_metadata = task.get("branch_metadata")
            if branch_metadata is not None and not isinstance(branch_metadata, dict):
                return (
                    f"Self-correction: sub-agent task {idx} has invalid `branch_metadata`. "
                    "`branch_metadata` must be a JSON object when provided."
                )
        return None

    @staticmethod
    def _hinted_tools(tools_hint: Any) -> list[str]:
        raw_tokens: list[str] = []
        if isinstance(tools_hint, str):
            raw_tokens = re.split(r"[\s,;:/()\[\]\n]+", tools_hint)
        elif isinstance(tools_hint, (list, tuple, set)):
            for item in tools_hint:
                if isinstance(item, str):
                    raw_tokens.extend(re.split(r"[\s,;:/()\[\]\n]+", item))
                elif item is not None:
                    raw_tokens.extend(re.split(r"[\s,;:/()\[\]\n]+", str(item)))
        elif tools_hint is None:
            return []
        else:
            raw_tokens = re.split(r"[\s,;:/()\[\]\n]+", str(tools_hint))

        if not raw_tokens:
            return []
        hinted: list[str] = []
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token):
                continue
            hinted.append(token)
        return hinted

    @classmethod
    def _derive_tools_hint_from_slot(
        cls,
        tools_hint: Any,
        *,
        slot_spec: dict[str, Any] | None,
        task_type: str,
        used_signatures: set[tuple[str, ...]] | None = None,
    ) -> list[str]:
        raw_hints = cls._hinted_tools(tools_hint)
        if not isinstance(slot_spec, dict):
            return raw_hints

        visible = [
            str(name).strip()
            for name in (slot_spec.get("visible_candidates") or [])
            if str(name).strip()
        ]
        if not visible:
            return raw_hints

        # New definition:
        # - If the main agent explicitly suggested a tool path, preserve that
        #   path shape and only keep the tools that are actually visible in the
        #   chosen slot.
        # - If none of the hinted tools are compatible with the slot, reject the
        #   path instead of silently rewriting it into a different route. That
        #   mismatch was causing "suggested path A, actual branch B" failures.
        if raw_hints:
            filtered_raw_hints: list[str] = []
            for tool in raw_hints:
                if tool in visible and tool not in filtered_raw_hints:
                    filtered_raw_hints.append(tool)
            if not filtered_raw_hints:
                return []
            signature = tuple(filtered_raw_hints)
            if used_signatures and signature in used_signatures:
                return []
            return filtered_raw_hints

        task_reasoning_tool = cls._task_llm_reasoning_tool(task_type)
        low_signal_tools = {
            "handle_missing",
            "normalize",
            "resample",
            "segment",
            "select_channels",
        }
        fragile_singleton_tools = {
            "window_stats",
            "autocorrelation",
        }
        informative = [tool for tool in visible if tool not in low_signal_tools]
        primary_tools = [
            tool for tool in informative
            if tool != task_reasoning_tool and tool not in fragile_singleton_tools
        ]
        fragile_tools = [
            tool for tool in informative
            if tool in fragile_singleton_tools
        ]

        candidates: list[list[str]] = []
        seen_candidates: set[tuple[str, ...]] = set()

        def _append_candidate(path: list[str]) -> None:
            normalized = tuple(
                str(tool).strip()
                for tool in path
                if str(tool).strip()
            )
            if not normalized or normalized in seen_candidates:
                return
            seen_candidates.add(normalized)
            candidates.append(list(normalized))

        for tool in primary_tools:
            _append_candidate([tool])
        if task_reasoning_tool and task_reasoning_tool in informative:
            _append_candidate([task_reasoning_tool])
        if task_reasoning_tool and task_reasoning_tool in informative:
            for tool in primary_tools:
                _append_candidate([tool, task_reasoning_tool])
        for fragile_tool in fragile_tools:
            for tool in primary_tools:
                _append_candidate([tool, fragile_tool])
        if len(primary_tools) >= 2:
            _append_candidate(primary_tools[:2])
        for tool in informative:
            if tool in fragile_singleton_tools:
                continue
            _append_candidate([tool])
        if not candidates and task_reasoning_tool and task_reasoning_tool in visible:
            _append_candidate([task_reasoning_tool])
        if not candidates and primary_tools and fragile_tools:
            _append_candidate([primary_tools[0], fragile_tools[0]])
        if not candidates and len(informative) >= 2:
            _append_candidate(informative)
        if not candidates:
            _append_candidate(visible)

        used = used_signatures or set()
        for path in candidates:
            signature = tuple(path)
            if signature not in used:
                return path
        return []

    @classmethod
    def _normalize_spawn_tasks(
        cls,
        tasks_spec: list[dict[str, Any]],
        *,
        plan: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str | None]:
        slot_map = {
            int(slot.get("slot_id")): slot
            for slot in (plan.get("subagent_slots", []) or [])
            if isinstance(slot, dict) and isinstance(slot.get("slot_id"), int)
        }
        used_signatures: set[tuple[str, ...]] = set()
        normalized_tasks: list[dict[str, Any]] = []
        task_type = str(plan.get("task_type") or "")

        for task in tasks_spec:
            slot_id = task.get("slot_id")
            slot_spec = slot_map.get(int(slot_id)) if isinstance(slot_id, int) else None
            raw_tools_hint = task.get("tools_hint", "")
            raw_hinted_tools = cls._hinted_tools(raw_tools_hint)
            allowed_reasoning_tools = cls._allowed_llm_reasoning_tools_for_task(task_type)
            hinted_reasoning_tools = [
                tool for tool in raw_hinted_tools
                if tool.startswith("llm_reasoning_")
            ]
            if (
                hinted_reasoning_tools
                and allowed_reasoning_tools
                and not any(tool in allowed_reasoning_tools for tool in hinted_reasoning_tools)
            ):
                return [], (
                    "Self-correction: this branch uses a task-mismatched reasoning tool. "
                    f"For task_type={task_type}, any hinted `llm_reasoning_*` tool must include "
                    f"one of {sorted(allowed_reasoning_tools)} rather than only {hinted_reasoning_tools}."
                )
            slot_visible = [
                str(name).strip()
                for name in (slot_spec.get("visible_candidates") or [])
                if isinstance(slot_spec, dict) and str(name).strip()
            ]
            if slot_visible and raw_hinted_tools:
                missing_tools = [
                    tool for tool in raw_hinted_tools
                    if tool not in slot_visible
                ]
                if missing_tools:
                    visible_text = ", ".join(slot_visible[:24])
                    recommended_text = cls._format_slot_recommended_paths(
                        cls._slot_recommended_paths(
                            slot_spec,
                            task_type=task_type,
                        )
                    )
                    return [], (
                        "Self-correction: the requested `tools_hint` is not fully supported by the "
                        f"chosen slot_id={slot_id}. Pick a slot whose visible tools support "
                        f"that entire path, or change the path. Missing from slot_id={slot_id}: "
                        f"{missing_tools}. Visible tools for slot_id={slot_id}: {visible_text}"
                        + (
                            f". Recommended supported slot-local path(s) for slot_id={slot_id}: "
                            f"{recommended_text}"
                            if recommended_text
                            else ""
                        )
                    )
                derived_tools_hint = []
                for tool in raw_hinted_tools:
                    if tool not in derived_tools_hint:
                        derived_tools_hint.append(tool)
                if tuple(derived_tools_hint) in used_signatures:
                    return [], (
                        "Self-correction: this spawn round produced a duplicate candidate path. "
                        f"Pick a different slot_id or a different `tools_hint` so the new branch "
                        f"is distinct. Duplicate path: {derived_tools_hint}"
                    )
            else:
                derived_tools_hint = cls._derive_tools_hint_from_slot(
                    raw_tools_hint,
                    slot_spec=slot_spec,
                    task_type=task_type,
                    used_signatures=used_signatures,
                )
            if slot_visible and not derived_tools_hint:
                continue
            if derived_tools_hint:
                used_signatures.add(tuple(derived_tools_hint))

            normalized_task = dict(task)
            normalized_task["raw_tools_hint"] = raw_tools_hint
            normalized_task["tools_hint"] = derived_tools_hint
            normalized_tasks.append(normalized_task)

        return normalized_tasks, None

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
    def _subagent_made_any_tool_call(
        execution_log: list[dict[str, Any]],
    ) -> bool:
        return any(
            isinstance(entry, dict) and str(entry.get("tool") or "").strip()
            for entry in execution_log
        )

    @classmethod
    def _subagent_must_start_with_hinted_tool(
        cls,
        execution_log: list[dict[str, Any]],
        tools_hint: Any,
        next_tool_name: str,
    ) -> bool:
        hinted = cls._hinted_tools(tools_hint)
        if not hinted:
            return False
        if cls._subagent_followed_tools_hint(execution_log, tools_hint):
            return False
        return str(next_tool_name or "").strip() not in set(hinted)

    @classmethod
    def _latest_task_valid_tool_answer(
        cls,
        *,
        task_type: str,
        execution_log: list[dict[str, Any]],
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any] | None:
        for entry in reversed(execution_log):
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name or tool_name in _ORCHESTRATION_ONLY_TOOLS:
                continue
            if not bool(entry.get("success")):
                continue
            output = entry.get("output")
            answer_candidate = cls._extract_task_answer_candidate(task_type, output)
            if task_type == "indicator":
                answer_candidate = cls._canonicalize_indicator_answer(
                    answer_candidate,
                    indicator_output_keys=indicator_output_keys,
                )
            if not cls._is_valid_final_answer(
                task_type,
                answer_candidate,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                continue
            return {
                "tool": tool_name,
                "answer": answer_candidate,
            }
        return None

    @classmethod
    def _finish_only_turn_message(
        cls,
        *,
        task_type: str,
        artifact: dict[str, Any],
    ) -> str:
        preview = cls._round_prompt_numbers(artifact.get("answer"))
        try:
            preview_text = json.dumps(preview, ensure_ascii=False, default=str)
        except TypeError:
            preview_text = str(preview)
        return (
            "Finish-only turn: a real tool in this branch already produced a task-valid "
            f"{task_type} answer via `{artifact.get('tool')}`. Do not call another tool. "
            f"Finish now by returning that exact answer unchanged: {preview_text[:800]}"
        )

    @classmethod
    def _build_subagent_tool_backed_result(
        cls,
        *,
        artifact: dict[str, Any],
        task_type: str,
        execution_log: list[dict[str, Any]],
        slot_id: int | None,
        task_description: str,
        tools_hint: Any,
        branch_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        tool_name = str(artifact.get("tool") or "").strip()
        answer = artifact.get("answer")
        tool_chain = [e.get("tool", "") for e in execution_log if e.get("tool")]
        reasoning = (
            f"1. `{tool_name}` produced this branch's {task_type} answer candidate.\n"
            "2. Near the branch step limit, this branch should stop here and keep that "
            "tool-produced result as its final candidate instead of adding another tool call."
        )
        return {
            "answer": answer,
            "answer_type": task_type,
            "confidence": None,
            "reasoning": reasoning,
            "execution_log": execution_log,
            "tool_chain": tool_chain,
            "slot_id": slot_id,
            "sub_task_description": task_description,
            "tools_hint": tools_hint,
            "branch_metadata": branch_metadata,
        }

    @staticmethod
    def _is_label_like_text(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if "\n" in text or "\r" in text:
            return False
        if len(text) > 80:
            return False
        if len(text.split()) > 8:
            return False
        lowered = text.lower()
        if lowered.startswith("step "):
            return False
        if any(marker in text for marker in [". ", ": ", "; "]):
            return False
        return True

    @staticmethod
    def _downstream_series_from_result(tool_name: str, result_data: Any) -> Any | None:
        """Return the series that downstream numeric tools should operate on.

        New definition:
            - Pure array/list outputs still flow through directly.
            - Forecast-like dict outputs now expose their numeric sequence as the
              next working series, so chains like `timesfm2_forecast -> window_stats`
              analyze the forecast rather than silently falling back to history.
            - Transform tools that return `{"data": ...}` (for example resample or
              handle_missing) also replace the working series with that transformed
              sequence.
            - Scalar summary dicts (for example indicator outputs like max/min/diff)
              still do not replace the working series.
        """
        if hasattr(result_data, "shape") or isinstance(result_data, (list, np.ndarray)):
            return result_data
        if not isinstance(result_data, dict):
            return None

        for key in ("data", "point_forecast", "forecast", "prediction", "values"):
            candidate = result_data.get(key)
            if candidate is None:
                continue
            try:
                arr = np.asarray(candidate, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 1 and arr.size > 0:
                return arr.tolist()
        return None

    @classmethod
    def _is_passthrough_transform(cls, tool_name: str, result_data: Any) -> bool:
        return cls._downstream_series_from_result(tool_name, result_data) is not None

    @staticmethod
    def _series_shape_info(data: Any) -> dict[str, Any]:
        try:
            arr = np.asarray(data, dtype=float)
        except (TypeError, ValueError):
            return {}
        if arr.ndim == 0 or arr.size == 0:
            return {}
        info: dict[str, Any] = {"length": int(arr.shape[0])}
        if arr.ndim > 1:
            info["num_channels"] = int(arr.shape[1])
        return info

    @staticmethod
    def _resolve_slice_bounds(start: Any, end: Any, length: int) -> tuple[int, int] | None:
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if length < 0:
            return None
        resolved_start = start + length if start < 0 else start
        resolved_end = end + length if end < 0 else end
        resolved_start = max(0, min(length, resolved_start))
        resolved_end = max(0, min(length, resolved_end))
        if resolved_end < resolved_start:
            resolved_end = resolved_start
        return resolved_start, resolved_end

    @classmethod
    def _series_length_from_state(cls, state: dict[str, Any] | None) -> int:
        shape = (state or {}).get("shape")
        if isinstance(shape, dict):
            try:
                length = int(shape.get("length"))
            except (TypeError, ValueError):
                length = 0
            if length > 0:
                return length
        try:
            return int((state or {}).get("length") or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _origin_space_from_state(state: dict[str, Any] | None) -> list[int] | None:
        coordinate_system = (state or {}).get("coordinate_system")
        if not isinstance(coordinate_system, dict):
            return None
        origin_space = coordinate_system.get("origin_space")
        if (
            isinstance(origin_space, list)
            and len(origin_space) == 2
            and all(isinstance(v, (int, float)) for v in origin_space)
        ):
            return [int(origin_space[0]), int(origin_space[1])]
        return None

    @staticmethod
    def _sanitized_derivation_args(tool_args: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(tool_args, dict):
            return {}
        sanitized: dict[str, Any] = {}
        for key, value in tool_args.items():
            if str(key).startswith("_"):
                continue
            sanitized[str(key)] = value
        return sanitized

    @classmethod
    def _series_coordinate_system(
        cls,
        *,
        length: int,
        origin_space: list[int] | None,
        relative_to: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "indexing": "positional",
            "current_space": [0, int(length)],
            "relative_to": relative_to,
        }
        if origin_space is not None:
            payload["origin_space"] = [int(origin_space[0]), int(origin_space[1])]
        return payload

    @classmethod
    def _make_artifact_state(
        cls,
        *,
        artifact_type: str,
        producer_tool: str,
        parent_artifact_type: str | None,
        shape_info: dict[str, Any] | None = None,
        coordinate_system: dict[str, Any] | None = None,
        derivation: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact: dict[str, Any] = {
            "artifact_type": artifact_type,
            "producer_tool": producer_tool,
            "parent_artifact_type": parent_artifact_type,
        }
        if shape_info:
            artifact["shape"] = {k: int(v) for k, v in shape_info.items()}
        if coordinate_system:
            artifact["coordinate_system"] = coordinate_system
        if derivation:
            artifact["derivation"] = derivation
        if extra:
            artifact.update(extra)
        return artifact

    @classmethod
    def _initial_series_state(cls, data: Any) -> dict[str, Any]:
        shape_info = cls._series_shape_info(data)
        length = int(shape_info.get("length") or 0)
        coordinate_system = None
        if length > 0:
            coordinate_system = cls._series_coordinate_system(
                length=length,
                origin_space=[0, length],
                relative_to="original_input",
            )
        return cls._make_artifact_state(
            artifact_type="series",
            producer_tool="initial_input",
            parent_artifact_type=None,
            shape_info=shape_info,
            coordinate_system=coordinate_system,
            derivation={"operation": "identity", "args": {}},
        )

    @classmethod
    def _artifact_state_from_result(
        cls,
        *,
        tool_name: str,
        result_data: Any,
        downstream_data: Any | None,
        previous_series_state: dict[str, Any] | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if tool_name == "segment" and isinstance(result_data, dict):
            segments = result_data.get("segments") or []
            segment_length = 0
            if segments and isinstance(segments, list) and isinstance(segments[0], list):
                segment_length = len(segments[0])
            num_segments = int(result_data.get("num_segments") or len(segments))
            return cls._make_artifact_state(
                artifact_type="segment_list",
                producer_tool=tool_name,
                parent_artifact_type=(previous_series_state or {}).get("artifact_type"),
                shape_info={
                    "num_segments": num_segments,
                    "segment_length": int(segment_length),
                },
                coordinate_system={
                    "indexing": "segment_local",
                    "current_space": [0, num_segments],
                    "relative_to": "current_input",
                },
                derivation={
                    "operation": tool_name,
                    "args": cls._sanitized_derivation_args(tool_args),
                },
            )

        if downstream_data is not None:
            shape_info = cls._series_shape_info(downstream_data)
            length = int(shape_info.get("length") or 0)
            parent_artifact_type = (previous_series_state or {}).get("artifact_type")
            previous_origin_space = cls._origin_space_from_state(previous_series_state)
            coordinate_system = None

            if length > 0:
                origin_space = previous_origin_space
                relative_to = "original_input"
                if tool_name.endswith("_forecast"):
                    origin_space = None
                    relative_to = "forecast_output"
                elif tool_name == "slice":
                    previous_length = cls._series_length_from_state(previous_series_state)
                    bounds = cls._resolve_slice_bounds(
                        (tool_args or {}).get("start"),
                        (tool_args or {}).get("end"),
                        previous_length,
                    )
                    if bounds is not None:
                        rel_start, rel_end = bounds
                        if previous_origin_space is not None:
                            origin_space = [
                                int(previous_origin_space[0]) + rel_start,
                                int(previous_origin_space[0]) + rel_end,
                            ]
                        else:
                            origin_space = [rel_start, rel_end]
                coordinate_system = cls._series_coordinate_system(
                    length=length,
                    origin_space=origin_space,
                    relative_to=relative_to,
                )
            return cls._make_artifact_state(
                artifact_type="series",
                producer_tool=tool_name,
                parent_artifact_type=parent_artifact_type,
                shape_info=shape_info,
                coordinate_system=coordinate_system,
                derivation={
                    "operation": tool_name,
                    "args": cls._sanitized_derivation_args(tool_args),
                },
            )

        if isinstance(result_data, dict):
            return cls._make_artifact_state(
                artifact_type="structured_artifact",
                producer_tool=tool_name,
                parent_artifact_type=(previous_series_state or {}).get("artifact_type"),
                derivation={
                    "operation": tool_name,
                    "args": cls._sanitized_derivation_args(tool_args),
                },
                extra={"keys": list(result_data.keys())[:6]},
            )
        return cls._make_artifact_state(
            artifact_type="scalar_or_label_artifact",
            producer_tool=tool_name,
            parent_artifact_type=(previous_series_state or {}).get("artifact_type"),
            derivation={
                "operation": tool_name,
                "args": cls._sanitized_derivation_args(tool_args),
            },
        )

    @staticmethod
    def _build_state_update_message(
        *,
        current_series_state: dict[str, Any],
        latest_artifact_state: dict[str, Any],
        task_answer_status: dict[str, Any] | None = None,
        branch_task_answer_status: dict[str, Any] | None = None,
        latest_redundancy_status: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "current_series_state": current_series_state,
            "latest_artifact_state": latest_artifact_state,
        }
        if isinstance(task_answer_status, dict) and task_answer_status:
            payload["task_answer_status"] = task_answer_status
        if isinstance(branch_task_answer_status, dict) and branch_task_answer_status:
            payload["branch_task_answer_status"] = branch_task_answer_status
        if isinstance(latest_redundancy_status, dict) and latest_redundancy_status:
            payload["latest_redundancy_status"] = latest_redundancy_status
        return (
            "## Current Working State\n"
            + json.dumps(
                ExecutorAgent._round_prompt_numbers(payload),
                ensure_ascii=False,
                default=str,
            )
            + "\n\n## Indexing Rule\n"
            + "- For index-sensitive tools such as `slice`, `value_at`, and `window_stats`, use indices from `current_series_state.coordinate_system.current_space`.\n"
            + "- Treat `origin_space` as provenance only. It explains where the current artifact came from; it is not the coordinate space for the next tool call.\n"
            + "- If the current working series is no longer the original input series, do not call index-sensitive tools with original-input indices. If you still need an original-input statistic, use values already present in the task/context or a reasoning tool that can combine original history with the current artifact.\n"
        )

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
        """Spawn one round of the provided parallel sub-agents.

        Each sub-agent is an independent mini-ReAct loop that inherits:
          - Full hierarchical memory (SOUL, MEMORY, SKILLS, tool_notes)
          - Same data and tool set as the main agent
          - Its own conversation context
        """
        tasks_spec = tool_args.get("tasks", [])
        if not tasks_spec:
            return {"n_tasks": 0, "n_success": 0, "results": [], "observation": "No tasks specified."}

        n_tasks = len(tasks_spec)
        logger.info(f"[Executor] Spawning {n_tasks} sub-agents")
        slot_map = {
            int(slot.get("slot_id")): slot
            for slot in (plan.get("subagent_slots", []) or [])
            if isinstance(slot, dict) and isinstance(slot.get("slot_id"), int)
        }
        normalized_tasks, normalize_error = self._normalize_spawn_tasks(
            tasks_spec,
            plan=plan,
        )
        if normalize_error:
            return {
                "n_tasks": n_tasks,
                "n_success": 0,
                "results": [],
                "observation": normalize_error,
            }
        if not normalized_tasks:
            return {
                "n_tasks": 0,
                "n_success": 0,
                "results": [],
                "observation": (
                    "No distinct sub-agent paths remained after deriving advisory paths "
                    "from the slots' visible tools."
                ),
            }

        trace.agent_step("executor", "spawn_subagents", {
            "n_tasks": n_tasks,
            "tasks": normalized_tasks,
        })

        coros = []
        for i, task in enumerate(normalized_tasks):
            sub_desc = task.get("task", f"sub-task-{i}")
            tools_hint = task.get("tools_hint", "")
            branch_metadata = task.get("branch_metadata")
            slot_id = task.get("slot_id")
            slot_spec = slot_map.get(int(slot_id)) if isinstance(slot_id, int) else None
            coros.append(self._run_sub_executor(
                i,
                sub_desc,
                tools_hint,
                branch_metadata if isinstance(branch_metadata, dict) else None,
                plan,
                envelope,
                slot_id=slot_id if isinstance(slot_id, int) else None,
                slot_spec=slot_spec,
            ))

        done_results = await asyncio.gather(*coros, return_exceptions=True)

        results = []
        n_success = 0
        observation_parts = [f"Sub-agent results ({n_tasks} tasks):\n"]

        for i, r in enumerate(done_results):
            if isinstance(r, Exception):
                error_type = type(r).__name__
                failure_reason = (
                    "max_steps_exhausted"
                    if isinstance(r, MaxStepsExhaustedError)
                    else error_type
                )
                results.append({
                    "sub_task": i,
                    "slot_id": normalized_tasks[i].get("slot_id"),
                    "branch_metadata": normalized_tasks[i].get("branch_metadata"),
                    "success": False,
                    "error": str(r),
                    "error_type": error_type,
                    "failure_reason": failure_reason,
                })
                observation_parts.append(
                    f"  Sub-agent {i} (slot_id={normalized_tasks[i].get('slot_id')}): FAILED — {failure_reason}; {str(r)[:500]}"
                )
            else:
                results.append({
                    "sub_task": i,
                    "slot_id": normalized_tasks[i].get("slot_id"),
                    "branch_metadata": normalized_tasks[i].get("branch_metadata"),
                    "success": True,
                    "result": r,
                })
                n_success += 1
                answer = r.get("answer", r.get("prediction"))
                reasoning = r.get("reasoning", "")
                tool_chain = [
                    str(tool).strip()
                    for tool in (r.get("tool_chain") or [])
                    if str(tool).strip()
                ]
                chain_text = " -> ".join(tool_chain) if tool_chain else "(none)"
                observation_parts.append(
                    f"  Sub-agent {i} (slot_id={normalized_tasks[i].get('slot_id')}): SUCCESS — tool_chain={chain_text}; "
                    f"answer={str(answer)[:2000]}; reasoning={reasoning[:1200]}"
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
        tools_hint: Any,
        branch_metadata: dict[str, Any] | None,
        plan: dict[str, Any],
        envelope: TaskEnvelope,
        *,
        slot_id: int | None = None,
        slot_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a single sub-agent ReAct loop with full memory injection."""
        trace.agent_step("executor", "subagent_start", {
            "sub_agent_idx": idx,
            "slot_id": slot_id,
            "sub_task_description": task_description,
            "tools_hint": tools_hint,
            "branch_metadata": branch_metadata,
        })
        data = envelope.environment.data
        text_data = envelope.environment.text_data or None
        images = envelope.environment.images or None
        candidates = self._build_subagent_candidates(
            idx=idx,
            task_description=task_description,
            tools_hint=tools_hint,
            plan=plan,
            envelope=envelope,
            slot_id=slot_id,
            slot_spec=slot_spec,
        )

        system_memory = plan.get("system_memory", "")
        system_prompt = executor_prompt.SUBAGENT_SYSTEM_PROMPT.format(
            max_steps=MAX_SUB_REACT_STEPS,
        )
        if system_memory:
            system_prompt += f"\n\n{system_memory}"
        system_prompt += plan.get("modality_hint", "")
        system_prompt += executor_prompt.build_visible_tool_guidance(
            candidates,
            defer_learning_record=False,
        )

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
            task_prompt=plan.get("task_description", ""),
            task_boundary_section=self._render_prompt_section(
                "Task Boundary",
                self._build_task_boundary_section(
                    plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ),
            ),
            fingerprint=self._format_prompt_payload(
                self._prepare_prompt_fingerprint(
                    plan["fingerprint"],
                    profiling_section=plan.get("profiling_section", ""),
                ),
                limit=3000,
            ),
            profiling_section=plan.get("profiling_section", ""),
            execution_context_section=self._render_prompt_section(
                "Execution Context",
                "",
            ),
            control_context_section=self._render_prompt_section(
                "Control Context",
                self._join_prompt_sections(
                    self._build_action_state_section(
                        state="need_evidence",
                        required_final_type=plan["task_type"],
                        execution_log=[],
                    ),
                    f"Branch Goal:\n{task_description}\n" if str(task_description).strip() else "",
                    self._build_subagent_tools_hint_section(
                        tools_hint,
                        candidates,
                    ),
                    self._build_parent_branch_metadata_section(branch_metadata),
                ),
            ),
            output_contract_section=self._render_prompt_section(
                "Output Contract",
                self._build_subagent_final_answer_contract_section(
                    plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ),
            ),
            support_section=self._render_prompt_section(
                "Support",
                self._join_prompt_sections(
                    hinted_tool_notes,
                    memory_section,
                ),
            ),
            decision_section=self._build_decision_section(
                allow_finish=True,
                spawn_available=False,
            ),
            tool_names=executor_prompt.build_tool_name_text(candidates),
        )

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]
        skip_generic_turn_prompt_once = False

        horizon = plan.get("horizon")
        current_data = data
        current_series_state = self._initial_series_state(current_data)
        final_result: dict[str, Any] = {}
        execution_log: list[dict[str, Any]] = []
        recoverable_answer: dict[str, Any] | None = None
        recoverable_final: dict[str, Any] | None = None
        repair_delivery_mode = False
        repair_delivery_feedback_messages: list[str] = []

        for step in range(1, MAX_SUB_REACT_STEPS + 1):
            turn_candidates = self._runtime_compatible_candidates(
                candidates,
                current_data=current_data,
                current_series_state=current_series_state,
            )
            tool_call_allowed = bool(step < MAX_SUB_REACT_STEPS) and not repair_delivery_mode
            if step > 1 and not skip_generic_turn_prompt_once:
                follow_up = executor_prompt.REACT_NEXT_TURN.format(
                    step=step,
                    max_steps=MAX_SUB_REACT_STEPS,
                    decision_section=self._build_decision_section(
                        allow_finish=True,
                        spawn_available=False,
                        tool_call_allowed=tool_call_allowed,
                    ),
                )
                conversation.append({"role": "user", "content": follow_up})
            if not skip_generic_turn_prompt_once:
                budget_msg = self._build_subagent_budget_message(
                    step=step,
                    max_steps=MAX_SUB_REACT_STEPS,
                )
                if budget_msg:
                    conversation.append({"role": "user", "content": budget_msg})
            skip_generic_turn_prompt_once = False
            conversation.append({
                "role": "user",
                "content": self._render_prompt_section(
                    "Decision State",
                    self._build_action_state_section(
                        state="repair_delivery" if repair_delivery_mode else "need_evidence",
                        required_final_type=plan["task_type"],
                        execution_log=execution_log,
                        recoverable_final=recoverable_answer or recoverable_final,
                    ),
                ).strip(),
            })

            decision_conversation = conversation
            if repair_delivery_mode:
                decision_conversation = self._build_clean_repair_delivery_conversation(
                    expected_type=plan["task_type"],
                    task_type=plan["task_type"],
                    task_description=task_description,
                    execution_log=execution_log,
                    recoverable_final=recoverable_answer or recoverable_final,
                    feedback_messages=repair_delivery_feedback_messages,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                    sub_task_description=task_description,
                    tools_hint=tools_hint,
                )
            response, decision = await self._request_structured_decision(
                decision_conversation,
                allow_tool_call=tool_call_allowed,
                allow_finish=True,
                ordinary_tool_names=[
                    name for name in turn_candidates
                    if name not in {
                        "spawn_subagent",
                        "evaluate_against_gt",
                        "evaluate_batch_against_gt",
                        "batch_evaluate_against_gt",
                    }
                ] if tool_call_allowed else [],
                evaluate_tool_names=[],
                spawn_available=False,
                fallback_thought="parse failed",
                finish_mode="task",
            )
            thought = decision.get("thought", "")
            parse_failed = bool(decision.get("_parse_failed"))
            trace.agent_step("executor", "subagent_react_step", {
                "sub_agent_idx": idx,
                "step": step,
                "decision": decision,
            })

            if parse_failed and step < MAX_SUB_REACT_STEPS:
                invalid_json_msg = self._invalid_json_self_correction_message(thought)
                if repair_delivery_mode:
                    repair_delivery_feedback_messages = [invalid_json_msg]
                conversation.append({
                    "role": "user",
                    "content": invalid_json_msg,
                })
                skip_generic_turn_prompt_once = True
                continue

            conversation.append({
                "role": "assistant",
                "content": self._decision_to_conversation_text(decision),
            })

            if not self._has_structured_thought(thought) and step < MAX_SUB_REACT_STEPS:
                structured_thought_msg = (
                    "Your `thought` must explicitly include `Reason:` and "
                    "`Action:`. Please continue with that structure."
                )
                if repair_delivery_mode:
                    repair_delivery_feedback_messages = [structured_thought_msg]
                conversation.append({
                    "role": "user",
                    "content": structured_thought_msg,
                })
                skip_generic_turn_prompt_once = True
                continue

            if decision.get("done", False):
                answer = decision.get("answer")
                answer_type = decision.get("answer_type", "")
                requested_type = plan["task_type"]
                if answer is None and step < MAX_SUB_REACT_STEPS:
                    missing_answer_msg = (
                        "Self-correction: you set `done=true` but did not provide "
                        "a final `answer`. Continue until you can return a valid "
                        "task-typed answer payload."
                    )
                    if repair_delivery_mode:
                        repair_delivery_feedback_messages = [missing_answer_msg]
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": "missing_final_answer",
                        "answer_type": answer_type,
                    })
                    conversation.append({
                        "role": "user",
                        "content": missing_answer_msg,
                    })
                    continue
                if (
                    not self._subagent_made_any_tool_call(execution_log)
                    and step < MAX_SUB_REACT_STEPS
                ):
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": "no_tool_used",
                    })
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Self-correction: every sub-task must use at least one real tool "
                            "before finishing. Pick the most relevant available tool, run it, "
                            "then continue."
                        ),
                    })
                    continue
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
                        requested_type,
                        answer,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    and step < MAX_SUB_REACT_STEPS
                ):
                    _, invalid_reason = self._validate_final_answer(
                        requested_type,
                        answer,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                if (
                    answer is not None
                    and not self._is_task_answer_backed_by_tool_output(
                        requested_type,
                        answer,
                        execution_log,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                ):
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": "final_answer_not_backed_by_tool_output",
                        "requested_type": requested_type,
                        "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                    })
                    if step < MAX_SUB_REACT_STEPS:
                        conversation.append({
                            "role": "user",
                            "content": (
                                "Self-correction: you may not invent a new task answer in the final step. "
                                "Finish only by returning the exact task answer that was already "
                                "produced by a real tool in this branch. Do not describe packaging, "
                                "JSON repair, or payload wrapping. If no tool has produced one yet, "
                                "call another tool."
                            ),
                        })
                    continue
                reasoning = decision.get("reasoning", decision.get("thought", ""))
                execution_context = self._normalize_execution_context(
                    decision.get("execution_context"),
                    execution_log=execution_log,
                )
                reasoning = self._normalize_reasoning_for_reuse(
                    reasoning,
                    execution_log,
                    execution_context,
                )
                control_context = self._normalize_control_context(
                    decision.get("control_context")
                )
                solve_ready_current = self._is_task_solve_ready_finish(
                    task_type=requested_type,
                    answer=answer,
                    answer_type=answer_type,
                    execution_log=execution_log,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                    require_tool_use=True,
                    tools_hint=tools_hint,
                )
                if solve_ready_current:
                    recoverable_answer = self._recoverable_final_record(
                        answer=answer,
                        answer_type=answer_type,
                        confidence=decision.get("confidence"),
                        reasoning=reasoning,
                        execution_context=execution_context,
                        control_context=control_context,
                    )
                execution_context_verdict = {"passed": True, "missing": [], "explanation": ""}
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                ):
                    execution_context_verdict = await self._execution_context_verdict(
                        execution_context,
                        execution_log,
                    )
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                    and not execution_context_verdict.get("passed", False)
                ):
                    repair_delivery_feedback_messages = [
                        self._execution_context_correction_message(
                            execution_log,
                            verdict=execution_context_verdict,
                        )
                    ]
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": "execution_context_not_reusable",
                        "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                        "execution_context_preview": json.dumps(
                            execution_context,
                            ensure_ascii=False,
                        )[:1200],
                        "execution_context_gate_missing": list(
                            execution_context_verdict.get("missing", [])
                        ),
                        "execution_context_gate_explanation": str(
                            execution_context_verdict.get("explanation") or ""
                        ),
                    })
                    if step < MAX_SUB_REACT_STEPS:
                        conversation.append({
                            "role": "user",
                            "content": repair_delivery_feedback_messages[0],
                        })
                    if recoverable_answer is not None:
                        repair_delivery_mode = True
                    continue

                if solve_ready_current:
                    recoverable_final = recoverable_answer

                reasoning_verdict = {"passed": True, "missing": [], "explanation": ""}
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                ):
                    reasoning_verdict = await self._stepwise_reasoning_verdict(
                        reasoning,
                        execution_log,
                    )
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                    and not reasoning_verdict.get("passed", False)
                ):
                    repair_delivery_feedback_messages = [
                        self._stepwise_reasoning_correction_message(
                            execution_log,
                            verdict=reasoning_verdict,
                        )
                    ]
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": "reasoning_too_shallow",
                        "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                        "reasoning_preview": str(reasoning)[:1200],
                        "reasoning_gate_missing": list(reasoning_verdict.get("missing", [])),
                        "reasoning_gate_explanation": str(reasoning_verdict.get("explanation") or ""),
                    })
                    if step < MAX_SUB_REACT_STEPS:
                        conversation.append({
                            "role": "user",
                            "content": repair_delivery_feedback_messages[0],
                        })
                    if recoverable_answer is not None:
                        repair_delivery_mode = True
                    continue
                final_result = {
                    "answer": answer,
                    "answer_type": answer_type,
                    "confidence": decision.get("confidence"),
                    "reasoning": reasoning,
                    "execution_context": execution_context,
                    "control_context": control_context,
                    "execution_log": execution_log,
                    "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                    "slot_id": slot_id,
                    "sub_task_description": task_description,
                    "tools_hint": tools_hint,
                    "branch_metadata": branch_metadata,
                }
                break

            tool_name = decision.get("tool", "")
            tool_args = decision.get("args", {})

            if tool_name == "spawn_subagent":
                conversation.append({"role": "user", "content": "Sub-agents cannot spawn further sub-agents."})
                continue

            if tool_name not in turn_candidates:
                conversation.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' not available.",
                })
                continue
            runtime_incompatibility_msg = self._tool_runtime_incompatibility_message(
                tool_name,
                current_data=current_data,
                current_series_state=current_series_state,
            )
            if runtime_incompatibility_msg and step < MAX_SUB_REACT_STEPS:
                trace.agent_step("executor", "subagent_self_correction", {
                    "sub_agent_idx": idx,
                    "step": step,
                    "reason": "runtime_incompatible_tool",
                    "tool": tool_name,
                })
                conversation.append({
                    "role": "user",
                    "content": runtime_incompatibility_msg,
                })
                continue
            mismatch_msg = self._task_mismatched_reasoning_tool_message(
                task_type=plan["task_type"],
                tool_name=tool_name,
            )
            if mismatch_msg and step < MAX_SUB_REACT_STEPS:
                trace.agent_step("executor", "subagent_self_correction", {
                    "sub_agent_idx": idx,
                    "step": step,
                    "reason": "task_mismatched_reasoning_tool",
                    "tool": tool_name,
                })
                conversation.append({
                    "role": "user",
                    "content": mismatch_msg,
                })
                continue
            if (
                self._subagent_must_start_with_hinted_tool(
                    execution_log,
                    tools_hint,
                    tool_name,
                )
                and step < MAX_SUB_REACT_STEPS
            ):
                trace.agent_step("executor", "subagent_self_correction", {
                    "sub_agent_idx": idx,
                    "step": step,
                    "reason": "must_start_with_hinted_tool",
                    "tool": tool_name,
                    "tools_hint": tools_hint,
                })
                hinted_text = ", ".join(self._hinted_tools(tools_hint)) or str(tools_hint)
                conversation.append({
                    "role": "user",
                    "content": (
                        "Self-correction: before using any other tool, this sub-task must first "
                        f"call at least one main-agent-assigned path tool: {hinted_text}."
                    ),
                })
                continue

            if horizon and tool_name.endswith("_forecast"):
                tool_args["horizon"] = horizon

            if plan.get("domain"):
                if tool_name.startswith("llm_reasoning_"):
                    tool_args["domain"] = plan["domain"]
                else:
                    tool_args.setdefault("domain", plan["domain"])

            if tool_name.startswith("llm_reasoning_"):
                tool_args["context"] = self._build_llm_reasoning_context(
                    plan=plan,
                    execution_log=execution_log,
                    existing_context=tool_args.get("context", ""),
                )

            pre_tool_series_state = dict(current_series_state)
            result = await self._execute_tool(tool_name, tool_args, current_data, text_data, images)
            success = result is not None and result.success
            downstream_data = None
            observation_payload = (
                result.data if success else {"error": result.error if result else "not found"}
            )
            log_entry = {
                "tool": tool_name,
                "args": tool_args,
                "success": success,
                "output": result.data if success else None,
                "observation": observation_payload,
                "thought": thought,
            }
            redundancy_status = self._latest_redundancy_status(
                task_type=plan["task_type"],
                execution_log=execution_log + [log_entry],
                latest_tool_name=tool_name,
                previous_series_state=pre_tool_series_state,
                current_series_state=current_series_state,
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
            )
            trace.agent_step("executor", "subagent_tool_result", {
                "sub_agent_idx": idx,
                "step": step,
                "tool": tool_name,
                "args": tool_args,
                "success": success,
                "observation": observation_payload,
            })

            observation = self._build_tool_observation_message(
                step=step,
                tool_name=tool_name,
                tool_args=tool_args,
                success=success,
                observation_payload=observation_payload,
                task_type=plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
                sub_agent_idx=idx,
            )
            conversation.append({"role": "user", "content": observation})

            if success:
                downstream_data = self._downstream_series_from_result(tool_name, result.data)
                if downstream_data is not None:
                    current_data = downstream_data
                    current_series_state = self._artifact_state_from_result(
                        tool_name=tool_name,
                        result_data=result.data,
                        downstream_data=downstream_data,
                        previous_series_state=pre_tool_series_state,
                        tool_args=tool_args,
                    )
            latest_artifact_state = self._artifact_state_from_result(
                tool_name=tool_name,
                result_data=observation_payload,
                downstream_data=downstream_data,
                previous_series_state=pre_tool_series_state,
                tool_args=tool_args,
            )
            conversation.append({
                "role": "user",
                "content": self._build_state_update_message(
                    current_series_state=current_series_state,
                    latest_artifact_state=latest_artifact_state,
                    task_answer_status=self._task_answer_status(
                        task_type=plan["task_type"],
                        candidate=observation_payload,
                        source_tool=tool_name,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ),
                    branch_task_answer_status=self._latest_branch_task_answer_status(
                        task_type=plan["task_type"],
                        execution_log=execution_log + [log_entry],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ),
                    latest_redundancy_status=redundancy_status,
                ),
            })

            if redundancy_status.get("redundant") and step < MAX_SUB_REACT_STEPS:
                trace.agent_step("executor", "subagent_self_correction", {
                    "sub_agent_idx": idx,
                    "step": step,
                    "reason": "repeated_task_valid_answer_without_new_evidence",
                    "tool": tool_name,
                    "repeat_count": redundancy_status.get("repeat_count"),
                })
                conversation.append({
                    "role": "user",
                    "content": (
                        "Self-correction: the latest tool call repeated the same task-valid answer "
                        "from the same tool without changing the working artifact. This did not "
                        "add a new candidate for this branch. If you already have the needed "
                        "answer, finish with that exact earlier answer. Otherwise, use a different tool that "
                        "changes the evidence basis before calling the same reasoning tool again."
                    ),
                })
                continue

            execution_log.append(log_entry)

        if not final_result:
            terminal_reuse_verdict = {
                "passed": True,
                "execution_context_verdict": {"passed": True, "missing": [], "explanation": ""},
                "reasoning_verdict": {"passed": True, "missing": [], "explanation": ""},
            }
            terminal_feedback_messages: list[str] = []
            for terminal_attempt in range(MAX_TERMINAL_REPAIR_ATTEMPTS):
                terminal_decision = await self._request_terminal_path_review(
                    expected_type=plan["task_type"],
                    task_type=plan["task_type"],
                    task_description=task_description,
                    execution_log=execution_log,
                    trace_phase="subagent_terminal_self_correction",
                    feedback_messages=terminal_feedback_messages,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                    sub_agent_idx=idx,
                    recoverable_final=recoverable_answer or recoverable_final,
                    sub_task_description=task_description,
                    tools_hint=tools_hint,
                )
                terminal_answer = terminal_decision.get("answer")
                terminal_answer_type = terminal_decision.get("answer_type", "")
                terminal_reasoning = terminal_decision.get(
                    "reasoning",
                    terminal_decision.get("thought", ""),
                )
                terminal_failure_reason = ""
                terminal_valid = bool(terminal_decision.get("done", False))
                if terminal_decision.get("_parse_failed"):
                    terminal_valid = False
                    terminal_failure_reason = "invalid_json_or_missing_required_fields"
                elif not terminal_valid:
                    terminal_failure_reason = "not_finished"
                elif terminal_answer is None:
                    terminal_valid = False
                    terminal_failure_reason = "missing_answer"
                elif terminal_answer_type != plan["task_type"]:
                    terminal_valid = False
                    terminal_failure_reason = "wrong_answer_type"
                elif not self._is_valid_final_answer(
                    plan["task_type"],
                    terminal_answer,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ):
                    terminal_valid = False
                    terminal_failure_reason = "invalid_final_answer"
                elif not self._is_task_answer_backed_by_tool_output(
                    plan["task_type"],
                    terminal_answer,
                    execution_log,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ):
                    terminal_valid = False
                    terminal_failure_reason = "unsupported_answer"
                elif not self._subagent_made_any_tool_call(execution_log):
                    terminal_valid = False
                    terminal_failure_reason = "no_tool_use"
                elif not self._subagent_followed_tools_hint(execution_log, tools_hint):
                    terminal_valid = False
                    terminal_failure_reason = "tools_hint_not_followed"
                if not terminal_valid:
                    trace.agent_step("executor", "subagent_terminal_self_correction_invalid", {
                        "sub_agent_idx": idx,
                        "slot_id": slot_id,
                        "reason": terminal_failure_reason or "invalid_terminal_delivery",
                        "done": bool(terminal_decision.get("done", False)),
                        "answer_type": terminal_answer_type,
                        "parse_failed": bool(terminal_decision.get("_parse_failed")),
                    })
                    if terminal_attempt + 1 >= MAX_TERMINAL_REPAIR_ATTEMPTS:
                        break
                    terminal_feedback_messages = [
                        self._terminal_delivery_correction_message(
                            expected_type=plan["task_type"],
                            failure_reason=terminal_failure_reason,
                        )
                    ]
                    continue

                terminal_execution_context = self._normalize_execution_context(
                    terminal_decision.get("execution_context"),
                    execution_log=execution_log,
                )
                terminal_reasoning = self._normalize_reasoning_for_reuse(
                    terminal_reasoning,
                    execution_log,
                    terminal_execution_context,
                )
                terminal_recoverable = self._recoverable_final_record(
                    answer=terminal_answer,
                    answer_type=terminal_answer_type,
                    confidence=terminal_decision.get("confidence"),
                    reasoning=terminal_reasoning,
                    execution_context=terminal_execution_context,
                    control_context=self._normalize_control_context(
                        terminal_decision.get("control_context")
                    ),
                )
                try:
                    terminal_reuse_verdict = await self._final_payload_reusability_verdict(
                        mode=plan.get("_mode"),
                        reasoning=terminal_reasoning,
                        execution_context=terminal_execution_context,
                        execution_log=execution_log,
                    )
                except Exception as e:
                    self._attach_partial_result_to_exception(
                        e,
                        answer=terminal_answer,
                        answer_type=terminal_answer_type,
                        confidence=terminal_decision.get("confidence"),
                        reasoning=terminal_reasoning,
                        execution_context=terminal_execution_context,
                        control_context=self._normalize_control_context(
                            terminal_decision.get("control_context")
                        ),
                        execution_log=execution_log,
                    )
                    raise
                if terminal_reuse_verdict.get("passed", False):
                    final_result = {
                        "answer": terminal_answer,
                        "answer_type": terminal_answer_type,
                        "confidence": terminal_decision.get("confidence"),
                        "reasoning": terminal_reasoning,
                        "execution_context": terminal_execution_context,
                        "control_context": terminal_recoverable.get("control_context"),
                        "execution_log": execution_log,
                        "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                        "slot_id": slot_id,
                        "sub_task_description": task_description,
                        "tools_hint": tools_hint,
                        "branch_metadata": branch_metadata,
                    }
                    trace.agent_step("executor", "subagent_terminal_self_correction_accepted", {
                        "sub_agent_idx": idx,
                        "slot_id": slot_id,
                        "answer_type": terminal_answer_type,
                        "tool_chain": final_result.get("tool_chain", []),
                    })
                    break

                recoverable_final = terminal_recoverable
                trace.agent_step("executor", "subagent_terminal_self_correction_rejected", {
                    "sub_agent_idx": idx,
                    "slot_id": slot_id,
                    "reason": (
                        "execution_context_not_reusable"
                        if not terminal_reuse_verdict["execution_context_verdict"].get("passed", False)
                        else "reasoning_too_shallow"
                    ),
                    "execution_context_gate_missing": list(
                        terminal_reuse_verdict["execution_context_verdict"].get("missing", [])
                    ),
                    "execution_context_gate_explanation": str(
                        terminal_reuse_verdict["execution_context_verdict"].get("explanation") or ""
                    ),
                    "reasoning_gate_missing": list(
                        terminal_reuse_verdict["reasoning_verdict"].get("missing", [])
                    ),
                    "reasoning_gate_explanation": str(
                        terminal_reuse_verdict["reasoning_verdict"].get("explanation") or ""
                    ),
                })
                if terminal_attempt + 1 >= MAX_TERMINAL_REPAIR_ATTEMPTS:
                    break
                terminal_feedback_messages = self._terminal_payload_repair_messages(
                    execution_log,
                    terminal_reuse_verdict,
                )

            if not final_result and recoverable_final is not None:
                final_result = {
                    "answer": recoverable_final.get("answer"),
                    "answer_type": str(recoverable_final.get("answer_type") or ""),
                    "confidence": recoverable_final.get("confidence"),
                    "reasoning": str(recoverable_final.get("reasoning") or ""),
                    "execution_context": self._normalize_execution_context(
                        recoverable_final.get("execution_context"),
                        execution_log=execution_log,
                    ),
                    "control_context": self._normalize_control_context(
                        recoverable_final.get("control_context")
                    ),
                    "execution_log": execution_log,
                    "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                    "slot_id": slot_id,
                    "sub_task_description": task_description,
                    "tools_hint": tools_hint,
                    "branch_metadata": branch_metadata,
                }
                trace.agent_step("executor", "subagent_recoverable_final_fallback", {
                    "sub_agent_idx": idx,
                    "slot_id": slot_id,
                    "answer_type": final_result.get("answer_type"),
                    "steps": len(execution_log),
                })

        if not final_result:
            trace.agent_step("executor", "subagent_incomplete", {
                "sub_agent_idx": idx,
                "slot_id": slot_id,
                "sub_task_description": task_description,
                "tools_hint": tools_hint,
                "reason": "max_steps_exhausted_without_final_answer",
                "steps": MAX_SUB_REACT_STEPS,
                "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
            })
            raise MaxStepsExhaustedError(
                agent_kind="subagent",
                max_steps=MAX_SUB_REACT_STEPS,
                sub_agent_idx=idx,
                partial_result={
                    "execution_log": list(execution_log),
                    "tool_chain": [e.get("tool", "") for e in execution_log if e.get("tool")],
                    "slot_id": slot_id,
                    "sub_task_description": task_description,
                    "tools_hint": tools_hint,
                    "branch_metadata": branch_metadata,
                    "failure_reason": "max_steps_exhausted_without_final_answer",
                },
            )

        logger.info(f"[Executor] Sub-agent {idx} done: {final_result.get('answer_type', '?')}")
        trace.agent_step("executor", "subagent_complete", {
            "sub_agent_idx": idx,
            "slot_id": slot_id,
            "sub_task_description": task_description,
            "tools_hint": tools_hint,
            "branch_metadata": branch_metadata,
            "answer_type": final_result.get("answer_type", ""),
            "answer_preview": str(final_result.get("answer", ""))[:2000],
            "tool_chain": final_result.get("tool_chain", []),
            "steps": len(execution_log),
        })
        return final_result

    def _build_subagent_candidates(
        self,
        idx: int,
        task_description: str,
        tools_hint: Any,
        plan: dict[str, Any],
        envelope: TaskEnvelope,
        *,
        slot_id: int | None = None,
        slot_spec: dict[str, Any] | None = None,
    ) -> list[str]:
        blocked_tools = {
            "spawn_subagent",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
        }
        base_candidates = [c for c in plan["candidates"] if c not in blocked_tools]
        if plan.get("_mode") != "train":
            return base_candidates

        if slot_spec is not None:
            visible = [
                name for name in (slot_spec.get("visible_candidates") or [])
                if name in base_candidates
            ]
            dropped = [name for name in base_candidates if name not in visible]
            trace.agent_step("executor", "subagent_tool_dropout", {
                "sub_agent_idx": idx,
                "slot_id": slot_id,
                "scope": str(plan.get("_tool_dropout_scope") or ""),
                "never_drop": sorted(slot_spec.get("never_drop") or []),
                "visible_candidates": visible,
                "dropped_candidates": dropped,
                "keep_probabilities": {
                    k: round(v, 4)
                    for k, v in (slot_spec.get("keep_probabilities") or {}).items()
                },
                "source": "precomputed_slot",
            })
            return visible

        scope = str(plan.get("_tool_dropout_scope") or "").strip()
        scope_counts = plan.get("_tool_dropout_counts") or {}
        if not scope or not isinstance(scope_counts, dict) or not base_candidates:
            return base_candidates

        hinted_tools = {
            tool for tool in self._hinted_tools(tools_hint)
            if tool in base_candidates
        }
        never_drop = {
            tool_name
            for tool_name in base_candidates
            if tool_name == self._task_llm_reasoning_tool(plan.get("task_type", ""))
        }
        never_drop.update(hinted_tools)

        rng = random.Random(f"{envelope.trace_id}:{scope}:{idx}:{task_description}")
        visible, keep_probs = generate_dropout_candidates(
            base_candidates,
            scope_counts,
            never_drop=never_drop,
            rng=rng,
        )
        dropped = [name for name in base_candidates if name not in visible]
        trace.agent_step("executor", "subagent_tool_dropout", {
            "sub_agent_idx": idx,
            "scope": scope,
            "never_drop": sorted(never_drop),
            "visible_candidates": visible,
            "dropped_candidates": dropped,
            "keep_probabilities": {k: round(v, 4) for k, v in keep_probs.items()},
        })
        return visible

    @staticmethod
    def _build_learning_summary_contract_section() -> str:
        return (
            '- In training mode, the main agent must finish with `answer_type` exactly `"learning_summary"`.\n'
            "- This learning-summary contract applies only on the final completion turn.\n"
            "- If the current turn is a tool call (including `spawn_subagent` or `evaluate_*`), do NOT include any `learning_summary`, winner selection, or final-answer payload in the same reply.\n"
            "- A placeholder `learning_summary` is invalid. Do not finish with text like 'pending', 'waiting for sub-agent results', 'need evaluation first', or any other not-yet-earned summary.\n"
            "- Only return `learning_summary` after actual candidate outputs already exist and you can name the winning candidate path plus decisive evidence.\n"
            "- `answer` must be a JSON object.\n"
            "- Required keys inside `answer`: `insight`, `recommendation`.\n"
            "- `insight` must be a short non-empty string that names the winning candidate path/tool chain and the decisive evidence that made it stronger than the compared candidates.\n"
            "- `recommendation` must be a short non-empty string stating one reusable next-time lesson tied to the same kind of evidence pattern; if the selected winner is incorrect overall, state a cautionary failure pattern instead.\n"
            "- `recommendation` is not the task prediction, not a forecast array, and not an indicator JSON object.\n"
            "- `reasoning` must be the numbered long-form version of the same execution-side evidence recorded in `execution_context.public_reasoning`; reuse the same literal tool/path names instead of introducing looser aliases.\n"
            "- Final-answer fields must also include `execution_context` and `control_context`.\n"
            "- `execution_context` must contain only execution-side evidence: the substantive `tool_chain`, reusable `public_reasoning`, and a concise `evidence_summary`.\n"
            "- `control_context` is the only place for completion/comparison mechanics; use an empty string if there is nothing extra to record.\n"
            "- Do not write about framework/orchestration mechanics such as `spawn_subagent`, `evaluate_*`, `execution context`, `assigned path`, `before finishing`, or ground truth revelation.\n"
            "- Do not justify the summary with control-flow phrases such as `task-valid`, `must call a tool`, `latest instruction requires`, `branch comparison`, `learning mode`, or `spawning independent branches`.\n"
            "- Do not write about answer packaging mechanics such as re-wrapping, payload preservation, JSON repair, or returning an unchanged saved answer.\n"
            "- Even if your top-level work was mostly orchestration, summarize the real candidate approaches, their concrete tool chains, the decisive evidence, and why the winner was selected.\n"
            "- The framework will derive winner fields from the actual candidate traces, but you should still summarize which candidate won and why.\n"
            '- Exact legacy JSON finish shape if you are not using a tool call: '
            '`{"thought":"Reason: ... Action: finish with learning summary.","done":true,'
            '"answer":{"insight":"...","recommendation":"..."},"answer_type":"learning_summary",'
            '"confidence":0.90,"reasoning":"1. ...","execution_context":{"tool_chain":["..."],"public_reasoning":["..."],"evidence_summary":"..."},'
            '"control_context":{"completion_notes":""}}`\n'
        )

    @staticmethod
    def _build_indicator_contract_section(indicator_output_keys: list[str] | None) -> str:
        keys = [str(key).strip() for key in (indicator_output_keys or []) if str(key).strip()]
        if not keys:
            return ""
        return (
            f"- The final indicator JSON keys must be exactly: {', '.join(keys)}.\n"
            "- Do not rename them, do not append suffixes like `_forecast`, and do not wrap them under another key.\n"
        )

    @staticmethod
    def _build_tool_backed_answer_contract_section() -> str:
        return (
            "- Do not invent new task content in the final step.\n"
            "- The final task answer must exactly match a task-valid output already produced by a real tool in this branch.\n"
            "- If no tool has produced a task-valid answer yet, call another tool instead of finishing.\n"
        )

    @staticmethod
    def _build_task_boundary_section(
        task_type: str,
        *,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> str:
        task_type = str(task_type or "").strip()
        if task_type == "forecast":
            horizon_text = (
                f"the next {int(horizon)} future step(s)"
                if isinstance(horizon, int) and horizon > 0
                else "the required future horizon"
            )
            return (
                "- Core task: predict a future numeric time-series sequence from the provided history and optional multimodal context.\n"
                f"- Task-valid candidate output: one JSON numeric array covering {horizon_text}.\n"
                "- Not valid for this task: trend labels, indicator JSON objects, or QA-style prose answers.\n"
            )
        if task_type == "indicator":
            keys = [str(key).strip() for key in (indicator_output_keys or []) if str(key).strip()]
            if keys:
                keys_text = ", ".join(keys)
            else:
                keys_text = "the required indicator keys"
            return (
                "- Core task: predict derived indicator values from the provided series and optional multimodal context, not the raw future series itself.\n"
                f"- Task-valid candidate output: one JSON object with exactly these indicator keys: {keys_text}.\n"
                "- Not valid for this task: raw forecast arrays, trend labels, or free-form QA answers.\n"
            )
        if task_type == "classification":
            return (
                "- Core task: classify the relevant trend/state label from the provided series and optional multimodal context.\n"
                "- Task-valid candidate output: one label string, or a minimal JSON object that contains that label.\n"
                "- Not valid for this task: forecast arrays, indicator JSON objects, or long explanatory QA prose in place of the label.\n"
            )
        if task_type == "qa":
            return (
                "- Core task: answer the question grounded in the provided series and optional context.\n"
                "- Task-valid candidate output: the final answer text itself, or the final option label if the question is multiple-choice.\n"
                "- Not valid for this task: forecast arrays, indicator JSON objects, or trend/correlation labels unless the question explicitly asks for them.\n"
            )
        if task_type == "correlation":
            return (
                "- Core task: judge the correlation/relationship label requested by the task from the provided evidence.\n"
                "- Task-valid candidate output: one correlation label string, or a minimal JSON object containing that label.\n"
                "- Not valid for this task: forecast arrays, indicator JSON objects, or long QA-style answers unrelated to the requested label.\n"
            )
        return ""

    @classmethod
    def _build_main_output_contract_section(
        cls,
        *,
        task_type: str,
        horizon: int | None = None,
        is_train: bool,
        defer_learning_record: bool,
        indicator_output_keys: list[str] | None,
        allow_finish_current_turn: bool,
    ) -> str:
        parts: list[str] = []
        if not allow_finish_current_turn:
            parts.append(
                "- This is not a completion turn.\n"
                "- Do NOT include any final-answer payload, `learning_summary`, winner selection, or backup `done=true` object in this reply.\n"
                "- Return only the current-turn tool-call JSON.\n"
            )
        elif is_train and not defer_learning_record:
            parts.append(cls._build_learning_summary_contract_section().strip())
        else:
            parts.append(
                cls._build_subagent_final_answer_contract_section(
                    task_type,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
                ).strip()
            )
        return "\n\n".join(parts) + ("\n" if parts else "")

    @staticmethod
    def _is_valid_learning_summary_answer(answer: Any) -> bool:
        if not isinstance(answer, dict):
            return False
        insight = answer.get("insight")
        recommendation = answer.get("recommendation")
        return (
            isinstance(insight, str) and bool(insight.strip())
            and isinstance(recommendation, str) and bool(recommendation.strip())
        )

    @staticmethod
    def _learning_summary_validation_message(answer: Any) -> str:
        if not isinstance(answer, dict):
            return (
                "Self-correction: a training `learning_summary` answer must be a JSON object "
                "with string fields `insight` and `recommendation`. Use this exact shape: "
                '`{"answer":{"insight":"...","recommendation":"..."},"answer_type":"learning_summary","reasoning":"1. ...","execution_context":{"tool_chain":["..."],"public_reasoning":["..."],"evidence_summary":"..."},"control_context":{"completion_notes":""}}`'
            )

        insight = answer.get("insight")
        recommendation = answer.get("recommendation")
        issues: list[str] = []
        if not isinstance(insight, str) or not insight.strip():
            issues.append("`insight` must be a non-empty string")
        if not isinstance(recommendation, str) or not recommendation.strip():
            issues.append(
                "`recommendation` must be a non-empty string that states the reusable lesson or cautionary failure pattern, "
                "not the task prediction object or numeric payload"
            )
        if not issues:
            issues.append("use non-empty string fields for both `insight` and `recommendation`")
        return (
            "Self-correction: a training `learning_summary` answer must be a JSON object with "
            + "; ".join(issues)
            + '. Use this exact shape: `{"answer":{"insight":"...","recommendation":"..."},"answer_type":"learning_summary","reasoning":"1. ...","execution_context":{"tool_chain":["..."],"public_reasoning":["..."],"evidence_summary":"..."},"control_context":{"completion_notes":""}}`. '
            + "Keep it concise, base it on the actually explored candidate paths, name the winning path/tool chain and decisive evidence, and do not mention framework wording such as spawn_subagent, evaluate_*, execution context, assigned path, before finishing, or ground truth."
        )

    @staticmethod
    def _invalid_json_self_correction_message(parse_thought: str) -> str:
        base = (
            "Self-correction: your reply did not produce exactly one valid decision action. "
            "Emit exactly one action for this turn: either one tool call or one final answer."
        )
        lowered = str(parse_thought or "").strip().lower()
        if lowered == "invalid_decision_tool_call_count":
            return (
                base
                + " The control channel must contain exactly one decision tool call."
            )
        if lowered == "invalid_decision_tool_call_payload":
            return (
                base
                + " The decision payload was malformed. Re-emit one well-formed action."
            )
        if lowered == "invalid_final_answer_payload_json":
            return (
                base
                + " Final answers must serialize the task payload as valid JSON in `answer_json`."
            )
        if lowered == "multiple_json_objects_in_one_reply":
            return (
                base
                + " Do not append a second JSON object in the same reply. "
                + "Choose exactly one action type this turn. Do not mix tool-call fields "
                + "(`tool`, `args`) with final-answer fields (`answer`, `answer_type`, "
                + "`confidence`, `reasoning`, `execution_context`, `control_context`) in the same reply."
            )
        if lowered == "json_followed_by_pseudo_tool_transcript":
            return (
                base
                + " Do not simulate tool execution transcripts such as `assistant to=tool`, "
                "`tool result=...`, markdown code fences, or any pseudo-log after the JSON. "
                + "Choose exactly one action type this turn, and do not mix tool-call fields "
                + "with final-answer fields in the same reply."
            )
        return (
            base
            + " If you are calling `evaluate_batch_against_gt` after recent sub-agent runs, prefer "
            "the compact form with only `task_type` and let the framework auto-fill `predictions`."
        )

    @staticmethod
    def _build_subagent_tools_hint_section(
        tools_hint: Any,
        candidates: list[str],
    ) -> str:
        hinted = [
            tool for tool in ExecutorAgent._hinted_tools(tools_hint)
            if tool in candidates
        ]
        if not hinted:
            return ""
        return (
            "Assigned Exploration Path:\n"
            f"- Prioritize this path: {', '.join(hinted)}.\n"
            "- Before returning `done=true`, you must have actually called at least one tool from this path.\n"
        )

    @staticmethod
    def _build_subagent_final_answer_contract_section(
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> str:
        task_type = str(task_type).strip()
        shared = ExecutorAgent._build_tool_backed_answer_contract_section()
        if task_type == "forecast":
            size_hint = (
                f"at least {int(horizon)} numbers"
                if isinstance(horizon, int) and horizon > 0
                else "the required forecast horizon"
            )
            return shared + (
                f"- Return ONLY a JSON numeric array with {size_hint}.\n"
                '- Example: [1.234, 1.567, 1.890]\n'
                "- Do NOT return prose like `Predicted Temperatures: ...`.\n"
            )
        if task_type == "indicator":
            keys = [str(key).strip() for key in (indicator_output_keys or []) if str(key).strip()]
            if keys:
                example_values = ", ".join("1.234" for _ in range(3))
                example = "{%s}" % ", ".join(
                    f'"{key}": [{example_values}]' for key in keys
                )
                keys_text = ", ".join(keys)
            else:
                example = '{"indicator_name": [1.234, 1.567, 1.890]}'
                keys_text = "the required indicator keys"
            return shared + (
                f"- Return ONLY a JSON object with exactly these keys: {keys_text}.\n"
                f"- Example: {example}\n"
                "- Do NOT return prose like `Predicted Prices: ...`.\n"
                "- Do not rename keys, do not append suffixes like `_forecast`, and do not wrap them under another key.\n"
            )
        if task_type in {"classification", "correlation"}:
            return shared + (
                "- Return a single label string or a minimal JSON object containing that label.\n"
                '- Example: "decreasing"\n'
            )
        if task_type == "qa":
            return shared + (
                "- Return the final answer text directly, without wrapping it in commentary about the process.\n"
            )
        return ""

    @staticmethod
    def _task_llm_reasoning_tool(task_type: str) -> str:
        mapping = {
            "forecast": "llm_reasoning_forecast",
            "indicator": "llm_reasoning_indicator",
            "classification": "llm_reasoning_classification",
            "qa": "llm_reasoning_qa",
            "correlation": "llm_reasoning_correlation",
        }
        return mapping.get(str(task_type).strip(), "")

    @classmethod
    def _allowed_llm_reasoning_tools_for_task(cls, task_type: str) -> set[str]:
        primary = cls._task_llm_reasoning_tool(task_type)
        allowed = {primary} if primary else set()
        return {tool for tool in allowed if tool}

    # ── Tool execution ────────────────────────────────────────────────

    _FORCE_INPROCESS = frozenset({
        "timesfm2_forecast", "chronos2_forecast", "sundial_base_128m_forecast", "moirai1_1_r_base_forecast",
        "aurora_forecast",
        "evaluate_against_gt",
        "evaluate_batch_against_gt",
        "batch_evaluate_against_gt",
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
        head-200 + tail-200. Dict values are included in full (no char cap).
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
                parts.append(f"{k}={v_str}")
            return ", ".join(parts)
        return str(data)

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
    def _is_valid_final_answer(
        task_type: str,
        answer: Any,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        valid, _ = ExecutorAgent._validate_final_answer(
            task_type,
            answer,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        return valid

    @staticmethod
    def _validate_final_answer(
        task_type: str,
        answer: Any,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> tuple[bool, str]:
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
            if isinstance(answer, dict) and "answer" in answer and isinstance(answer["answer"], dict):
                answer = answer["answer"]
            expected_keys = [
                str(key).strip()
                for key in (indicator_output_keys or [])
                if str(key).strip()
            ]
            if expected_keys and isinstance(answer, dict):
                raw_keys = [str(key).strip() for key in answer.keys() if str(key).strip()]
                if raw_keys != expected_keys:
                    return (
                        False,
                        f"indicator answer keys must be exactly {expected_keys}, got {raw_keys}",
                    )
            answer = ExecutorAgent._canonicalize_indicator_answer(
                answer,
                indicator_output_keys=indicator_output_keys,
            )
            if not isinstance(answer, dict) or not answer:
                return False, "indicator answer is not a non-empty JSON object"
            if expected_keys:
                actual_keys = [str(key).strip() for key in answer.keys() if str(key).strip()]
                if actual_keys != expected_keys:
                    return (
                        False,
                        f"indicator answer keys must be exactly {expected_keys}, got {actual_keys}",
                    )
            numeric_found = False
            for value in answer.values():
                try:
                    float(value)
                    numeric_found = True
                    continue
                except (ValueError, TypeError):
                    pass
                try:
                    arr = np.asarray(value, dtype=float).ravel()
                except (ValueError, TypeError):
                    return False, "indicator answer contains non-numeric value"
                if arr.size == 0:
                    return False, "indicator answer contains empty numeric array"
                if not np.isfinite(arr).any():
                    return False, "indicator answer contains no finite numeric values"
                numeric_found = True
            return numeric_found, "" if numeric_found else "indicator answer has no numeric values"

        if task_type in {"classification", "correlation"}:
            if isinstance(answer, str):
                if not answer.strip():
                    return False, "classification/correlation answer is empty string"
                if not ExecutorAgent._is_label_like_text(answer):
                    return False, "classification/correlation answer must be a short label-like string"
                return True, ""
            if isinstance(answer, dict):
                for key in ("trend", "correlation", "answer", "label", "prediction"):
                    if key in answer and ExecutorAgent._is_label_like_text(answer[key]):
                        return True, ""
                return False, "classification/correlation JSON has no non-empty label field"
            return False, "classification/correlation answer is neither string nor dict"

        if task_type == "qa":
            if isinstance(answer, str):
                return bool(answer.strip()), "" if answer.strip() else "qa answer is empty string"
            if isinstance(answer, dict):
                for key in ("answer", "text", "response", "summary", "prediction", "label"):
                    value = answer.get(key)
                    if isinstance(value, str) and value.strip():
                        return True, ""
                return False, "qa answer dict has no textual answer field"
            return False, "qa answer is neither string nor dict"

        return True, ""

    @staticmethod
    def _normalize_text_answer(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return " ".join(text.split())

    @classmethod
    def _normalize_answer_for_comparison(
        cls,
        task_type: str,
        value: Any,
        *,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> Any | None:
        if task_type == "forecast":
            candidate = value
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    return None
            if isinstance(candidate, dict):
                for key in ("answer", "forecast", "point_forecast", "prediction", "values", "predictions", "data"):
                    if key in candidate:
                        candidate = candidate[key]
                        break
            try:
                arr = np.asarray(candidate, dtype=float).ravel()
            except (ValueError, TypeError):
                return None
            if arr.size == 0:
                return None
            if horizon is not None:
                if arr.size < horizon:
                    return None
                arr = arr[:horizon]
            return arr

        if task_type == "indicator":
            candidate = value
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    return None
            if isinstance(candidate, dict) and "answer" in candidate:
                candidate = candidate["answer"]
            candidate = cls._canonicalize_indicator_answer(
                candidate,
                indicator_output_keys=indicator_output_keys,
            )
            if not isinstance(candidate, dict) or not candidate:
                return None
            expected_keys = [
                str(key).strip()
                for key in (indicator_output_keys or [])
                if str(key).strip()
            ]
            actual_keys = [str(key).strip() for key in candidate.keys() if str(key).strip()]
            if expected_keys and actual_keys != expected_keys:
                return None
            normalized: dict[str, Any] = {}
            for key in actual_keys:
                raw_value = candidate.get(key)
                try:
                    scalar = float(raw_value)
                except (ValueError, TypeError):
                    try:
                        arr = np.asarray(raw_value, dtype=float).ravel()
                    except (ValueError, TypeError):
                        return None
                    if arr.size == 0:
                        return None
                    normalized[key] = arr
                else:
                    normalized[key] = scalar
            return normalized

        if task_type in {"classification", "correlation"}:
            candidate = value
            if isinstance(candidate, dict):
                for key in ("answer", "label", "prediction", "trend", "direction", "correlation", "classification"):
                    if key in candidate:
                        candidate = candidate[key]
                        break
            normalized = cls._normalize_text_answer(candidate)
            if normalized is None or not cls._is_label_like_text(normalized):
                return None
            return normalized.lower()

        if task_type == "qa":
            candidate = value
            if isinstance(candidate, dict):
                for key in ("answer", "text", "response", "summary", "prediction"):
                    if key in candidate:
                        candidate = candidate[key]
                        break
            return cls._normalize_text_answer(candidate)

        return None

    @classmethod
    def _extract_task_answer_candidate(
        cls,
        task_type: str,
        value: Any,
    ) -> Any:
        task_type = str(task_type or "").strip()
        candidate = value

        if task_type in {"forecast", "indicator"}:
            if isinstance(candidate, dict) and "answer" in candidate:
                return candidate["answer"]
            return candidate

        if task_type in {"classification", "correlation"} and isinstance(candidate, dict):
            for key in ("answer", "label", "prediction", "trend", "direction", "correlation", "classification"):
                if key in candidate:
                    return candidate[key]
            return candidate

        if task_type == "qa" and isinstance(candidate, dict):
            for key in ("answer", "text", "response", "summary", "prediction"):
                if key in candidate:
                    return candidate[key]
            return candidate

        return candidate

    @classmethod
    def _answers_match_semantically(
        cls,
        task_type: str,
        left: Any,
        right: Any,
        *,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        left_norm = cls._normalize_answer_for_comparison(
            task_type,
            left,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        right_norm = cls._normalize_answer_for_comparison(
            task_type,
            right,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if left_norm is None or right_norm is None:
            return False
        if task_type == "forecast":
            try:
                left_arr = np.asarray(left_norm, dtype=float).ravel()
                right_arr = np.asarray(right_norm, dtype=float).ravel()
            except (ValueError, TypeError):
                return False
            if left_arr.shape != right_arr.shape:
                return False
            return bool(np.allclose(left_arr, right_arr, atol=5e-3, rtol=1e-5))
        if task_type == "indicator":
            if set(left_norm.keys()) != set(right_norm.keys()):
                return False
            for key in left_norm:
                left_value = left_norm[key]
                right_value = right_norm[key]
                if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
                    try:
                        left_arr = np.asarray(left_value, dtype=float).ravel()
                        right_arr = np.asarray(right_value, dtype=float).ravel()
                    except (ValueError, TypeError):
                        return False
                    if left_arr.shape != right_arr.shape:
                        return False
                    if not np.allclose(left_arr, right_arr, atol=5e-3, rtol=1e-5):
                        return False
                else:
                    try:
                        if abs(float(left_value) - float(right_value)) > 5e-3:
                            return False
                    except (ValueError, TypeError):
                        return False
            return True
        return left_norm == right_norm

    @classmethod
    def _is_task_answer_backed_by_tool_output(
        cls,
        task_type: str,
        answer: Any,
        execution_log: list[dict[str, Any]],
        *,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        for entry in execution_log:
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name:
                continue
            if tool_name == "spawn_subagent":
                for item in entry.get("output") or []:
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    sub_log = result.get("execution_log") or []
                    if cls._is_task_answer_backed_by_tool_output(
                        task_type,
                        answer,
                        sub_log,
                        horizon=horizon,
                        indicator_output_keys=indicator_output_keys,
                    ):
                        return True
                continue
            if tool_name in _ORCHESTRATION_ONLY_TOOLS:
                continue
            if not bool(entry.get("success")):
                continue
            output = entry.get("output")
            if cls._answers_match_semantically(
                task_type,
                answer,
                output,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                return True
        return False

    @classmethod
    def _task_answer_status(
        cls,
        *,
        task_type: str | None,
        candidate: Any,
        source_tool: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        task_type_text = str(task_type or "").strip()
        if not task_type_text:
            return {"available": False}

        answer_candidate = cls._extract_task_answer_candidate(task_type_text, candidate)
        if not cls._is_valid_final_answer(
            task_type_text,
            answer_candidate,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
            return {"available": False}

        preview_value = cls._round_prompt_numbers(answer_candidate)
        try:
            preview_text = json.dumps(preview_value, ensure_ascii=False, default=str)
        except TypeError:
            preview_text = str(preview_value)

        status = {
            "available": True,
            "task_type": task_type_text,
            "source_tool": str(source_tool or "").strip(),
            "note": "A task-valid answer is already available from a real tool result. You may finish by returning that exact answer unchanged.",
            "answer_preview": preview_text[:240],
        }
        return status

    @classmethod
    def _latest_branch_task_answer_status(
        cls,
        *,
        task_type: str | None,
        execution_log: list[dict[str, Any]],
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        task_type_text = str(task_type or "").strip()
        if not task_type_text:
            return {"available": False}

        for entry in reversed(execution_log):
            if not isinstance(entry, dict) or not entry.get("success"):
                continue
            status = cls._task_answer_status(
                task_type=task_type_text,
                candidate=entry.get("output"),
                source_tool=entry.get("tool"),
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
            if status.get("available"):
                branch_status = dict(status)
                branch_status["availability_scope"] = "branch"
                branch_status["note"] = (
                    "A task-valid answer remains available from an earlier real tool result in "
                    "this branch. You may finish by returning that exact answer unchanged."
                )
                return branch_status
        return {"available": False}

    @staticmethod
    def _state_signature(state: dict[str, Any] | None) -> str:
        if not isinstance(state, dict):
            return ""
        try:
            return json.dumps(
                ExecutorAgent._round_prompt_numbers(state),
                ensure_ascii=False,
                default=str,
                sort_keys=True,
            )
        except TypeError:
            return str(state)

    @classmethod
    def _latest_redundancy_status(
        cls,
        *,
        task_type: str | None,
        execution_log: list[dict[str, Any]],
        latest_tool_name: str | None,
        previous_series_state: dict[str, Any] | None,
        current_series_state: dict[str, Any] | None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        task_type_text = str(task_type or "").strip()
        latest_tool = str(latest_tool_name or "").strip()
        if not task_type_text or not latest_tool or len(execution_log) < 2:
            return {"redundant": False}
        if cls._state_signature(previous_series_state) != cls._state_signature(current_series_state):
            return {"redundant": False}

        latest_entry = execution_log[-1]
        if not isinstance(latest_entry, dict) or not latest_entry.get("success"):
            return {"redundant": False}
        if str(latest_entry.get("tool") or "").strip() != latest_tool:
            return {"redundant": False}

        latest_answer = latest_entry.get("output")
        latest_status = cls._task_answer_status(
            task_type=task_type_text,
            candidate=latest_answer,
            source_tool=latest_tool,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if not latest_status.get("available"):
            return {"redundant": False}

        repeat_count = 1
        for earlier in reversed(execution_log[:-1]):
            if not isinstance(earlier, dict) or not earlier.get("success"):
                continue
            if str(earlier.get("tool") or "").strip() != latest_tool:
                continue
            earlier_status = cls._task_answer_status(
                task_type=task_type_text,
                candidate=earlier.get("output"),
                source_tool=latest_tool,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
            if not earlier_status.get("available"):
                continue
            if not cls._answers_match_semantically(
                task_type_text,
                latest_answer,
                earlier.get("output"),
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                continue
            repeat_count += 1

        if repeat_count <= 1:
            return {"redundant": False}

        redundancy = dict(latest_status)
        redundancy["redundant"] = True
        redundancy["repeat_count"] = repeat_count
        redundancy["note"] = (
            "The latest tool call repeated the same task-valid answer from the same real tool "
            "while the current working artifact did not change. This did not add a new "
            "candidate or new evidence. Either finish with the existing answer, or "
            "switch to a tool that changes the evidence basis before asking for the label again."
        )
        return redundancy

    @staticmethod
    def _thought_schema() -> dict[str, Any]:
        return {
            "type": "string",
            "description": "Must contain both labeled parts in this order: `Reason:` then `Action:`.",
            "pattern": r"^Reason:[\s\S]*Action:[\s\S]*$",
        }

    @staticmethod
    def _execution_context_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_chain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The substantive tool chain that directly produced this answer. "
                        "Do not include orchestration-only tools."
                    ),
                },
                "public_reasoning": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Reusable evidence sentences grounded in actual tool outputs or data characteristics. "
                        "Do not mention branch control, completion mechanics, or packaging."
                    ),
                },
                "evidence_summary": {
                    "type": "string",
                    "description": "One concise execution-side summary of why the answer is supported.",
                },
            },
            "required": ["tool_chain", "public_reasoning", "evidence_summary"],
            "additionalProperties": False,
        }

    @staticmethod
    def _control_context_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "completion_notes": {
                    "type": "string",
                    "description": (
                        "Flow-control or completion notes reserved for controller/trace use. "
                        "Use an empty string when there is nothing extra to note."
                    ),
                },
            },
            "required": ["completion_notes"],
            "additionalProperties": False,
        }

    @classmethod
    def _normalize_execution_context(
        cls,
        payload: Any,
        *,
        execution_log: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        observed_tools = (
            set(cls._available_substantive_tool_names(execution_log))
            if execution_log is not None
            else None
        )
        tool_chain: list[str] = []
        seen_tools: set[str] = set()
        for raw_tool in payload.get("tool_chain") or []:
            tool = cls._normalize_execution_context_tool_chain_entry(
                raw_tool,
                observed_tools=observed_tools,
            )
            if not tool:
                continue
            if tool in seen_tools:
                continue
            seen_tools.add(tool)
            tool_chain.append(tool)
        if execution_log:
            primary_chain = cls._reasoning_substantive_tool_chain(execution_log)
            plain_tools = [tool for tool in tool_chain if "->" not in tool]
            path_labels = [tool for tool in tool_chain if "->" in tool]
            if path_labels and plain_tools:
                tool_chain = path_labels
            elif (
                len(primary_chain) >= 2
                and len(plain_tools) > len(primary_chain)
                and all(tool in plain_tools for tool in primary_chain)
            ):
                tool_chain = list(primary_chain)

        public_reasoning = [
            cls._sanitize_execution_context_text(str(item).strip())
            for item in payload.get("public_reasoning") or []
            if str(item).strip()
        ]
        public_reasoning = [item for item in public_reasoning if item]
        evidence_summary = cls._sanitize_execution_context_text(
            str(payload.get("evidence_summary") or "").strip()
        )
        if not evidence_summary and public_reasoning:
            evidence_summary = "; ".join(public_reasoning[:2])[:400].strip()
        if execution_log:
            tool_chain = cls._canonicalize_execution_context_tool_chain(
                tool_chain,
                public_reasoning=public_reasoning,
                evidence_summary=evidence_summary,
                execution_log=execution_log,
            )
        return {
            "tool_chain": tool_chain,
            "public_reasoning": public_reasoning,
            "evidence_summary": evidence_summary,
        }

    @staticmethod
    def _sanitize_execution_context_text(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        sentence_re = re.compile(r"(?<=[.!?])\s+|;\s+")
        replacements = [
            (r"(?i)\btask-valid\b", ""),
            (r"(?i)\bexact task-valid format\b", "exact format"),
            (r"(?i)\breturn(?:ed|ing)? unchanged\b", ""),
            (r"(?i)\bused unchanged\b", ""),
            (r"(?i)\bprovided reuse answer\b", ""),
            (r"(?i)\breuse answer\b", ""),
            (r"(?i)\baccepted (?:result|answer|forecast|payload)\b", "result"),
            (r"(?i)\bwinning (?:branch|candidate|path)\b", "path"),
            (r"(?i)\bbest candidate among explored branches\b", "path"),
            (r"(?i)\bexplored branches\b", "tested paths"),
            (r"(?i)\brepair-only\b", ""),
            (r"(?i)\bfinish-only\b", ""),
            (r"(?i)\bfinal JSON delivery\b", ""),
            (r"(?i)\bfinal answer\b", ""),
            (r"(?i)\bthe user requested\b", ""),
            (r"(?i)\bexecution context\b", ""),
            (r"(?i)\bcompletion mechanics\b", ""),
            (r"(?i)\bpayload wrapping\b", ""),
            (r"(?i)\bcontroller-side\b", ""),
            (r"(?i)\bcontroller\b", ""),
        ]
        hard_control_markers = (
            "ground truth",
            "ground_truth",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
            "spawn_subagent",
            "json repair",
            "before finishing",
            " repair ",
            "completion contract",
            "branch instructions",
            "return the corrected final answer",
        )
        kept_parts: list[str] = []
        for part in sentence_re.split(cleaned):
            chunk = str(part or "").strip()
            if not chunk:
                continue
            for pattern, repl in replacements:
                chunk = re.sub(pattern, repl, chunk)
            lowered = f" {chunk.lower()} "
            if any(marker in lowered for marker in hard_control_markers):
                continue
            chunk = re.sub(r"\s+", " ", chunk)
            chunk = re.sub(r"\s+([,.;:])", r"\1", chunk)
            chunk = chunk.strip(" -|,;:")
            if re.match(r"(?i)^(use|return|wrap|resend|copy)\b", chunk):
                continue
            if chunk:
                kept_parts.append(chunk)
        if not kept_parts:
            return ""
        return ". ".join(kept_parts).strip().rstrip(".")

    @staticmethod
    def _normalize_control_context(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "completion_notes": str(payload.get("completion_notes") or "").strip(),
        }

    @staticmethod
    def _normalize_execution_context_tool_chain_entry(
        raw_tool: Any,
        *,
        observed_tools: set[str] | None,
    ) -> str:
        tool = str(raw_tool or "").strip()
        if not tool or tool in _ORCHESTRATION_ONLY_TOOLS:
            return ""
        if observed_tools is None:
            return tool
        if tool in observed_tools:
            return tool
        if "->" not in tool:
            return ""
        segments = [segment.strip() for segment in tool.split("->")]
        if len(segments) < 2:
            return ""
        if any(
            not segment
            or segment in _ORCHESTRATION_ONLY_TOOLS
            or segment not in observed_tools
            for segment in segments
        ):
            return ""
        return " -> ".join(segments)

    @staticmethod
    def _normalize_label_text(text: Any) -> str:
        normalized = str(text or "").strip()
        normalized = re.sub(r"\s*->\s*", " -> ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def _mentioned_candidate_path_labels(
        cls,
        text: str,
        candidate_labels: list[str],
    ) -> list[str]:
        haystack = cls._normalize_label_text(text).lower()
        if not haystack:
            return []
        matches: list[str] = []
        for label in candidate_labels:
            normalized_label = cls._normalize_label_text(label)
            if normalized_label and normalized_label.lower() in haystack:
                matches.append(normalized_label)
        return matches

    @classmethod
    def _mentioned_substantive_tools(
        cls,
        text: str,
        substantive_tools: list[str],
    ) -> list[str]:
        haystack = str(text or "").lower()
        if not haystack:
            return []
        mentioned: list[str] = []
        for tool in substantive_tools:
            aliases = cls._reasoning_tool_aliases(tool)
            if any(alias.lower() in haystack for alias in aliases):
                mentioned.append(tool)
        return mentioned

    @classmethod
    def _canonicalize_execution_context_tool_chain(
        cls,
        tool_chain: list[str],
        *,
        public_reasoning: list[str],
        evidence_summary: str,
        execution_log: list[dict[str, Any]],
    ) -> list[str]:
        candidate_labels = cls._reasoning_candidate_labels(execution_log)
        substantive_tools = cls._available_substantive_tool_names(execution_log)
        evidence_text = " ".join([*public_reasoning, evidence_summary]).strip()
        mentioned_labels = cls._mentioned_candidate_path_labels(evidence_text, candidate_labels)
        if mentioned_labels:
            return [mentioned_labels[0]]

        path_labels = [item for item in tool_chain if "->" in item]
        plain_tools = [item for item in tool_chain if "->" not in item]
        if path_labels:
            return path_labels

        primary_chain = cls._reasoning_substantive_tool_chain(execution_log)
        if len(primary_chain) >= 2 and all(tool in plain_tools for tool in primary_chain):
            return list(primary_chain)

        mentioned_tools = cls._mentioned_substantive_tools(evidence_text, substantive_tools)
        if mentioned_tools:
            filtered_plain = [tool for tool in plain_tools if tool in mentioned_tools]
            if filtered_plain:
                tool_chain = filtered_plain

        if (
            len(primary_chain) >= 2
            and len(tool_chain) > len(primary_chain)
            and all(tool in tool_chain for tool in primary_chain)
        ):
            return list(primary_chain)
        return tool_chain

    @classmethod
    def _selected_candidate_label_from_context(
        cls,
        execution_context: dict[str, Any],
        execution_log: list[dict[str, Any]],
    ) -> str:
        candidate_labels = cls._reasoning_candidate_labels(execution_log)
        if not candidate_labels:
            return ""
        tool_chain = [
            cls._normalize_label_text(item)
            for item in execution_context.get("tool_chain") or []
            if cls._normalize_label_text(item)
        ]
        for entry in tool_chain:
            for label in candidate_labels:
                normalized_label = cls._normalize_label_text(label)
                if entry == normalized_label:
                    return normalized_label
        joined = " ".join([
            *(
                str(item).strip()
                for item in execution_context.get("public_reasoning") or []
                if str(item).strip()
            ),
            str(execution_context.get("evidence_summary") or "").strip(),
        ])
        mentioned = cls._mentioned_candidate_path_labels(joined, candidate_labels)
        return mentioned[0] if mentioned else ""

    @classmethod
    def _normalize_reasoning_for_reuse(
        cls,
        reasoning: Any,
        execution_log: list[dict[str, Any]],
        execution_context: dict[str, Any],
    ) -> str:
        text = str(reasoning or "").strip()
        if not text:
            return ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        cleaned_lines: list[str] = []
        for raw_line in lines:
            line = raw_line
            line = re.sub(r"(?i)\btask-valid\b", "", line)
            line = re.sub(
                r"(?i)\b(spawn_subagent|evaluate_against_gt|evaluate_batch_against_gt|batch_evaluate_against_gt|execution context|control context)\b",
                "",
                line,
            )
            line = re.sub(r"\s+", " ", line)
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        normalized = "\n".join(cleaned_lines).strip()
        candidate_labels = cls._reasoning_candidate_labels(execution_log)
        selected_label = cls._selected_candidate_label_from_context(
            execution_context,
            execution_log,
        )
        if len(candidate_labels) >= 2 and selected_label:
            lowered = normalized.lower()
            selection_markers = (" was selected", " winner", " won", " best ", " outperformed", " lowest ")
            mentions_selected = selected_label.lower() in lowered
            has_selection = any(marker in lowered for marker in selection_markers)
            if not (mentions_selected and has_selection):
                step_count = len(re.findall(r"(?:^|\n)\s*\d+\.\s+", normalized))
                next_step = step_count + 1 if step_count > 0 else 1
                evidence_summary = cls._sanitize_execution_context_text(
                    str(execution_context.get("evidence_summary") or "").strip()
                )
                support = evidence_summary or "it best matched the decisive evidence above"
                addition = f"{next_step}. {selected_label} was selected because {support.rstrip('.')}."
                normalized = f"{normalized}\n{addition}".strip() if normalized else addition
        return normalized

    @classmethod
    def _is_task_answer_present_in_assistant_history(
        cls,
        task_type: str,
        answer: Any,
        conversation: list[dict[str, str]],
        *,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        for message in conversation:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            parsed = cls._coerce_llm_decision(
                None,
                str(message.get("content") or ""),
                fallback_thought="assistant_history_parse_failed",
            )
            if cls._is_valid_final_answer(
                task_type,
                parsed,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ) and cls._answers_match_semantically(
                task_type,
                answer,
                parsed,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                return True
        return False

    @staticmethod
    def _coerce_prediction_for_evaluation(task_type: str, answer: Any) -> Any:
        """Restore structured payloads from JSON-looking string answers."""
        if isinstance(answer, str) and task_type in {"forecast", "indicator"}:
            try:
                answer = json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                return answer
        return answer

    @staticmethod
    def _unwrap_task_terminal_answer(task_type: str, answer: Any) -> Any:
        if not isinstance(answer, dict):
            return answer
        inner = answer.get("answer")
        wrapped_type = str(answer.get("answer_type") or "").strip()
        if wrapped_type == str(task_type).strip() and inner is not None:
            return inner
        return answer

    @classmethod
    def _collect_batch_eval_predictions(
        cls,
        execution_log: list[dict[str, Any]],
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        framework_tools = {
            "spawn_subagent",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
            "bound_check",
            "consistency_check",
        }
        predictions: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        pending_chain: list[str] = []
        for entry in execution_log:
            tool_name = str(entry.get("tool") or "").strip()
            if tool_name == "spawn_subagent":
                pending_chain = []
                output = entry.get("output") or []
                for idx, item in enumerate(output):
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    if not isinstance(result, dict):
                        continue
                    answer = result.get("answer", result.get("prediction"))
                    answer = cls._unwrap_task_terminal_answer(task_type, answer)
                    if task_type == "indicator":
                        answer = cls._canonicalize_indicator_answer(
                            answer,
                            indicator_output_keys=indicator_output_keys,
                        )
                    if not cls._is_valid_final_answer(
                        task_type,
                        answer,
                        horizon=horizon,
                        indicator_output_keys=indicator_output_keys,
                    ):
                        continue
                    name = cls._candidate_branch_id(result, fallback_idx=idx)
                    if name in seen_names:
                        continue
                    seen_names.add(name)
                    eval_prediction = cls._coerce_prediction_for_evaluation(task_type, answer)
                    if task_type == "indicator":
                        eval_prediction = cls._canonicalize_indicator_answer(
                            eval_prediction,
                            indicator_output_keys=indicator_output_keys,
                        )
                    predictions.append({"name": name, "prediction": eval_prediction})
                continue
            if not tool_name:
                continue
            if tool_name in framework_tools:
                pending_chain = []
                continue

            pending_chain.append(tool_name)
            answer = cls._candidate_answer_from_entry(entry, task_type=task_type)
            if task_type == "indicator":
                answer = cls._canonicalize_indicator_answer(
                    answer,
                    indicator_output_keys=indicator_output_keys,
                )
            if not cls._is_valid_final_answer(
                task_type,
                answer,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                continue

            name = cls._top_level_candidate_path_name(pending_chain)
            if name in seen_names:
                pending_chain = []
                continue
            seen_names.add(name)
            eval_prediction = cls._coerce_prediction_for_evaluation(task_type, answer)
            if task_type == "indicator":
                eval_prediction = cls._canonicalize_indicator_answer(
                    eval_prediction,
                    indicator_output_keys=indicator_output_keys,
                )
            predictions.append({"name": name, "prediction": eval_prediction})
            pending_chain = []
        return predictions

    @staticmethod
    def _candidate_branch_id(result: dict[str, Any], *, fallback_idx: int) -> str:
        """Return a stable branch identity for one sub-agent candidate.

        New definition:
            Candidate identity is branch-based, not tool-name-based. Different
            spawned sub-agents should remain different candidates even if they
            converge to the same first tool or final tool. We therefore prefer
            `slot_id`, then the assigned branch/task description, and only fall
            back to tool-chain text when branch metadata is missing.
        """
        slot_id = result.get("slot_id")
        if isinstance(slot_id, int):
            return f"slot_{slot_id}"

        branch_metadata = result.get("branch_metadata")
        if isinstance(branch_metadata, dict):
            branch_id = str(branch_metadata.get("branch_id") or "").strip()
            if branch_id:
                return branch_id

        sub_task_description = str(result.get("sub_task_description") or "").strip()
        if sub_task_description:
            return sub_task_description

        hinted_tools = ExecutorAgent._hinted_tools(result.get("tools_hint"))
        if hinted_tools:
            return ", ".join(hinted_tools)

        candidate_name = str(result.get("candidate_name") or "").strip()
        if candidate_name:
            return candidate_name

        sub_execution_log = result.get("execution_log") or []
        tool_chain = [
            str(log_entry.get("tool") or "").strip()
            for log_entry in sub_execution_log
            if isinstance(log_entry, dict) and str(log_entry.get("tool") or "").strip()
        ]
        if tool_chain:
            return " -> ".join(tool_chain)

        return f"candidate_{fallback_idx}"

    @staticmethod
    def _build_profiling_section(profiling_results: dict[str, Any]) -> str:
        if not isinstance(profiling_results, dict) or not profiling_results:
            return ""
        keys = [
            "modality_report",
            "compute_statistics",
            "basic_stats",
            "autocorrelation",
            "stationarity_test",
            "detect_trend",
            "detect_anomaly",
        ]
        lines: list[str] = []
        for key in keys:
            if key not in profiling_results:
                continue
            value = profiling_results.get(key)
            try:
                value_text = json.dumps(
                    ExecutorAgent._round_prompt_numbers(value),
                    ensure_ascii=False,
                    default=str,
                )
            except TypeError:
                value_text = str(value)
            if value_text:
                lines.append(f"- {key}: {value_text[:1200]}")
        if not lines:
            return ""
        return "## Profiling Results\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _round_prompt_numbers(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return [ExecutorAgent._round_prompt_numbers(item) for item in value]
        if isinstance(value, tuple):
            return [ExecutorAgent._round_prompt_numbers(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): ExecutorAgent._round_prompt_numbers(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _has_structured_thought(thought: str) -> bool:
        if not thought:
            return False
        return all(label in thought for label in ("Reason:", "Action:"))

    @staticmethod
    def _reasoning_substantive_tool_chain(
        execution_log: list[dict[str, Any]],
    ) -> list[str]:
        return [
            str(entry.get("tool", "")).strip()
            for entry in execution_log
            if (
                str(entry.get("tool", "")).strip()
                and str(entry.get("tool", "")).strip() not in _ORCHESTRATION_ONLY_TOOLS
            )
        ]

    @classmethod
    def _available_substantive_tool_names(
        cls,
        execution_log: list[dict[str, Any]],
    ) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        def _add_tool(raw_tool: Any) -> None:
            tool = str(raw_tool or "").strip()
            if not tool or tool in _ORCHESTRATION_ONLY_TOOLS or tool in seen:
                return
            seen.add(tool)
            names.append(tool)

        for entry in execution_log:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool") or "").strip()
            _add_tool(tool_name)
            if tool_name != "spawn_subagent":
                continue
            output = entry.get("output") or []
            for item in output:
                if not isinstance(item, dict) or not item.get("success"):
                    continue
                result = item.get("result") or {}
                if not isinstance(result, dict):
                    continue
                for log_entry in result.get("execution_log") or []:
                    if isinstance(log_entry, dict):
                        _add_tool(log_entry.get("tool"))
        return names

    @classmethod
    def _reasoning_candidate_tool_chains(
        cls,
        execution_log: list[dict[str, Any]],
    ) -> list[list[str]]:
        chains: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for entry in execution_log:
            if entry.get("tool") != "spawn_subagent":
                continue
            output = entry.get("output") or []
            for item in output:
                if not isinstance(item, dict) or not item.get("success"):
                    continue
                result = item.get("result") or {}
                if not isinstance(result, dict):
                    continue
                sub_execution_log = result.get("execution_log") or []
                sub_chain = cls._reasoning_substantive_tool_chain(sub_execution_log)
                if not sub_chain:
                    continue
                key = tuple(sub_chain)
                if key in seen:
                    continue
                seen.add(key)
                chains.append(sub_chain)
        return chains

    @staticmethod
    def _reasoning_tool_aliases(tool_name: str) -> list[str]:
        tool = str(tool_name or "").strip()
        if not tool or tool in _ORCHESTRATION_ONLY_TOOLS:
            return []
        aliases: list[str] = [tool]
        canonical_aliases = {
            "chronos2_forecast": ["Chronos-2"],
            "timesfm2_forecast": ["TimesFM 2.0"],
            "moirai1_1_r_base_forecast": ["Moirai 1.1 R-base"],
            "sundial_base_128m_forecast": ["Sundial Base 128M"],
            "aurora_forecast": ["Aurora"],
            "arima_forecast": ["ARIMA"],
            "ets_forecast": ["ETS"],
            "prophet_forecast": ["Prophet"],
        }
        for alias in canonical_aliases.get(tool, []):
            if alias not in aliases:
                aliases.append(alias)
        return aliases

    @classmethod
    def _reasoning_explicit_name_hints(
        cls,
        execution_log: list[dict[str, Any]],
    ) -> list[str]:
        seen: set[str] = set()
        hints: list[str] = []
        for chain in cls._reasoning_candidate_tool_chains(execution_log):
            for tool in chain:
                for alias in cls._reasoning_tool_aliases(tool):
                    key = alias.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    hints.append(f"{tool}: {alias}")
        for tool in cls._reasoning_substantive_tool_chain(execution_log):
            for alias in cls._reasoning_tool_aliases(tool):
                key = alias.lower()
                if key in seen:
                    continue
                seen.add(key)
                hints.append(f"{tool}: {alias}")
        return hints

    @classmethod
    def _reasoning_candidate_labels(
        cls,
        execution_log: list[dict[str, Any]],
    ) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
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
                sub_execution_log = result.get("execution_log") or []
                sub_chain = cls._reasoning_substantive_tool_chain(sub_execution_log)
                label = (
                    " -> ".join(sub_chain)
                    or str(result.get("candidate_name") or "").strip()
                    or " -> ".join(cls._hinted_tools(result.get("tools_hint")))
                    or str(result.get("sub_task_description") or "").strip()
                    or f"candidate_{idx}"
                )
                lowered = label.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                labels.append(label)
        return labels

    async def _stepwise_reasoning_verdict(
        self,
        reasoning: Any,
        execution_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        text = str(reasoning or "").strip()
        candidate_labels = self._reasoning_candidate_labels(execution_log)
        substantive_chain = self._reasoning_substantive_tool_chain(execution_log)
        explicit_name_hints = self._reasoning_explicit_name_hints(execution_log)
        mode = "candidate_comparison" if len(candidate_labels) >= 2 else "single_path"
        allowed_missing = {
            "numbered_steps",
            "explicit_names",
            "decisive_evidence",
            "winner_selection",
            "tool_purpose_or_result",
            "orchestration_only_language",
        }
        if not text:
            return {
                "passed": False,
                "missing": ["numbered_steps", "explicit_names"],
                "explanation": "reasoning is empty",
            }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON judge for TimeClaw final reasoning quality. "
                    "This reasoning will be consumed by downstream agents and memory, so "
                    "it must explicitly name real executed paths/tools.\n\n"
                    "Return exactly one JSON object with keys `passed`, `missing`, and "
                    "`explanation`.\n"
                    "`passed` must be a boolean.\n"
                    "`missing` must be an array containing only items from "
                    "[`numbered_steps`, `explicit_names`, `decisive_evidence`, "
                    "`winner_selection`, `tool_purpose_or_result`, "
                    "`orchestration_only_language`].\n"
                    "`explanation` must be a short string.\n\n"
                    "Judging rules:\n"
                    "1. Be strict.\n"
                    "2. Treat a name as explicit only if the reasoning literally includes "
                    "either (a) an exact concrete tool/path name string from the provided "
                    "lists, or (b) an exact canonical alias from the provided explicit-name "
                    "hint list. Do not invent new aliases.\n"
                    "3. Never require orchestration-only tools such as spawn_subagent or "
                    "evaluate_*.\n"
                    "4. If the reasoning talks mainly about orchestration/framework actions, "
                    "fail with `orchestration_only_language`.\n"
                    "5. In `candidate_comparison` mode, pass only if the reasoning has "
                    "numbered steps, explicitly names at least two compared real candidate "
                    "paths or concrete tool names from those paths, states the decisive "
                    "evidence, and states why the winner was selected.\n"
                    "6. In `single_path` mode, pass only if the reasoning has numbered "
                    "steps, explicitly names the real substantive tools used in this branch, "
                    "and explains why each named tool mattered or what key result it produced.\n"
                    "7. If all requirements are satisfied, return `passed=true` and "
                    "`missing=[]`."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Mode: {mode}\n"
                    "Exact candidate path labels:\n"
                    + (
                        "\n".join(f"- {label}" for label in candidate_labels)
                        if candidate_labels else "- (none)"
                    )
                    + "\n\nExact substantive tool chain for this branch:\n"
                    + (
                        "\n".join(f"- {tool}" for tool in substantive_chain)
                        if substantive_chain else "- (none)"
                    )
                    + "\n\nExplicit-name hints that still count as concrete if quoted literally:\n"
                    + (
                        "\n".join(f"- {hint}" for hint in explicit_name_hints)
                        if explicit_name_hints else "- (none)"
                    )
                    + "\n\nReasoning to judge:\n"
                    + text
                ),
            },
        ]
        parsed = await self._complete_json_judge_response(
            messages,
            error_message="stepwise reasoning judge returned invalid JSON",
        )
        missing = [
            str(item).strip()
            for item in (parsed.get("missing") or [])
            if str(item).strip() in allowed_missing
        ]
        passed = bool(parsed.get("passed"))
        if passed:
            missing = []
        return {
            "passed": passed,
            "missing": missing,
            "explanation": str(parsed.get("explanation") or "").strip(),
        }

    async def _execution_context_verdict(
        self,
        execution_context: dict[str, Any],
        execution_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_labels = self._reasoning_candidate_labels(execution_log)
        substantive_tools = self._available_substantive_tool_names(execution_log)
        allowed_missing = {
            "tool_chain_alignment",
            "public_reasoning",
            "evidence_summary",
            "explicit_tool_names",
            "decisive_evidence",
            "execution_only_language",
        }
        if not substantive_tools:
            return {"passed": True, "missing": [], "explanation": ""}
        execution_text = "\n".join([
            *(
                str(item).strip()
                for item in execution_context.get("public_reasoning") or []
                if str(item).strip()
            ),
            str(execution_context.get("evidence_summary") or "").strip(),
        ]).lower()
        hard_control_markers = (
            "task-valid",
            "ground truth",
            "ground_truth",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
            "spawn_subagent",
            "return unchanged",
            "execution context",
            "json repair",
            "payload wrapping",
            "before finishing",
        )
        if any(marker in execution_text for marker in hard_control_markers):
            missing = ["execution_only_language"]
            if any(
                str(tool).strip() in _ORCHESTRATION_ONLY_TOOLS
                for tool in execution_context.get("tool_chain") or []
            ):
                missing.insert(0, "tool_chain_alignment")
            return {
                "passed": False,
                "missing": missing,
                "explanation": "execution_context contains controller-only language",
            }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON judge for TimeClaw execution_context quality. "
                    "This payload will be reused by downstream memory and agents, so it must "
                    "contain execution-side evidence only.\n\n"
                    "Return exactly one JSON object with keys `passed`, `missing`, and "
                    "`explanation`.\n"
                    "`passed` must be a boolean.\n"
                    "`missing` must be an array containing only items from "
                    "[`tool_chain_alignment`, `public_reasoning`, `evidence_summary`, "
                    "`explicit_tool_names`, `decisive_evidence`, "
                    "`execution_only_language`].\n"
                    "`explanation` must be a short string.\n\n"
                    "Judging rules:\n"
                    "1. Be strict.\n"
                    "2. `tool_chain_alignment` passes only if each entry in "
                    "`execution_context.tool_chain` is either (a) an exact real "
                    "substantive tool name from the provided executed-tool list, or "
                    "(b) an exact candidate path label from the provided list whose "
                    "segments are all real substantive executed tools. Never allow "
                    "orchestration-only tools such as spawn_subagent or evaluate_*.\n"
                    "3. `public_reasoning` passes only if it is non-empty and each item is "
                    "reusable execution-side evidence grounded in real tool results or data "
                    "properties, not controller instructions.\n"
                    "4. `evidence_summary` passes only if it is a concise execution-side "
                    "summary grounded in the same evidence.\n"
                    "5. `explicit_tool_names` passes only if the public reasoning or "
                    "evidence summary literally names concrete tools or candidate paths from "
                    "the provided lists. Do not invent aliases.\n"
                    "6. `decisive_evidence` passes only if the payload explains the actual "
                    "evidence that made the accepted answer or winner convincing.\n"
                    "7. Fail with `execution_only_language` if the payload talks about "
                    "task-valid status, ground truth, evaluate_*, spawn_subagent, branch "
                    "control requirements, completion mechanics, future steps, packaging, "
                    "JSON repair, or returning an answer unchanged.\n"
                    "8. If all requirements are satisfied, return `passed=true` and "
                    "`missing=[]`."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Exact candidate path labels:\n"
                    + (
                        "\n".join(f"- {label}" for label in candidate_labels)
                        if candidate_labels else "- (none)"
                    )
                    + "\n\nExact executed substantive tools:\n"
                    + "\n".join(f"- {tool}" for tool in substantive_tools)
                    + "\n\nExecution context to judge:\n"
                    + json.dumps(execution_context, ensure_ascii=False, indent=2)
                ),
            },
        ]
        parsed = await self._complete_json_judge_response(
            messages,
            error_message="execution_context judge returned invalid JSON",
        )
        missing = [
            str(item).strip()
            for item in (parsed.get("missing") or [])
            if str(item).strip() in allowed_missing
        ]
        passed = bool(parsed.get("passed"))
        if passed:
            missing = []
        return {
            "passed": passed,
            "missing": missing,
            "explanation": str(parsed.get("explanation") or "").strip(),
        }

    async def _final_payload_reusability_verdict(
        self,
        *,
        mode: Any,
        reasoning: Any,
        execution_context: dict[str, Any],
        execution_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self._should_enforce_stepwise_reasoning(mode):
            return {
                "passed": True,
                "execution_context_verdict": {"passed": True, "missing": [], "explanation": ""},
                "reasoning_verdict": {"passed": True, "missing": [], "explanation": ""},
            }

        execution_context_verdict = await self._execution_context_verdict(
            execution_context,
            execution_log,
        )
        if not execution_context_verdict.get("passed", False):
            return {
                "passed": False,
                "execution_context_verdict": execution_context_verdict,
                "reasoning_verdict": {"passed": True, "missing": [], "explanation": ""},
            }

        reasoning_verdict = await self._stepwise_reasoning_verdict(
            reasoning,
            execution_log,
        )
        return {
            "passed": bool(reasoning_verdict.get("passed", False)),
            "execution_context_verdict": execution_context_verdict,
            "reasoning_verdict": reasoning_verdict,
        }

    async def _complete_json_judge_response(
        self,
        messages: list[dict[str, str]],
        *,
        error_message: str,
    ) -> dict[str, Any]:
        for attempt in range(MAX_JSON_JUDGE_ATTEMPTS):
            response = await self.llm.complete(messages, response_format="json")
            parsed = self._json_dict_from_response(response)
            if isinstance(parsed, dict) and "passed" in parsed:
                return parsed
            response_text = str(getattr(response, "text", "") or "").strip()
            if response_text or attempt + 1 >= MAX_JSON_JUDGE_ATTEMPTS:
                break
        raise RuntimeError(error_message)

    @staticmethod
    def _attach_partial_result_to_exception(
        exc: Exception,
        *,
        answer: Any,
        answer_type: str,
        confidence: Any,
        reasoning: str,
        execution_context: dict[str, Any],
        control_context: dict[str, Any],
        execution_log: list[dict[str, Any]],
    ) -> None:
        """Preserve terminal evidence when a late judge fails.

        New definition:
            Some failures happen after the executor already has a real terminal
            payload, but before it can be accepted. In that case the exception
            should carry a ``partial_result`` so train/test artifacts can still
            show the actual answer candidate and evidence instead of an empty
            shell row.
        """
        partial = getattr(exc, "partial_result", None)
        payload = dict(partial) if isinstance(partial, dict) else {}
        if answer is not None:
            payload["answer"] = answer
        if answer_type:
            payload["answer_type"] = str(answer_type)
        payload["confidence"] = confidence
        if reasoning:
            payload["reasoning"] = str(reasoning)
        if isinstance(execution_context, dict):
            payload["execution_context"] = execution_context
        if isinstance(control_context, dict):
            payload["control_context"] = control_context
        if isinstance(execution_log, list):
            payload["execution_log"] = list(execution_log)
        setattr(exc, "partial_result", payload)
        setattr(exc, "execution_log", payload.get("execution_log", []))
        setattr(exc, "raw_answer", payload.get("answer"))
        setattr(exc, "answer_type", payload.get("answer_type", ""))
        setattr(exc, "confidence", payload.get("confidence"))
        setattr(exc, "reasoning", payload.get("reasoning", ""))
        setattr(exc, "execution_context", payload.get("execution_context", {}))
        setattr(exc, "control_context", payload.get("control_context", {}))

    @staticmethod
    def _json_dict_from_response(response: Any) -> dict[str, Any]:
        parsed = response.parsed if isinstance(getattr(response, "parsed", None), dict) else {}
        if not parsed and isinstance(getattr(response, "text", None), str) and response.text.strip():
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError:
                parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    @classmethod
    def _terminal_payload_repair_messages(
        cls,
        execution_log: list[dict[str, Any]],
        terminal_reuse_verdict: dict[str, Any] | None,
    ) -> list[str]:
        verdict = terminal_reuse_verdict or {}
        execution_context_verdict = verdict.get("execution_context_verdict") or {}
        reasoning_verdict = verdict.get("reasoning_verdict") or {}
        messages: list[str] = []
        if not execution_context_verdict.get("passed", False):
            messages.append(
                cls._execution_context_correction_message(
                    execution_log,
                    verdict=execution_context_verdict,
                )
            )
        elif not reasoning_verdict.get("passed", False):
            messages.append(
                cls._stepwise_reasoning_correction_message(
                    execution_log,
                    verdict=reasoning_verdict,
                )
            )
        return messages

    @staticmethod
    def _terminal_delivery_correction_message(
        *,
        expected_type: str,
        failure_reason: str,
    ) -> str:
        detail_map = {
            "invalid_json_or_missing_required_fields": (
                "The last reply was not a valid final JSON object with all required fields."
            ),
            "not_finished": (
                "The last reply returned `done=false`; this turn must either finish correctly or explain why completion is impossible."
            ),
            "missing_answer": "The last reply did not include a final `answer` payload.",
            "wrong_answer_type": (
                f"The last reply used the wrong `answer_type`; it must be `{expected_type}`."
            ),
            "invalid_learning_summary": (
                "For `learning_summary`, `answer.insight` and `answer.recommendation` must both be short non-empty strings."
            ),
            "training_contract_unsatisfied": (
                "In training, a `learning_summary` is valid only after the required candidate-generation and evaluation steps are already complete."
            ),
            "invalid_final_answer": (
                "The last `answer` did not satisfy the task-specific final answer shape."
            ),
            "unsupported_answer": (
                "The last `answer` was not grounded in the completed path's evidence."
            ),
            "no_tool_use": (
                "A final answer here must be backed by at least one actual tool call in this path."
            ),
            "tools_hint_not_followed": (
                "This branch must finish using only evidence from its assigned visible tools."
            ),
        }
        detail = detail_map.get(
            failure_reason,
            "The last terminal reply was not an acceptable final delivery for this run.",
        )
        return (
            "Self-correction: your last terminal reply was not an acceptable final delivery. "
            "Do not call any tool. Return exactly one JSON object with "
            "`thought`, `done=true`, `answer`, "
            f"`answer_type='{expected_type}'`, `confidence`, `reasoning`, "
            "`execution_context`, and `control_context`. "
            "`reasoning` must explain the evidence in numbered steps. "
            "`execution_context` must contain execution-only evidence, and "
            "`control_context` is only for completion notes. "
            f"{detail} If completion is genuinely impossible from the existing path, "
            "return `done=false` and explain why."
        )

    @staticmethod
    def _has_stepwise_reasoning(
        reasoning: Any,
        execution_log: list[dict[str, Any]],
    ) -> bool:
        text = str(reasoning or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if any(tool in lowered for tool in _ORCHESTRATION_ONLY_TOOLS):
            return False

        candidate_names = ExecutorAgent._reasoning_candidate_names(execution_log)
        if candidate_names:
            numbered_items = re.findall(r"(?:^|\n)\s*\d+\.\s+", text)
            if len(numbered_items) < 2:
                return False
            mentioned = sum(
                1 for name in candidate_names
                if ExecutorAgent._reasoning_mentions_candidate(lowered, name)
            )
            return mentioned >= min(2, len(candidate_names))

        tool_chain = [
            str(entry.get("tool", "")).strip()
            for entry in execution_log
            if (
                str(entry.get("tool", "")).strip()
                and str(entry.get("tool", "")).strip() not in _ORCHESTRATION_ONLY_TOOLS
            )
        ]
        numbered_items = re.findall(r"(?:^|\n)\s*\d+\.\s+", text)
        if tool_chain:
            if len(numbered_items) < len(tool_chain):
                return False
            return all(tool.lower() in lowered for tool in tool_chain)

        return len(text) >= 40

    @staticmethod
    def _stepwise_reasoning_correction_message(
        execution_log: list[dict[str, Any]],
        verdict: dict[str, Any] | None = None,
    ) -> str:
        missing_map = {
            "numbered_steps": "numbered steps",
            "explicit_names": "explicit concrete candidate/tool names",
            "decisive_evidence": "the decisive evidence",
            "winner_selection": "why the winner was selected",
            "tool_purpose_or_result": "why each named tool mattered or what key result it produced",
            "orchestration_only_language": "removing orchestration/framework wording",
        }
        missing = [
            missing_map[item]
            for item in (verdict or {}).get("missing", [])
            if item in missing_map
        ]
        missing_text = ""
        if missing:
            missing_text = " The last draft was missing: " + ", ".join(missing) + "."

        candidate_labels = ExecutorAgent._reasoning_candidate_labels(execution_log)
        if len(candidate_labels) >= 2:
            names_text = ", ".join(candidate_labels)
            return (
                "Self-correction: your final `reasoning` is not explicit enough for downstream "
                "reuse. Rewrite it as numbered steps. Explicitly name at least two real candidate "
                f"paths using these concrete names where relevant: {names_text}. "
                "State the decisive evidence and why the winner was selected. Do not mention "
                "spawn_subagent, evaluate_*, execution context, assigned path, before finishing, "
                "or other framework wording in the reasoning. Do not call another tool; return "
                f"the corrected final answer now.{missing_text}"
            )

        substantive_chain = ExecutorAgent._reasoning_substantive_tool_chain(execution_log)
        if substantive_chain:
            chain_text = " -> ".join(substantive_chain)
            return (
                "Self-correction: your final `reasoning` is not explicit enough for downstream "
                "reuse. Rewrite it as numbered steps and explicitly name the real substantive "
                f"tools from this branch: {chain_text}. For each named tool, state why it was "
                "chosen and what key result it contributed to the final answer. Do not mention "
                "orchestration-only tools such as spawn_subagent or evaluate_* in the reasoning. "
                f"Do not call another tool; return the corrected final answer now.{missing_text}"
            )

        return (
            "Self-correction: your final `reasoning` is not explicit enough for downstream "
            "reuse. Rewrite it as numbered steps focused on substantive evidence, not "
            "orchestration-only tools such as spawn_subagent or evaluate_*. Do not "
            f"call another tool; return the corrected final answer now.{missing_text}"
        )

    @staticmethod
    def _execution_context_correction_message(
        execution_log: list[dict[str, Any]],
        verdict: dict[str, Any] | None = None,
    ) -> str:
        missing_map = {
            "tool_chain_alignment": "a tool_chain containing only real substantive tools",
            "public_reasoning": "non-empty reusable public_reasoning",
            "evidence_summary": "a concise execution-side evidence_summary",
            "explicit_tool_names": "explicit concrete tool/path names",
            "decisive_evidence": "the decisive evidence",
            "execution_only_language": "removing control-flow and packaging language",
        }
        missing = [
            missing_map[item]
            for item in (verdict or {}).get("missing", [])
            if item in missing_map
        ]
        missing_text = ""
        if missing:
            missing_text = " The last draft was missing: " + ", ".join(missing) + "."

        candidate_labels = ExecutorAgent._reasoning_candidate_labels(execution_log)
        substantive_tools = ExecutorAgent._available_substantive_tool_names(execution_log)
        names_text = ", ".join(candidate_labels) if candidate_labels else ", ".join(substantive_tools)
        return (
            "Self-correction: your final `execution_context` is not reusable yet. "
            "Rewrite the final answer now without calling another tool. Keep "
            "`execution_context.tool_chain` limited to the real substantive tools that "
            f"produced the accepted result, using these concrete names where relevant: {names_text or '(none)'}. "
            "Treat previous controller, evaluator, and completion wording as non-evidence; "
            "rebuild `execution_context` from tool outputs and observed data properties only. "
            "Rewrite `execution_context.public_reasoning` as 1-3 short execution-side "
            "evidence statements, and rewrite `execution_context.evidence_summary` as one "
            "concise execution-side summary. Move any task-valid wording, ground-truth "
            "notes, evaluation/control mechanics, packaging language, or 'return unchanged' "
            "phrasing into `control_context.completion_notes` or omit it. Then rewrite "
            "`reasoning` as numbered steps expanding the same execution-side evidence. "
            f"Do not call another tool; return the corrected final answer now.{missing_text}"
        )

    @staticmethod
    def _reasoning_candidate_names(execution_log: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
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
                sub_execution_log = result.get("execution_log") or []
                sub_chain = [
                    str(log_entry.get("tool") or "").strip()
                    for log_entry in sub_execution_log
                    if (
                        isinstance(log_entry, dict)
                        and str(log_entry.get("tool") or "").strip()
                        and str(log_entry.get("tool") or "").strip() not in _ORCHESTRATION_ONLY_TOOLS
                    )
                ]
                name = (
                    (sub_chain[0] if sub_chain else "")
                    or str(result.get("candidate_name") or "").strip()
                    or ", ".join(ExecutorAgent._hinted_tools(result.get("tools_hint")))
                    or str(result.get("sub_task_description") or "").strip()
                    or f"candidate_{idx}"
                )
                lowered = name.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                names.append(name)
        return names

    @staticmethod
    def _reasoning_mentions_candidate(reasoning_lower: str, candidate_name: str) -> bool:
        name = str(candidate_name or "").strip().lower()
        if not name:
            return False
        aliases = {name}
        if name.endswith("_forecast"):
            base = name[: -len("_forecast")].strip("_")
            if base:
                aliases.add(base)
                aliases.add(f"{base}-based")
        if name.startswith("llm_reasoning_"):
            suffix = name[len("llm_reasoning_"):].strip("_")
            aliases.add("llm")
            if suffix:
                aliases.add(f"llm {suffix}")
                aliases.add(f"direct {suffix} llm reasoning")
                aliases.add(f"direct multimodal llm {suffix} reasoning")
        if name == "ets_forecast":
            aliases.update({"ets", "ets-based", "exponential smoothing", "holt-winters"})
        if name == "arima_forecast":
            aliases.update({"arima", "arima-based"})
        return any(alias and alias in reasoning_lower for alias in aliases)

    @staticmethod
    def _decision_tools(
        *,
        allow_tool_call: bool,
        allow_finish: bool,
        ordinary_tool_names: list[str] | None = None,
        evaluate_tool_names: list[str] | None = None,
        spawn_available: bool = False,
        finish_mode: str = "task",
    ) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        ordinary_tool_names = [
            str(name).strip()
            for name in (ordinary_tool_names or [])
            if str(name).strip()
        ]
        evaluate_tool_names = [
            str(name).strip()
            for name in (evaluate_tool_names or [])
            if str(name).strip()
        ]
        if allow_tool_call:
            if ordinary_tool_names:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": _DECISION_TOOL_CALL,
                        "strict": True,
                        "description": (
                            "Emit the single ordinary tool call to execute for the current turn. "
                            "Use this only for ordinary tools in the allowed list below."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "thought": ExecutorAgent._thought_schema(),
                                "tool": {
                                    "type": "string",
                                    "enum": ordinary_tool_names,
                                },
                                "args_json": {
                                    "type": "string",
                                    "description": (
                                        "Arguments for the chosen ordinary tool serialized as a valid JSON "
                                        "object string. Use \"{}\" when the tool takes no arguments."
                                    ),
                                },
                            },
                            "required": ["thought", "tool", "args_json"],
                            "additionalProperties": False,
                        },
                    },
                })
            if spawn_available:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": _DECISION_SPAWN,
                        "strict": True,
                        "description": (
                            "Emit a `spawn_subagent` action for the current turn. Use this only when "
                            "you are opening parallel candidate branches."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "thought": ExecutorAgent._thought_schema(),
                                "tasks": {
                                    "type": "array",
                                    "description": "Distinct sub-agent branch tasks for `spawn_subagent`.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "slot_id": {
                                                "type": "integer",
                                                "description": (
                                                    "One chosen candidate slot id whose visible tools fully "
                                                    "support every tool named in this branch's tools_hint."
                                                ),
                                            },
                                            "task": {
                                                "type": "string",
                                                "description": "Short branch description for that slot.",
                                            },
                                            "slot_local_path_tools": {
                                                "type": "array",
                                                "description": (
                                                    "Ordered slot-local tool path for that branch. Every named "
                                                    "tool must appear in that slot's visible_tools. Use an empty "
                                                    "array when there is no explicit path."
                                                ),
                                                "items": {
                                                    "type": "string",
                                                },
                                            },
                                            "branch_metadata_json": {
                                                "type": "string",
                                                "description": (
                                                    "Structured parent branch metadata serialized as a valid JSON "
                                                    "object string. Use an empty string when absent."
                                                ),
                                            },
                                        },
                                        "required": ["slot_id", "task", "slot_local_path_tools", "branch_metadata_json"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["thought", "tasks"],
                            "additionalProperties": False,
                        },
                    },
                })
            if evaluate_tool_names:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": _DECISION_EVALUATE,
                        "strict": True,
                        "description": (
                            "Emit one `evaluate_*` action for the current turn. Use this only for "
                            "ground-truth evaluation tools in the allowed list below."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "thought": ExecutorAgent._thought_schema(),
                                "tool": {
                                    "type": "string",
                                    "enum": evaluate_tool_names,
                                },
                                "task_type": {
                                    "type": "string",
                                    "description": "Current task type to evaluate against ground truth.",
                                },
                                "prediction_json": {
                                    "type": "string",
                                    "description": (
                                        "Single prediction payload serialized as valid JSON. "
                                        "Use an empty string when omitted."
                                    ),
                                },
                                "predictions_json": {
                                    "type": "string",
                                    "description": (
                                        "Multiple prediction payloads serialized as a JSON array string. "
                                        "Use an empty string when omitted."
                                    ),
                                },
                            },
                            "required": ["thought", "tool", "task_type", "prediction_json", "predictions_json"],
                            "additionalProperties": False,
                        },
                    },
                })
        if allow_finish:
            if finish_mode == "learning_summary":
                tools.append({
                    "type": "function",
                    "function": {
                        "name": _DECISION_FINISH,
                        "strict": True,
                        "description": "Emit the single training learning summary to finish the current turn.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "thought": ExecutorAgent._thought_schema(),
                                "insight": {
                                    "type": "string",
                                    "description": "A short factual summary of which explored path won and why.",
                                },
                                "recommendation": {
                                    "type": "string",
                                    "description": "A short reusable lesson or cautionary failure pattern, not the task payload.",
                                },
                                "confidence": {"type": "number"},
                                "reasoning": {"type": "string"},
                                "execution_context": ExecutorAgent._execution_context_schema(),
                                "control_context": ExecutorAgent._control_context_schema(),
                            },
                            "required": [
                                "thought",
                                "insight",
                                "recommendation",
                                "confidence",
                                "reasoning",
                                "execution_context",
                                "control_context",
                            ],
                            "additionalProperties": False,
                        },
                    },
                })
                return tools
            tools.append({
                "type": "function",
                "function": {
                    "name": _DECISION_FINISH,
                    "strict": True,
                    "description": "Emit the single final answer to finish the current turn.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": ExecutorAgent._thought_schema(),
                            "answer_type": {"type": "string"},
                            "answer_json": {
                                "type": "string",
                                "description": (
                                    "The final answer payload serialized as valid JSON. "
                                    "Examples: \"\\\"B\\\"\", \"[1.0, 2.0]\", or "
                                    "\"{\\\"insight\\\": \\\"...\\\", \\\"recommendation\\\": \\\"...\\\"}\"."
                                ),
                            },
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                            "execution_context": ExecutorAgent._execution_context_schema(),
                            "control_context": ExecutorAgent._control_context_schema(),
                        },
                        "required": [
                            "thought",
                            "answer_type",
                            "answer_json",
                            "confidence",
                            "reasoning",
                            "execution_context",
                            "control_context",
                        ],
                        "additionalProperties": False,
                    },
                },
            })
        return tools

    @staticmethod
    def _decision_response_preview(response: Any) -> str:
        parts: list[str] = []
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            try:
                parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))
            except TypeError:
                parts.append(str(tool_calls))
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            parts.append(text)
        return "\n".join(parts)[:4000]

    @staticmethod
    def _decision_to_conversation_text(decision: dict[str, Any]) -> str:
        try:
            return json.dumps(decision, ensure_ascii=False, default=str)
        except TypeError:
            return str(decision)

    @classmethod
    def _coerce_structured_decision(
        cls,
        response: Any,
        *,
        allow_tool_call: bool,
        allow_finish: bool,
        ordinary_tool_names: list[str] | None = None,
        evaluate_tool_names: list[str] | None = None,
        spawn_available: bool = False,
        fallback_thought: str,
        finish_mode: str = "task",
    ) -> dict[str, Any]:
        tool_calls = getattr(response, "tool_calls", None) or []
        if len(tool_calls) != 1:
            legacy = cls._coerce_llm_decision(
                getattr(response, "parsed", None),
                getattr(response, "text", None),
                fallback_thought="invalid_decision_tool_call_count",
            )
            normalized = cls._normalize_legacy_decision_payload(
                legacy,
                allow_tool_call=allow_tool_call,
                allow_finish=allow_finish,
                finish_mode=finish_mode,
            )
            if normalized is not None:
                return normalized
            return {
                "done": False,
                "_parse_failed": True,
                "thought": "invalid_decision_tool_call_count",
            }
        call = tool_calls[0] or {}
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            return {
                "done": False,
                "_parse_failed": True,
                "thought": "invalid_decision_tool_call_payload",
            }
        ordinary_tool_set = {
            str(name).strip() for name in (ordinary_tool_names or []) if str(name).strip()
        }
        evaluate_tool_set = {
            str(name).strip() for name in (evaluate_tool_names or []) if str(name).strip()
        }
        if name == _DECISION_TOOL_CALL and allow_tool_call:
            tool_name = str(arguments.get("tool") or "")
            if ordinary_tool_set and tool_name not in ordinary_tool_set:
                return {
                    "done": False,
                    "_parse_failed": True,
                    "thought": "invalid_decision_wrong_tool_family",
                }
            parsed_args: dict[str, Any] = {}
            args_json = arguments.get("args_json")
            if isinstance(args_json, str):
                try:
                    loaded_args = json.loads(args_json)
                except Exception:
                    loaded_args = None
                if isinstance(loaded_args, dict):
                    parsed_args = loaded_args
            return {
                "thought": str(arguments.get("thought") or ""),
                "done": False,
                "tool": tool_name,
                "args": parsed_args,
            }
        if name == _DECISION_SPAWN and allow_tool_call:
            tasks = arguments.get("tasks")
            if not spawn_available:
                return {
                    "done": False,
                    "_parse_failed": True,
                    "thought": "invalid_decision_wrong_tool_family",
                }
            normalized_tasks: list[dict[str, Any]] = []
            if isinstance(tasks, list):
                for item in tasks:
                    if not isinstance(item, dict):
                        continue
                    normalized: dict[str, Any] = {}
                    slot_id = item.get("slot_id")
                    if isinstance(slot_id, int):
                        normalized["slot_id"] = slot_id
                    task = item.get("task")
                    if isinstance(task, str):
                        normalized["task"] = task
                    slot_local_path_tools = item.get("slot_local_path_tools")
                    if isinstance(slot_local_path_tools, list):
                        normalized_path = [
                            str(tool).strip()
                            for tool in slot_local_path_tools
                            if str(tool).strip()
                        ]
                        if normalized_path:
                            normalized["tools_hint"] = normalized_path
                    elif isinstance(slot_local_path_tools, tuple):
                        normalized_path = [
                            str(tool).strip()
                            for tool in slot_local_path_tools
                            if str(tool).strip()
                        ]
                        if normalized_path:
                            normalized["tools_hint"] = normalized_path
                    else:
                        tools_hint = item.get("tools_hint")
                        if isinstance(tools_hint, str) and tools_hint.strip():
                            normalized["tools_hint"] = tools_hint
                    branch_metadata_json = item.get("branch_metadata_json")
                    if isinstance(branch_metadata_json, str) and branch_metadata_json.strip():
                        try:
                            parsed_branch_metadata = json.loads(branch_metadata_json)
                        except Exception:
                            parsed_branch_metadata = None
                        if isinstance(parsed_branch_metadata, dict):
                            normalized["branch_metadata"] = parsed_branch_metadata
                    if "slot_id" in normalized and "task" in normalized:
                        normalized_tasks.append(normalized)
            return {
                "thought": str(arguments.get("thought") or ""),
                "done": False,
                "tool": "spawn_subagent",
                "args": {"tasks": normalized_tasks},
            }
        if name == _DECISION_EVALUATE and allow_tool_call:
            tool_name = str(arguments.get("tool") or "")
            if evaluate_tool_set and tool_name not in evaluate_tool_set:
                return {
                    "done": False,
                    "_parse_failed": True,
                    "thought": "invalid_decision_wrong_tool_family",
                }
            normalized_args: dict[str, Any] = {}
            task_type = arguments.get("task_type")
            if isinstance(task_type, str) and task_type.strip():
                normalized_args["task_type"] = task_type
            prediction_json = arguments.get("prediction_json")
            if isinstance(prediction_json, str) and prediction_json.strip():
                try:
                    parsed_prediction = json.loads(prediction_json)
                except Exception:
                    parsed_prediction = None
                if parsed_prediction is not None:
                    normalized_args["prediction"] = parsed_prediction
            predictions_json = arguments.get("predictions_json")
            if isinstance(predictions_json, str) and predictions_json.strip():
                try:
                    parsed_predictions = json.loads(predictions_json)
                except Exception:
                    parsed_predictions = None
                if isinstance(parsed_predictions, list):
                    normalized_args["predictions"] = parsed_predictions
            return {
                "thought": str(arguments.get("thought") or ""),
                "done": False,
                "tool": tool_name,
                "args": normalized_args,
            }
        if name == _DECISION_FINISH and allow_finish:
            if finish_mode == "learning_summary":
                insight = arguments.get("insight")
                recommendation = arguments.get("recommendation")
                reasoning = arguments.get("reasoning")
                if not (
                    isinstance(insight, str)
                    and isinstance(recommendation, str)
                    and isinstance(reasoning, str)
                ):
                    return {
                        "done": False,
                        "_parse_failed": True,
                        "thought": "invalid_learning_summary_payload",
                    }
                return {
                    "thought": str(arguments.get("thought") or ""),
                    "done": True,
                    "answer": {
                        "insight": insight,
                        "recommendation": recommendation,
                    },
                    "answer_type": "learning_summary",
                    "confidence": arguments.get("confidence"),
                    "reasoning": reasoning,
                    "execution_context": cls._normalize_execution_context(
                        arguments.get("execution_context")
                    ),
                    "control_context": cls._normalize_control_context(
                        arguments.get("control_context")
                    ),
                }
            answer_json = arguments.get("answer_json")
            if not isinstance(answer_json, str):
                return {
                    "done": False,
                    "_parse_failed": True,
                    "thought": "invalid_final_answer_payload_json",
                }
            try:
                answer = json.loads(answer_json)
            except Exception:
                return {
                    "done": False,
                    "_parse_failed": True,
                    "thought": "invalid_final_answer_payload_json",
                }
            return {
                "thought": str(arguments.get("thought") or ""),
                "done": True,
                "answer": answer,
                "answer_type": str(arguments.get("answer_type") or ""),
                "confidence": arguments.get("confidence"),
                "reasoning": str(arguments.get("reasoning") or ""),
                "execution_context": cls._normalize_execution_context(
                    arguments.get("execution_context")
                ),
                "control_context": cls._normalize_control_context(
                    arguments.get("control_context")
                ),
            }
        return {"done": False, "_parse_failed": True, "thought": fallback_thought}

    @classmethod
    def _normalize_legacy_decision_payload(
        cls,
        decision: dict[str, Any],
        *,
        allow_tool_call: bool,
        allow_finish: bool,
        finish_mode: str = "task",
    ) -> dict[str, Any] | None:
        if not isinstance(decision, dict) or decision.get("_parse_failed"):
            return None

        thought = str(decision.get("thought") or "")
        is_done = bool(decision.get("done", False))

        if is_done:
            answer = decision.get("answer")
            answer_type = str(decision.get("answer_type") or "")
            confidence = decision.get("confidence")
            reasoning = str(decision.get("reasoning") or "")
            if not allow_finish or answer is None or not answer_type or not reasoning:
                return None
            if "tool" in decision or "args" in decision:
                return None
            # Structural normalization should preserve any legacy final-answer
            # payload that is internally coherent. Task-vs-learning-summary
            # semantics are enforced later in the main/sub-agent loops, where we
            # can emit the correct self-correction reason instead of collapsing
            # everything into a parse failure.
            return {
                "thought": thought,
                "done": True,
                "answer": answer,
                "answer_type": answer_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "execution_context": cls._normalize_execution_context(
                    decision.get("execution_context")
                ),
                "control_context": cls._normalize_control_context(
                    decision.get("control_context")
                ),
            }

        tool = str(decision.get("tool") or "")
        args = decision.get("args")
        if not allow_tool_call or not tool or not isinstance(args, dict):
            return None
        if (
            "answer" in decision
            or "answer_type" in decision
            or "reasoning" in decision
            or "execution_context" in decision
            or "control_context" in decision
        ):
            return None
        return {
            "thought": thought,
            "done": False,
            "tool": tool,
            "args": args,
        }

    async def _request_structured_decision(
        self,
        conversation: list[dict[str, str]],
        *,
        allow_tool_call: bool,
        allow_finish: bool,
        ordinary_tool_names: list[str] | None = None,
        evaluate_tool_names: list[str] | None = None,
        spawn_available: bool = False,
        fallback_thought: str,
        finish_mode: str = "task",
    ) -> tuple[Any, dict[str, Any]]:
        response = await self.llm.complete_with_tools(
            conversation,
            tools=self._decision_tools(
                allow_tool_call=allow_tool_call,
                allow_finish=allow_finish,
                ordinary_tool_names=ordinary_tool_names,
                evaluate_tool_names=evaluate_tool_names,
                spawn_available=spawn_available,
                finish_mode=finish_mode,
            ),
            tool_choice="required",
        )
        decision = self._coerce_structured_decision(
            response,
            allow_tool_call=allow_tool_call,
            allow_finish=allow_finish,
            ordinary_tool_names=ordinary_tool_names,
            evaluate_tool_names=evaluate_tool_names,
            spawn_available=spawn_available,
            fallback_thought=fallback_thought,
            finish_mode=finish_mode,
        )
        return response, decision


    @staticmethod
    def _coerce_llm_decision(
        parsed: Any,
        text: str | None,
        *,
        fallback_thought: str,
    ) -> dict[str, Any]:
        def _normalize_terminal_finish_shape(decision: Any) -> Any:
            if not isinstance(decision, dict) or not bool(decision.get("done", False)):
                return decision
            answer = decision.get("answer")
            if not isinstance(answer, dict):
                return decision
            leaked_fields = {
                key: answer.get(key)
                for key in ("answer_type", "confidence", "reasoning")
                if key in answer
            }
            if not leaked_fields:
                return decision
            normalized = dict(decision)
            cleaned_answer = dict(answer)
            for key in ("answer_type", "confidence", "reasoning"):
                value = cleaned_answer.pop(key, None)
                if key == "answer_type":
                    if not normalized.get("answer_type") and isinstance(value, str) and value.strip():
                        normalized["answer_type"] = value.strip()
                elif key == "reasoning":
                    if not str(normalized.get("reasoning") or "").strip() and isinstance(value, str):
                        normalized["reasoning"] = value
                else:
                    if normalized.get("confidence") is None and value is not None:
                        normalized["confidence"] = value
            normalized["answer"] = cleaned_answer
            return normalized

        raw = (text or "").strip()
        if raw:
            try:
                decoder = json.JSONDecoder()
                recovered, end = decoder.raw_decode(raw)
            except Exception:
                recovered = None
                end = -1
            else:
                trailing = raw[end:].strip()
                if trailing.startswith("{") or trailing.startswith("["):
                    return {
                        "done": False,
                        "_parse_failed": True,
                        "thought": "multiple_json_objects_in_one_reply",
                    }
                trailing_lower = trailing.lower()
                if trailing and (
                    "assistant to=" in trailing_lower
                    or "tool result=" in trailing_lower
                    or trailing.startswith("```")
                ):
                    return {
                        "done": False,
                        "_parse_failed": True,
                        "thought": "json_followed_by_pseudo_tool_transcript",
                    }
            if isinstance(parsed, dict):
                return _normalize_terminal_finish_shape(parsed)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return _normalize_terminal_finish_shape(parsed[0])
            if isinstance(recovered, dict):
                return _normalize_terminal_finish_shape(recovered)
            if isinstance(recovered, list) and recovered and isinstance(recovered[0], dict):
                return _normalize_terminal_finish_shape(recovered[0])
        if isinstance(parsed, dict):
            return _normalize_terminal_finish_shape(parsed)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return _normalize_terminal_finish_shape(parsed[0])
        if raw:
            try:
                recovered = json.loads(raw)
            except Exception:
                recovered = None
            if isinstance(recovered, dict):
                return _normalize_terminal_finish_shape(recovered)
            if isinstance(recovered, list) and recovered and isinstance(recovered[0], dict):
                return _normalize_terminal_finish_shape(recovered[0])
        return {"done": False, "_parse_failed": True, "thought": fallback_thought}

    async def _request_terminal_path_review(
        self,
        *,
        expected_type: str,
        trace_phase: str,
        task_type: str | None = None,
        task_description: str = "",
        execution_log: list[dict[str, Any]] | None = None,
        feedback_messages: list[str] | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
        sub_agent_idx: int | None = None,
        recoverable_final: dict[str, Any] | None = None,
        sub_task_description: str = "",
        tools_hint: Any = None,
        conversation: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del conversation
        task_type = str(task_type or expected_type or "").strip()
        execution_log = execution_log or []
        trace_payload = {
            "reason": "path_review_after_budget_exhaustion",
            "expected_type": expected_type,
        }
        if sub_agent_idx is not None:
            trace_payload["sub_agent_idx"] = sub_agent_idx
        trace.agent_step("executor", trace_phase, trace_payload)

        clean_conversation = [
            {
                "role": "system",
                "content": (
                    "You are repairing the final JSON delivery for a TimeClaw run after the "
                    "normal step budget was exhausted. Use only the execution-side evidence "
                    "provided below. Return exactly one JSON object. Do not call any tool. "
                    "Do not copy controller-side repair wording into `execution_context`."
                ),
            },
            {
                "role": "user",
                "content": self._build_clean_repair_prompt(
                    expected_type=expected_type,
                    task_type=task_type,
                    task_description=task_description,
                    execution_log=execution_log,
                    recoverable_final=recoverable_final,
                    feedback_messages=feedback_messages,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
                    sub_task_description=sub_task_description,
                    tools_hint=tools_hint,
                    terminal_json=True,
                ),
            },
        ]
        retry_response = await self.llm.complete(clean_conversation, response_format="json")
        retry_decision = self._coerce_llm_decision(
            retry_response.parsed,
            retry_response.text,
            fallback_thought="LLM parse failed after terminal path review",
        )

        retry_trace = {
            "reason": "terminal_path_review_response",
            "parse_failed": bool(retry_decision.get("_parse_failed")),
            "done": bool(retry_decision.get("done", False)),
            "answer_type": retry_decision.get("answer_type", ""),
        }
        if sub_agent_idx is not None:
            retry_trace["sub_agent_idx"] = sub_agent_idx
        trace.agent_step("executor", trace_phase, retry_trace)
        return retry_decision

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
        return "Tool Notes:\n" + "\n".join(lines) + "\n"

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
        trace_lines: list[str] = []
        for idx, entry in enumerate(execution_log, start=1):
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name or tool_name in _ORCHESTRATION_ONLY_TOOLS:
                continue
            if not bool(entry.get("success")):
                continue
            args_text = self._format_prompt_payload(entry.get("args", {}), limit=1200)
            result_text = self._format_prompt_payload(entry.get("observation"), limit=6000)
            trace_lines.append(
                "\n".join(
                    [
                        f"- Step {idx}",
                        f"  Tool: {tool_name}",
                        f"  Args: {args_text}",
                        f"  Result: {result_text}",
                    ]
                )
            )
        if trace_lines:
            parts.append("## Prior Tool Trace\n" + "\n\n".join(trace_lines))
        if existing_context:
            parts.append(f"## Caller Context\n{existing_context}")
        return "\n\n".join(p for p in parts if p).strip()

    @staticmethod
    def _prepare_prompt_fingerprint(
        fingerprint: Any,
        *,
        profiling_section: str = "",
    ) -> Any:
        if not isinstance(fingerprint, dict):
            return fingerprint

        prompt_fp = dict(fingerprint)
        profiling_text = profiling_section or ""

        # `detect_trend` is duplicated exactly in profiling output.
        if "detect_trend" in profiling_text:
            prompt_fp.pop("trend", None)

        # `stationarity_test` is a richer version of the compact fingerprint stationarity.
        if "stationarity_test" in profiling_text:
            prompt_fp.pop("stationarity", None)

        return prompt_fp

    @staticmethod
    def _format_prompt_payload(value: Any, *, limit: int) -> str:
        try:
            text = json.dumps(
                ExecutorAgent._round_prompt_numbers(value),
                ensure_ascii=False,
                default=str,
            )
        except TypeError:
            text = str(value)
        return text[:limit]

    @staticmethod
    def _format_observation_for_prompt(observation: Any) -> str:
        try:
            text = json.dumps(
                ExecutorAgent._round_prompt_numbers(observation),
                ensure_ascii=False,
                default=str,
            )
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
