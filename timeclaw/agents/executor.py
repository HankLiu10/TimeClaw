"""ExecutorAgent: mini-ReAct tool orchestration layer.

Instead of planning the entire tool chain upfront, the Executor runs a
step-by-step Think→Act→Observe loop. Each step, the LLM sees the results
of all previous tools and decides what to do next — enabling dynamic
parameter passing (e.g., detect_seasonality → stl_decompose(period=N)).
"""

from __future__ import annotations

from timeclaw.core.errors import ExecutionContractError

import asyncio
import json
import random
import re
import time
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.agents.prompts import executor_prompt
from timeclaw.agents.tool_dropout import generate_dropout_candidates
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.clawhub.tool_registry import get_tool_instance
from timeclaw.core import trace_logger as trace
from timeclaw.core.agent_base import AgentBase
from timeclaw.core.confidence import coerce_confidence
from timeclaw.core.evidence import EvidenceEntry
from timeclaw.core.indicator_answers import canonical_indicator_key, canonicalize_indicator_answer
from timeclaw.core.learning_text import sanitize_learning_evidence_text
from timeclaw.core.message import AgentRole
from timeclaw.core.retry_control import RetryLater, current_work_item_id, should_defer_retry_to_scheduler
from timeclaw.core.tea_protocol import TaskEnvelope
from timeclaw.llm.base import (
    _is_retryable,
    _is_supplier_pool_retry,
    retry_delay_seconds_for_attempt,
)
from timeclaw.benchmark import mtbench_prompts

MAX_REACT_STEPS = 10
MAX_SUB_REACT_STEPS = 6
SUBAGENT_SLOT_POOL_SIZE = 10
MAX_CONCURRENT_SUBAGENTS = SUBAGENT_SLOT_POOL_SIZE
MIN_TRAINING_CANDIDATES = 2
SUBAGENT_TIMEOUT_SECONDS = 120
MAX_JSON_JUDGE_ATTEMPTS = 4
MAX_SKILL_SELECTOR_ATTEMPTS = 3
MAX_TERMINAL_REPAIR_ATTEMPTS = 5
MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP = 4
MAX_DECISION_RECOVERY_EXHAUSTIONS_PER_STEP = 4
_ORCHESTRATION_ONLY_TOOLS = frozenset({
    "spawn_subagent",
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
})

_MAX_TOOL_RETRY_ATTEMPTS = 5
_TOOL_DEFERRED_RETRY_STATE: dict[str, dict[str, int]] = {}
_ACTIVE_EXPERIENCE_MAX_RULES = 5
_ACTIVE_EXPERIENCE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "if",
    "in", "into", "is", "it", "its", "of", "on", "or", "the", "their", "then",
    "this", "that", "to", "use", "using", "when", "with", "without", "task",
    "asks", "asked", "based", "before", "after", "than", "over", "under",
    "only", "just", "also", "where", "while", "across", "about", "should",
    "must", "need", "needs", "data", "series", "window", "prompt", "text",
    "future", "past", "observed", "prediction",
})
_MATCHED_SKILL_DEFAULT_BLOCKS = 2
_MATCHED_SKILL_MAX_BLOCKS = 3
_DECISION_SKILL_DEFAULT_BLOCKS = 1
_DECISION_SKILL_MAX_BLOCKS = 2


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


class DecisionOutputExhaustedError(RuntimeError):
    """Raised when decision delivery keeps failing before any new evidence is gathered."""

    def __init__(
        self,
        *,
        agent_kind: str,
        failure_reason: str,
        sub_agent_idx: int | None = None,
        partial_result: dict[str, Any] | None = None,
    ) -> None:
        self.agent_kind = str(agent_kind)
        self.failure_reason = str(failure_reason)
        self.sub_agent_idx = sub_agent_idx
        self.partial_result = dict(partial_result or {})
        detail = f"{self.agent_kind} failed before gathering new evidence: {self.failure_reason}"
        if sub_agent_idx is not None:
            detail = (
                f"{self.agent_kind} {sub_agent_idx} failed before gathering new evidence: "
                f"{self.failure_reason}"
            )
        super().__init__(detail)


class ExecutorAgent(AgentBase):
    role = AgentRole.EXECUTOR

    def __init__(
        self,
        *args: Any,
        text_backends: dict[str, Any] | None = None,
        mcp_manager: Any | None = None,
        disable_tool_dropout: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._text_backends = text_backends
        self._mcp_manager = mcp_manager
        self._disable_tool_dropout = bool(disable_tool_dropout)

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
    def _parse_memory_tool_chains(cls, value: str) -> list[list[str]]:
        chains: list[list[str]] = []
        for raw_chain in str(value or "").split("||"):
            chain = [
                token
                for token in (
                    item.strip()
                    for item in raw_chain.split("->")
                )
                if token and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token)
            ]
            if chain:
                chains.append(chain)
        return chains

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
            preferred_paths = cls._parse_memory_tool_chains(fields.get("Prefer Path", ""))
            if not preferred_paths:
                continue
            support_match = re.search(
                r"(\d+)\s+note\(s\)",
                str(fields.get("Evidence", "")).strip(),
            )
            support = int(support_match.group(1)) if support_match else 0
            for preferred in preferred_paths:
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
            if "Prefer Path" not in fields and "Avoid Path" not in fields:
                lines.append(line)
                continue
            kept_parts: list[str] = []
            for key in ("Rule", "When", "Prefer", "Avoid", "Why", "Evidence"):
                value = fields.get(key)
                if value:
                    kept_parts.append(f"{key}: {value}")
            if kept_parts:
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}- " + " | ".join(kept_parts))
        return "\n".join(lines).strip()

    @staticmethod
    def _memory_entry_support_count(entry: dict[str, Any]) -> int:
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        refs = evidence.get("sample_refs")
        if isinstance(refs, list):
            return len([str(item).strip() for item in refs if str(item).strip()])
        return int(evidence.get("support_count") or 0)

    @staticmethod
    def _memory_entry_conflict_ids(entry: dict[str, Any]) -> list[str]:
        raw = entry.get("conflict_rule_ids")
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            rule_id = str(item).strip()
            if not rule_id or rule_id in seen:
                continue
            seen.add(rule_id)
            out.append(rule_id)
        return out

    @staticmethod
    def _experience_tokens(text: Any) -> set[str]:
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", str(text or ""))
        }
        return {token for token in tokens if token not in _ACTIVE_EXPERIENCE_STOPWORDS}

    @classmethod
    def _memory_entry_text(cls, entry: dict[str, Any]) -> str:
        parts: list[str] = [str(entry.get("summary") or "").strip(), str(entry.get("rationale") or "").strip()]
        parts.extend(
            str(item).strip()
            for item in entry.get("conditions") or []
            if str(item).strip()
        )
        parts.extend(
            str(item).strip()
            for item in entry.get("preferred_tools") or []
            if str(item).strip()
        )
        parts.extend(
            str(item).strip()
            for item in entry.get("avoid_tools") or []
            if str(item).strip()
        )
        for chain in entry.get("preferred_tool_chains") or []:
            if isinstance(chain, list):
                parts.append(" ".join(str(item).strip() for item in chain if str(item).strip()))
        for chain in entry.get("avoid_tool_chains") or []:
            if isinstance(chain, list):
                parts.append(" ".join(str(item).strip() for item in chain if str(item).strip()))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _extract_tools_from_text(text: Any) -> list[str]:
        candidates = re.findall(r"`([A-Za-z0-9_]+)`", str(text or ""))
        if not candidates:
            candidates = re.findall(r"\b([A-Za-z][A-Za-z0-9_]+(?:_[A-Za-z0-9_]+)+)\b", str(text or ""))
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    @classmethod
    def _parse_skill_blocks(cls, skill_prompt: str) -> list[dict[str, Any]]:
        text = str(skill_prompt or "").strip()
        if not text:
            return []
        lines = text.splitlines()
        blocks: list[dict[str, Any]] = []
        current_title = ""
        current_sections: dict[str, list[str]] = {}
        current_section = ""

        def _flush() -> None:
            nonlocal current_title, current_sections, current_section
            if not current_title and not current_sections:
                return
            raw_lines: list[str] = []
            if current_title:
                raw_lines.append(current_title)
            ordered_sections = ("Trigger", "Core Idea", "Workflow", "Avoid", "Evidence")
            for section_name in ordered_sections:
                values = current_sections.get(section_name, [])
                if not values:
                    continue
                raw_lines.append(f"{section_name}:")
                raw_lines.extend(values)
                raw_lines.append("")
            raw_text = "\n".join(raw_lines).strip()
            section_text = {
                name.lower().replace(" ", "_"): "\n".join(current_sections.get(name, [])).strip()
                for name in ordered_sections
            }
            tool_names = cls._extract_tools_from_text(raw_text)
            blocks.append({
                "title": current_title.strip(),
                "raw_text": raw_text,
                "trigger_text": section_text.get("trigger", ""),
                "core_idea_text": section_text.get("core_idea", ""),
                "workflow_text": section_text.get("workflow", ""),
                "avoid_text": section_text.get("avoid", ""),
                "evidence_text": section_text.get("evidence", ""),
                "tool_names": tool_names,
            })
            current_title = ""
            current_sections = {}
            current_section = ""

        section_headers = {
            "trigger:": "Trigger",
            "core idea:": "Core Idea",
            "workflow:": "Workflow",
            "avoid:": "Avoid",
            "evidence:": "Evidence",
        }
        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("## SKILL"):
                continue
            if stripped.startswith("### Skill "):
                _flush()
                current_title = stripped
                current_sections = {}
                current_section = ""
                continue
            header = section_headers.get(stripped.lower())
            if header:
                current_section = header
                current_sections.setdefault(header, [])
                continue
            if not current_title:
                continue
            if not current_section:
                continue
            current_sections.setdefault(current_section, []).append(line)
        _flush()
        return blocks

    @classmethod
    def _parse_decision_skill_blocks(cls, skill_prompt: str) -> list[dict[str, Any]]:
        text = str(skill_prompt or "").strip()
        if not text:
            return []
        lines = text.splitlines()
        blocks: list[dict[str, Any]] = []
        current_title = ""
        current_sections: dict[str, list[str]] = {}
        current_section = ""

        def _flush() -> None:
            nonlocal current_title, current_sections, current_section
            if not current_title and not current_sections:
                return
            raw_lines: list[str] = []
            if current_title:
                raw_lines.append(current_title)
            ordered_sections = (
                "When",
                "Why This Usually Supports The Judgment",
                "How To Explain It To A User",
                "Do Not Overclaim",
                "Evidence Strength",
            )
            for section_name in ordered_sections:
                values = current_sections.get(section_name, [])
                if not values:
                    continue
                raw_lines.append(f"{section_name}:")
                raw_lines.extend(values)
                raw_lines.append("")
            raw_text = "\n".join(raw_lines).strip()
            section_text = {
                name.lower().replace(" ", "_"): "\n".join(current_sections.get(name, [])).strip()
                for name in ordered_sections
            }
            blocks.append({
                "title": current_title.strip(),
                "raw_text": raw_text,
                "when_text": section_text.get("when", ""),
                "why_text": section_text.get("why_this_usually_supports_the_judgment", ""),
                "explain_text": section_text.get("how_to_explain_it_to_a_user", ""),
                "guardrail_text": section_text.get("do_not_overclaim", ""),
                "evidence_text": section_text.get("evidence_strength", ""),
                "tool_names": cls._extract_tools_from_text(raw_text),
            })
            current_title = ""
            current_sections = {}
            current_section = ""

        section_headers = {
            "when:": "When",
            "why this usually supports the judgment:": "Why This Usually Supports The Judgment",
            "how to explain it to a user:": "How To Explain It To A User",
            "do not overclaim:": "Do Not Overclaim",
            "evidence strength:": "Evidence Strength",
        }
        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("### Decision Skill "):
                _flush()
                current_title = stripped
                current_sections = {}
                current_section = ""
                continue
            header = section_headers.get(stripped.lower())
            if header:
                current_section = header
                current_sections.setdefault(header, [])
                continue
            if not current_title or not current_section:
                continue
            current_sections.setdefault(current_section, []).append(line)
        _flush()
        return blocks

    @classmethod
    def _extract_avoid_tool_chains_from_skill_block(
        cls,
        block: dict[str, Any],
    ) -> list[list[str]]:
        avoid_text = str(block.get("avoid_text") or "").strip()
        if not avoid_text:
            return []
        chains: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for line in avoid_text.splitlines():
            tool_names = cls._extract_tools_from_text(line)
            if len(tool_names) < 2 or "->" not in line:
                continue
            chain = tuple(tool_names)
            if chain in seen:
                continue
            seen.add(chain)
            chains.append(list(chain))
        return chains

    @classmethod
    def _sanitize_tool_lanes_against_selected_skill_avoids(
        cls,
        *,
        blocks: list[dict[str, Any]],
        selected_titles: list[str],
        tool_lanes: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        selected_title_set = {
            str(title).strip()
            for title in selected_titles
            if str(title).strip()
        }
        if not selected_title_set:
            return {
                "primary_path_tools": list(tool_lanes.get("primary_path_tools") or []),
                "challenger_path_tools": list(tool_lanes.get("challenger_path_tools") or []),
                "tool_prep_tools": list(tool_lanes.get("tool_prep_tools") or []),
            }

        primary_path_tools = [
            str(tool_name).strip()
            for tool_name in (tool_lanes.get("primary_path_tools") or [])
            if str(tool_name).strip()
        ]
        if not primary_path_tools:
            return {
                "primary_path_tools": primary_path_tools,
                "challenger_path_tools": list(tool_lanes.get("challenger_path_tools") or []),
                "tool_prep_tools": list(tool_lanes.get("tool_prep_tools") or []),
            }

        avoid_chains: list[list[str]] = []
        for block in blocks:
            title = str(block.get("title") or "").strip()
            if title not in selected_title_set:
                continue
            avoid_chains.extend(cls._extract_avoid_tool_chains_from_skill_block(block))

        if any(primary_path_tools == chain for chain in avoid_chains):
            primary_set = set(primary_path_tools)
            challenger_path_tools = [
                str(tool_name).strip()
                for tool_name in (tool_lanes.get("challenger_path_tools") or [])
                if str(tool_name).strip() and str(tool_name).strip() not in primary_set
            ]
            return {
                "primary_path_tools": [],
                "challenger_path_tools": challenger_path_tools,
                "tool_prep_tools": list(tool_lanes.get("tool_prep_tools") or []),
            }

        return {
            "primary_path_tools": primary_path_tools,
            "challenger_path_tools": list(tool_lanes.get("challenger_path_tools") or []),
            "tool_prep_tools": list(tool_lanes.get("tool_prep_tools") or []),
        }

    @classmethod
    def _selector_guidance_from_selected_skills(
        cls,
        *,
        blocks: list[dict[str, Any]],
        selected_titles: list[str],
        existing_guidance: str,
    ) -> str:
        guidance_parts: list[str] = []
        base_guidance = str(existing_guidance or "").strip()
        if base_guidance:
            guidance_parts.append(base_guidance)

        selected_title_set = {
            str(title).strip()
            for title in selected_titles
            if str(title).strip()
        }
        if not selected_title_set:
            return "\n".join(guidance_parts).strip()

        for block in blocks:
            title = str(block.get("title") or "").strip()
            if title not in selected_title_set:
                continue
            core_idea = str(block.get("core_idea_text") or "").strip()
            workflow = str(block.get("workflow_text") or "").strip()
            avoid_text = str(block.get("avoid_text") or "").strip()

            if core_idea and "mean of the first future 24 hours" in core_idea and "mean of the last observed 24 hours" in core_idea:
                guidance_parts.append(
                    "Selected-skill execution requirement: before final classification, explicitly form the required "
                    "future-vs-observed evidence by comparing the mean of the first future 24 hours against the mean "
                    "of the last observed 24 hours. Do not finalize from observed-only structure or weather narrative alone."
                )
            if workflow and "If the shift looks close to ±0.5" in workflow:
                guidance_parts.append(
                    "Selected-skill execution requirement: if the implied future-vs-observed mean shift is close to the ±0.5 threshold, "
                    "check more than one plausible continuation before deciding the label."
                )
            if avoid_text and "segment -> llm_reasoning_classification" in avoid_text:
                guidance_parts.append(
                    "Selected-skill execution requirement: do not use `segment -> llm_reasoning_classification` alone as the decisive path "
                    "when the skill marks that chain as an avoid case."
                )

        deduped: list[str] = []
        seen: set[str] = set()
        for part in guidance_parts:
            text = str(part).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return "\n".join(deduped).strip()

    @staticmethod
    def _tool_chain_has_forecast_step(tool_names: list[str]) -> bool:
        return any(str(tool_name).strip().endswith("_forecast") for tool_name in tool_names)

    @classmethod
    def _selected_packet_requires_explicit_future_day_evidence(
        cls,
        *,
        blocks: list[dict[str, Any]],
        selected_titles: list[str],
        memory_entries: list[dict[str, Any]],
        selected_rule_ids: list[str],
    ) -> bool:
        """New definition:
        Some selected experience does not merely prefer a tool family; it defines
        the task judgment as an explicit comparison between the first future day
        and the last observed day. In those cases, the packet's primary path must
        expose a future-evidence-producing step instead of stopping at an
        observed-only summary plus final classification.
        """
        selected_title_set = {
            str(title).strip()
            for title in selected_titles
            if str(title).strip()
        }
        for block in blocks:
            title = str(block.get("title") or "").strip()
            if title not in selected_title_set:
                continue
            combined = "\n".join(
                [
                    str(block.get("core_idea_text") or "").strip(),
                    str(block.get("workflow_text") or "").strip(),
                ]
            ).lower()
            if (
                "first future 24 hours" in combined
                and "last observed 24 hours" in combined
            ):
                return True

        selected_rule_id_set = {
            str(rule_id).strip()
            for rule_id in selected_rule_ids
            if str(rule_id).strip()
        }
        indexed_entries = {
            str(entry.get("rule_id") or "").strip(): entry
            for entry in memory_entries
            if isinstance(entry, dict) and str(entry.get("rule_id") or "").strip()
        }
        for rule_id in selected_rule_id_set:
            entry = indexed_entries.get(rule_id)
            if not isinstance(entry, dict):
                continue
            combined = "\n".join(
                [
                    str(entry.get("summary") or "").strip(),
                    str(entry.get("rationale") or "").strip(),
                ]
            ).lower()
            if (
                "first predicted 24 hours" in combined
                and "last observed 24 hours" in combined
            ):
                return True
        return False

    @classmethod
    def _derive_learned_primary_path_from_rule_ids(
        cls,
        *,
        memory_entries: list[dict[str, Any]],
        candidate_rule_ids: list[str],
        requires_future_day_evidence: bool,
        available_tool_names: list[str],
    ) -> list[str]:
        """New definition:
        if the selector leaves the primary lane empty after pruning an invalid
        observed-only path, we may still rebuild one from already learned
        rule-linked tool chains. This is not a fallback heuristic; it is a
        direct reuse of learned path guidance tied to the selected sample's
        skill-linked rules.
        """
        available_set = {
            str(tool_name).strip()
            for tool_name in available_tool_names
            if str(tool_name).strip()
        }
        indexed_entries = {
            str(entry.get("rule_id") or "").strip(): dict(entry)
            for entry in memory_entries
            if isinstance(entry, dict) and str(entry.get("rule_id") or "").strip()
        }
        for rule_id in candidate_rule_ids:
            entry = indexed_entries.get(str(rule_id).strip())
            if not isinstance(entry, dict):
                continue
            for raw_chain in entry.get("preferred_tool_chains") or []:
                if not isinstance(raw_chain, list):
                    continue
                chain = [
                    str(tool_name).strip()
                    for tool_name in raw_chain
                    if str(tool_name).strip()
                ]
                if not chain:
                    continue
                if requires_future_day_evidence and not cls._tool_chain_has_forecast_step(chain):
                    continue
                if any(tool_name not in available_set for tool_name in chain):
                    continue
                return chain
        return []

    @staticmethod
    def _skill_metadata_by_title(skill_metadata: Any) -> dict[str, dict[str, Any]]:
        payload = skill_metadata or {}
        scopes = payload.get("scopes") if isinstance(payload, dict) else None
        raw_items: list[Any] = []
        if isinstance(scopes, list):
            for scope_payload in scopes:
                if isinstance(scope_payload, dict):
                    raw_items.extend(scope_payload.get("skills") or [])
        elif isinstance(payload, dict):
            raw_items.extend(payload.get("skills") or [])
        indexed: dict[str, dict[str, Any]] = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            title = str(raw_item.get("title") or "").strip()
            if not title:
                continue
            source_rule_ids = [
                str(rule_id).strip()
                for rule_id in raw_item.get("source_rule_ids") or []
                if str(rule_id).strip()
            ]
            indexed[title] = {
                "title": title,
                "source_rule_ids": source_rule_ids,
            }
        return indexed

    @classmethod
    def _score_skill_block_for_sample(
        cls,
        block: dict[str, Any],
        *,
        query_tokens: set[str],
        candidate_set: set[str],
    ) -> tuple[float, dict[str, Any]]:
        text = "\n".join(
            str(block.get(key) or "").strip()
            for key in ("trigger_text", "core_idea_text", "workflow_text", "avoid_text")
        )
        block_tokens = cls._experience_tokens(text)
        overlap = query_tokens & block_tokens
        overlap_score = min(0.60, 0.04 * len(overlap))
        tool_names = {
            str(name).strip()
            for name in block.get("tool_names") or []
            if str(name).strip()
        }
        tool_hits = sorted(tool_names & candidate_set)
        tool_score = min(0.24, 0.06 * len(tool_hits))
        workflow_bonus = 0.0
        workflow_text = str(block.get("workflow_text") or "").strip()
        avoid_text = str(block.get("avoid_text") or "").strip()
        if workflow_text:
            workflow_bonus += 0.08
        if avoid_text:
            workflow_bonus += 0.04
        score = overlap_score + tool_score + workflow_bonus
        return score, {
            "title": str(block.get("title") or "").strip(),
            "score": round(score, 6),
            "query_overlap_tokens": sorted(overlap)[:12],
            "tool_hits": tool_hits,
        }

    @classmethod
    def _assemble_matched_skills_package(
        cls,
        *,
        blocks: list[dict[str, Any]],
        selected_titles: list[str],
        skill_metadata: Any,
        selector_audit: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not blocks:
            return {
                "text": "",
                "selected_titles": [],
                "selected_count": 0,
                "scored_blocks": [],
                "selected_blocks": [],
                "selected_rule_ids": [],
            }

        metadata_by_title = cls._skill_metadata_by_title(skill_metadata)
        block_by_title = {
            str(block.get("title") or "").strip(): block
            for block in blocks
            if str(block.get("title") or "").strip()
        }
        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        selector_audit_by_title = {
            str(item.get("title") or "").strip(): dict(item)
            for item in (selector_audit or [])
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        }
        for idx, title in enumerate(selected_titles, start=1):
            block = block_by_title.get(str(title).strip())
            if block is None:
                continue
            audit = dict(selector_audit_by_title.get(str(title).strip()) or {})
            audit.setdefault("title", str(title).strip())
            audit.setdefault("rank", idx)
            selected.append((block, audit))

        lines = ["Sample-Matched Skills:"]
        selected_rule_ids: list[str] = []
        seen_selected_rule_ids: set[str] = set()
        selected_blocks: list[dict[str, Any]] = []
        for idx, (block, _audit) in enumerate(selected, start=1):
            role = "Primary Skill" if idx == 1 else "Backup Skill"
            raw_text = str(block.get("raw_text") or "").strip()
            if not raw_text:
                continue
            block_payload = dict(block)
            metadata = metadata_by_title.get(str(block.get("title") or "").strip(), {})
            source_rule_ids = [
                str(rule_id).strip()
                for rule_id in metadata.get("source_rule_ids") or []
                if str(rule_id).strip()
            ]
            block_payload["source_rule_ids"] = source_rule_ids
            selected_blocks.append(block_payload)
            for rule_id in source_rule_ids:
                if rule_id in seen_selected_rule_ids:
                    continue
                seen_selected_rule_ids.add(rule_id)
                selected_rule_ids.append(rule_id)
            lines.append(f"{role}:\n{raw_text}")
        fallback_blocks = selected_blocks or blocks
        fallback_text = "\n\n".join(
            str(block.get("raw_text") or "").strip()
            for block in fallback_blocks
            if str(block.get("raw_text") or "").strip()
        ).strip()
        text = "\n\n".join(lines).strip() if len(lines) > 1 else fallback_text
        return {
            "text": text,
            "selected_titles": [
                str(block.get("title") or "").strip()
                for block, _audit in selected
                if str(block.get("title") or "").strip()
            ],
            "selected_count": len(selected),
            "scored_blocks": [audit for _block, audit in selected],
            "selected_blocks": selected_blocks,
            "selected_rule_ids": selected_rule_ids,
        }

    @classmethod
    def _assemble_decision_skills_package(
        cls,
        *,
        blocks: list[dict[str, Any]],
        selected_titles: list[str],
        selector_audit: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not blocks:
            return {
                "text": "",
                "selected_titles": [],
                "selected_count": 0,
                "scored_blocks": [],
                "selected_blocks": [],
            }

        block_by_title = {
            str(block.get("title") or "").strip(): block
            for block in blocks
            if str(block.get("title") or "").strip()
        }
        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        selector_audit_by_title = {
            str(item.get("title") or "").strip(): dict(item)
            for item in (selector_audit or [])
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        }
        for idx, title in enumerate(selected_titles, start=1):
            block = block_by_title.get(str(title).strip())
            if block is None:
                continue
            audit = dict(selector_audit_by_title.get(str(title).strip()) or {})
            audit.setdefault("title", str(title).strip())
            audit.setdefault("rank", idx)
            selected.append((block, audit))

        lines = ["Decision Guidance:"]
        selected_blocks: list[dict[str, Any]] = []
        for idx, (block, _audit) in enumerate(selected, start=1):
            role = "Primary Decision Skill" if idx == 1 else "Backup Decision Skill"
            raw_text = str(block.get("raw_text") or "").strip()
            if not raw_text:
                continue
            selected_blocks.append(dict(block))
            lines.append(f"{role}:\n{raw_text}")
        fallback_blocks = selected_blocks or blocks
        fallback_text = "\n\n".join(
            str(block.get("raw_text") or "").strip()
            for block in fallback_blocks
            if str(block.get("raw_text") or "").strip()
        ).strip()
        text = "\n\n".join(lines).strip() if len(lines) > 1 else fallback_text
        return {
            "text": text,
            "selected_titles": [
                str(block.get("title") or "").strip()
                for block, _audit in selected
                if str(block.get("title") or "").strip()
            ],
            "selected_count": len(selected),
            "scored_blocks": [audit for _block, audit in selected],
            "selected_blocks": selected_blocks,
        }

    @staticmethod
    def _block_raw_text(block: dict[str, Any]) -> str:
        return str(block.get("raw_text") or "").strip()

    @classmethod
    def _render_selected_skill_blocks(
        cls,
        *,
        selected_blocks: list[dict[str, Any]],
        primary_label: str,
        backup_label: str,
        heading: str,
    ) -> str:
        lines = [heading]
        visible = 0
        for idx, block in enumerate(selected_blocks, start=1):
            raw_text = cls._block_raw_text(block)
            if not raw_text:
                continue
            visible += 1
            role = primary_label if visible == 1 else backup_label
            lines.append(f"{role}:\n{raw_text}")
        if visible == 0:
            return ""
        return "\n\n".join(lines).strip()

    @classmethod
    def _rebuild_matched_skills_package_from_selected_blocks(
        cls,
        package: dict[str, Any],
        *,
        selected_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rebuilt = dict(package)
        rebuilt["selected_blocks"] = [dict(block) for block in selected_blocks]
        rebuilt["selected_titles"] = [
            str(block.get("title") or "").strip()
            for block in selected_blocks
            if str(block.get("title") or "").strip()
        ]
        rebuilt["selected_count"] = len(rebuilt["selected_titles"])
        rebuilt["selected_rule_ids"] = [
            str(rule_id).strip()
            for block in selected_blocks
            for rule_id in (block.get("source_rule_ids") or [])
            if str(rule_id).strip()
        ]
        rebuilt["text"] = cls._render_selected_skill_blocks(
            selected_blocks=selected_blocks,
            primary_label="Primary Skill",
            backup_label="Backup Skill",
            heading="Sample-Matched Skills:",
        )
        return rebuilt

    @classmethod
    def _rebuild_decision_skills_package_from_selected_blocks(
        cls,
        package: dict[str, Any],
        *,
        selected_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rebuilt = dict(package)
        rebuilt["selected_blocks"] = [dict(block) for block in selected_blocks]
        rebuilt["selected_titles"] = [
            str(block.get("title") or "").strip()
            for block in selected_blocks
            if str(block.get("title") or "").strip()
        ]
        rebuilt["selected_count"] = len(rebuilt["selected_titles"])
        rebuilt["text"] = cls._render_selected_skill_blocks(
            selected_blocks=selected_blocks,
            primary_label="Primary Decision Skill",
            backup_label="Backup Decision Skill",
            heading="Decision Guidance:",
        )
        return rebuilt

    @staticmethod
    def _correlation_tail_abstention_text_markers() -> tuple[str, ...]:
        return (
            "do not infer a signed correlation",
            "do not make a signed correlation inference from tone alone",
            "do not make a signed correlation call from tone alone",
            "use only a true post-publication price window",
            "post-publication prices are observable",
            "post-publication price window",
            "future response window is absent",
            "partial next-day window",
            "defaulting to no correlation",
            "prefer no correlation",
            "required unusually clean post-publication evidence",
            "unusually clean future evidence before any signed correlation call",
            "before any signed correlation call",
        )

    @staticmethod
    def _correlation_block_drop_text_markers() -> tuple[str, ...]:
        return (
            "signed correlation requires observable post-publication prices",
            "there are no observable post-publication prices",
            "there are no true post-publication prices",
            "a signed correlation judgment needs a real reaction window",
            "the evidence is too incomplete for a confident positive or negative call",
            "if future coverage is missing, partial, or only ambiguously visible near the series boundary",
        )

    @staticmethod
    def _correlation_line_strip_text_markers() -> tuple[str, ...]:
        return (
            "stay conservative rather than forcing a sign",
            "if only a partial next-day window is visible",
            "if none are visible, do not infer a signed correlation",
            "verify whether any true post-publication prices are visible after the publication timestamp",
            "the future response window is absent, partial, or hard to align near the series boundary",
            "do not make a signed correlation call from tone alone",
        )

    @classmethod
    def _is_correlation_tail_abstention_text(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        return any(marker in lowered for marker in cls._correlation_tail_abstention_text_markers())

    @classmethod
    def _correlation_block_should_drop(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        return any(marker in lowered for marker in cls._correlation_block_drop_text_markers())

    @classmethod
    def _correlation_line_should_strip(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        return any(marker in lowered for marker in cls._correlation_line_strip_text_markers())

    @classmethod
    def _entry_is_harmful_for_predictive_correlation(cls, entry: dict[str, Any]) -> bool:
        summary = str(entry.get("summary") or "").strip()
        rationale = str(entry.get("rationale") or "").strip()
        conditions = " ".join(
            str(item).strip()
            for item in (entry.get("conditions") or [])
            if str(item).strip()
        )
        combined = "\n".join(part for part in [summary, rationale, conditions] if part)
        return cls._is_correlation_tail_abstention_text(combined)

    @classmethod
    def _block_is_harmful_for_predictive_correlation(cls, block: dict[str, Any]) -> bool:
        return cls._is_correlation_tail_abstention_text(cls._block_raw_text(block))

    @classmethod
    def _sanitize_correlation_skill_block(
        cls,
        block: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw_text = cls._block_raw_text(block)
        if not raw_text:
            return None
        if cls._correlation_block_should_drop(raw_text):
            return None
        kept_lines: list[str] = []
        for line in raw_text.splitlines():
            if (
                cls._is_correlation_tail_abstention_text(line)
                or cls._correlation_line_should_strip(line)
            ):
                continue
            kept_lines.append(line)
        sanitized_text = "\n".join(kept_lines).strip()
        if not sanitized_text:
            return None
        sanitized = dict(block)
        sanitized["raw_text"] = sanitized_text
        return sanitized

    @classmethod
    def _rebuild_active_experience_package_from_selected_entries(
        cls,
        package: dict[str, Any],
        *,
        selected_entries: list[dict[str, Any]],
        selector_guidance: str,
        primary_path_tools: list[str],
        challenger_path_tools: list[str],
        tool_prep_tools: list[str],
    ) -> dict[str, Any]:
        rebuilt = dict(package)
        rebuilt["selected_entries"] = [dict(entry) for entry in selected_entries]
        rebuilt["selected_rule_ids"] = [
            str(entry.get("rule_id") or "").strip()
            for entry in selected_entries
            if str(entry.get("rule_id") or "").strip()
        ]
        rebuilt["selector_guidance"] = selector_guidance
        sections: list[str] = []
        role_guidance = cls._build_experience_role_guidance(selected_entries)
        if role_guidance:
            sections.append(role_guidance)
        tool_lane_section = cls._render_selected_tool_lanes(
            primary_path_tools=primary_path_tools,
            challenger_path_tools=challenger_path_tools,
            tool_prep_tools=tool_prep_tools,
        )
        if tool_lane_section:
            sections.append(tool_lane_section)
        rendered_lines = [
            f"- {rendered}"
            for entry in selected_entries
            if (rendered := cls._render_active_experience_entry(entry))
        ]
        if rendered_lines:
            sections.append(
                "## Sample Experience Cards\n"
                "Use these cards as concise reference only. Do not override direct evidence from the current sample.\n"
                + "\n".join(rendered_lines)
            )
        rebuilt["text"] = "\n\n".join(section for section in sections if section).strip()
        return rebuilt

    @classmethod
    def _sanitize_inference_sample_experience_packet(
        cls,
        *,
        task_type: str,
        matched_skills_package: dict[str, Any],
        decision_skills_package: dict[str, Any],
        active_experience_package: dict[str, Any],
        primary_path_tools: list[str],
        challenger_path_tools: list[str],
        tool_prep_tools: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """New definition:
        inference packet sanitization is a narrow post-selector cleanup step.
        It only removes inference-time experience items whose semantics are
        known to conflict with the benchmark task definition.
        """
        matched_skills_package = cls._trim_redundant_challenger_only_backup_skills(
            task_type=task_type,
            matched_skills_package=matched_skills_package,
            primary_path_tools=primary_path_tools,
            challenger_path_tools=challenger_path_tools,
        )
        active_experience_package = cls._trim_active_experience_for_direct_reasoning(
            task_type=task_type,
            active_experience_package=active_experience_package,
            primary_path_tools=primary_path_tools,
            challenger_path_tools=challenger_path_tools,
            tool_prep_tools=tool_prep_tools,
        )
        decision_skills_package = cls._trim_backup_decision_skills_for_direct_reasoning(
            task_type=task_type,
            decision_skills_package=decision_skills_package,
            primary_path_tools=primary_path_tools,
        )
        if str(task_type or "").strip() != "correlation":
            return matched_skills_package, decision_skills_package, active_experience_package

        # New definition:
        # keep correlation skills, but strip only the abstention-style lines
        # that wrongly collapse directional prediction into `No Correlation`.
        filtered_skill_blocks = [
            sanitized
            for block in (matched_skills_package.get("selected_blocks") or [])
            if isinstance(block, dict)
            for sanitized in [cls._sanitize_correlation_skill_block(dict(block))]
            if sanitized is not None
        ]
        filtered_decision_blocks = [
            sanitized
            for block in (decision_skills_package.get("selected_blocks") or [])
            if isinstance(block, dict)
            for sanitized in [cls._sanitize_correlation_skill_block(dict(block))]
            if sanitized is not None
        ]
        filtered_entries = [
            dict(entry)
            for entry in (active_experience_package.get("selected_entries") or [])
            if isinstance(entry, dict) and not cls._entry_is_harmful_for_predictive_correlation(entry)
        ]
        selector_guidance = (
            "Use the selected experience to extract the target-specific event signal and "
            "judge the most plausible next-window reaction. Missing or partial visible "
            "post-publication tail is not, by itself, a reason to force `No Correlation`. "
            "Use the pre-publication price regime only as supporting context; do not let it "
            "by itself override a clear target-specific directional article signal. "
            "If the target-specific signal is directional, prefer a signed correlation label "
            "and express uncertainty through strength, rather than collapsing to `No Correlation`."
        )

        return (
            cls._rebuild_matched_skills_package_from_selected_blocks(
                matched_skills_package,
                selected_blocks=filtered_skill_blocks,
            ),
            cls._rebuild_decision_skills_package_from_selected_blocks(
                decision_skills_package,
                selected_blocks=filtered_decision_blocks,
            ),
            cls._rebuild_active_experience_package_from_selected_entries(
                active_experience_package,
                selected_entries=filtered_entries,
                selector_guidance=selector_guidance,
                primary_path_tools=primary_path_tools,
                challenger_path_tools=challenger_path_tools,
                tool_prep_tools=tool_prep_tools,
            ),
        )

    @classmethod
    def _trim_redundant_challenger_only_backup_skills(
        cls,
        *,
        task_type: str,
        matched_skills_package: dict[str, Any],
        primary_path_tools: list[str],
        challenger_path_tools: list[str],
    ) -> dict[str, Any]:
        """New definition:
        if inference already has a single direct task-answer path, a backup
        skill that only describes the advisory challenger lane should stay in
        lane metadata but should not be injected as a second full operational
        SOP block.
        """
        task_reasoning_tool = cls._task_llm_reasoning_tool(task_type)
        if not task_reasoning_tool:
            return matched_skills_package
        primary_path = [str(tool).strip() for tool in primary_path_tools if str(tool).strip()]
        if primary_path != [task_reasoning_tool]:
            return matched_skills_package

        selected_blocks = [
            dict(block)
            for block in (matched_skills_package.get("selected_blocks") or [])
            if isinstance(block, dict)
        ]
        if len(selected_blocks) <= 1:
            return matched_skills_package

        primary_block = selected_blocks[0]
        kept_blocks = [primary_block]
        return cls._rebuild_matched_skills_package_from_selected_blocks(
            matched_skills_package,
            selected_blocks=kept_blocks,
        )

    @classmethod
    def _trim_active_experience_for_direct_reasoning(
        cls,
        *,
        task_type: str,
        active_experience_package: dict[str, Any],
        primary_path_tools: list[str],
        challenger_path_tools: list[str],
        tool_prep_tools: list[str],
    ) -> dict[str, Any]:
        """New definition:
        when inference already has a single direct task-answer path, active
        experience should keep only the primary driver card plus any explicit
        output contract cards. Guardrails stay in decision skills instead of
        being repeated as extra sample cards.
        """
        task_reasoning_tool = cls._task_llm_reasoning_tool(task_type)
        primary_path = [str(tool).strip() for tool in primary_path_tools if str(tool).strip()]
        if not task_reasoning_tool or primary_path != [task_reasoning_tool]:
            return active_experience_package

        selected_entries = [
            dict(entry)
            for entry in (active_experience_package.get("selected_entries") or [])
            if isinstance(entry, dict)
        ]
        if len(selected_entries) <= 1:
            return active_experience_package

        kept_entries: list[dict[str, Any]] = []
        if selected_entries:
            kept_entries.append(selected_entries[0])
        for entry in selected_entries[1:]:
            if str(entry.get("kind") or "").strip().lower() == "result_constraint":
                kept_entries.append(entry)

        if len(kept_entries) == len(selected_entries):
            return active_experience_package
        return cls._rebuild_active_experience_package_from_selected_entries(
            active_experience_package,
            selected_entries=kept_entries,
            selector_guidance=str(active_experience_package.get("selector_guidance") or "").strip(),
            primary_path_tools=primary_path_tools,
            challenger_path_tools=challenger_path_tools,
            tool_prep_tools=tool_prep_tools,
        )

    @classmethod
    def _trim_backup_decision_skills_for_direct_reasoning(
        cls,
        *,
        task_type: str,
        decision_skills_package: dict[str, Any],
        primary_path_tools: list[str],
    ) -> dict[str, Any]:
        """New definition:
        for single-tool direct reasoning, retain one full decision guardrail.
        Additional decision blocks tend to restate narrower cautions and can
        tilt the direct reasoner away from the primary judgment target.
        """
        task_reasoning_tool = cls._task_llm_reasoning_tool(task_type)
        primary_path = [str(tool).strip() for tool in primary_path_tools if str(tool).strip()]
        if not task_reasoning_tool or primary_path != [task_reasoning_tool]:
            return decision_skills_package

        selected_blocks = [
            dict(block)
            for block in (decision_skills_package.get("selected_blocks") or [])
            if isinstance(block, dict)
        ]
        if len(selected_blocks) <= 1:
            return decision_skills_package
        return cls._rebuild_decision_skills_package_from_selected_blocks(
            decision_skills_package,
            selected_blocks=[selected_blocks[0]],
        )

    @staticmethod
    def _normalize_selected_skill_titles(
        selected_titles: Any,
        *,
        available_titles: list[str],
    ) -> list[str]:
        if not isinstance(selected_titles, list):
            return []
        title_map: dict[str, str] = {}
        for title in available_titles:
            stripped = str(title).strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            title_map.setdefault(lowered, stripped)
            title_map.setdefault(stripped.lstrip("#").strip().lower(), stripped)
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_title in selected_titles:
            key = str(raw_title or "").strip().lower()
            normalized_title = title_map.get(key)
            if not normalized_title or normalized_title in seen:
                continue
            seen.add(normalized_title)
            normalized.append(normalized_title)
        return normalized

    @staticmethod
    def _normalize_selected_rule_ids(
        selected_rule_ids: Any,
        *,
        available_rule_ids: list[str],
    ) -> list[str]:
        if not isinstance(selected_rule_ids, list):
            return []
        available = {
            str(rule_id).strip(): str(rule_id).strip()
            for rule_id in available_rule_ids
            if str(rule_id).strip()
        }
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_rule_id in selected_rule_ids:
            rule_id = available.get(str(raw_rule_id or "").strip())
            if not rule_id or rule_id in seen:
                continue
            seen.add(rule_id)
            normalized.append(rule_id)
        return normalized

    @staticmethod
    def _normalize_selected_tool_names(
        selected_tool_names: Any,
        *,
        available_tool_names: list[str],
    ) -> list[str]:
        if not isinstance(selected_tool_names, list):
            return []
        available = {
            str(tool_name).strip(): str(tool_name).strip()
            for tool_name in available_tool_names
            if str(tool_name).strip()
        }
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_tool_name in selected_tool_names:
            tool_name = available.get(str(raw_tool_name or "").strip())
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            normalized.append(tool_name)
        return normalized

    @classmethod
    def _normalize_selected_tool_lanes(
        cls,
        payload: dict[str, Any],
        *,
        available_tool_names: list[str],
    ) -> dict[str, list[str]]:
        primary_path_tools = cls._normalize_selected_tool_names(
            payload.get("primary_path_tools"),
            available_tool_names=available_tool_names,
        )
        challenger_path_tools = cls._normalize_selected_tool_names(
            payload.get("challenger_path_tools"),
            available_tool_names=available_tool_names,
        )
        tool_prep_tools = cls._normalize_selected_tool_names(
            payload.get("tool_prep_tools"),
            available_tool_names=available_tool_names,
        )
        challenger_path_tools = [
            tool_name for tool_name in challenger_path_tools if tool_name not in set(primary_path_tools)
        ]
        tool_prep_tools = [
            tool_name
            for tool_name in tool_prep_tools
            if tool_name not in set(primary_path_tools) and tool_name not in set(challenger_path_tools)
        ]
        return {
            "primary_path_tools": primary_path_tools,
            "challenger_path_tools": challenger_path_tools,
            "tool_prep_tools": tool_prep_tools,
        }

    @staticmethod
    def _normalize_challenger_requirement(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"compare_before_finish", "advisory"}:
            return normalized
        return "advisory"

    @staticmethod
    def _flatten_selected_tool_lanes(
        *,
        primary_path_tools: list[str],
        challenger_path_tools: list[str],
        tool_prep_tools: list[str],
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for tool_name in [*tool_prep_tools, *primary_path_tools, *challenger_path_tools]:
            name = str(tool_name).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    @staticmethod
    def _render_selected_tool_lanes(
        *,
        primary_path_tools: list[str],
        challenger_path_tools: list[str],
        tool_prep_tools: list[str],
    ) -> str:
        lines: list[str] = []
        if tool_prep_tools:
            lines.append("- Prep tools: " + ", ".join(tool_prep_tools))
        if primary_path_tools:
            lines.append("- Primary path tools: " + ", ".join(primary_path_tools))
        if challenger_path_tools:
            lines.append("- Challenger path tools: " + ", ".join(challenger_path_tools))
        if not lines:
            return ""
        return "## Suggested Execution Lanes\n" + "\n".join(lines)

    @classmethod
    def _memory_entry_selector_line(cls, entry: dict[str, Any]) -> str:
        rule_id = str(entry.get("rule_id") or "").strip()
        if not rule_id:
            return ""
        parts: list[str] = []
        kind = str(entry.get("kind") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        if kind:
            parts.append(f"kind={kind}")
        if summary:
            parts.append(f"summary={summary}")
        conditions = [str(item).strip() for item in entry.get("conditions") or [] if str(item).strip()]
        if conditions:
            parts.append(f"when={'; '.join(conditions[:2])}")
        preferred_tools = [str(item).strip() for item in entry.get("preferred_tools") or [] if str(item).strip()]
        if preferred_tools:
            parts.append(f"prefer={', '.join(preferred_tools[:3])}")
        avoid_tools = [str(item).strip() for item in entry.get("avoid_tools") or [] if str(item).strip()]
        if avoid_tools:
            parts.append(f"avoid={', '.join(avoid_tools[:3])}")
        preferred_paths: list[str] = []
        for chain in entry.get("preferred_tool_chains") or []:
            if not isinstance(chain, list):
                continue
            tools = [str(item).strip() for item in chain if str(item).strip()]
            if tools:
                preferred_paths.append(" -> ".join(tools))
        if preferred_paths:
            parts.append(f"prefer_path={preferred_paths[0]}")
        avoid_paths: list[str] = []
        for chain in entry.get("avoid_tool_chains") or []:
            if not isinstance(chain, list):
                continue
            tools = [str(item).strip() for item in chain if str(item).strip()]
            if tools:
                avoid_paths.append(" -> ".join(tools))
        if avoid_paths:
            parts.append(f"avoid_path={avoid_paths[0]}")
        support = cls._memory_entry_support_count(entry)
        if support > 0:
            parts.append(f"evidence={support}")
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        if confidence > 0:
            parts.append(f"confidence={confidence:.2f}")
        return f"[{rule_id}] " + " | ".join(parts)

    @classmethod
    def _build_selected_experience_package(
        cls,
        *,
        memory_entries: list[dict[str, Any]],
        selected_rule_ids: list[str],
        selector_guidance: str = "",
        selector_audit: dict[str, Any] | None = None,
        primary_path_tools: list[str] | None = None,
        challenger_path_tools: list[str] | None = None,
        tool_prep_tools: list[str] | None = None,
        challenger_requirement: str = "advisory",
    ) -> dict[str, Any]:
        indexed_entries = {
            str(entry.get("rule_id") or "").strip(): dict(entry)
            for entry in memory_entries
            if isinstance(entry, dict)
            and bool(entry.get("injectable", True))
            and str(entry.get("rule_id") or "").strip()
        }
        selected_entries = [
            indexed_entries[rule_id]
            for rule_id in selected_rule_ids
            if rule_id in indexed_entries
        ]
        sections: list[str] = []
        guidance = str(selector_guidance or "").strip()
        role_guidance = cls._build_experience_role_guidance(selected_entries)
        if role_guidance:
            sections.append(role_guidance)
        tool_lane_section = cls._render_selected_tool_lanes(
            primary_path_tools=list(primary_path_tools or []),
            challenger_path_tools=list(challenger_path_tools or []),
            tool_prep_tools=list(tool_prep_tools or []),
        )
        if tool_lane_section:
            sections.append(tool_lane_section)
        rendered_lines = [
            f"- {rendered}"
            for entry in selected_entries
            if (rendered := cls._render_active_experience_entry(entry))
        ]
        if rendered_lines:
            sections.append(
                "## Sample Experience Cards\n"
                "Use these cards as concise reference only. Do not override direct evidence from the current sample.\n"
                + "\n".join(rendered_lines)
            )
        return {
            "text": "\n\n".join(section for section in sections if section).strip(),
            "selected_rule_ids": [
                str(entry.get("rule_id") or "").strip()
                for entry in selected_entries
                if str(entry.get("rule_id") or "").strip()
            ],
            "suppressed_conflict_rule_ids": [],
            "source_rule_count": len(indexed_entries),
            "scored_rules": [],
            "selected_entries": [dict(entry) for entry in selected_entries],
            "selector_guidance": guidance,
            "selector_audit": dict(selector_audit or {}),
            "primary_path_tools": list(primary_path_tools or []),
            "challenger_path_tools": list(challenger_path_tools or []),
            "tool_prep_tools": list(tool_prep_tools or []),
            "challenger_requirement": cls._normalize_challenger_requirement(
                challenger_requirement
            ),
        }

    def _build_selected_packet_tool_notes_section(
        self,
        tool_notes: dict[str, Any],
        *,
        primary_path_tools: list[str],
        challenger_path_tools: list[str],
        tool_prep_tools: list[str],
        task_type: str,
    ) -> str:
        """New definition:
        `Packet Tool Notes` are the focused tool notes for the tools explicitly
        selected into the sample packet. This is narrower than the full
        candidate tool-notes dump and is meant to reinforce the chosen path.
        """
        lane_tools = self._flatten_selected_tool_lanes(
            primary_path_tools=primary_path_tools,
            challenger_path_tools=challenger_path_tools,
            tool_prep_tools=tool_prep_tools,
        )
        if not lane_tools:
            return ""
        lines: list[str] = []
        for tool_name in lane_tools:
            matched_note = None
            for key in self._tool_note_candidates(tool_name, task_type):
                if key in tool_notes and str(tool_notes[key]).strip():
                    matched_note = str(tool_notes[key]).strip()
                    break
            if matched_note:
                lines.append(f"- {tool_name}: {matched_note}")
        if not lines:
            return ""
        return "## Packet Tool Notes\n" + "\n".join(lines)

    @classmethod
    def _build_selected_experience_decision_summary(
        cls,
        selected_entries: list[dict[str, Any]],
    ) -> str:
        """New definition:
        `Experience Decision Summary` is the compact, execution-time checklist
        distilled from the already-selected memory entries. It does not add new
        rules; it makes the learned prefer/avoid/path/contract guidance explicit
        right before tool or answer selection.
        """
        prioritized: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in selected_entries:
            if not isinstance(entry, dict):
                continue
            rule_id = str(entry.get("rule_id") or "").strip()
            if rule_id and rule_id in seen:
                continue
            if rule_id:
                seen.add(rule_id)
            if cls._entry_has_path_guidance(entry) or str(entry.get("kind") or "").strip() == "result_constraint":
                prioritized.append(entry)
            else:
                fallback.append(entry)
        chosen = (prioritized + fallback)[:5]
        lines = [
            f"- {rendered}"
            for entry in chosen
            if (rendered := cls._memory_entry_selector_line(entry))
        ]
        if not lines:
            return ""
        return (
            "## Experience Decision Summary\n"
            "Before choosing the next tool or final answer, explicitly reconcile your move against these packet rules. "
            "If you override one, only do so because stronger current-sample evidence contradicts it.\n"
            + "\n".join(lines)
        )

    @classmethod
    def _build_sample_packet_text(
        cls,
        *,
        matched_skills_package: dict[str, Any],
        decision_skills_package: dict[str, Any],
        active_experience_package: dict[str, Any],
        selected_packet_tool_notes: str,
    ) -> str:
        """New definition:
        `Sample Packet` is the compact execution scaffold built from the
        already-selected sample-level experience. It names the chosen skills,
        judgment guardrails, linked rules, and tool lanes in one place so the
        executor can act on the packet instead of treating experience as loose
        background text.
        """
        matched_skill_titles = [
            str(item).strip()
            for item in matched_skills_package.get("selected_titles") or []
            if str(item).strip()
        ]
        decision_skill_titles = [
            str(item).strip()
            for item in decision_skills_package.get("selected_titles") or []
            if str(item).strip()
        ]
        active_rule_ids = [
            str(item).strip()
            for item in active_experience_package.get("selected_rule_ids") or []
            if str(item).strip()
        ]
        matched_skill_rule_ids = [
            str(item).strip()
            for item in matched_skills_package.get("selected_rule_ids") or []
            if str(item).strip()
        ]
        primary_path_tools = [
            str(item).strip()
            for item in active_experience_package.get("primary_path_tools") or []
            if str(item).strip()
        ]
        challenger_path_tools = [
            str(item).strip()
            for item in active_experience_package.get("challenger_path_tools") or []
            if str(item).strip()
        ]
        tool_prep_tools = [
            str(item).strip()
            for item in active_experience_package.get("tool_prep_tools") or []
            if str(item).strip()
        ]
        selected_entries = [
            dict(entry)
            for entry in active_experience_package.get("selected_entries") or []
            if isinstance(entry, dict)
        ]

        lines: list[str] = ["## Sample Packet"]
        if matched_skill_titles:
            lines.append("- Operational skills: " + ", ".join(matched_skill_titles))
        if decision_skill_titles:
            lines.append("- Judgment guardrails: " + ", ".join(decision_skill_titles))
        if matched_skill_rule_ids:
            lines.append("- Skill-linked rule ids: " + ", ".join(matched_skill_rule_ids))
        if active_rule_ids:
            lines.append("- Selected memory rule ids: " + ", ".join(active_rule_ids))
        if tool_prep_tools:
            lines.append("- Prep tools: " + ", ".join(tool_prep_tools))
        if primary_path_tools:
            lines.append("- Primary path tools: " + ", ".join(primary_path_tools))
        if challenger_path_tools:
            lines.append("- Challenger path tools: " + ", ".join(challenger_path_tools))
        decision_summary = cls._build_selected_experience_decision_summary(selected_entries)
        if decision_summary:
            lines.append(decision_summary)
        if selected_packet_tool_notes:
            lines.append(selected_packet_tool_notes.strip())

        if len(lines) == 1:
            return ""
        return "\n".join(lines).strip()

    async def _build_sample_experience_packet(
        self,
        *,
        task_type: str,
        skill_prompt: str,
        skill_decision_prompt: str,
        skill_metadata: Any,
        memory_entries: list[dict[str, Any]],
        task_description: str,
        profiling_section: str,
        candidates: list[str],
        tool_notes_section: str,
    ) -> dict[str, Any]:
        blocks = self._parse_skill_blocks(skill_prompt)
        decision_blocks = self._parse_decision_skill_blocks(skill_decision_prompt)
        available_titles = [
            str(block.get("title") or "").strip()
            for block in blocks
            if str(block.get("title") or "").strip()
        ]
        available_decision_titles = [
            str(block.get("title") or "").strip()
            for block in decision_blocks
            if str(block.get("title") or "").strip()
        ]
        injectable_entries = [
            dict(entry)
            for entry in (memory_entries or [])
            if isinstance(entry, dict)
            and bool(entry.get("injectable", True))
            and str(entry.get("rule_id") or "").strip()
        ]
        available_rule_ids = [
            str(entry.get("rule_id") or "").strip()
            for entry in injectable_entries
            if str(entry.get("rule_id") or "").strip()
        ]
        available_tool_names = [
            str(tool_name).strip()
            for tool_name in candidates
            if str(tool_name).strip()
        ]
        if not available_titles and not available_decision_titles and not available_rule_ids:
            return {
                "matched_skills_package": {
                    "text": str(skill_prompt or "").strip(),
                    "selected_titles": [],
                    "selected_count": 0,
                    "scored_blocks": [],
                    "selected_blocks": [],
                    "selected_rule_ids": [],
                },
                "decision_skills_package": {
                    "text": str(skill_decision_prompt or "").strip(),
                    "selected_titles": [],
                    "selected_count": 0,
                    "scored_blocks": [],
                    "selected_blocks": [],
                },
                "active_experience_package": {
                    "text": "",
                    "selected_rule_ids": [],
                    "suppressed_conflict_rule_ids": [],
                    "source_rule_count": 0,
                    "scored_rules": [],
                    "selected_entries": [],
                    "selector_guidance": "",
                    "selector_audit": {},
                    "primary_path_tools": [],
                    "challenger_path_tools": [],
                    "tool_prep_tools": [],
                    "challenger_requirement": "advisory",
                },
                "primary_path_tools": [],
                "challenger_path_tools": [],
                "challenger_requirement": "advisory",
                "tool_prep_tools": [],
                "prioritized_tool_names": [],
                "selector_guidance": "",
                "selector_audit": {},
            }

        skill_block_text = "\n\n".join(
            "\n".join(
                [
                    f"Title: {str(block.get('title') or '').strip()}",
                    f"Tools: {', '.join(block.get('tool_names') or []) or '(none)'}",
                    str(block.get("raw_text") or "").strip(),
                ]
            ).strip()
            for block in blocks
        ).strip() or "(none)"
        decision_block_text = "\n\n".join(
            "\n".join(
                [
                    f"Title: {str(block.get('title') or '').strip()}",
                    f"Tools: {', '.join(block.get('tool_names') or []) or '(none)'}",
                    str(block.get("raw_text") or "").strip(),
                ]
            ).strip()
            for block in decision_blocks
        ).strip() or "(none)"
        memory_catalog = "\n".join(
            line
            for entry in injectable_entries
            if (line := self._memory_entry_selector_line(entry))
        ).strip() or "(none)"
        messages = [
            {
                "role": "system",
                "content": (
                    "Build one sample-level experience packet for the current inference sample. "
                    "Return JSON only. Select the exact operational skills, decision skills, memory rule_ids, and priority tools that should be injected together. "
                    "Skills should drive the packet; selected rules should either support the chosen skills directly or provide a necessary general constraint for this sample. "
                    "Decision skills should sharpen what to judge and what not to overclaim. "
                    "Keep the packet compact and non-redundant. Do not invent titles, rule_ids, or tool names. "
                    "Do not stack multiple rules that all express the same conservative caution. Prefer a clear primary path/anchor, then at most one distinct guardrail."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Task Description:\n"
                    f"{str(task_description or '').strip() or '(none)'}\n\n"
                    "Profiling Section:\n"
                    f"{str(profiling_section or '').strip() or '(none)'}\n\n"
                    "Available Tool Names:\n"
                    f"{', '.join(available_tool_names) or '(none)'}\n\n"
                    "Available Skills:\n"
                    f"{skill_block_text}\n\n"
                    "Available Decision Skills:\n"
                    f"{decision_block_text}\n\n"
                    "Available Memory Rules:\n"
                    f"{memory_catalog}\n\n"
                    "Scoped Tool Notes:\n"
                    f"{str(tool_notes_section or '').strip() or '(none)'}\n\n"
                    "Return JSON with this schema:\n"
                    "{\n"
                    '  "selected_titles": ["exact skill title"],\n'
                    '  "selected_decision_titles": ["exact decision skill title"],\n'
                    '  "selected_rule_ids": ["exact_rule_id"],\n'
                    '  "primary_path_tools": ["exact_tool_name"],\n'
                    '  "challenger_path_tools": ["exact_tool_name"],\n'
                    '  "challenger_requirement": "advisory or compare_before_finish",\n'
                    '  "tool_prep_tools": ["exact_tool_name"],\n'
                    '  "selector_guidance": "short execution guidance for how to use the selected experience",\n'
                    '  "selection_audit": {\n'
                    '    "why_skills": "short reason",\n'
                    '    "why_decision_skills": "short reason",\n'
                    '    "why_rules": "short reason",\n'
                    '    "why_tools": "short reason"\n'
                    "  }\n"
                    "}\n"
                    f"If skills are available, choose 1 to {min(_MATCHED_SKILL_MAX_BLOCKS, max(1, len(available_titles)))} exact skill titles. "
                    f"Usually choose {_MATCHED_SKILL_DEFAULT_BLOCKS}; choose 1 only if one skill already covers the needed path; "
                    f"choose {_MATCHED_SKILL_MAX_BLOCKS} only if the third skill adds a materially different role.\n"
                    f"If decision skills are available, choose 1 to {min(_DECISION_SKILL_MAX_BLOCKS, max(1, len(available_decision_titles)))} exact decision skill titles. "
                    f"Usually choose {_DECISION_SKILL_DEFAULT_BLOCKS}; choose {_DECISION_SKILL_MAX_BLOCKS} only if the second one adds a distinct guardrail.\n"
                    f"Choose 0 to {_ACTIVE_EXPERIENCE_MAX_RULES} exact rule_ids.\n"
                    "Choose 0 to 3 exact primary_path_tools, 0 to 3 exact challenger_path_tools, and 0 to 3 exact tool_prep_tools.\n"
                    "Primary path tools are the best first path for this sample. Challenger path tools are a different path that may test the same judgment before finishing. Tool prep tools are setup tools that help either path.\n"
                    "Set `challenger_requirement` to `compare_before_finish` only when the selected skills/rules make two-path comparison materially necessary for this sample, such as threshold-sensitive cases where comparing two plausible continuations is part of the judgment. Otherwise use `advisory`.\n"
                    "Important: the selected operational skills must constrain the tool lanes. "
                    "Do not choose a primary or challenger path that the selected skills explicitly place in Avoid. "
                    "If a skill says a chain should be avoided under the current trigger conditions, do not return that chain as the primary path."
                ),
            },
        ]
        last_failure = "invalid sample experience packet JSON"
        for attempt in range(MAX_SKILL_SELECTOR_ATTEMPTS):
            response = await self.llm.complete(
                messages,
                response_format="json",
                temperature=0.0,
            )
            parsed = self._json_dict_from_response(
                response,
                required_keys={"selected_titles", "selected_decision_titles", "selected_rule_ids"},
            )
            selected_titles = self._normalize_selected_skill_titles(
                parsed.get("selected_titles"),
                available_titles=available_titles,
            )
            selected_decision_titles = self._normalize_selected_skill_titles(
                parsed.get("selected_decision_titles"),
                available_titles=available_decision_titles,
            )
            selected_rule_ids = self._normalize_selected_rule_ids(
                parsed.get("selected_rule_ids"),
                available_rule_ids=available_rule_ids,
            )
            tool_lanes = self._normalize_selected_tool_lanes(
                parsed,
                available_tool_names=available_tool_names,
            )
            challenger_requirement = self._normalize_challenger_requirement(
                parsed.get("challenger_requirement")
            )
            tool_lanes = self._sanitize_tool_lanes_against_selected_skill_avoids(
                blocks=blocks,
                selected_titles=selected_titles,
                tool_lanes=tool_lanes,
            )
            requires_future_day_evidence = self._selected_packet_requires_explicit_future_day_evidence(
                blocks=blocks,
                selected_titles=selected_titles,
                memory_entries=injectable_entries,
                selected_rule_ids=selected_rule_ids,
            )
            if requires_future_day_evidence and tool_lanes["primary_path_tools"] and not self._tool_chain_has_forecast_step(
                tool_lanes["primary_path_tools"]
            ):
                last_failure = (
                    "primary_path_tools must include an explicit forecast-producing tool for this sample "
                    "because the selected experience defines the judgment as first future day versus last observed day"
                )
                if attempt + 1 >= MAX_SKILL_SELECTOR_ATTEMPTS:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON picked an observed-only primary path for a sample whose selected "
                            "experience requires explicit future-day evidence. Return corrected JSON now. "
                            "Keep the same compact packet style, but make `primary_path_tools` include a forecast-producing tool "
                            "or leave it empty if no valid primary path exists."
                        ),
                    }
                )
                continue
            if requires_future_day_evidence and not tool_lanes["primary_path_tools"]:
                tool_lanes["primary_path_tools"] = self._derive_learned_primary_path_from_rule_ids(
                    memory_entries=injectable_entries,
                    candidate_rule_ids=list(
                        (
                            matched_skills_rule_ids := [
                                str(rule_id).strip()
                                for block in blocks
                                if str(block.get("title") or "").strip() in set(selected_titles)
                                for rule_id in (block.get("rule_ids") or [])
                                if str(rule_id).strip()
                            ]
                        )
                    )
                    or list(selected_rule_ids),
                    requires_future_day_evidence=True,
                    available_tool_names=available_tool_names,
                )
            prioritized_tool_names = self._flatten_selected_tool_lanes(
                primary_path_tools=tool_lanes["primary_path_tools"],
                challenger_path_tools=tool_lanes["challenger_path_tools"],
                tool_prep_tools=tool_lanes["tool_prep_tools"],
            )
            valid_skill_count = (
                len(selected_titles) == 0
                if not available_titles
                else 1 <= len(selected_titles) <= min(_MATCHED_SKILL_MAX_BLOCKS, len(available_titles))
            )
            valid_decision_count = (
                len(selected_decision_titles) == 0
                if not available_decision_titles
                else 1 <= len(selected_decision_titles) <= min(
                    _DECISION_SKILL_MAX_BLOCKS,
                    len(available_decision_titles),
                )
            )
            if (
                valid_skill_count
                and valid_decision_count
                and len(selected_rule_ids) <= _ACTIVE_EXPERIENCE_MAX_RULES
            ):
                selector_guidance = self._selector_guidance_from_selected_skills(
                    blocks=blocks,
                    selected_titles=selected_titles,
                    existing_guidance=str(parsed.get("selector_guidance") or "").strip(),
                )
                raw_audit = parsed.get("selection_audit") or {}
                selector_audit = raw_audit if isinstance(raw_audit, dict) else {}
                matched_skills_package = (
                    self._assemble_matched_skills_package(
                        blocks=blocks,
                        selected_titles=selected_titles,
                        skill_metadata=skill_metadata,
                        selector_audit=[
                            {
                                "title": title,
                                "rank": idx,
                                "role": "selected",
                                "why_selected": str(selector_audit.get("why_skills") or "").strip(),
                            }
                            for idx, title in enumerate(selected_titles, start=1)
                        ],
                    )
                    if selected_titles
                    else {
                        "text": str(skill_prompt or "").strip(),
                        "selected_titles": [],
                        "selected_count": 0,
                        "scored_blocks": [],
                        "selected_blocks": [],
                        "selected_rule_ids": [],
                    }
                )
                decision_skills_package = (
                    self._assemble_decision_skills_package(
                        blocks=decision_blocks,
                        selected_titles=selected_decision_titles,
                        selector_audit=[
                            {
                                "title": title,
                                "rank": idx,
                                "role": "selected",
                                "why_selected": str(
                                    selector_audit.get("why_decision_skills") or ""
                                ).strip(),
                            }
                            for idx, title in enumerate(selected_decision_titles, start=1)
                        ],
                    )
                    if selected_decision_titles
                    else {
                        "text": str(skill_decision_prompt or "").strip(),
                        "selected_titles": [],
                        "selected_count": 0,
                        "scored_blocks": [],
                        "selected_blocks": [],
                    }
                )
                active_experience_package = self._build_selected_experience_package(
                    memory_entries=injectable_entries,
                    selected_rule_ids=selected_rule_ids,
                    selector_guidance=selector_guidance,
                    selector_audit=selector_audit,
                    primary_path_tools=tool_lanes["primary_path_tools"],
                    challenger_path_tools=tool_lanes["challenger_path_tools"],
                    tool_prep_tools=tool_lanes["tool_prep_tools"],
                    challenger_requirement=challenger_requirement,
                )
                return {
                    "matched_skills_package": matched_skills_package,
                    "decision_skills_package": decision_skills_package,
                    "active_experience_package": active_experience_package,
                    "primary_path_tools": list(tool_lanes["primary_path_tools"]),
                    "challenger_path_tools": list(tool_lanes["challenger_path_tools"]),
                    "challenger_requirement": challenger_requirement,
                    "tool_prep_tools": list(tool_lanes["tool_prep_tools"]),
                    "prioritized_tool_names": prioritized_tool_names,
                    "selector_guidance": selector_guidance,
                    "selector_audit": selector_audit,
                }
            last_failure = (
                f"selected_titles must contain 1 to {min(_MATCHED_SKILL_MAX_BLOCKS, len(available_titles))} exact titles"
                if available_titles and not valid_skill_count
                else (
                    f"selected_decision_titles must contain 1 to {min(_DECISION_SKILL_MAX_BLOCKS, len(available_decision_titles))} exact titles"
                    if available_decision_titles and not valid_decision_count
                    else f"selected_rule_ids must contain at most {_ACTIVE_EXPERIENCE_MAX_RULES} exact rule_ids"
                )
            )
            if attempt + 1 >= MAX_SKILL_SELECTOR_ATTEMPTS:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous JSON was invalid. "
                        f"{last_failure}. "
                        "Return corrected JSON now using exact provided identifiers only."
                    ),
                }
            )
        raise RuntimeError(f"sample experience selector returned {last_failure}")

    async def _build_matched_skills_package(
        self,
        *,
        skill_prompt: str,
        skill_metadata: Any,
        task_description: str,
        profiling_section: str,
        candidates: list[str],
    ) -> dict[str, Any]:
        blocks = self._parse_skill_blocks(skill_prompt)
        if not blocks:
            return {
                "text": str(skill_prompt or "").strip(),
                "selected_titles": [],
                "selected_count": 0,
                "scored_blocks": [],
                "selected_blocks": [],
                "selected_rule_ids": [],
            }
        if len(blocks) == 1:
            only_title = str(blocks[0].get("title") or "").strip()
            return self._assemble_matched_skills_package(
                blocks=blocks,
                selected_titles=[only_title] if only_title else [],
                skill_metadata=skill_metadata,
                selector_audit=[{
                    "title": only_title,
                    "rank": 1,
                    "role": "primary",
                    "why_selected": "only available skill",
                }] if only_title else [],
            )

        available_titles = [
            str(block.get("title") or "").strip()
            for block in blocks
            if str(block.get("title") or "").strip()
        ]
        skill_block_text = "\n\n".join(
            "\n".join(
                [
                    f"Title: {str(block.get('title') or '').strip()}",
                    f"Tools: {', '.join(block.get('tool_names') or []) or '(none)'}",
                    str(block.get("raw_text") or "").strip(),
                ]
            ).strip()
            for block in blocks
        ).strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "Select the best sample-matched skills for the current inference sample. "
                    "Return JSON only. Choose 1 to 3 exact skill titles from the provided list. "
                    f"Usually choose {_MATCHED_SKILL_DEFAULT_BLOCKS}. Choose 1 only if one skill fully covers the operational strategy. "
                    f"Choose {_MATCHED_SKILL_MAX_BLOCKS} only if the third skill adds a materially different role such as a constraint, caveat, or alternative path. "
                    "Do not choose redundant restatements. Do not invent titles."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Task Description:\n"
                    f"{str(task_description or '').strip() or '(none)'}\n\n"
                    "Profiling Section:\n"
                    f"{str(profiling_section or '').strip() or '(none)'}\n\n"
                    "Available Tool Names:\n"
                    f"{', '.join(str(item).strip() for item in candidates if str(item).strip()) or '(none)'}\n\n"
                    "Available Skills:\n"
                    f"{skill_block_text}\n\n"
                    "Return JSON with this schema:\n"
                    "{\n"
                    '  "selected_titles": ["exact title 1", "exact title 2"],\n'
                    '  "selection_audit": [\n'
                    '    {"title": "exact title 1", "role": "primary", "why_selected": "short reason"},\n'
                    '    {"title": "exact title 2", "role": "constraint", "why_selected": "short reason"}\n'
                    "  ]\n"
                    "}\n"
                    "The titles must exactly match the provided skill titles."
                ),
            },
        ]
        last_failure = "invalid skill selection JSON"
        for attempt in range(MAX_SKILL_SELECTOR_ATTEMPTS):
            response = await self.llm.complete(
                messages,
                response_format="json",
                temperature=0.0,
            )
            parsed = self._json_dict_from_response(response, required_keys={"selected_titles"})
            selected_titles = self._normalize_selected_skill_titles(
                parsed.get("selected_titles"),
                available_titles=available_titles,
            )
            if 1 <= len(selected_titles) <= min(_MATCHED_SKILL_MAX_BLOCKS, len(available_titles)):
                raw_audit = parsed.get("selection_audit") or []
                selector_audit: list[dict[str, Any]] = []
                if isinstance(raw_audit, list):
                    selected_title_set = set(selected_titles)
                    for item in raw_audit:
                        if not isinstance(item, dict):
                            continue
                        normalized_title_list = self._normalize_selected_skill_titles(
                            [item.get("title")],
                            available_titles=available_titles,
                        )
                        if not normalized_title_list:
                            continue
                        normalized_title = normalized_title_list[0]
                        if normalized_title not in selected_title_set:
                            continue
                        selector_audit.append(
                            {
                                "title": normalized_title,
                                "role": str(item.get("role") or "").strip(),
                                "why_selected": str(item.get("why_selected") or "").strip(),
                            }
                        )
                return self._assemble_matched_skills_package(
                    blocks=blocks,
                    selected_titles=selected_titles,
                    skill_metadata=skill_metadata,
                    selector_audit=selector_audit,
                )
            last_failure = (
                f"selected_titles must contain 1 to {min(_MATCHED_SKILL_MAX_BLOCKS, len(available_titles))} "
                "exact titles from the provided list"
            )
            if attempt + 1 >= MAX_SKILL_SELECTOR_ATTEMPTS:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous JSON was invalid. "
                        f"{last_failure}. "
                        "Return corrected JSON now using exact provided titles only."
                    ),
                }
            )
        raise RuntimeError(f"skill selector returned {last_failure}")

    @staticmethod
    def _entry_has_path_guidance(entry: dict[str, Any]) -> bool:
        return bool(
            entry.get("preferred_tools")
            or entry.get("avoid_tools")
            or entry.get("preferred_tool_chains")
            or entry.get("avoid_tool_chains")
        )

    @staticmethod
    def _chain_coverage_score(chain: Any, candidate_set: set[str]) -> float:
        if not isinstance(chain, list):
            return 0.0
        tools = [str(item).strip() for item in chain if str(item).strip()]
        if not tools:
            return 0.0
        return sum(1 for tool in tools if tool in candidate_set) / len(tools)

    @classmethod
    def _score_memory_entry_for_sample(
        cls,
        entry: dict[str, Any],
        *,
        query_tokens: set[str],
        candidate_set: set[str],
        skill_rule_ids: set[str],
    ) -> tuple[float, dict[str, Any]]:
        entry_tokens = cls._experience_tokens(cls._memory_entry_text(entry))
        overlap = query_tokens & entry_tokens
        overlap_score = min(0.35, 0.03 * len(overlap))
        rule_id = str(entry.get("rule_id") or "").strip()
        skill_linked = bool(rule_id and rule_id in skill_rule_ids)

        preferred_tools = {
            str(item).strip()
            for item in entry.get("preferred_tools") or []
            if str(item).strip()
        }
        avoid_tools = {
            str(item).strip()
            for item in entry.get("avoid_tools") or []
            if str(item).strip()
        }
        preferred_tool_hits = sorted(preferred_tools & candidate_set)
        avoid_tool_hits = sorted(avoid_tools & candidate_set)

        preferred_chains = [
            chain for chain in entry.get("preferred_tool_chains") or []
            if isinstance(chain, list) and chain
        ]
        avoid_chains = [
            chain for chain in entry.get("avoid_tool_chains") or []
            if isinstance(chain, list) and chain
        ]
        preferred_chain_score = max(
            (cls._chain_coverage_score(chain, candidate_set) for chain in preferred_chains),
            default=0.0,
        )
        avoid_chain_score = max(
            (cls._chain_coverage_score(chain, candidate_set) for chain in avoid_chains),
            default=0.0,
        )

        candidate_score = 0.0
        if preferred_tool_hits:
            candidate_score += min(0.18, 0.06 * len(preferred_tool_hits))
        if avoid_tool_hits:
            candidate_score += min(0.18, 0.06 * len(avoid_tool_hits))
        candidate_score += 0.18 * preferred_chain_score
        candidate_score += 0.18 * avoid_chain_score

        support = cls._memory_entry_support_count(entry)
        support_score = min(0.20, 0.02 * max(0, min(support, 10)))
        confidence_score = 0.20 * float(entry.get("confidence", 0.0) or 0.0)

        score = overlap_score + candidate_score + support_score + confidence_score
        if str(entry.get("kind") or "").strip() == "result_constraint":
            score += 0.03

        return score, {
            "rule_id": rule_id,
            "kind": str(entry.get("kind") or "").strip(),
            "summary": str(entry.get("summary") or "").strip(),
            "support_count": support,
            "confidence": float(entry.get("confidence", 0.0) or 0.0),
            "score": round(score, 6),
            "query_overlap_tokens": sorted(overlap)[:12],
            "preferred_tool_hits": preferred_tool_hits,
            "avoid_tool_hits": avoid_tool_hits,
            "preferred_chain_hit_ratio": round(preferred_chain_score, 4),
            "avoid_chain_hit_ratio": round(avoid_chain_score, 4),
            "has_path_guidance": cls._entry_has_path_guidance(entry),
            "conflict_rule_ids": cls._memory_entry_conflict_ids(entry),
            "skill_linked": skill_linked,
        }

    @classmethod
    def _render_active_experience_entry(cls, entry: dict[str, Any]) -> str:
        parts: list[str] = []
        summary = str(entry.get("summary") or "").strip()
        if summary:
            parts.append(f"rule: {summary}")
        conditions = [str(item).strip() for item in entry.get("conditions") or [] if str(item).strip()]
        if conditions:
            parts.append(f"when: {', '.join(conditions[:2])}")
        rationale = str(entry.get("rationale") or "").strip()
        if rationale:
            parts.append(f"why: {rationale}")

        path_guidance = ""
        preferred_paths = []
        for chain in entry.get("preferred_tool_chains") or []:
            if isinstance(chain, list):
                tools = [str(item).strip() for item in chain if str(item).strip()]
                if tools:
                    preferred_paths.append(" -> ".join(tools))
        if preferred_paths:
            path_guidance = f"prefer_path: {' || '.join(preferred_paths[:1])}"
        else:
            avoid_paths = []
            for chain in entry.get("avoid_tool_chains") or []:
                if isinstance(chain, list):
                    tools = [str(item).strip() for item in chain if str(item).strip()]
                    if tools:
                        avoid_paths.append(" -> ".join(tools))
            if avoid_paths:
                path_guidance = f"avoid_path: {' || '.join(avoid_paths[:1])}"
        if path_guidance:
            parts.append(path_guidance)

        support = cls._memory_entry_support_count(entry)
        if support > 0:
            parts.append(f"evidence: {support} notes")
        rule_id = str(entry.get("rule_id") or "").strip()
        if rule_id:
            return f"[{rule_id}] " + " | ".join(part for part in parts if part)
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _memory_entry_role_tag(entry: dict[str, Any]) -> str:
        kind = str(entry.get("kind") or "").strip().lower()
        summary = str(entry.get("summary") or "").strip().lower()
        rationale = str(entry.get("rationale") or "").strip().lower()
        combined = f"{summary}\n{rationale}"
        has_prefer = bool(entry.get("preferred_tools") or entry.get("preferred_tool_chains"))
        has_avoid = bool(entry.get("avoid_tools") or entry.get("avoid_tool_chains"))

        if kind == "result_constraint":
            return "contract"
        if has_avoid or any(
            marker in combined
            for marker in (
                "avoid ",
                "do not ",
                "reserve ",
                "step back",
                "not by itself",
                "weaker than",
                "should not override",
            )
        ):
            return "guardrail"
        if has_prefer:
            return "path"
        if any(
            marker in combined
            for marker in (
                "anchor ",
                "classify from",
                "use the price path as the primary signal",
                "use endpoint",
                "match the chosen bucket",
                "treat ",
            )
        ):
            return "anchor"
        return "support"

    @classmethod
    def _build_experience_role_guidance(
        cls,
        selected_entries: list[dict[str, Any]],
    ) -> str:
        if not selected_entries:
            return ""
        role_to_ids: dict[str, list[str]] = {
            "path": [],
            "anchor": [],
            "support": [],
            "guardrail": [],
            "contract": [],
        }
        for entry in selected_entries:
            role = cls._memory_entry_role_tag(entry)
            rule_id = str(entry.get("rule_id") or "").strip()
            if rule_id and role in role_to_ids:
                role_to_ids[role].append(rule_id)

        if not any(role_to_ids[role] for role in ("path", "anchor", "guardrail")):
            return ""

        lines = [
            "## Experience Role Discipline",
            "Use selected experience by role, not as one blended instruction block.",
        ]
        contract_anchor_ids = role_to_ids["contract"][:1]
        if contract_anchor_ids:
            lines.append(
                "- Judgment anchor: satisfy "
                + ", ".join(contract_anchor_ids)
                + " first when forming the first candidate judgment. This anchor defines the task target and threshold; path advice only helps gather the evidence needed to satisfy it."
            )
        primary_ids = role_to_ids["anchor"][:2] + role_to_ids["path"][:1]
        if primary_ids:
            lines.append(
                "- Primary drivers: form the first candidate path/judgment from "
                + ", ".join(primary_ids)
                + "."
            )
        if role_to_ids["support"]:
            lines.append(
                "- Secondary support: use "
                + ", ".join(role_to_ids["support"][:2])
                + " only to sharpen or tilt an existing candidate, not to replace it."
            )
        if role_to_ids["guardrail"]:
            lines.append(
                "- Guardrails: use "
                + ", ".join(role_to_ids["guardrail"][:2])
                + " only as sanity checks after a concrete candidate exists. They may reject overclaim, but should not erase a clear candidate unless they expose a direct mismatch with the task target, numeric threshold, horizon, or actually produced tool evidence."
            )
        if role_to_ids["contract"]:
            lines.append(
                "- Output constraints: still satisfy "
                + ", ".join(role_to_ids["contract"][:1])
                + " exactly at delivery time. Do not treat this as a delivery-only reminder if it already defines the sample's judgment target."
            )
        return "\n".join(lines).strip()

    @classmethod
    def _build_active_experience_package(
        cls,
        *,
        memory_entries: Any,
        task_description: str,
        profiling_section: str,
        candidates: list[str],
        skill_rule_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(memory_entries, list) or not memory_entries:
            return {
                "text": "",
                "selected_rule_ids": [],
                "suppressed_conflict_rule_ids": [],
                "source_rule_count": 0,
                "scored_rules": [],
                "selected_entries": [],
            }

        candidate_set = {str(item).strip() for item in candidates if str(item).strip()}
        skill_rule_id_set = {
            str(rule_id).strip()
            for rule_id in (skill_rule_ids or [])
            if str(rule_id).strip()
        }
        query_tokens = cls._experience_tokens(
            "\n".join(part for part in [task_description, profiling_section] if str(part).strip())
        )

        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for raw_entry in memory_entries:
            if not isinstance(raw_entry, dict) or not bool(raw_entry.get("injectable", True)):
                continue
            entry = dict(raw_entry)
            score, audit = cls._score_memory_entry_for_sample(
                entry,
                query_tokens=query_tokens,
                candidate_set=candidate_set,
                skill_rule_ids=skill_rule_id_set,
            )
            scored.append((score, entry, audit))

        scored.sort(
            key=lambda item: (
                -item[0],
                -float(item[1].get("confidence", 0.0) or 0.0),
                -cls._memory_entry_support_count(item[1]),
                str(item[1].get("rule_id") or ""),
            )
        )

        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        selected_ids: set[str] = set()
        suppressed_conflicts: set[str] = set()
        linked_scored = [
            (score, entry, audit)
            for score, entry, audit in scored
            if bool(audit.get("skill_linked"))
        ]
        general_scored = [
            (score, entry, audit)
            for score, entry, audit in scored
            if not bool(audit.get("skill_linked"))
        ]

        def _try_select(pool: list[tuple[float, dict[str, Any], dict[str, Any]]], limit: int | None = None) -> None:
            picked = 0
            for _score, entry, audit in pool:
                nonlocal selected, selected_ids, suppressed_conflicts
                rule_id = str(entry.get("rule_id") or "").strip()
                if rule_id and rule_id in selected_ids:
                    continue
                conflict_ids = cls._memory_entry_conflict_ids(entry)
                if rule_id and rule_id in suppressed_conflicts:
                    continue
                if any(conflict_id in selected_ids for conflict_id in conflict_ids):
                    if rule_id:
                        suppressed_conflicts.add(rule_id)
                    continue
                selected.append((entry, audit))
                if rule_id:
                    selected_ids.add(rule_id)
                suppressed_conflicts.update(conflict_ids)
                picked += 1
                if len(selected) >= _ACTIVE_EXPERIENCE_MAX_RULES:
                    return
                if limit is not None and picked >= limit:
                    return

        if skill_rule_id_set:
            _try_select(linked_scored, limit=2)
        if len(selected) < _ACTIVE_EXPERIENCE_MAX_RULES:
            _try_select(general_scored)
        if len(selected) < _ACTIVE_EXPERIENCE_MAX_RULES and not general_scored:
            _try_select(linked_scored)

        if not selected:
            for _score, entry, audit in scored:
                selected.append((entry, audit))
                rule_id = str(entry.get("rule_id") or "").strip()
                if rule_id:
                    selected_ids.add(rule_id)
                break

        if scored and not any(cls._entry_has_path_guidance(entry) for entry, _ in selected):
            for _score, entry, audit in scored:
                rule_id = str(entry.get("rule_id") or "").strip()
                if not cls._entry_has_path_guidance(entry) or rule_id in selected_ids:
                    continue
                if selected:
                    selected[-1] = (entry, audit)
                else:
                    selected.append((entry, audit))
                selected_ids = {
                    str(item.get("rule_id") or "").strip()
                    for item, _audit in selected
                    if str(item.get("rule_id") or "").strip()
                }
                break

        rendered_lines: list[str] = []
        for entry, _audit in selected:
            rendered = cls._render_active_experience_entry(entry)
            if rendered:
                rendered_lines.append(f"- {rendered}")
        text = ""
        if rendered_lines:
            text = (
                "## Sample Experience Cards\n"
                "Use these cards as concise reference only. Do not override direct evidence from the current sample.\n"
                + "\n".join(rendered_lines)
                + "\n"
            )

        return {
            "text": text,
            "selected_rule_ids": [
                str(entry.get("rule_id") or "").strip()
                for entry, _audit in selected
                if str(entry.get("rule_id") or "").strip()
            ],
            "suppressed_conflict_rule_ids": sorted(suppressed_conflicts - selected_ids),
            "source_rule_count": len(scored),
            "scored_rules": [audit for _score, _entry, audit in scored[:12]],
            "selected_entries": [dict(entry) for entry, _audit in selected],
        }

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
                f"- Historical preferred branch path: {' -> '.join(tools)}.\n"
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
                f"- Historical preferred support path for branch design: {' -> '.join(tools)}.\n"
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
                f"- Historical preferred branch path for design: {' -> '.join(tools)}.\n"
            )
        elif anchor_tools:
            guide_line = (
                "- On the first `spawn_subagent` round, include at least one branch anchored "
                f"by `{anchor_tools[0]}` and at least one distinct alternative branch.\n"
                f"- Make that prior-guided branch explicit by starting its `tools_hint` with `{anchor_tools[0]}`.\n"
            )
            title_line = (
                "Training First-Spawn Prior:\n"
                f"- Historical preferred branch path: {' -> '.join(tools)}.\n"
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
                f"- Historical preferred direct reasoning path: {' -> '.join(tools)}.\n"
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

    @staticmethod
    def _looks_like_timestamp_literal(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        text = value.strip()
        if not text:
            return False
        if text.lstrip("+-").isdigit():
            return False
        return bool(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?", text)
        )

    @staticmethod
    def _current_source_supports_timestamp_slicing(data: Any) -> bool:
        try:
            import pandas as pd
        except ImportError:
            return False
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            return False
        index = getattr(data, "index", None)
        return bool(index is not None and pd.api.types.is_datetime64_any_dtype(index))

    @classmethod
    def _tool_argument_coordinate_misuse_message(
        cls,
        tool_name: str,
        *,
        tool_args: dict[str, Any] | None,
        current_data: Any,
        source_ref: str,
    ) -> str | None:
        """Detect when timestamp evidence is being misused as positional tool coordinates.

        New definition:
            `timestamp` fields returned by tools such as `window_stats` and `value_at`
            are evidence labels only. They are not automatically valid arguments for
            later index-sensitive tools. Unless the selected `source_ref` currently
            points to a pandas object with a datetime index, downstream coordinates
            must stay in that source's positional current-space.
        """
        name = str(tool_name or "").strip()
        args = tool_args if isinstance(tool_args, dict) else {}
        positional_fields: list[str] = []
        if name == "slice":
            positional_fields = ["start", "end"]
        elif name == "window_stats":
            positional_fields = ["start", "end"]
        elif name == "value_at":
            positional_fields = ["index"]
        else:
            return None

        offending = [
            field for field in positional_fields
            if cls._looks_like_timestamp_literal(args.get(field))
        ]
        if not offending:
            return None
        if name == "slice" and cls._current_source_supports_timestamp_slicing(current_data):
            return None

        field_text = ", ".join(offending)
        return (
            "Self-correction: you are using a timestamp-like value as a positional coordinate "
            f"for `{name}` (`{field_text}`). The selected source `{source_ref or 'current'}` "
            "currently uses positional current-space for index-sensitive tools. "
            "Treat `start_timestamp`, `end_timestamp`, and `timestamp` fields from earlier tool "
            "results as evidence labels only, not as the next tool arguments. "
            "Use integer indices relative to the selected source's `current_space`, or switch to "
            "`source_ref=\"original_input\"` only if you truly need to work on the raw history."
        )

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
            "skill_decision_prompt": envelope.task.metadata.get("skill_decision_prompt", ""),
            "skill_metadata": envelope.task.metadata.get("skill_metadata", {}),
            "memory_l2": envelope.task.metadata.get("memory_l2", ""),
            "memory_entries": envelope.task.metadata.get("memory_entries", []),
            "tool_notes": envelope.task.metadata.get("tool_notes", {}),
            "domain": envelope.task.metadata.get("domain", ""),
            "scope_keys": envelope.task.metadata.get("scope_keys", []),
            "reanalysis_context": envelope.task.metadata.get("reanalysis_context"),
            "_mode": envelope.task.metadata.get("_mode", "test"),
            "_defer_learning_record": envelope.task.metadata.get("_defer_learning_record", False),
            "_ground_truth": envelope.task.metadata.get("_ground_truth"),
            "indicator_output_keys": envelope.task.metadata.get("indicator_output_keys", []),
            "_tool_dropout_scope": envelope.task.metadata.get("_tool_dropout_scope", ""),
            "_tool_dropout_counts": envelope.task.metadata.get("_tool_dropout_counts", {}),
            "_disable_tool_dropout": envelope.task.metadata.get("_disable_tool_dropout", False),
            "_multi_agent_mode": envelope.task.metadata.get("_multi_agent_mode", True),
            "allowed_tools": envelope.task.metadata.get("allowed_tools"),
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

        # Explicit smoke/ablation setting: keep the main ReAct controller but
        # remove parallel sub-agent branches, which otherwise multiply provider
        # requests and can turn transient backpressure into misleading branch
        # evidence.
        if not bool(perception.get("_multi_agent_mode", True)):
            candidates = [c for c in candidates if c != "spawn_subagent"]

        candidates = self._filter_task_specific_candidates(
            candidates,
            task_type=task_type,
            fingerprint=perception.get("fingerprint"),
        )
        if perception.get("allowed_tools") is not None:
            allowed_tools = set(perception["allowed_tools"])
            candidates = [name for name in candidates if name in allowed_tools]

        subagent_slots = self._precompute_subagent_slots(
            trace_id=str(perception.get("trace_id") or ""),
            candidates=candidates,
            task_type=task_type,
            run_mode=run_mode,
            scope=str(perception.get("_tool_dropout_scope") or ""),
            scope_counts=perception.get("_tool_dropout_counts") or {},
            disable_tool_dropout=bool(
                perception.get("_disable_tool_dropout", self._disable_tool_dropout)
            ),
        )
        subagent_slot_section = self._build_subagent_slot_section(
            subagent_slots,
            task_type=task_type,
        )

        # SOUL (fixed) + injectable per-scope MEMORY → system prompt
        system_memory = ""
        soul = perception.get("soul_prompt", "")
        raw_memory = perception.get("memory_l2", "")
        memory_entries = perception.get("memory_entries", [])
        mem = raw_memory
        train_path_priors: list[dict[str, Any]] = []
        if run_mode == "train":
            train_path_priors = self._extract_train_path_priors(raw_memory)
            mem = self._strip_train_path_priors_from_memory(raw_memory)
        else:
            mem = ""
        profiling_section = self._build_profiling_section(
            perception.get("profiling_results", {}),
        )
        active_experience_package = {
            "text": "",
            "selected_rule_ids": [],
            "suppressed_conflict_rule_ids": [],
            "source_rule_count": 0,
            "scored_rules": [],
            "selected_entries": [],
            "selector_guidance": "",
            "selector_audit": {},
            "challenger_requirement": "advisory",
        }
        mem_parts = [p for p in [soul, mem] if p]
        if mem_parts:
            system_memory = "\n".join(mem_parts)

        experience_packet = {
            "matched_skills_package": {
                "text": str(perception.get("skill_prompt") or "").strip(),
                "selected_titles": [],
                "selected_count": 0,
                "scored_blocks": [],
                "selected_blocks": [],
                "selected_rule_ids": [],
            },
            "decision_skills_package": {
                "text": str(perception.get("skill_decision_prompt") or "").strip(),
                "selected_titles": [],
                "selected_count": 0,
                "scored_blocks": [],
                "selected_blocks": [],
            },
            "active_experience_package": active_experience_package,
            "primary_path_tools": [],
            "challenger_path_tools": [],
            "tool_prep_tools": [],
            "prioritized_tool_names": [],
            "selector_guidance": "",
            "selector_audit": {},
        }
        matched_skills_package = {
            "text": str(perception.get("skill_prompt") or "").strip(),
            "selected_titles": [],
            "selected_count": 0,
            "scored_blocks": [],
            "selected_blocks": [],
            "selected_rule_ids": [],
        }
        decision_skills_package = {
            "text": str(perception.get("skill_decision_prompt") or "").strip(),
            "selected_titles": [],
            "selected_count": 0,
            "scored_blocks": [],
            "selected_blocks": [],
        }
        pre_selector_tool_notes_section = self._build_tool_notes_section(
            perception.get("tool_notes", {}),
            candidates,
            task_type,
        )
        if run_mode != "train":
            experience_packet = await self._build_sample_experience_packet(
                task_type=task_type,
                skill_prompt=perception.get("skill_prompt", ""),
                skill_decision_prompt=perception.get("skill_decision_prompt", ""),
                skill_metadata=perception.get("skill_metadata", {}),
                memory_entries=memory_entries,
                task_description=perception.get("task_description", ""),
                profiling_section=profiling_section,
                candidates=candidates,
                tool_notes_section=pre_selector_tool_notes_section,
            )
            matched_skills_package = dict(experience_packet.get("matched_skills_package") or {})
            decision_skills_package = dict(experience_packet.get("decision_skills_package") or {})
            active_experience_package = dict(experience_packet.get("active_experience_package") or {})
            (
                matched_skills_package,
                decision_skills_package,
                active_experience_package,
            ) = self._sanitize_inference_sample_experience_packet(
                task_type=task_type,
                matched_skills_package=matched_skills_package,
                decision_skills_package=decision_skills_package,
                active_experience_package=active_experience_package,
                primary_path_tools=list(experience_packet.get("primary_path_tools") or []),
                challenger_path_tools=list(experience_packet.get("challenger_path_tools") or []),
                tool_prep_tools=list(experience_packet.get("tool_prep_tools") or []),
            )
            prioritized_tool_names = [
                str(tool_name).strip()
                for tool_name in experience_packet.get("prioritized_tool_names") or []
                if str(tool_name).strip()
            ]
            if prioritized_tool_names:
                priority_set = set(prioritized_tool_names)
                candidates = prioritized_tool_names + [
                    tool_name for tool_name in candidates if tool_name not in priority_set
                ]
        tool_names_text = executor_prompt.build_tool_name_text(candidates)
        selected_tool_note_keys = self._selected_tool_note_keys(
            perception.get("tool_notes", {}),
            candidates,
            task_type,
        )
        tool_notes_section = self._build_tool_notes_section(
            perception.get("tool_notes", {}),
            candidates,
            task_type,
        )
        selected_packet_tool_notes = self._build_selected_packet_tool_notes_section(
            perception.get("tool_notes", {}),
            primary_path_tools=list(active_experience_package.get("primary_path_tools") or []),
            challenger_path_tools=list(active_experience_package.get("challenger_path_tools") or []),
            tool_prep_tools=list(active_experience_package.get("tool_prep_tools") or []),
            task_type=task_type,
        )
        sample_packet_text = self._build_sample_packet_text(
            matched_skills_package=matched_skills_package,
            decision_skills_package=decision_skills_package,
            active_experience_package=active_experience_package,
            selected_packet_tool_notes=selected_packet_tool_notes,
        )

        # SKILL → first-turn user message (scene-specific SOP)
        skills_section = str(matched_skills_package.get("text") or "").strip()
        decision_skills_section = str(decision_skills_package.get("text") or "").strip()

        logger.debug(
            f"[Executor.plan] Memory injection: SOUL={len(soul)}, "
            f"MEMORY={len(mem)}, RAW_MEMORY={len(raw_memory)}, "
            f"SKILL={len(skills_section)}, "
            f"DECISION_SKILL={len(decision_skills_section)}, "
            f"ToolNotes={len(perception.get('tool_notes', {}))} tools"
        )

        modality_hint = ""

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
            "selected_packet_tool_notes": selected_packet_tool_notes,
            "sample_packet_text": sample_packet_text,
            "skills_section": skills_section,
            "decision_skills_section": decision_skills_section,
            "system_memory": system_memory,
            "memory_audit": {
                "soul_text": soul,
                "memory_text": mem,
                "raw_memory_text": raw_memory,
                "system_memory_text": system_memory,
                "skills_text": skills_section,
                "skills_decision_text": decision_skills_section,
                "decision_skills_text": decision_skills_section,
                "tool_notes_text": tool_notes_section,
                "tool_notes_section": tool_notes_section,
                "selected_packet_tool_notes": selected_packet_tool_notes,
                "sample_packet_text": sample_packet_text,
                "tool_note_keys": selected_tool_note_keys,
                "tool_note_count": len(selected_tool_note_keys),
                "matched_skills_text": skills_section,
                "matched_skill_titles": list(matched_skills_package.get("selected_titles") or []),
                "matched_skill_count": int(matched_skills_package.get("selected_count") or 0),
                "matched_skill_scored_blocks": list(matched_skills_package.get("scored_blocks") or []),
                "matched_skill_selected_blocks": list(matched_skills_package.get("selected_blocks") or []),
                "matched_skill_rule_ids": list(matched_skills_package.get("selected_rule_ids") or []),
                "decision_skill_titles": list(decision_skills_package.get("selected_titles") or []),
                "decision_skill_count": int(decision_skills_package.get("selected_count") or 0),
                "decision_skill_scored_blocks": list(decision_skills_package.get("scored_blocks") or []),
                "decision_skill_selected_blocks": list(decision_skills_package.get("selected_blocks") or []),
                "active_experience_text": str(active_experience_package.get("text") or "").strip(),
                "active_rule_ids": list(active_experience_package.get("selected_rule_ids") or []),
                "active_rule_count": len(active_experience_package.get("selected_rule_ids") or []),
                "active_suppressed_conflict_rule_ids": list(
                    active_experience_package.get("suppressed_conflict_rule_ids") or []
                ),
                "memory_entry_count": int(active_experience_package.get("source_rule_count") or 0),
                "active_experience_scored_rules": list(
                    active_experience_package.get("scored_rules") or []
                ),
                "active_experience_selected_entries": list(
                    active_experience_package.get("selected_entries") or []
                ),
                "primary_path_tools": list(experience_packet.get("primary_path_tools") or []),
                "challenger_path_tools": list(experience_packet.get("challenger_path_tools") or []),
                "tool_prep_tools": list(experience_packet.get("tool_prep_tools") or []),
                "experience_selector_guidance": str(
                    active_experience_package.get("selector_guidance") or ""
                ).strip(),
                "experience_selector_audit": dict(
                    active_experience_package.get("selector_audit") or {}
                ),
                "prioritized_tool_names": list(experience_packet.get("prioritized_tool_names") or []),
                "challenger_requirement": self._normalize_challenger_requirement(
                    experience_packet.get("challenger_requirement")
                    or active_experience_package.get("challenger_requirement")
                ),
            },
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
            "scope_keys": perception.get("scope_keys", []),
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
        series_sources = self._initial_series_sources(current_data, current_series_state)
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

        task_boundary_section = self._render_prompt_section(
            "Task Boundary",
            self._build_task_boundary_section(
                plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
                scope_keys=plan.get("scope_keys", []),
            ),
        )
        task_semantic_section = self._render_prompt_section(
            "Task Semantic Guidance",
            self._build_task_semantic_guidance_section(
                task_type=plan["task_type"],
                scope_keys=plan.get("scope_keys", []),
            ),
        )
        objective_sections = self._join_prompt_sections(
            task_boundary_section,
            task_semantic_section,
        )
        execution_context_section = self._render_prompt_section(
            "Execution Context",
            "",
        )
        control_context_section = self._render_prompt_section(
            "Control Context",
            self._join_prompt_sections(
                self._build_action_state_section(
                    state="need_evidence",
                    required_final_type=expected_final_type,
                    execution_log=execution_log,
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ),
                self._build_train_spawn_prior_section(
                    plan.get("train_spawn_prior_path", []),
                ) if is_train else "",
                plan.get("subagent_slot_section", ""),
            ),
        )
        output_contract_section = self._render_prompt_section(
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
                scope_keys=plan.get("scope_keys", []),
            ),
        )
        support_section = self._render_prompt_section(
            "Support",
            self._join_prompt_sections(
                plan.get("reanalysis_section", ""),
                plan.get("sample_packet_text", ""),
                plan.get("selected_packet_tool_notes", "") or plan.get("tool_notes_section", ""),
            ),
        )
        decision_section = self._build_decision_section(
            allow_finish=self._main_finish_allowed_this_turn(
                execution_log=execution_log,
                plan=plan,
            ),
            spawn_available="spawn_subagent" in candidates,
            task_type=plan["task_type"],
        )

        first_user = executor_prompt.REACT_FIRST_TURN.format(
            task_prompt=plan.get("task_description", ""),
            task_boundary_section=objective_sections,
            fingerprint=self._format_prompt_payload(
                self._prepare_prompt_fingerprint(
                    plan["fingerprint"],
                    profiling_section=plan.get("profiling_section", ""),
                ),
                limit=None,
            ),
            profiling_section=plan.get("profiling_section", ""),
            execution_context_section=execution_context_section,
            control_context_section=control_context_section,
            output_contract_section=output_contract_section,
            support_section=support_section,
            decision_section=decision_section,
            tool_names=plan["tool_names"],
        )

        base_control_context = self._normalize_control_context({
            "completion_notes": "",
            "audit": self._build_control_context_audit(
                memory_injected=plan.get("memory_audit"),
                prompt_sections=[
                    self._build_prompt_section_audit(name="system_prompt", text=system_prompt),
                    self._build_prompt_section_audit(name="first_user", text=first_user),
                    self._build_prompt_section_audit(name="task_boundary_section", text=task_boundary_section),
                    self._build_prompt_section_audit(name="task_semantic_section", text=task_semantic_section),
                    self._build_prompt_section_audit(name="execution_context_section", text=execution_context_section),
                    self._build_prompt_section_audit(name="control_context_section", text=control_context_section),
                    self._build_prompt_section_audit(name="output_contract_section", text=output_contract_section),
                    self._build_prompt_section_audit(name="support_section", text=support_section),
                    self._build_prompt_section_audit(name="decision_section", text=decision_section),
                ],
            ),
        })
        accepted_control_context = base_control_context

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
                scope_keys=plan.get("scope_keys", []),
            ) if is_train else 0
            spawned_candidate_count_now = self._count_training_spawn_candidates(
                execution_log,
                plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
                scope_keys=plan.get("scope_keys", []),
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
                allow_finish_for_parse = True
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
                if eval_only_turn:
                    spawn_only_turn = False
                    pending_spawn_requirement = False
                allow_finish = self._main_finish_allowed_this_turn(
                    execution_log=execution_log,
                    plan=plan,
                )
                allow_finish_for_parse = True
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
                        task_type=plan["task_type"],
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
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    ),
                ).strip(),
            })

            response = None
            decision: dict[str, Any] = {}
            thought = ""
            is_done = False
            advance_to_next_step = False
            non_action_self_corrections = 0
            decision_recovery_exhaustions = 0
            while True:
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
                        scope_keys=plan.get("scope_keys", []),
                    )
                response, decision = await self._request_structured_decision(
                    decision_conversation,
                    allow_tool_call=not repair_delivery_mode,
                    allow_finish=allow_finish_for_parse,
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

                if parse_failed:
                    parse_failure_reason = (
                        "invalid_decision_wrong_tool_family"
                        if str(thought or "").strip() == "invalid_decision_wrong_tool_family"
                        else "invalid_json_response"
                    )
                    invalid_json_msg = self._invalid_json_self_correction_message(
                        thought,
                        allow_tool_call=not repair_delivery_mode,
                        allow_finish=allow_finish,
                        spawn_available=spawn_available,
                        ordinary_tool_names=ordinary_tool_names,
                        evaluate_tool_names=evaluate_tool_names,
                    )
                    if repair_delivery_mode:
                        repair_delivery_feedback_messages = [invalid_json_msg]
                    trace.agent_step("executor", f"react_step_{step}", {
                        "self_correction": True,
                        "reason": parse_failure_reason,
                        "thought": thought,
                        "response_preview": self._decision_response_preview(response),
                        "same_step_retry": (
                            non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP
                        ),
                    })
                    conversation.append({
                        "role": "user",
                        "content": invalid_json_msg,
                    })
                    skip_generic_turn_prompt_once = True
                    if non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP:
                        non_action_self_corrections += 1
                        continue
                    if decision_recovery_exhaustions < MAX_DECISION_RECOVERY_EXHAUSTIONS_PER_STEP:
                        decision_recovery_exhaustions += 1
                        non_action_self_corrections = 0
                        continue
                    if repair_delivery_mode:
                        recoverable_decision = self._decision_from_recoverable_final(
                            self._recoverable_failure_record(
                                expected_type=expected_final_type,
                                task_type=plan["task_type"],
                                execution_log=execution_log,
                                horizon=plan.get("horizon"),
                                indicator_output_keys=plan.get("indicator_output_keys", []),
                            ),
                            thought=(
                                "Reason: Repair delivery already has a fully execution-grounded final "
                                "payload available, so the unresolved judgment is only delivery format. "
                                "Check: No additional tool result is needed because the final payload is "
                                "already recoverable from the execution trace. "
                                "Self-check: The required final payload artifact is already present in the "
                                "recoverable execution trace, and its basic sanity condition is simply that "
                                "no new evidence is needed to complete delivery. "
                                "Update: Keep the current recovered candidate and stop retrying format-only "
                                "turns because no rival candidate or new evidence is being introduced. "
                                "Action: Finish with that recovered payload unchanged."
                            ),
                        )
                        if recoverable_decision is not None:
                            decision = recoverable_decision
                            thought = str(decision.get("thought") or thought)
                            is_done = True
                            break
                    failure_reason = "decision_parse_exhausted"
                    trace.agent_step("executor", f"react_step_{step}", {
                        "self_correction": True,
                        "reason": failure_reason,
                        "thought": thought,
                        "response_preview": self._decision_response_preview(response),
                    })
                    raise DecisionOutputExhaustedError(
                        agent_kind="executor",
                        failure_reason=failure_reason,
                        partial_result=self._failure_partial_result(
                            expected_type=expected_final_type,
                            task_type=plan["task_type"],
                            execution_log=execution_log,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                            recoverable_final=recoverable_answer or recoverable_final,
                            failure_reason=failure_reason,
                        ),
                    )

                if not parse_failed:
                    conversation.append({
                        "role": "assistant",
                        "content": self._decision_to_conversation_text(decision),
                    })

                if not self._has_structured_thought(thought):
                    structured_thought_msg = (
                        "Self-correction: your `thought` must explicitly include "
                        "`Reason:`, `Check:`, `Self-check:`, `Update:`, and `Action:` in that order. "
                        "Please continue with the required structure."
                    )
                    if repair_delivery_mode:
                        repair_delivery_feedback_messages = [structured_thought_msg]
                    trace.agent_step("executor", f"react_step_{step}", {
                        "self_correction": True,
                        "reason": "unstructured_or_unparsed_response",
                        "decision_type": type(decision).__name__,
                        "decision_is_dict": isinstance(decision, dict),
                        "thought": thought,
                        "response_preview": self._decision_response_preview(response),
                        "same_step_retry": (
                            non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP
                        ),
                    })
                    conversation.append({
                        "role": "user",
                        "content": structured_thought_msg,
                    })
                    skip_generic_turn_prompt_once = True
                    if non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP:
                        non_action_self_corrections += 1
                        continue
                    if decision_recovery_exhaustions < MAX_DECISION_RECOVERY_EXHAUSTIONS_PER_STEP:
                        decision_recovery_exhaustions += 1
                        non_action_self_corrections = 0
                        continue
                    if repair_delivery_mode:
                        recoverable_decision = self._decision_from_recoverable_final(
                            self._recoverable_failure_record(
                                expected_type=expected_final_type,
                                task_type=plan["task_type"],
                                execution_log=execution_log,
                                horizon=plan.get("horizon"),
                                indicator_output_keys=plan.get("indicator_output_keys", []),
                            ),
                            thought=(
                                "Reason: Repair delivery already has an execution-grounded final payload, "
                                "and repeated formatting retries are no longer changing the judgment. "
                                "Check: The trace already contains the needed final evidence and no further "
                                "tool call would change it. "
                                "Self-check: The required final payload artifact is already present, and "
                                "its basic sanity condition is that another retry would not add any missing "
                                "evidence or change the payload content. "
                                "Update: Keep the recovered final candidate rather than attempting another "
                                "format-only retry, because no competing candidate or new evidence remains. "
                                "Action: Finish with that recovered payload."
                            ),
                        )
                        if recoverable_decision is not None:
                            decision = recoverable_decision
                            thought = str(decision.get("thought") or thought)
                            is_done = True
                            break
                    failure_reason = "decision_format_exhausted"
                    trace.agent_step("executor", f"react_step_{step}", {
                        "self_correction": True,
                        "reason": failure_reason,
                        "thought": thought,
                        "response_preview": self._decision_response_preview(response),
                    })
                    raise DecisionOutputExhaustedError(
                        agent_kind="executor",
                        failure_reason=failure_reason,
                        partial_result=self._failure_partial_result(
                            expected_type=expected_final_type,
                            task_type=plan["task_type"],
                            execution_log=execution_log,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                            recoverable_final=recoverable_answer or recoverable_final,
                            failure_reason=failure_reason,
                        ),
                    )

                tool_name = str(decision.get("tool") or "").strip()
                tool_args = decision.get("args", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}
                    decision["args"] = tool_args

                if (
                    not is_done
                    and tool_name == "spawn_subagent"
                    and not repair_delivery_mode
                ):
                    prepared_spawn_tasks, spawn_validation_reason, spawn_validation_detail = (
                        self._prepare_main_spawn_subagent_tasks(
                            tool_args=tool_args,
                            execution_log=execution_log,
                            plan=plan,
                            is_train=is_train,
                            needs_more_candidates=needs_more_candidates,
                        )
                    )
                    if spawn_validation_detail:
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "tool": "spawn_subagent",
                            "self_correction": True,
                            "reason": spawn_validation_reason or "invalid_subagent_slots",
                            "detail": spawn_validation_detail,
                            "same_step_retry": (
                                non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP
                            ),
                        })
                        conversation.append({"role": "user", "content": spawn_validation_detail})
                        skip_generic_turn_prompt_once = True
                        if non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP:
                            non_action_self_corrections += 1
                            continue
                        if decision_recovery_exhaustions < MAX_DECISION_RECOVERY_EXHAUSTIONS_PER_STEP:
                            decision_recovery_exhaustions += 1
                            non_action_self_corrections = 0
                            continue
                        failure_reason = str(
                            spawn_validation_reason or "invalid_subagent_slots_exhausted"
                        )
                        raise DecisionOutputExhaustedError(
                            agent_kind="executor",
                            failure_reason=failure_reason,
                            partial_result=self._failure_partial_result(
                                expected_type=expected_final_type,
                                task_type=plan["task_type"],
                                execution_log=execution_log,
                                horizon=plan.get("horizon"),
                                indicator_output_keys=plan.get("indicator_output_keys", []),
                                recoverable_final=recoverable_answer or recoverable_final,
                                failure_reason=failure_reason,
                            ),
                        )
                    decision["_prepared_spawn_tasks"] = list(prepared_spawn_tasks or [])
                break

            if advance_to_next_step:
                continue

            if is_done:
                repair_record = (
                    recoverable_answer if isinstance(recoverable_answer, dict) and recoverable_answer
                    else recoverable_final if isinstance(recoverable_final, dict) and recoverable_final
                    else None
                )
                agent_answer = decision.get("answer")
                agent_answer_type = decision.get("answer_type", "")
                agent_confidence = decision.get("confidence")
                agent_reasoning = self._coerce_reasoning_text(decision.get("reasoning", thought))
                agent_execution_context = self._normalize_execution_context(
                    decision.get("execution_context"),
                    execution_log=execution_log,
                )
                if (
                    repair_delivery_mode
                    and isinstance(repair_record, dict)
                    and not any(
                        agent_execution_context.get(key)
                        for key in ("tool_chain", "public_reasoning", "evidence_summary")
                    )
                ):
                    agent_execution_context = self._normalize_execution_context(
                        repair_record.get("execution_context"),
                        execution_log=execution_log,
                    )
                if (
                    repair_delivery_mode
                    and isinstance(repair_record, dict)
                    and not agent_reasoning
                ):
                    agent_reasoning = self._coerce_reasoning_text(
                        repair_record.get("reasoning", thought)
                    )
                agent_reasoning = self._normalize_reasoning_for_reuse(
                    agent_reasoning,
                    execution_log,
                    agent_execution_context,
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
                repair_base_control_context = base_control_context
                if repair_delivery_mode and isinstance(repair_record, dict):
                    repair_base_control_context = self._merge_control_context(
                        base_control_context,
                        repair_record.get("control_context"),
                    )
                agent_control_context = self._merge_control_context(
                    repair_base_control_context,
                    decision.get("control_context"),
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
                        scope_keys=plan.get("scope_keys", []),
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
                        recoverable_learning_summary = self._recoverable_learning_summary_record_from_execution_log(
                            execution_log=execution_log,
                            task_type=plan["task_type"],
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                        )
                        if recoverable_learning_summary is not None:
                            recoverable_final = recoverable_learning_summary
                            repair_delivery_mode = True
                            repair_delivery_feedback_messages = [correction_msg]
                        else:
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
                    recoverable_learning_summary = self._recoverable_learning_summary_record_from_execution_log(
                        execution_log=execution_log,
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    if recoverable_learning_summary is not None:
                        logger.info(
                            "[Executor] Normalized malformed learning_summary from execution log"
                        )
                        agent_answer = recoverable_learning_summary.get("answer")
                        agent_answer_type = str(
                            recoverable_learning_summary.get("answer_type") or agent_answer_type
                        )
                        agent_confidence = recoverable_learning_summary.get("confidence")
                        agent_reasoning = self._coerce_reasoning_text(
                            recoverable_learning_summary.get("reasoning", agent_reasoning)
                        )
                        agent_execution_context = self._normalize_execution_context(
                            recoverable_learning_summary.get("execution_context"),
                            execution_log=execution_log,
                        )
                        agent_control_context = self._merge_control_context(
                            agent_control_context,
                            recoverable_learning_summary.get("control_context"),
                        )
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "done": True,
                            "self_correction": True,
                            "reason": "normalized_invalid_learning_summary_from_execution_log",
                            "answer_type": agent_answer_type,
                            "answer_python_type": type(agent_answer).__name__,
                        })
                    if not self._is_valid_learning_summary_answer(agent_answer):
                        correction_msg = self._learning_summary_validation_message(agent_answer)
                        if step < MAX_REACT_STEPS:
                            logger.info(
                                "[Executor] Self-correction: malformed learning_summary payload, retrying"
                            )
                            if recoverable_learning_summary is not None:
                                recoverable_final = recoverable_learning_summary
                                repair_delivery_mode = True
                                repair_delivery_feedback_messages = [correction_msg]
                            else:
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
                        domain=plan.get("domain", ""),
                        scope_keys=plan.get("scope_keys", []),
                    )
                    and step < MAX_REACT_STEPS
                )
                if invalid_answer:
                    _, invalid_reason = self._validate_final_answer(
                        requested_type,
                        agent_answer,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        domain=plan.get("domain", ""),
                        scope_keys=plan.get("scope_keys", []),
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
                        f"Reason: {invalid_reason}. "
                        f"For forecast, return a numeric array. For indicator, return a numeric JSON object. "
                        f"For classification/correlation, return a label or JSON containing the label. "
                        f"Please continue and produce a valid final answer."
                    )
                    if requested_type == "forecast" and plan.get("horizon") is not None:
                        correction_msg += (
                            f" The forecast must contain exactly {plan['horizon']} time steps "
                            "of finite numeric values, with all requested channels preserved. "
                            "Return the complete tool forecast without appending or omitting values."
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

                if not is_train and not learning_summary_allowed:
                    lane_status = self._inference_packet_lane_status(
                        execution_log=execution_log,
                        memory_audit=plan.get("memory_audit"),
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    challenger_status = self._inference_challenger_finish_status(
                        execution_log=execution_log,
                        memory_audit=plan.get("memory_audit"),
                    )
                    if challenger_status.get("needs_prep_before_finish"):
                        correction_msg = self._prep_finish_correction_message(
                            challenger_status,
                        )
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "done": True,
                            "self_correction": True,
                            "reason": "prep_path_not_completed",
                            "completed_prep_tools": list(
                                challenger_status.get("completed_prep_tools") or []
                            ),
                            "remaining_prep_tools": list(
                                challenger_status.get("remaining_prep_tools") or []
                            ),
                            "executed_tools": list(challenger_status.get("executed_tools") or []),
                        })
                        if step < MAX_REACT_STEPS:
                            conversation.append({"role": "user", "content": correction_msg})
                        continue
                    if challenger_status.get("needs_anchor_evidence_before_finish"):
                        correction_msg = self._anchor_finish_correction_message(
                            challenger_status,
                        )
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "done": True,
                            "self_correction": True,
                            "reason": "anchor_evidence_not_completed",
                            "completed_anchor_tools": list(
                                challenger_status.get("completed_anchor_tools") or []
                            ),
                            "remaining_anchor_tools": list(
                                challenger_status.get("remaining_anchor_tools") or []
                            ),
                            "executed_tools": list(challenger_status.get("executed_tools") or []),
                        })
                        if step < MAX_REACT_STEPS:
                            conversation.append({"role": "user", "content": correction_msg})
                        continue
                    if (
                        challenger_status.get("needs_challenger_before_finish")
                        and not self._has_challenger_skip_note(agent_control_context)
                    ):
                        correction_msg = self._challenger_finish_correction_message(
                            challenger_status,
                        )
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "done": True,
                            "self_correction": True,
                            "reason": "challenger_path_not_tested",
                            "primary_path_tools": list(
                                challenger_status.get("primary_path_tools") or []
                            ),
                            "challenger_path_tools": list(
                                challenger_status.get("challenger_path_tools") or []
                            ),
                            "executed_tools": list(challenger_status.get("executed_tools") or []),
                        })
                        if step < MAX_REACT_STEPS:
                            conversation.append({"role": "user", "content": correction_msg})
                        continue
                    refresh_status = self._inference_packet_answer_refresh_status(
                        task_type=plan["task_type"],
                        execution_log=execution_log,
                        memory_audit=plan.get("memory_audit"),
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    if (refresh_status.get("needs_refresh_before_finish")
                        and not self._finish_payload_covers_stale_packet_evidence(
                            execution_context=agent_execution_context, refresh_status=refresh_status,
                        )):
                        correction_msg = self._stale_packet_answer_correction_message(
                            refresh_status,
                        )
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "done": True,
                            "self_correction": True,
                            "reason": "stale_packet_answer_after_new_evidence",
                            "latest_task_answer_tool": refresh_status.get("latest_task_answer_tool"),
                            "stale_packet_tools_after_answer": list(
                                refresh_status.get("stale_packet_tools_after_answer") or []
                            ),
                            "executed_tools": [e.get("tool", "") for e in execution_log if e.get("tool")],
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
                if solve_ready_current and not learning_summary_allowed:
                    agent_execution_context = self._recoverable_execution_context_from_trace(
                        tool_chain=self._reasoning_substantive_tool_chain(execution_log),
                        answer=agent_answer,
                        reasoning=agent_reasoning,
                        execution_context=agent_execution_context,
                        execution_log=execution_log,
                    )
                    agent_reasoning = self._normalize_reasoning_for_reuse(
                        agent_reasoning,
                        execution_log,
                        agent_execution_context,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                    execution_context_verdict = await self._call_execution_context_verdict(
                        agent_execution_context,
                        execution_log,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                                task_type=requested_type,
                                horizon=plan.get("horizon"),
                                indicator_output_keys=plan.get("indicator_output_keys", []),
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
                    if not learning_summary_allowed:
                        refresh_status_now = self._inference_packet_answer_refresh_status(
                            task_type=plan["task_type"],
                            execution_log=execution_log,
                            memory_audit=plan.get("memory_audit"),
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                        )
                        if (refresh_status_now.get("needs_refresh_before_finish")
                            and not self._finish_payload_covers_stale_packet_evidence(
                                execution_context=agent_execution_context, refresh_status=refresh_status_now,
                            )):
                            recoverable_final = None
                            recoverable_answer = None

                reasoning_verdict = {"passed": True, "missing": [], "explanation": ""}
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                ):
                    reasoning_verdict = await self._call_stepwise_reasoning_verdict(
                        agent_reasoning,
                        execution_log,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                                task_type=requested_type,
                                horizon=plan.get("horizon"),
                                indicator_output_keys=plan.get("indicator_output_keys", []),
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
                if is_train:
                    if needs_more_candidates:
                        pending_spawn_requirement = True
                normalized_tasks = decision.get("_prepared_spawn_tasks")
                if not isinstance(normalized_tasks, list):
                    normalized_tasks, spawn_validation_reason, spawn_validation_detail = (
                        self._prepare_main_spawn_subagent_tasks(
                            tool_args=tool_args,
                            execution_log=execution_log,
                            plan=plan,
                            is_train=is_train,
                            needs_more_candidates=needs_more_candidates,
                        )
                    )
                    if spawn_validation_detail:
                        conversation.append({"role": "user", "content": spawn_validation_detail})
                        trace.agent_step("executor", f"react_step_{step}", {
                            "thought": thought,
                            "tool": "spawn_subagent",
                            "self_correction": True,
                            "reason": spawn_validation_reason or "invalid_subagent_slots",
                            "detail": spawn_validation_detail,
                        })
                        continue
                sub_results = await self._spawn_subagents(
                    {"tasks": normalized_tasks}, plan, envelope,
                )
                observation_msg = sub_results["observation"]
                trace.policy_transition(
                    state_before=current_series_state,
                    action="spawn_subagent",
                    args={"n_tasks": sub_results["n_tasks"]},
                    result=sub_results["results"],
                    state_after=current_series_state,
                    llm_called=True,
                )
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
                            scope_keys=plan.get("scope_keys", []),
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

            selected_data, selected_state, selected_source_ref, source_error = self._resolve_series_source(
                self._tool_series_source_ref(tool_name, tool_args),
                current_data=current_data,
                current_series_state=current_series_state,
                series_sources=series_sources,
            )
            if source_error:
                conversation.append({"role": "user", "content": source_error})
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought,
                    "tool": tool_name,
                    "self_correction": True,
                    "reason": "unknown_source_ref",
                    "source_ref": tool_args.get("source_ref"),
                })
                continue

            runtime_incompatibility_msg = self._tool_runtime_incompatibility_message(
                tool_name,
                current_data=selected_data,
                current_series_state=selected_state,
            )
            if runtime_incompatibility_msg:
                conversation.append({"role": "user", "content": runtime_incompatibility_msg})
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought,
                    "tool": tool_name,
                    "self_correction": True,
                    "reason": "runtime_incompatible_tool",
                    "source_ref": selected_source_ref,
                })
                continue
            coordinate_misuse_msg = self._tool_argument_coordinate_misuse_message(
                tool_name,
                tool_args=tool_args,
                current_data=selected_data,
                source_ref=selected_source_ref,
            )
            if coordinate_misuse_msg:
                conversation.append({"role": "user", "content": coordinate_misuse_msg})
                trace.agent_step("executor", f"react_step_{step}", {
                    "thought": thought,
                    "tool": tool_name,
                    "self_correction": True,
                    "reason": "coordinate_argument_misuse",
                    "source_ref": selected_source_ref,
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

            if not is_train:
                lane_status = self._inference_packet_lane_status(
                    execution_log=execution_log,
                    memory_audit=plan.get("memory_audit"),
                    task_type=plan["task_type"],
                )
                if (
                    lane_status.get("enforce")
                    and str(tool_name or "").strip() not in set(lane_status.get("allowed_now") or [])
                ):
                    correction_msg = self._packet_lane_correction_message(lane_status)
                    conversation.append({"role": "user", "content": correction_msg})
                    trace.agent_step("executor", f"react_step_{step}", {
                        "thought": thought,
                        "tool": tool_name,
                        "self_correction": True,
                        "reason": "packet_lane_not_followed",
                        "allowed_now": list(lane_status.get("allowed_now") or []),
                        "primary_path_tools": list(lane_status.get("primary_path_tools") or []),
                        "challenger_path_tools": list(lane_status.get("challenger_path_tools") or []),
                        "tool_prep_tools": list(lane_status.get("tool_prep_tools") or []),
                        "executed_tools": list(lane_status.get("executed_tools") or []),
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
                    scope_keys=plan.get("scope_keys", []),
                )
                spawned_candidate_count = self._count_training_spawn_candidates(
                    execution_log,
                    plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                    scope_keys=plan.get("scope_keys", []),
                )
                terminal_single_evaluation = step >= MAX_REACT_STEPS - 1 and self._single_execution_candidate(
                    execution_log, task_type=plan["task_type"], horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []), scope_keys=plan.get("scope_keys", []), require_score=False,
                ) is not None
                if (
                    candidate_count < MIN_TRAINING_CANDIDATES
                    or spawned_candidate_count < MIN_TRAINING_CANDIDATES
                ) and not terminal_single_evaluation:
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
                        domain=plan.get("domain", ""),
                        scope_keys=plan.get("scope_keys", []),
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
                tool_name, tool_args, selected_data, text_data, images,
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
                    self._register_series_source(
                        series_sources,
                        ref=self._series_source_ref_for_step(step, tool_name),
                        data=downstream_data,
                        state=current_series_state,
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
            # Policy-compilation audit boundary.  These are the states exposed
            # by the executor itself; do not reconstruct hidden state from the
            # LLM transcript.  Every main-loop action was selected by the
            # structured controller call immediately above, so it is marked as
            # LLM-dependent for the ReAct cost baseline.
            trace.policy_transition(
                state_before=pre_tool_series_state,
                action=tool_name,
                args=tool_args,
                result=observation_payload,
                state_after=current_series_state,
                llm_called=True,
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
                    accessible_series_sources=self._series_source_prompt_payload(
                        series_sources=series_sources,
                        current_series_state=current_series_state,
                    ),
                    task_answer_status=self._task_answer_status(
                        task_type=plan["task_type"],
                        candidate=observation_payload,
                        source_tool=tool_name,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        scope_keys=plan.get("scope_keys", []),
                    ),
                    branch_task_answer_status=self._latest_branch_task_answer_status(
                        task_type=plan["task_type"],
                        execution_log=execution_log,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        scope_keys=plan.get("scope_keys", []),
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
                "source_ref": selected_source_ref,
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
                    accessible_series_sources=self._series_source_prompt_payload(
                        series_sources=series_sources,
                        current_series_state=current_series_state,
                    ),
                    task_answer_status=self._task_answer_status(
                        task_type=plan["task_type"],
                        candidate=observation_payload,
                        source_tool=tool_name,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        scope_keys=plan.get("scope_keys", []),
                    ),
                    branch_task_answer_status=self._latest_branch_task_answer_status(
                        task_type=plan["task_type"],
                        execution_log=execution_log + [log_entry],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        scope_keys=plan.get("scope_keys", []),
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

            if not is_train:
                refresh_status_after_tool = self._inference_packet_answer_refresh_status(
                    task_type=plan["task_type"],
                    execution_log=execution_log + [log_entry],
                    memory_audit=plan.get("memory_audit"),
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
                if refresh_status_after_tool.get("needs_refresh_before_finish") and step < MAX_REACT_STEPS:
                    trace.agent_step("executor", "self_correction", {
                        "step": step,
                        "reason": "stale_packet_answer_after_tool",
                        "tool": tool_name,
                        "latest_task_answer_tool": refresh_status_after_tool.get("latest_task_answer_tool"),
                        "stale_packet_tools_after_answer": list(
                            refresh_status_after_tool.get("stale_packet_tools_after_answer") or []
                        ),
                    })
                    conversation.append({
                        "role": "user",
                        "content": self._stale_packet_answer_correction_message(
                            refresh_status_after_tool
                        ),
                    })
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
            training_contract_satisfied = (
                not learning_summary_required
                or self._training_completion_contract_satisfied(
                    execution_log,
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
            )
            single_execution = (
                self._single_execution_candidate(
                    execution_log, task_type=plan["task_type"], horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []), scope_keys=plan.get("scope_keys", []),
                ) if learning_summary_required and not training_contract_satisfied else None
            )
            if single_execution is not None:
                training_contract_satisfied = True  # terminal recording, not comparative success
                terminal_feedback_messages = [
                    "Exploration has ended with exactly one valid, actually scored path. "
                    "Record single-execution evidence only. Do not claim a winner, "
                    "relative advantage, or failure of any unscored alternative. "
                    "Return a learning_summary describing this path and its observed score."
                ]
            terminal_valid = False
            recovered_terminal = self._recoverable_failure_record(
                expected_type=expected_final_type,
                task_type=plan["task_type"],
                execution_log=execution_log,
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
            )
            if recovered_terminal is not None and (
                recoverable_final is None
                or str(recoverable_final.get("answer_type") or "") != expected_final_type
            ):
                recoverable_final = recovered_terminal
            locked_recoverable_final = None if single_execution is not None else (recoverable_answer or recoverable_final)
            if learning_summary_required and not training_contract_satisfied:
                trace.agent_step("executor", "act_terminal_self_correction_skipped", {
                    "reason": "training_contract_unsatisfied",
                    "training_contract": self._training_contract_status(
                        execution_log,
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        scope_keys=plan.get("scope_keys", []),
                    ),
                })
            else:
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
                        scope_keys=plan.get("scope_keys", []),
                        recoverable_final=locked_recoverable_final,
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
                        elif not training_contract_satisfied:
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
                            domain=plan.get("domain", ""),
                            scope_keys=plan.get("scope_keys", []),
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
                        elif not self._terminal_answer_preserves_recoverable_answer(
                            expected_type=expected_final_type,
                            task_type=plan["task_type"],
                            recoverable_final=locked_recoverable_final,
                            candidate_answer=terminal_answer,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                        ):
                            terminal_valid = False
                            terminal_failure_reason = "changed_recoverable_answer"
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
                    if not learning_summary_required:
                        terminal_execution_context = self._recoverable_execution_context_from_trace(
                            tool_chain=self._reasoning_substantive_tool_chain(execution_log),
                            answer=terminal_answer,
                            reasoning=terminal_reasoning,
                            execution_context=terminal_execution_context,
                            execution_log=execution_log,
                        )
                    terminal_reasoning = self._normalize_reasoning_for_reuse(
                        terminal_reasoning,
                        execution_log,
                        terminal_execution_context,
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                    terminal_recoverable = self._recoverable_final_record(
                        answer=terminal_answer,
                        answer_type=terminal_answer_type,
                        confidence=terminal_decision.get("confidence"),
                        reasoning=terminal_reasoning,
                        execution_context=terminal_execution_context,
                        control_context=self._merge_control_context(
                            base_control_context,
                            terminal_decision.get("control_context"),
                        ),
                    )
                    try:
                        terminal_reuse_verdict = await self._final_payload_reusability_verdict(
                            mode=plan.get("_mode"),
                            reasoning=terminal_reasoning,
                            execution_context=terminal_execution_context,
                            execution_log=execution_log,
                            task_type=plan["task_type"],
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                        )
                    except Exception as e:
                        self._attach_partial_result_to_exception(
                            e,
                            answer=terminal_answer,
                            answer_type=terminal_answer_type,
                            confidence=terminal_decision.get("confidence"),
                            reasoning=terminal_reasoning,
                            execution_context=terminal_execution_context,
                            control_context=self._merge_control_context(
                                base_control_context,
                                terminal_decision.get("control_context"),
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
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
            if (
                not accepted_answer_type
                and recoverable_final is not None
                and single_execution is None
                and not (learning_summary_required and not training_contract_satisfied)
            ):
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
                accepted_control_context = self._merge_control_context(
                    base_control_context,
                    recoverable_final.get("control_context"),
                )
                trace.agent_step("executor", "act_recoverable_final_fallback", {
                    "answer_type": accepted_answer_type,
                    "confidence": accepted_confidence,
                    "has_answer": accepted_answer is not None,
                })

        if not accepted_answer_type:
            failure_reason = (
                "training_contract_unsatisfied"
                if (
                    plan.get("_mode") == "train"
                    and not bool(plan.get("_defer_learning_record", False))
                    and not self._training_completion_contract_satisfied(
                        execution_log,
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                )
                else "max_steps_exhausted_without_final_answer"
            )
            trace.agent_step("executor", "act_incomplete", {
                "reason": failure_reason,
                "steps": MAX_REACT_STEPS,
                "last_tool_result_preview": str(last_tool_result)[:5000] if last_tool_result is not None else None,
            })
            partial_result = self._failure_partial_result(
                expected_type=expected_final_type,
                task_type=plan["task_type"],
                execution_log=execution_log,
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
                recoverable_final=recoverable_answer or recoverable_final,
                failure_reason=failure_reason,
                extra={
                    "prediction": accepted_answer,
                    "answer": accepted_answer,
                    "answer_type": accepted_answer_type,
                    "confidence": accepted_confidence,
                    "reasoning": accepted_reasoning,
                    "execution_context": accepted_execution_context,
                    "control_context": accepted_control_context,
                    "last_tool_result_preview": (
                        str(last_tool_result)[:5000] if last_tool_result is not None else ""
                    ),
                },
            )
            if failure_reason == "training_contract_unsatisfied":
                raise DecisionOutputExhaustedError(
                    agent_kind="executor",
                    failure_reason=failure_reason,
                    partial_result=partial_result,
                )
            raise MaxStepsExhaustedError(
                agent_kind="executor",
                max_steps=MAX_REACT_STEPS,
                partial_result=partial_result,
            )

        if is_train and isinstance(accepted_answer, dict) and accepted_answer_type == "learning_summary":
            single = self._single_execution_candidate(
                execution_log, task_type=plan["task_type"], horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []), scope_keys=plan.get("scope_keys", []),
            )
            accepted_answer = dict(accepted_answer)
            accepted_answer["evidence_kind"] = "single_execution" if single else "comparative"
            if single:
                accepted_answer["observed_candidate"] = single
        accepted_confidence = coerce_confidence(accepted_confidence)
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
        scope_keys: list[str] | None = None,
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
                        scope_keys=scope_keys,
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
        scope_keys: list[str] | None = None,
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
                    scope_keys=scope_keys,
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

    @classmethod
    def _reasoning_candidate_records(
        cls,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_labels: set[str] = set()

        def _append(label: Any, tool_chain: Any) -> None:
            normalized_label = cls._normalize_label_text(label)
            if not normalized_label:
                return
            key = normalized_label.lower()
            if key in seen_labels:
                return
            seen_labels.add(key)
            records.append({
                "label": normalized_label,
                "tool_chain": [
                    str(tool).strip()
                    for tool in (tool_chain or [])
                    if str(tool).strip()
                ],
            })

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
                _append(label, sub_chain)

        task_type_text = str(task_type or "").strip()
        if task_type_text:
            for record in cls._top_level_candidate_records(
                execution_log,
                task_type=task_type_text,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                tool_chain = [
                    str(tool).strip()
                    for tool in record.get("tool_chain") or []
                    if str(tool).strip()
                ]
                label = " -> ".join(tool_chain) or str(record.get("name") or "").strip()
                _append(label, tool_chain)
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
        return canonical_indicator_key(key)

    @classmethod
    def _canonicalize_indicator_answer(
        cls,
        answer: Any,
        *,
        indicator_output_keys: list[str] | None = None,
    ) -> Any:
        return canonicalize_indicator_answer(
            answer,
            indicator_output_keys=indicator_output_keys,
        )

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
    def _training_contract_status(
        cls,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
        scope_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        candidate_count = cls._count_training_candidates(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
            scope_keys=scope_keys,
        )
        spawned_candidate_count = cls._count_training_spawn_candidates(
            execution_log,
            task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
            scope_keys=scope_keys,
        )
        has_eval = cls._has_training_evaluation(execution_log)
        return {
            "candidate_count": candidate_count,
            "spawned_candidate_count": spawned_candidate_count,
            "has_eval": has_eval,
            "satisfied": (
                candidate_count >= MIN_TRAINING_CANDIDATES
                and spawned_candidate_count >= MIN_TRAINING_CANDIDATES
                and has_eval
            ),
        }

    @classmethod
    def _training_completion_contract_satisfied(
        cls,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        return bool(
            cls._training_contract_status(
                execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ).get("satisfied", False)
        )

    @classmethod
    def _single_execution_candidate(
        cls, execution_log: list[dict[str, Any]], *, task_type: str,
        horizon: int | None = None, indicator_output_keys: list[str] | None = None,
        require_score: bool = True, scope_keys: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """A sole tool-backed, scored path after a real multi-branch attempt.

        This is terminal factual evidence, not satisfaction of the comparative
        exploration contract. A failed/absent score or an unexecuted answer
        cannot create a Note. Normal exploration still seeks two valid paths.
        """
        attempts = 0
        for entry in execution_log:
            if entry.get("tool") != "spawn_subagent":
                continue
            output = entry.get("output") or []
            if isinstance(output, dict):
                output = output.get("results") or []
            attempts += len(output)
        if attempts < MIN_TRAINING_CANDIDATES:
            return None
        if cls._count_training_candidates(execution_log, task_type, horizon=horizon,
                indicator_output_keys=indicator_output_keys, scope_keys=scope_keys) != 1:
            return None
        candidates = cls._recoverable_training_candidates(
            execution_log=execution_log, task_type=task_type, horizon=horizon,
            indicator_output_keys=indicator_output_keys, scored_only=require_score,
        )
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        if not cls._is_valid_final_answer(task_type, candidate["prediction"], horizon=horizon,
                indicator_output_keys=indicator_output_keys, scope_keys=scope_keys):
            return None
        metrics = candidate["metrics"]
        if any(metrics.get(key) for key in ("error", "skipped", "parse_error", "type_error")):
            return None
        if not cls._is_task_answer_backed_by_tool_output(task_type, candidate["prediction"],
                execution_log, horizon=horizon, indicator_output_keys=indicator_output_keys):
            return None
        return candidate

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
        if is_train and step >= MAX_REACT_STEPS - 1 and cls._single_execution_candidate(
            execution_log, task_type=plan["task_type"], horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []), scope_keys=plan.get("scope_keys", []), require_score=False,
        ) is not None:
            return ("Exploration budget is ending with one valid path after a multi-branch attempt. "
                    "Evaluate that exact path if not already scored. Terminal recording will mark "
                    "single-execution evidence, with no comparison or winner claim.")
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

        contract_status = cls._training_contract_status(
            execution_log,
            task_type=plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
            scope_keys=plan.get("scope_keys", []),
        )
        candidate_count = int(contract_status.get("candidate_count") or 0)
        spawned_candidate_count = int(contract_status.get("spawned_candidate_count") or 0)
        has_eval = bool(contract_status.get("has_eval", False))

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

    @staticmethod
    def _memory_audit_tool_list(memory_audit: Any, key: str) -> list[str]:
        if not isinstance(memory_audit, dict):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in memory_audit.get(key) or []:
            tool_name = str(item).strip()
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            out.append(tool_name)
        return out

    @staticmethod
    def _lane_support_tools(
        lane_tools: list[str],
        *,
        task_reasoning_tool: str,
    ) -> list[str]:
        reasoning_tool = str(task_reasoning_tool or "").strip()
        return [
            str(tool_name).strip()
            for tool_name in (lane_tools or [])
            if str(tool_name).strip() and str(tool_name).strip() != reasoning_tool
        ]

    @classmethod
    def _lane_materialized_for_answer(
        cls,
        *,
        lane_tools: list[str],
        executed_set: set[str],
        task_reasoning_tool: str,
    ) -> bool:
        support_tools = cls._lane_support_tools(
            lane_tools,
            task_reasoning_tool=task_reasoning_tool,
        )
        if support_tools:
            return any(tool_name in executed_set for tool_name in support_tools)
        return any(str(tool_name).strip() in executed_set for tool_name in (lane_tools or []))

    @staticmethod
    def _packet_evidence_anchor_tools(selected_tools: list[str]) -> list[str]:
        """New definition:
        `evidence-anchor tools` are lightweight packet-selected tools that
        materialize the concrete intermediate evidence the learned packet is
        pointing at. Unlike full alternate forecast challengers, these anchors
        are safe to require before LLM-only final reasoning.
        """
        anchor_candidates = {
            "segment",
            "window_stats",
            "value_at",
            "temporal_align_text",
        }
        anchors: list[str] = []
        seen: set[str] = set()
        for tool_name in selected_tools:
            name = str(tool_name).strip()
            if not name or name not in anchor_candidates or name in seen:
                continue
            seen.add(name)
            anchors.append(name)
        return anchors

    @staticmethod
    def _has_challenger_skip_note(control_context: Any) -> bool:
        if not isinstance(control_context, dict):
            return False
        note = str(control_context.get("completion_notes") or "").strip()
        if not note.lower().startswith("challenger_skip:"):
            return False
        return len(note.split(":", 1)[1].strip()) >= 16

    @classmethod
    def _inference_challenger_finish_status(
        cls,
        *,
        execution_log: list[dict[str, Any]],
        memory_audit: Any,
    ) -> dict[str, Any]:
        """New definition:
        `challenger path` remains an advisory alternate lane for inference,
        but it is not a mandatory finish gate. We still audit whether the
        primary/challenger lanes were touched so traces stay inspectable.
        """
        primary_path_tools = cls._memory_audit_tool_list(memory_audit, "primary_path_tools")
        challenger_path_tools = cls._memory_audit_tool_list(memory_audit, "challenger_path_tools")
        tool_prep_tools = cls._memory_audit_tool_list(memory_audit, "tool_prep_tools")
        lane_union = cls._flatten_selected_tool_lanes(
            primary_path_tools=primary_path_tools,
            challenger_path_tools=challenger_path_tools,
            tool_prep_tools=tool_prep_tools,
        )
        evidence_anchor_tools = cls._packet_evidence_anchor_tools(lane_union)
        executed_tools = cls._available_substantive_tool_names(execution_log)
        executed_set = set(executed_tools)
        primary_hit = any(tool_name in executed_set for tool_name in primary_path_tools)
        challenger_hit = any(tool_name in executed_set for tool_name in challenger_path_tools)
        completed_prep_tools = [
            tool_name for tool_name in tool_prep_tools
            if tool_name in executed_set
        ]
        remaining_prep_tools = [
            tool_name for tool_name in tool_prep_tools
            if tool_name not in executed_set
        ]
        completed_anchor_tools = [
            tool_name for tool_name in evidence_anchor_tools
            if tool_name in executed_set
        ]
        remaining_anchor_tools = [
            tool_name for tool_name in evidence_anchor_tools
            if tool_name not in executed_set
        ]
        prep_hit = bool(completed_prep_tools)
        prep_complete = not remaining_prep_tools
        needs_prep_before_finish = bool(remaining_prep_tools)
        return {
            "primary_path_tools": primary_path_tools,
            "challenger_path_tools": challenger_path_tools,
            "tool_prep_tools": tool_prep_tools,
            "evidence_anchor_tools": evidence_anchor_tools,
            "executed_tools": executed_tools,
            "primary_hit": primary_hit,
            "challenger_hit": challenger_hit,
            "prep_hit": prep_hit,
            "prep_complete": prep_complete,
            "completed_prep_tools": completed_prep_tools,
            "remaining_prep_tools": remaining_prep_tools,
            "completed_anchor_tools": completed_anchor_tools,
            "remaining_anchor_tools": remaining_anchor_tools,
            "needs_prep_before_finish": needs_prep_before_finish,
            "needs_anchor_evidence_before_finish": bool(remaining_anchor_tools),
            "needs_challenger_before_finish": False,
        }

    @classmethod
    def _inference_packet_answer_refresh_status(
        cls,
        *,
        task_type: str,
        execution_log: list[dict[str, Any]],
        memory_audit: Any,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """New definition:
        if packet-selected evidence arrives after the latest task-valid answer,
        the current branch is carrying a stale task label. In that case
        inference must refresh the task-valid answer with the later packet
        evidence before finishing, rather than only mentioning the new evidence
        in prose.
        """
        lane_union = cls._flatten_selected_tool_lanes(
            primary_path_tools=cls._memory_audit_tool_list(memory_audit, "primary_path_tools"),
            challenger_path_tools=cls._memory_audit_tool_list(memory_audit, "challenger_path_tools"),
            tool_prep_tools=cls._memory_audit_tool_list(memory_audit, "tool_prep_tools"),
        )
        successful_entries: list[dict[str, Any]] = []
        for entry in execution_log:
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name or tool_name in _ORCHESTRATION_ONLY_TOOLS:
                continue
            if not bool(entry.get("success")):
                continue
            successful_entries.append(entry)

        latest_task_answer_idx: int | None = None
        latest_task_answer_tool = ""
        for idx, entry in enumerate(successful_entries):
            status = cls._task_answer_status(
                task_type=task_type,
                candidate=entry.get("output"),
                source_tool=entry.get("tool"),
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
            if bool(status.get("available")):
                latest_task_answer_idx = idx
                latest_task_answer_tool = str(status.get("source_tool") or entry.get("tool") or "").strip()

        if latest_task_answer_idx is None or not lane_union:
            return {
                "needs_refresh_before_finish": False,
                "latest_task_answer_tool": latest_task_answer_tool,
                "latest_task_answer_idx": latest_task_answer_idx,
                "stale_packet_tools_after_answer": [],
            }

        stale_packet_tools_after_answer: list[str] = []
        for entry in successful_entries[latest_task_answer_idx + 1 :]:
            tool_name = str(entry.get("tool") or "").strip()
            if tool_name and tool_name in lane_union:
                stale_packet_tools_after_answer.append(tool_name)

        return {
            "needs_refresh_before_finish": bool(stale_packet_tools_after_answer),
            "latest_task_answer_tool": latest_task_answer_tool,
            "latest_task_answer_idx": latest_task_answer_idx,
            "stale_packet_tools_after_answer": stale_packet_tools_after_answer,
        }

    @classmethod
    def _prep_finish_correction_message(
        cls,
        challenger_status: dict[str, Any],
    ) -> str:
        completed_text = ", ".join(challenger_status.get("completed_prep_tools") or []) or "(none)"
        remaining_text = ", ".join(challenger_status.get("remaining_prep_tools") or []) or "(none)"
        return (
            "Self-correction: the injected sample experience marked prep evidence that must be gathered "
            "before finishing this sample. "
            f"Completed prep tools: {completed_text}. Remaining prep tools: {remaining_text}. "
            "Call the remaining prep tool(s) before finishing."
        )

    @classmethod
    def _anchor_finish_correction_message(
        cls,
        challenger_status: dict[str, Any],
    ) -> str:
        completed_text = ", ".join(challenger_status.get("completed_anchor_tools") or []) or "(none)"
        remaining_text = ", ".join(challenger_status.get("remaining_anchor_tools") or []) or "(none)"
        return (
            "Self-correction: the injected sample experience selected concrete evidence-anchor tools "
            "that must be materialized before an LLM-only final answer is acceptable. "
            f"Completed evidence anchors: {completed_text}. Remaining evidence anchors: {remaining_text}. "
            "Call the remaining evidence-anchor tool(s) before finishing."
        )

    @classmethod
    def _challenger_finish_correction_message(
        cls,
        challenger_status: dict[str, Any],
    ) -> str:
        primary_text = ", ".join(challenger_status.get("primary_path_tools") or []) or "(none)"
        challenger_text = ", ".join(challenger_status.get("challenger_path_tools") or []) or "(none)"
        return (
            "Self-correction: the injected sample experience marked a primary path and a challenger path. "
            f"You already used the primary path ({primary_text}) but have not tested the challenger path "
            f"({challenger_text}). Before finishing, either call at least one challenger-path tool or, "
            "if it is genuinely unnecessary for this sample, finish later with "
            "`control_context.completion_notes` starting exactly with `challenger_skip:` followed by a "
            "concrete reason grounded in the current evidence."
        )

    @classmethod
    def _primary_only_answer_needs_challenger_message(
        cls,
        lane_status: dict[str, Any],
    ) -> str:
        primary_text = ", ".join(lane_status.get("primary_path_tools") or []) or "(none)"
        challenger_text = ", ".join(lane_status.get("challenger_path_tools") or []) or "(none)"
        latest_tool = str(lane_status.get("latest_task_answer_tool") or "").strip() or "(unknown)"
        candidate_count = int(lane_status.get("candidate_count") or 0)
        return (
            "Self-correction: the current branch is trying to finish after only the primary lane "
            "produced the current task-valid answer. "
            f"Primary path: {primary_text}. Challenger path: {challenger_text}. "
            f"Latest task-valid answer tool: {latest_tool}. Current candidate_count={candidate_count}. "
            "Before finishing, test the challenger lane so the new evidence can either preserve or "
            "disconfirm the current primary answer. Do not keep packaging or defending the same primary-only answer."
        )

    @classmethod
    def _stale_packet_answer_correction_message(
        cls,
        refresh_status: dict[str, Any],
    ) -> str:
        latest_tool = str(refresh_status.get("latest_task_answer_tool") or "").strip() or "(unknown)"
        stale_tools = ", ".join(refresh_status.get("stale_packet_tools_after_answer") or []) or "(none)"
        return (
            "Self-correction: after the latest task-valid answer was produced, you gathered new "
            "packet-selected evidence. "
            f"Latest task-valid answer tool: {latest_tool}. "
            f"Later packet-selected evidence tools: {stale_tools}. "
            "Before finishing, call the task-answering tool again so the final label/forecast is "
            "recomputed from the later packet-selected evidence, instead of presenting only the "
            "earlier answer state."
        )

    @classmethod
    def _inference_packet_lane_status(
        cls,
        *,
        execution_log: list[dict[str, Any]],
        memory_audit: Any,
        task_type: str,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        primary_path_tools = cls._memory_audit_tool_list(memory_audit, "primary_path_tools")
        challenger_path_tools = cls._memory_audit_tool_list(memory_audit, "challenger_path_tools")
        tool_prep_tools = cls._memory_audit_tool_list(memory_audit, "tool_prep_tools")
        lane_union = cls._flatten_selected_tool_lanes(
            primary_path_tools=primary_path_tools,
            challenger_path_tools=challenger_path_tools,
            tool_prep_tools=tool_prep_tools,
        )
        evidence_anchor_tools = cls._packet_evidence_anchor_tools(lane_union)
        executed_tools = cls._available_substantive_tool_names(execution_log)
        executed_set = set(executed_tools)
        completed_prep_tools = [
            tool_name for tool_name in tool_prep_tools
            if tool_name in executed_set
        ]
        remaining_prep_tools = [
            tool_name for tool_name in tool_prep_tools
            if tool_name not in executed_set
        ]
        completed_anchor_tools = [
            tool_name for tool_name in evidence_anchor_tools
            if tool_name in executed_set
        ]
        remaining_anchor_tools = [
            tool_name for tool_name in evidence_anchor_tools
            if tool_name not in executed_set
        ]
        prep_hit = bool(completed_prep_tools)
        prep_complete = not remaining_prep_tools
        primary_hit = any(tool_name in executed_set for tool_name in primary_path_tools)
        challenger_hit = any(tool_name in executed_set for tool_name in challenger_path_tools)
        task_reasoning_tool = cls._task_llm_reasoning_tool(task_type)
        latest_task_answer_status = cls._latest_branch_task_answer_status(
            task_type=task_type,
            execution_log=execution_log,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        candidate_records = cls._reasoning_candidate_records(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        candidate_count = len(candidate_records)
        latest_candidate_chain = [
            str(tool_name).strip()
            for tool_name in ((candidate_records[-1] if candidate_records else {}).get("tool_chain") or [])
            if str(tool_name).strip()
        ]
        latest_candidate_executed_set = set(latest_candidate_chain)
        latest_candidate_primary_hit = cls._lane_materialized_for_answer(
            lane_tools=primary_path_tools,
            executed_set=latest_candidate_executed_set,
            task_reasoning_tool=task_reasoning_tool,
        )
        latest_candidate_challenger_hit = cls._lane_materialized_for_answer(
            lane_tools=challenger_path_tools,
            executed_set=latest_candidate_executed_set,
            task_reasoning_tool=task_reasoning_tool,
        )
        latest_task_answer_tool = str(latest_task_answer_status.get("source_tool") or "").strip()
        latest_answer_from_primary_only = bool(
            latest_task_answer_status.get("available")
            and latest_candidate_primary_hit
            and not latest_candidate_challenger_hit
        )

        if not lane_union:
            return {
                "primary_path_tools": primary_path_tools,
                "challenger_path_tools": challenger_path_tools,
                "tool_prep_tools": tool_prep_tools,
                "evidence_anchor_tools": evidence_anchor_tools,
                "lane_union": lane_union,
                "executed_tools": executed_tools,
                "prep_hit": prep_hit,
                "prep_complete": prep_complete,
                "completed_prep_tools": completed_prep_tools,
                "remaining_prep_tools": remaining_prep_tools,
                "completed_anchor_tools": completed_anchor_tools,
                "remaining_anchor_tools": remaining_anchor_tools,
                "primary_hit": primary_hit,
                "challenger_hit": challenger_hit,
                "allowed_now": [],
                "reason": "no_packet_lanes",
                "task_reasoning_tool": task_reasoning_tool,
                "latest_task_answer_available": bool(latest_task_answer_status.get("available")),
                "latest_task_answer_tool": latest_task_answer_tool,
                "candidate_count": candidate_count,
                "latest_answer_from_primary_only": latest_answer_from_primary_only,
                "latest_candidate_chain": latest_candidate_chain,
                "enforce": False,
            }

        if tool_prep_tools and not prep_complete:
            allowed_now = cls._flatten_selected_tool_lanes(
                primary_path_tools=primary_path_tools,
                challenger_path_tools=[],
                tool_prep_tools=tool_prep_tools,
            )
            reason = "must_complete_prep_before_leaving_lane"
        elif remaining_anchor_tools:
            allowed_now = list(remaining_anchor_tools)
            reason = "must_materialize_anchor_evidence_before_reasoning"
        elif primary_path_tools and not primary_hit:
            allowed_now = list(primary_path_tools)
            reason = "must_start_primary_path"
        elif (
            challenger_path_tools
            and primary_hit
            and not challenger_hit
            and latest_answer_from_primary_only
            and candidate_count < 2
        ):
            """New definition:
            once the primary lane has already produced the only current
            task-valid candidate, inference should stop treating the challenger
            lane as rhetorical. The next move must test the challenger lane so
            fresh evidence can actually disconfirm or preserve the primary
            answer, instead of repeatedly restating the same primary result.
            """
            allowed_now = list(challenger_path_tools)
            reason = "must_test_challenger_after_primary_answer"
        elif challenger_path_tools and not challenger_hit:
            allowed_now = cls._flatten_selected_tool_lanes(
                primary_path_tools=primary_path_tools,
                challenger_path_tools=challenger_path_tools,
                tool_prep_tools=[],
            )
            reason = "must_continue_primary_or_test_challenger"
        else:
            allowed_now = list(lane_union)
            reason = "stay_within_selected_lanes"

        if task_reasoning_tool:
            non_reasoning_lane_tools = [
                tool_name for tool_name in lane_union
                if tool_name != task_reasoning_tool
            ]
            has_any_lane_evidence = prep_hit or primary_hit or challenger_hit
            if remaining_prep_tools or remaining_anchor_tools:
                if task_reasoning_tool in allowed_now:
                    allowed_now = [
                        tool_name for tool_name in allowed_now
                        if tool_name != task_reasoning_tool
                    ]
            elif has_any_lane_evidence or not non_reasoning_lane_tools:
                if task_reasoning_tool not in allowed_now:
                    allowed_now.append(task_reasoning_tool)
            elif task_reasoning_tool in allowed_now:
                allowed_now = [
                    tool_name for tool_name in allowed_now
                    if tool_name != task_reasoning_tool
                ]

        return {
            "primary_path_tools": primary_path_tools,
            "challenger_path_tools": challenger_path_tools,
            "tool_prep_tools": tool_prep_tools,
            "evidence_anchor_tools": evidence_anchor_tools,
            "lane_union": lane_union,
            "executed_tools": executed_tools,
            "prep_hit": prep_hit,
            "prep_complete": prep_complete,
            "completed_prep_tools": completed_prep_tools,
            "remaining_prep_tools": remaining_prep_tools,
            "completed_anchor_tools": completed_anchor_tools,
            "remaining_anchor_tools": remaining_anchor_tools,
            "primary_hit": primary_hit,
            "challenger_hit": challenger_hit,
            "allowed_now": allowed_now,
            "reason": reason,
            "task_reasoning_tool": task_reasoning_tool,
            "latest_task_answer_available": bool(latest_task_answer_status.get("available")),
            "latest_task_answer_tool": latest_task_answer_tool,
            "candidate_count": candidate_count,
            "latest_answer_from_primary_only": latest_answer_from_primary_only,
            "latest_candidate_chain": latest_candidate_chain,
            "enforce": bool(allowed_now),
        }

    @classmethod
    def _packet_lane_correction_message(
        cls,
        lane_status: dict[str, Any],
    ) -> str:
        allowed_now = ", ".join(lane_status.get("allowed_now") or []) or "(none)"
        prep_text = ", ".join(lane_status.get("tool_prep_tools") or []) or "(none)"
        anchor_text = ", ".join(lane_status.get("evidence_anchor_tools") or []) or "(none)"
        primary_text = ", ".join(lane_status.get("primary_path_tools") or []) or "(none)"
        challenger_text = ", ".join(lane_status.get("challenger_path_tools") or []) or "(none)"
        remaining_prep_text = ", ".join(lane_status.get("remaining_prep_tools") or []) or "(none)"
        remaining_anchor_text = ", ".join(lane_status.get("remaining_anchor_tools") or []) or "(none)"
        reason = str(lane_status.get("reason") or "").strip()
        if reason == "must_complete_prep_before_leaving_lane":
            focus = (
                "Finish the remaining prep evidence before leaving this packet lane or finishing the sample."
            )
        elif reason == "must_materialize_anchor_evidence_before_reasoning":
            focus = (
                "Materialize the remaining packet-selected evidence anchors before jumping to final reasoning."
            )
        elif reason == "must_start_primary_path":
            focus = (
                "Start the packet's primary path before switching to an unrelated tool."
            )
        elif reason == "must_test_challenger_after_primary_answer":
            focus = (
                "The current primary lane has already produced the only task-valid candidate so far; test the challenger lane now before you keep packaging or defending that same primary answer."
            )
        elif reason == "must_continue_primary_or_test_challenger":
            focus = (
                "Continue the primary path or explicitly test the challenger path next; do not open an unrelated side path."
            )
        else:
            focus = (
                "Stay within the packet-selected lanes unless you later finish with an explicit evidence-based reason."
            )
        return (
            "Self-correction: the injected sample experience selected explicit execution lanes for this sample. "
            f"Prep tools: {prep_text}. Evidence anchors: {anchor_text}. Primary path: {primary_text}. "
            f"Challenger path: {challenger_text}. Remaining prep tools: {remaining_prep_text}. "
            f"Remaining evidence anchors: {remaining_anchor_text}. {focus} The next tool must be one of: {allowed_now}. "
            "In your next `thought`, make `Check:` name the specific missing packet evidence you are about to gather, "
            "`Self-check:` say how you will decide that this packet obligation is now satisfied, and `Update:` say whether "
            "that result would keep the current path or force you to test the rival path."
        )

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
    def _finish_payload_covers_stale_packet_evidence(
        cls,
        *,
        execution_context: dict[str, Any] | None,
        refresh_status: dict[str, Any],
    ) -> bool:
        """New definition:
        if later packet-selected evidence exists, the finish payload counts as
        refreshed when its adopted execution_context.tool_chain explicitly
        carries at least one of those later packet-selected tools.
        """
        stale_tools = {
            str(tool).strip()
            for tool in (refresh_status.get("stale_packet_tools_after_answer") or [])
            if str(tool).strip()
        }
        if not stale_tools:
            return True
        normalized = cls._normalize_execution_context(execution_context)
        final_tool_chain = {
            str(tool).strip()
            for tool in (normalized.get("tool_chain") or [])
            if str(tool).strip()
        }
        return bool(final_tool_chain & stale_tools)

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

        contract_status = cls._training_contract_status(
            execution_log,
            task_type=plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
            scope_keys=plan.get("scope_keys", []),
        )
        candidate_count = int(contract_status.get("candidate_count") or 0)
        spawned_candidate_count = int(contract_status.get("spawned_candidate_count") or 0)
        has_eval = bool(contract_status.get("has_eval", False))
        if (
            candidate_count >= MIN_TRAINING_CANDIDATES
            and spawned_candidate_count >= MIN_TRAINING_CANDIDATES
            and not has_eval
        ):
            return True
        if step >= MAX_REACT_STEPS - 1 and not has_eval and cls._single_execution_candidate(
            execution_log, task_type=plan["task_type"], horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []), scope_keys=plan.get("scope_keys", []), require_score=False,
        ) is not None:
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

        contract_status = cls._training_contract_status(
            execution_log,
            task_type=plan["task_type"],
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
            scope_keys=plan.get("scope_keys", []),
        )
        candidate_count = int(contract_status.get("candidate_count") or 0)
        spawned_candidate_count = int(contract_status.get("spawned_candidate_count") or 0)
        has_eval = bool(contract_status.get("has_eval", False))
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
        task_type: str | None = None,
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
            lines.append(f"{idx}. The only valid tool for this turn is `spawn_subagent`.")
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
        self_check_hint = ExecutorAgent._decision_self_check_hint(task_type)
        if self_check_hint:
            lines.append(self_check_hint)
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
                "This is a spawn-only turn: use `spawn_subagent` now so later turns can perform GT evaluation."
            )
            lines.append(
                "Do not call `evaluate_*` in this turn, even if a task-valid answer already exists."
            )
            lines.append(
                "Do not restate, package, or justify the existing answer in this turn; return only a `spawn_subagent` tool-call JSON."
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
    def _decision_self_check_hint(task_type: str | None) -> str:
        normalized = str(task_type or "").strip()
        if normalized in {"forecast", "indicator", "classification"}:
            return (
                "In `Self-check:`, name the missing or newly acquired task-valid artifact and state one "
                "minimal sanity/consistency test for it, such as continuity with the last observed values, "
                "correct target window, correct label space, or agreement with the chosen path's evidence."
            )
        if normalized in {"qa", "mcqa", "correlation"}:
            return (
                "In `Self-check:`, focus on whether the current evidence actually covers the decisive claim, "
                "option semantics, or requested relationship label. Do not treat a raw numeric artifact as "
                "sufficient unless it directly resolves that final judgment."
            )
        return (
            "In `Self-check:`, state what key evidence or artifact is still missing or newly confirmed, and "
            "what minimal condition it must satisfy before you trust the current path."
        )

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
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
        memory_audit: Any = None,
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
        candidate_records = cls._reasoning_candidate_records(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if len(candidate_records) >= 2:
            lines.append(
                "- Task-valid candidate paths already available: "
                + ", ".join(record["label"] for record in candidate_records)
            )
        if task_type and memory_audit:
            lane_status = cls._inference_packet_lane_status(
                execution_log=execution_log,
                memory_audit=memory_audit,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
            lane_union = list(lane_status.get("lane_union") or [])
            if lane_union:
                lines.append("- Packet-selected lane tools: " + ", ".join(lane_union))
                prep_tools = list(lane_status.get("tool_prep_tools") or [])
                if prep_tools:
                    lines.append(
                        "- Packet prep status: completed="
                        + (", ".join(lane_status.get("completed_prep_tools") or []) or "(none)")
                        + "; remaining="
                        + (", ".join(lane_status.get("remaining_prep_tools") or []) or "(none)")
                    )
                anchor_tools = list(lane_status.get("evidence_anchor_tools") or [])
                if anchor_tools:
                    lines.append(
                        "- Packet anchor status: completed="
                        + (", ".join(lane_status.get("completed_anchor_tools") or []) or "(none)")
                        + "; remaining="
                        + (", ".join(lane_status.get("remaining_anchor_tools") or []) or "(none)")
                    )
                primary_tools = list(lane_status.get("primary_path_tools") or [])
                challenger_tools = list(lane_status.get("challenger_path_tools") or [])
                if primary_tools:
                    lines.append(
                        "- Packet primary path: "
                        + ", ".join(primary_tools)
                        + f" | touched={'yes' if lane_status.get('primary_hit') else 'no'}"
                    )
                if challenger_tools:
                    lines.append(
                        "- Packet challenger path: "
                        + ", ".join(challenger_tools)
                        + f" | touched={'yes' if lane_status.get('challenger_hit') else 'no'}"
                    )
                allowed_now = list(lane_status.get("allowed_now") or [])
                if lane_status.get("enforce") and allowed_now:
                    lines.append("- Packet next-tool envelope now: " + ", ".join(allowed_now))
                latest_task_answer_tool = str(lane_status.get("latest_task_answer_tool") or "").strip()
                if latest_task_answer_tool:
                    lines.append(
                        "- Latest task-valid answer tool: "
                        + latest_task_answer_tool
                        + f" | candidate_count={int(lane_status.get('candidate_count') or 0)}"
                    )
                unresolved_packet_state = bool(
                    lane_status.get("remaining_prep_tools")
                    or lane_status.get("remaining_anchor_tools")
                )
                if lane_status.get("reason") == "must_test_challenger_after_primary_answer":
                    lines.append(
                        "- Finish gate from packet state: not yet justified because only the primary lane has produced a task-valid candidate and the challenger lane remains untested."
                    )
                if unresolved_packet_state:
                    lines.append(
                        "- Finish gate from packet state: not yet justified because required packet evidence is still missing. Next move must gather the missing packet evidence before finishing."
                    )
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
        scope_keys: list[str] | None = None,
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
        candidate_labels = cls._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )

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
                    "Keep this exact answer payload unchanged. Repair mode may rewrite only "
                    "`confidence`, `reasoning`, `execution_context`, and `control_context`; it "
                    "must not change the answer content itself.\n"
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
            "- `thought` must keep the normal decision shape and explicitly contain these five labeled parts in order: `Reason:` then `Check:` then `Self-check:` then `Update:` then `Action:`.",
            "- `Self-check:` must say which required artifact or decisive evidence is already present and what minimal sanity, consistency, or semantic-coverage condition it passed.",
            "- `Update:` must explain whether the repaired final payload remains the best current candidate, replaces a rival candidate, or needs no rival because execution evidence is already decisive.",
        ]
        if terminal_json:
            output_lines.append(
                "- Return exactly one final JSON object with `thought`, `done=true`, `answer`, "
                f"`answer_type='{expected_type}'`, `confidence`, `reasoning`, `execution_context`, and `control_context`."
            )
        else:
            output_lines.append(
                "- Return exactly one finish-action JSON object with `thought`, `done=true`, `answer`, "
                f"`answer_type='{expected_type}'`, `confidence`, `reasoning`, `execution_context`, and `control_context`."
            )
            output_lines.append(
                "- Do not return a bare final payload without the `thought` and `done=true` wrapper, because this repair path still goes through the normal decision parser."
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
                    scope_keys=scope_keys,
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
        scope_keys: list[str] | None = None,
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
                    scope_keys=scope_keys,
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
            "confidence": coerce_confidence(confidence),
            "reasoning": reasoning,
            "execution_context": execution_context,
            "control_context": control_context,
        }

    @classmethod
    def _recoverable_terminal_record(
        cls,
        *,
        expected_type: str,
        task_type: str,
        execution_log: list[dict[str, Any]],
        horizon: int | None,
        indicator_output_keys: list[str] | None,
    ) -> dict[str, Any] | None:
        """Recover a terminal payload directly from executed evidence.

        New definition:
            Terminal recovery is execution-log-based, not prompt-based. When
            the last LLM delivery step fails after a real tool path already
            solved the task, recover the terminal payload from executed tool
            evidence instead of requiring one more model rewrite.
        """
        if str(expected_type).strip() == "learning_summary":
            return cls._recoverable_learning_summary_record_from_execution_log(
                execution_log=execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        return cls._recoverable_task_record_from_execution_log(
            task_type=task_type,
            execution_log=execution_log,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )

    @classmethod
    def _recoverable_failure_record(
        cls,
        *,
        expected_type: str,
        task_type: str,
        execution_log: list[dict[str, Any]],
        horizon: int | None,
        indicator_output_keys: list[str] | None,
    ) -> dict[str, Any] | None:
        expected_type_text = str(expected_type or "").strip()
        if expected_type_text == "learning_summary" and not cls._training_completion_contract_satisfied(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
            return cls._recoverable_task_record_from_execution_log(
                task_type=task_type,
                execution_log=execution_log,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        return cls._recoverable_terminal_record(
            expected_type=expected_type,
            task_type=task_type,
            execution_log=execution_log,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )

    @classmethod
    def _failure_partial_result(
        cls,
        *,
        expected_type: str,
        task_type: str,
        execution_log: list[dict[str, Any]],
        horizon: int | None,
        indicator_output_keys: list[str] | None,
        recoverable_final: dict[str, Any] | None = None,
        failure_reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        record = recoverable_final if isinstance(recoverable_final, dict) and recoverable_final else None
        if record is None:
            record = cls._recoverable_failure_record(
                expected_type=expected_type,
                task_type=task_type,
                execution_log=execution_log,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        if isinstance(record, dict):
            payload.update(record)
        payload["execution_log"] = list(execution_log)
        payload["tool_chain"] = [
            str(entry.get("tool") or "").strip()
            for entry in execution_log
            if isinstance(entry, dict) and str(entry.get("tool") or "").strip()
        ]
        if failure_reason:
            payload["failure_reason"] = str(failure_reason)
        if str(expected_type or "").strip() == "learning_summary":
            payload["training_contract"] = cls._training_contract_status(
                execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    @classmethod
    def _decision_from_recoverable_final(
        cls,
        record: dict[str, Any] | None,
        *,
        thought: str,
    ) -> dict[str, Any] | None:
        if not isinstance(record, dict) or not record:
            return None
        answer_type = str(record.get("answer_type") or "").strip()
        if not answer_type:
            return None
        return {
            "thought": thought,
            "done": True,
            "answer": record.get("answer"),
            "answer_type": answer_type,
            "confidence": record.get("confidence"),
            "reasoning": record.get("reasoning"),
            "execution_context": record.get("execution_context"),
            "control_context": record.get("control_context"),
        }

    @staticmethod
    def _recoverable_preview_text(value: Any, *, limit: int = 240) -> str:
        try:
            text = json.dumps(
                ExecutorAgent._round_prompt_numbers(value),
                ensure_ascii=False,
                default=str,
            )
        except TypeError:
            text = str(value)
        return text if len(text) <= limit else text[:limit] + "..."

    @classmethod
    def _recoverable_reasoning_snippets(cls, text: Any, *, limit: int = 3) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        snippets: list[str] = []
        seen: set[str] = set()
        for part in re.split(r"(?<=[.!?])\s+|[;\n]+", raw):
            cleaned = sanitize_learning_evidence_text(str(part), drop_imperatives=True)
            cleaned = cleaned.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            snippets.append(cleaned)
            if len(snippets) >= limit:
                break
        return snippets

    @classmethod
    def _recoverable_execution_context_from_trace(
        cls,
        *,
        tool_chain: list[str],
        answer: Any,
        reasoning: Any,
        execution_context: Any = None,
        execution_log: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized = cls._normalize_execution_context(
            execution_context,
            execution_log=execution_log,
        )
        if cls._has_reusable_execution_context(
            normalized,
            execution_log or [],
        ):
            return normalized

        clean_chain = [str(tool).strip() for tool in tool_chain if str(tool).strip()]
        snippets = list(normalized.get("public_reasoning") or [])
        if not snippets:
            snippets = cls._recoverable_reasoning_snippets(reasoning, limit=3)
        if not snippets:
            preview = cls._recoverable_preview_text(answer, limit=180)
            if clean_chain:
                snippets = [f"`{clean_chain[-1]}` produced {preview}."]
            elif preview:
                snippets = [f"The executed path produced {preview}."]
        evidence_summary = str(normalized.get("evidence_summary") or "").strip()
        if not evidence_summary:
            if snippets:
                evidence_summary = snippets[0]
            elif clean_chain:
                evidence_summary = (
                    f"The executed path `{' -> '.join(clean_chain)}` produced the accepted result."
                )
        return {
            "tool_chain": clean_chain or list(normalized.get("tool_chain") or []),
            "public_reasoning": snippets[:3],
            "evidence_summary": evidence_summary[:400].strip(),
        }

    @classmethod
    def _has_reusable_execution_context(
        cls,
        execution_context: dict[str, Any] | None,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        normalized = cls._normalize_execution_context(
            execution_context,
            execution_log=execution_log,
        )
        substantive_tools = cls._available_substantive_tool_names(execution_log)
        if not substantive_tools:
            return True

        tool_chain = [
            str(item).strip()
            for item in normalized.get("tool_chain") or []
            if str(item).strip()
        ]
        public_reasoning = [
            str(item).strip()
            for item in normalized.get("public_reasoning") or []
            if str(item).strip()
        ]
        evidence_summary = str(normalized.get("evidence_summary") or "").strip()
        if not tool_chain or not public_reasoning or not evidence_summary:
            return False

        execution_text = "\n".join([*public_reasoning, evidence_summary]).lower()
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
            return False

        candidate_labels = cls._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        mentioned_tools = cls._mentioned_substantive_tools(execution_text, substantive_tools)
        mentioned_labels = cls._mentioned_candidate_path_labels(execution_text, candidate_labels)
        if not mentioned_tools and not mentioned_labels:
            return False

        evidence_fragments = [*public_reasoning, evidence_summary]
        if max((len(fragment) for fragment in evidence_fragments), default=0) < 60:
            return False
        if sum(len(fragment) for fragment in evidence_fragments) < 140:
            return False
        return True

    @classmethod
    def _recoverable_task_record_from_execution_log(
        cls,
        *,
        task_type: str,
        execution_log: list[dict[str, Any]],
        horizon: int | None,
        indicator_output_keys: list[str] | None,
    ) -> dict[str, Any] | None:
        for entry in reversed(execution_log):
            tool_name = str(entry.get("tool") or "").strip()
            if tool_name == "spawn_subagent":
                output = entry.get("output") or []
                for idx in range(len(output) - 1, -1, -1):
                    item = output[idx]
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    if not isinstance(result, dict):
                        continue
                    sub_execution_log = result.get("execution_log") or []
                    answer = cls._unwrap_task_terminal_answer(
                        task_type,
                        result.get("answer", result.get("prediction")),
                    )
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
                    if not cls._is_task_answer_backed_by_tool_output(
                        task_type,
                        answer,
                        sub_execution_log,
                        horizon=horizon,
                        indicator_output_keys=indicator_output_keys,
                    ):
                        continue
                    reasoning = str(result.get("reasoning") or "").strip()
                    tool_chain = cls._reasoning_substantive_tool_chain(sub_execution_log)
                    execution_context = cls._recoverable_execution_context_from_trace(
                        tool_chain=tool_chain,
                        answer=answer,
                        reasoning=reasoning,
                        execution_context=result.get("execution_context"),
                        execution_log=sub_execution_log,
                    )
                    reasoning_text = cls._normalize_reasoning_for_reuse(
                        reasoning or (
                            f"1. `{' -> '.join(tool_chain) if tool_chain else task_type}` produced "
                            f"the task answer {cls._recoverable_preview_text(answer, limit=120)}."
                        ),
                        sub_execution_log,
                        execution_context,
                    )
                    return cls._recoverable_final_record(
                        answer=answer,
                        answer_type=task_type,
                        confidence=result.get("confidence"),
                        reasoning=reasoning_text,
                        execution_context=execution_context,
                        control_context=cls._normalize_control_context(
                            result.get("control_context")
                        ),
                    )

        top_level_records = cls._top_level_candidate_records(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if not top_level_records:
            return None
        record = top_level_records[-1]
        answer = record.get("prediction")
        tool_chain = [
            str(tool).strip()
            for tool in record.get("tool_chain") or []
            if str(tool).strip()
        ]
        execution_context = cls._recoverable_execution_context_from_trace(
            tool_chain=tool_chain,
            answer=answer,
            reasoning="",
            execution_context=None,
            execution_log=execution_log,
        )
        reasoning = (
            f"1. `{' -> '.join(tool_chain) if tool_chain else task_type}` produced "
            f"the task answer {cls._recoverable_preview_text(answer, limit=120)}."
        )
        return cls._recoverable_final_record(
            answer=answer,
            answer_type=task_type,
            confidence=None,
            reasoning=reasoning,
            execution_context=execution_context,
            control_context={"completion_notes": ""},
        )

    @classmethod
    def _observed_task_answer_from_execution_log(
        cls,
        *,
        task_type: str,
        execution_log: list[dict[str, Any]],
        horizon: int | None,
        indicator_output_keys: list[str] | None,
    ) -> Any | None:
        """Recover the task answer directly observed in substantive tool output.

        New definition:
            When a branch-level wrapped `result.answer` disagrees with the
            branch's real tool observations, batch evaluation should use the
            observed tool-backed answer instead of the wrapped payload.
        """
        top_level_records = cls._top_level_candidate_records(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if not top_level_records:
            return None
        return top_level_records[-1].get("prediction")

    @staticmethod
    def _recoverable_metrics_score(metrics: dict[str, Any]) -> tuple[int, float]:
        if isinstance(metrics.get("accuracy"), (int, float)):
            return (0, -float(metrics["accuracy"]))
        if isinstance(metrics.get("correct"), bool):
            return (0, 0.0 if metrics["correct"] else 1.0)
        for key in ("mae", "avg_mae", "rmse", "mse", "max_mae"):
            if isinstance(metrics.get(key), (int, float)):
                return (1, float(metrics[key]))
        return (9, 0.0)

    @classmethod
    def _recoverable_metric_summary(cls, metrics: dict[str, Any]) -> str:
        if not isinstance(metrics, dict):
            return ""
        parts: list[str] = []
        for key in ("accuracy", "correct", "mae", "avg_mae", "rmse", "mse", "max_mae", "mape"):
            value = metrics.get(key)
            if isinstance(value, bool):
                parts.append(f"{key}={str(value).lower()}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.3f}")
        return ", ".join(parts[:4])

    @staticmethod
    def _recoverable_candidate_identity_key(name: Any, tool_chain: list[str], index: int) -> str:
        clean_chain = [str(tool).strip() for tool in tool_chain if str(tool).strip()]
        if clean_chain:
            return " -> ".join(clean_chain)
        clean_name = str(name or "").strip()
        if clean_name:
            return clean_name
        return f"candidate_{index}"

    @classmethod
    def _recoverable_branch_anchored_candidate_key(
        cls,
        candidate: dict[str, Any],
        index: int,
    ) -> str:
        path = " -> ".join(
            str(tool).strip()
            for tool in candidate.get("tool_chain") or []
            if str(tool).strip()
        )
        stable = str(candidate.get("name") or "").strip()
        base = str(candidate.get("candidate_key") or "").strip()
        if stable:
            if not path:
                return stable
            if stable.lower() == path.lower() or path.lower() in stable.lower():
                return stable
            return f"{stable} :: {path}"
        if path:
            return path
        if base:
            return base
        return f"candidate_{index}"

    @classmethod
    def _disambiguate_recoverable_candidate_keys(
        cls,
        candidates: list[dict[str, Any]],
    ) -> None:
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, candidate in enumerate(candidates):
            key = str(candidate.get("candidate_key") or candidate.get("name") or "").strip()
            if not key:
                continue
            grouped.setdefault(key, []).append((index, candidate))
        for group in grouped.values():
            if len(group) <= 1:
                continue
            for index, candidate in group:
                candidate["candidate_key"] = cls._recoverable_branch_anchored_candidate_key(
                    candidate,
                    index,
                )

    @classmethod
    def _recoverable_training_candidates(
        cls,
        *,
        execution_log: list[dict[str, Any]],
        task_type: str,
        horizon: int | None,
        indicator_output_keys: list[str] | None,
        scored_only: bool = True,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        def _trace_keys(candidate: dict[str, Any]) -> set[str]:
            keys = {
                str(candidate.get("name") or "").strip().lower(),
                str(candidate.get("candidate_key") or "").strip().lower(),
                " -> ".join(candidate.get("tool_chain") or []).lower(),
            }
            return {key for key in keys if key}

        def _match_candidate(name: Any, prediction: Any, evaluation_index: int) -> dict[str, Any] | None:
            for candidate in candidates:
                if candidate["entry_index"] >= evaluation_index:
                    continue
                if cls._answers_match_semantically(
                    task_type,
                    candidate.get("prediction"),
                    prediction,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
                ):
                    return candidate
            return None

        for entry_idx, entry in enumerate(execution_log):
            if str(entry.get("tool") or "").strip() != "spawn_subagent":
                continue
            output = entry.get("output") or []
            if isinstance(output, dict):
                output = output.get("results") or []
            for idx, item in enumerate(output):
                if not isinstance(item, dict) or not item.get("success"):
                    continue
                result = item.get("result") or {}
                if not isinstance(result, dict):
                    continue
                answer = cls._unwrap_task_terminal_answer(
                    task_type,
                    result.get("answer", result.get("prediction")),
                )
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
                sub_execution_log = result.get("execution_log") or []
                tool_chain = cls._reasoning_substantive_tool_chain(sub_execution_log)
                public_reasoning = cls._recoverable_execution_context_from_trace(
                    tool_chain=tool_chain,
                    answer=answer,
                    reasoning=result.get("reasoning"),
                    execution_context=result.get("execution_context"),
                    execution_log=sub_execution_log,
                ).get("public_reasoning", [])
                candidates.append(
                    {
                        "name": cls._candidate_branch_id(result, fallback_idx=idx),
                        "candidate_key": cls._recoverable_candidate_identity_key(
                            cls._candidate_branch_id(result, fallback_idx=idx),
                            tool_chain,
                            idx,
                        ),
                        "tool_chain": tool_chain,
                        "prediction": answer,
                        "metrics": {},
                        "public_reasoning": public_reasoning,
                        "confidence": result.get("confidence"),
                        "entry_index": entry_idx,
                        "steps": sub_execution_log,
                    }
                )

        for evaluation_index, entry in enumerate(execution_log):
            if not entry.get("success"):
                continue
            tool_name = str(entry.get("tool") or "").strip()
            observation = entry.get("observation")
            if observation is None:
                observation = {}
            if tool_name in {"evaluate_batch_against_gt", "batch_evaluate_against_gt"}:
                if not isinstance(observation, dict):
                    continue
                for item in observation.get("results") or []:
                    if not isinstance(item, dict):
                        continue
                    matched = _match_candidate(item.get("name"), item.get("prediction"), evaluation_index)
                    if matched is not None and isinstance(item.get("metrics"), dict):
                        matched["metrics"] = dict(item.get("metrics") or {})
                continue
            if tool_name != "evaluate_against_gt" or not isinstance(observation, dict):
                continue
            metrics = observation.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                continue
            args = entry.get("args") or {}
            matched = _match_candidate("", args.get("prediction"), evaluation_index)
            if matched is not None:
                matched["metrics"] = dict(metrics)
        cls._disambiguate_recoverable_candidate_keys(candidates)
        return [
            candidate
            for candidate in candidates
            if not scored_only or (isinstance(candidate.get("metrics"), dict) and candidate.get("metrics"))
        ]

    @classmethod
    def _recoverable_learning_summary_record_from_execution_log(
        cls,
        *,
        execution_log: list[dict[str, Any]],
        task_type: str,
        horizon: int | None,
        indicator_output_keys: list[str] | None,
    ) -> dict[str, Any] | None:
        if not cls._training_completion_contract_satisfied(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
            return None

        candidate_traces = cls._recoverable_training_candidates(
            execution_log=execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if not candidate_traces:
            return None

        ranked = sorted(
            candidate_traces,
            key=lambda trace: cls._recoverable_metrics_score(trace.get("metrics") or {}),
        )
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        winner_chain = [
            str(tool).strip()
            for tool in winner.get("tool_chain") or []
            if str(tool).strip()
        ]
        winner_label = " -> ".join(winner_chain) or str(winner.get("name") or "candidate").strip()
        winner_reason = ""
        public_reasoning = [
            str(item).strip()
            for item in winner.get("public_reasoning") or []
            if str(item).strip()
        ]
        if public_reasoning:
            winner_reason = public_reasoning[0]
        metric_text = cls._recoverable_metric_summary(dict(winner.get("metrics") or {}))
        prediction_text = cls._recoverable_preview_text(winner.get("prediction"), limit=120)

        insight_parts = [f"Strongest path on this sample: `{winner_label}`."]
        if winner_reason:
            insight_parts.append(f"Execution evidence: {winner_reason.rstrip('.')}.")
        if prediction_text:
            insight_parts.append(f"Observed output: {prediction_text}.")
        if metric_text:
            insight_parts.append(f"Observed evaluation: {metric_text}.")
        insight = " ".join(part.strip() for part in insight_parts if part).strip()

        recommendation = (
            f"Reusable lesson: prefer `{winner_label}` when similar evidence supports the same pattern"
        )
        if winner_reason:
            recommendation += f", because {winner_reason.rstrip('.')}"
        if runner_up is not None:
            runner_up_chain = [
                str(tool).strip()
                for tool in runner_up.get("tool_chain") or []
                if str(tool).strip()
            ]
            runner_up_label = " -> ".join(runner_up_chain) or str(runner_up.get("name") or "runner-up").strip()
            recommendation += f". Compare it against `{runner_up_label}` when the alternative path is also plausible."
        else:
            recommendation += "."

        reasoning_lines = [
            f"1. `{winner_label}` produced the reusable candidate answer {prediction_text}.",
        ]
        if winner_reason:
            reasoning_lines.append(f"2. {winner_reason.rstrip('.')}.")
        if metric_text:
            reasoning_lines.append(f"3. Its recorded evaluation was {metric_text}.")
        if runner_up is not None:
            runner_up_chain = [
                str(tool).strip()
                for tool in runner_up.get("tool_chain") or []
                if str(tool).strip()
            ]
            runner_up_label = " -> ".join(runner_up_chain) or str(runner_up.get("name") or "runner-up").strip()
            reasoning_lines.append(
                f"4. Compared explored candidates, `{winner_label}` remained the strongest evaluated path over `{runner_up_label}`."
            )
        reasoning = "\n".join(reasoning_lines)

        summary_answer = {
            "insight": insight,
            "recommendation": recommendation,
            "winner_candidate": str(
                winner.get("candidate_key") or winner.get("name") or ""
            ).strip(),
            "winner_prediction": winner.get("prediction"),
            "winner_tool_chain": winner_chain,
            "winner_metrics": dict(winner.get("metrics") or {}),
            "compared_candidates": [
                {
                    "name": str(trace.get("candidate_key") or trace.get("name") or "").strip(),
                    "tool_chain": list(trace.get("tool_chain") or []),
                    "prediction": trace.get("prediction"),
                    "metrics": dict(trace.get("metrics") or {}),
                }
                for trace in ranked
            ],
        }
        execution_context = cls._recoverable_execution_context_from_trace(
            tool_chain=winner_chain,
            answer=winner.get("prediction"),
            reasoning=winner_reason or reasoning,
            execution_context={
                "tool_chain": winner_chain,
                "public_reasoning": public_reasoning[:2],
                "evidence_summary": winner_reason or metric_text,
            },
            execution_log=execution_log,
        )
        return cls._recoverable_final_record(
            answer=summary_answer,
            answer_type="learning_summary",
            confidence=winner.get("confidence"),
            reasoning=reasoning,
            execution_context=execution_context,
            control_context={
                "completion_notes": "Recovered from executed candidate traces after terminal delivery failed.",
            },
        )

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
        disable_tool_dropout: bool = False,
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
            or disable_tool_dropout
            or not scope
            or not isinstance(scope_counts, dict)
            or not base_candidates
        ):
            return []

        # The final answer must be tool-backed. Keep the typed answer operation
        # available while sampling the interchangeable evidence/model tools.
        # Otherwise a hot answer tool eventually disappears from every slot,
        # making the multi-branch contract impossible to satisfy.
        never_drop = cls._required_answer_tools(task_type, base_candidates)
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
            "- Each branch `task` may propose a different evidence-gathering angle, but it must preserve the original task target, judgment window, and answer contract; explore a different path, not a different task.",
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
            slot_id = ExecutorAgent._coerce_slot_id(task.get("slot_id"))
            if slot_id is None:
                return (
                    "Self-correction: each sub-agent task must include a valid "
                    "`slot_id` chosen from the provided candidate slots. "
                    "Accepted forms are an integer like `1` or a string like `slot_1`."
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

    def _prepare_main_spawn_subagent_tasks(
        self,
        *,
        tool_args: dict[str, Any],
        execution_log: list[dict[str, Any]],
        plan: dict[str, Any],
        is_train: bool,
        needs_more_candidates: bool,
    ) -> tuple[list[dict[str, Any]] | None, str, str]:
        """Validate and normalize a main-agent spawn decision before execution.

        New definition:
            Spawn validation failures in the main loop are non-actions. They
            should be retried within the same top-level step instead of burning
            a full step budget before any real branch executes.
        """
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

        spawn_min_tasks = MIN_TRAINING_CANDIDATES if is_train else 0
        slot_error = self._validate_spawn_subagent_tasks(
            tool_args.get("tasks", []),
            plan.get("subagent_slots", []),
            min_tasks=spawn_min_tasks,
            disallowed_slot_ids=used_slot_ids if is_train and spawn_rounds > 0 else None,
        )
        if slot_error:
            return None, "invalid_subagent_slots", slot_error

        normalized_tasks, normalize_error = self._normalize_spawn_tasks(
            tool_args.get("tasks", []),
            plan=plan,
        )
        if normalize_error:
            return None, "non_distinct_subagent_paths_after_dropout", normalize_error

        if is_train and spawn_rounds == 0:
            prior_mix_error = self._validate_first_spawn_prior_mix(
                normalized_tasks,
                prior_path=plan.get("train_spawn_prior_path", []),
            )
            if prior_mix_error:
                return None, "first_spawn_prior_mix_missing", prior_mix_error

        if len(normalized_tasks) < max(1, spawn_min_tasks):
            detail = (
                "Self-correction: after deriving advisory paths from the selected "
                "slots' visible tools, only "
                f"{len(normalized_tasks)} distinct branch(es) remain. Choose more "
                "distinct `slot_id`s so this spawn round keeps enough different paths."
            )
            return None, "insufficient_distinct_subagent_paths_after_dropout", detail

        if is_train and needs_more_candidates:
            # Explicitly keep the training branch-count requirement attached to
            # the prepared spawn so the caller can preserve its existing control
            # flow without recomputing slot state.
            return list(normalized_tasks), "", ""
        return list(normalized_tasks), "", ""

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

    @staticmethod
    def _coerce_slot_id(slot_id: Any) -> int | None:
        if isinstance(slot_id, int):
            return slot_id
        text = str(slot_id or "").strip()
        if not text:
            return None
        matched = re.fullmatch(r"(?:slot_)?(\d+)", text)
        if matched is None:
            return None
        return int(matched.group(1))

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
            slot_id = cls._coerce_slot_id(task.get("slot_id"))
            slot_spec = slot_map.get(slot_id) if slot_id is not None else None
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
            slot_visible = []
            if isinstance(slot_spec, dict):
                slot_visible = [
                    str(name).strip()
                    for name in (slot_spec.get("visible_candidates") or [])
                    if str(name).strip()
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
            if slot_id is not None:
                normalized_task["slot_id"] = slot_id
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
    def _series_source_ref_for_step(step: int, tool_name: str) -> str:
        safe_tool = re.sub(r"[^a-zA-Z0-9_]+", "_", str(tool_name or "").strip()) or "artifact"
        return f"step_{int(step)}_{safe_tool}"

    @classmethod
    def _initial_series_sources(
        cls,
        data: Any,
        current_series_state: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        return {
            "original_input": {
                "data": data,
                "state": dict(current_series_state),
            },
        }

    @classmethod
    def _register_series_source(
        cls,
        series_sources: dict[str, dict[str, Any]],
        *,
        ref: str,
        data: Any,
        state: dict[str, Any] | None,
    ) -> None:
        if not ref or data is None or not isinstance(state, dict):
            return
        if str(state.get("artifact_type") or "").strip() != "series":
            return
        series_sources[ref] = {
            "data": data,
            "state": dict(state),
        }

    @classmethod
    def _series_source_prompt_payload(
        cls,
        *,
        series_sources: dict[str, dict[str, Any]],
        current_series_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        def _summary(ref: str, state: dict[str, Any], *, is_current: bool) -> dict[str, Any]:
            coordinate_system = state.get("coordinate_system")
            payload: dict[str, Any] = {
                "ref": ref,
                "is_current": bool(is_current),
                "producer_tool": state.get("producer_tool"),
                "artifact_type": state.get("artifact_type"),
                "parent_artifact_type": state.get("parent_artifact_type"),
            }
            length = cls._series_length_from_state(state)
            if length:
                payload["length"] = length
            if isinstance(coordinate_system, dict):
                if coordinate_system.get("current_space") is not None:
                    payload["current_space"] = coordinate_system.get("current_space")
                if coordinate_system.get("origin_space") is not None:
                    payload["origin_space"] = coordinate_system.get("origin_space")
                if coordinate_system.get("relative_to") is not None:
                    payload["relative_to"] = coordinate_system.get("relative_to")
            return payload

        items.append(_summary("current", current_series_state, is_current=True))
        for ref, bundle in series_sources.items():
            if ref == "current":
                continue
            state = bundle.get("state")
            if not isinstance(state, dict):
                continue
            items.append(_summary(ref, state, is_current=False))
        return items

    @staticmethod
    def _tool_series_source_ref(tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Resolve an omitted routing field from the tool's declared default.

        Explicit values still reach normal routing and tool validation unchanged.
        Tools without a declared default retain the executor's current-series input.
        """
        if "source_ref" in tool_args:
            return tool_args["source_ref"]
        schema = getattr(get_tool_instance(tool_name), "parameters_schema", {})
        return schema.get("properties", {}).get("source_ref", {}).get("default")

    @classmethod
    def _resolve_series_source(
        cls,
        source_ref: Any,
        *,
        current_data: Any,
        current_series_state: dict[str, Any],
        series_sources: dict[str, dict[str, Any]],
    ) -> tuple[Any, dict[str, Any], str, str | None]:
        ref = str(source_ref or "").strip()
        if not ref or ref == "current":
            return current_data, current_series_state, "current", None
        bundle = series_sources.get(ref)
        if not isinstance(bundle, dict):
            available = ["current", *sorted(series_sources.keys())]
            return (
                current_data,
                current_series_state,
                "current",
                "Self-correction: unknown `source_ref`. "
                f"Choose one of: {', '.join(available)}.",
            )
        state = bundle.get("state")
        if not isinstance(state, dict):
            return current_data, current_series_state, "current", None
        return bundle.get("data"), state, ref, None

    @staticmethod
    def _build_state_update_message(
        *,
        current_series_state: dict[str, Any],
        latest_artifact_state: dict[str, Any],
        accessible_series_sources: list[dict[str, Any]] | None = None,
        task_answer_status: dict[str, Any] | None = None,
        branch_task_answer_status: dict[str, Any] | None = None,
        latest_redundancy_status: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "current_series_state": current_series_state,
            "latest_artifact_state": latest_artifact_state,
        }
        if accessible_series_sources:
            payload["accessible_series_sources"] = accessible_series_sources
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
            + "- By default, index-sensitive tools such as `slice`, `value_at`, and `window_stats` use indices from `current_series_state.coordinate_system.current_space`.\n"
            + "- If you pass `source_ref`, the tool reads that selected series source instead, and indices must follow that source's own `current_space` shown in `accessible_series_sources`.\n"
            + "- Treat `origin_space` as provenance only. It explains where a series came from; it is not the coordinate space for the next tool call.\n"
            + "- Treat tool-returned timestamp fields such as `timestamp`, `start_timestamp`, and `end_timestamp` as evidence labels only. Do not feed them back into `slice`, `value_at`, or `window_stats` unless the selected source itself is explicitly timestamp-indexed.\n"
            + "- If the current working series is no longer the original input series, do not call index-sensitive tools with original-input indices against the current series.\n"
            + "- Use `source_ref=\"original_input\"` when you need the raw history after the working series has already moved to a forecast, slice, or other derived artifact.\n"
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

        async def retain_ordinary_failure(coro):
            try:
                return await coro
            except ExecutionContractError:
                raise
            except Exception as error:
                return error

        tasks = [asyncio.create_task(retain_ordinary_failure(coro)) for coro in coros]
        try:
            done_results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        results = []
        n_success = 0
        observation_parts = [f"Sub-agent results ({n_tasks} tasks):\n"]

        for i, r in enumerate(done_results):
            if isinstance(r, Exception):
                error_type = type(r).__name__
                failure_reason = str(
                    getattr(r, "failure_reason", "")
                    or (
                        "max_steps_exhausted"
                        if isinstance(r, MaxStepsExhaustedError)
                        else error_type
                    )
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

        memory_section = str(plan.get("sample_packet_text") or "").strip()

        hinted_tool_notes = self._build_tool_notes_section(
            plan.get("tool_notes", {}),
            [tool for tool in self._hinted_tools(tools_hint) if tool in candidates],
            plan["task_type"],
        )
        hinted_tool_note_keys = self._selected_tool_note_keys(
            plan.get("tool_notes", {}),
            [tool for tool in self._hinted_tools(tools_hint) if tool in candidates],
            plan["task_type"],
        )

        task_boundary_section = self._render_prompt_section(
            "Task Boundary",
            self._build_task_boundary_section(
                plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
                scope_keys=plan.get("scope_keys", []),
            ),
        )
        task_semantic_section = self._render_prompt_section(
            "Task Semantic Guidance",
            self._build_task_semantic_guidance_section(
                task_type=plan["task_type"],
                scope_keys=plan.get("scope_keys", []),
            ),
        )
        execution_context_section = self._render_prompt_section(
            "Execution Context",
            "",
        )
        control_context_section = self._render_prompt_section(
            "Control Context",
            self._join_prompt_sections(
                self._build_action_state_section(
                    state="need_evidence",
                    required_final_type=plan["task_type"],
                    execution_log=[],
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ),
                f"Branch Goal:\n{task_description}\n" if str(task_description).strip() else "",
                self._build_subagent_tools_hint_section(
                    tools_hint,
                    candidates,
                ),
                self._build_parent_branch_metadata_section(branch_metadata),
            ),
        )
        output_contract_section = self._render_prompt_section(
            "Output Contract",
            self._build_subagent_final_answer_contract_section(
                plan["task_type"],
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
                scope_keys=plan.get("scope_keys", []),
            ),
        )
        support_section = self._render_prompt_section(
            "Support",
            self._join_prompt_sections(
                hinted_tool_notes,
                memory_section,
            ),
        )
        decision_section = self._build_decision_section(
            allow_finish=True,
            spawn_available=False,
            task_type=plan["task_type"],
        )
        objective_sections = self._join_prompt_sections(
            task_boundary_section,
            task_semantic_section,
        )

        first_user = executor_prompt.SUBAGENT_FIRST_TURN.format(
            task_prompt=plan.get("task_description", ""),
            task_boundary_section=objective_sections,
            fingerprint=self._format_prompt_payload(
                self._prepare_prompt_fingerprint(
                    plan["fingerprint"],
                    profiling_section=plan.get("profiling_section", ""),
                ),
                limit=None,
            ),
            profiling_section=plan.get("profiling_section", ""),
            execution_context_section=execution_context_section,
            control_context_section=control_context_section,
            output_contract_section=output_contract_section,
            support_section=support_section,
            decision_section=decision_section,
            tool_names=executor_prompt.build_tool_name_text(candidates),
        )

        base_control_context = self._normalize_control_context({
            "completion_notes": "",
            "audit": self._build_control_context_audit(
                memory_injected={
                    "soul_text": str((plan.get("memory_audit") or {}).get("soul_text") or "").strip(),
                    "memory_text": str((plan.get("memory_audit") or {}).get("memory_text") or "").strip(),
                    "system_memory_text": system_memory,
                    "skills_text": memory_section,
                    "tool_notes_section": hinted_tool_notes,
                    "tool_note_keys": hinted_tool_note_keys,
                    "tool_note_count": len(hinted_tool_note_keys),
                },
                prompt_sections=[
                    self._build_prompt_section_audit(name="system_prompt", text=system_prompt),
                    self._build_prompt_section_audit(name="first_user", text=first_user),
                    self._build_prompt_section_audit(name="task_boundary_section", text=task_boundary_section),
                    self._build_prompt_section_audit(name="task_semantic_section", text=task_semantic_section),
                    self._build_prompt_section_audit(name="execution_context_section", text=execution_context_section),
                    self._build_prompt_section_audit(name="control_context_section", text=control_context_section),
                    self._build_prompt_section_audit(name="output_contract_section", text=output_contract_section),
                    self._build_prompt_section_audit(name="support_section", text=support_section),
                    self._build_prompt_section_audit(name="decision_section", text=decision_section),
                ],
            ),
        })

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]
        skip_generic_turn_prompt_once = False

        horizon = plan.get("horizon")
        current_data = data
        current_series_state = self._initial_series_state(current_data)
        series_sources = self._initial_series_sources(current_data, current_series_state)
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
                        task_type=plan["task_type"],
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
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                    scope_keys=plan.get("scope_keys", []),
                    sub_task_description=task_description,
                    tools_hint=tools_hint,
                )
            response = None
            decision: dict[str, Any] = {}
            thought = ""
            advance_to_next_step = False
            non_action_self_corrections = 0
            decision_recovery_exhaustions = 0
            while True:
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
                    "same_step_retry_count": non_action_self_corrections,
                })

                if parse_failed:
                    parse_failure_reason = (
                        "invalid_decision_wrong_tool_family"
                        if str(thought or "").strip() == "invalid_decision_wrong_tool_family"
                        else "invalid_json_response"
                    )
                    invalid_json_msg = self._invalid_json_self_correction_message(
                        thought,
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
                    )
                    if repair_delivery_mode:
                        repair_delivery_feedback_messages = [invalid_json_msg]
                    conversation.append({
                        "role": "user",
                        "content": invalid_json_msg,
                    })
                    skip_generic_turn_prompt_once = True
                    if non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP:
                        non_action_self_corrections += 1
                        continue
                    if decision_recovery_exhaustions < MAX_DECISION_RECOVERY_EXHAUSTIONS_PER_STEP:
                        decision_recovery_exhaustions += 1
                        non_action_self_corrections = 0
                        continue
                    if repair_delivery_mode:
                        recoverable_decision = self._decision_from_recoverable_final(
                            self._recoverable_failure_record(
                                expected_type=plan["task_type"],
                                task_type=plan["task_type"],
                                execution_log=execution_log,
                                horizon=plan.get("horizon"),
                                indicator_output_keys=plan.get("indicator_output_keys", []),
                            ),
                            thought=(
                                "Reason: This branch already has an execution-grounded recoverable final "
                                "payload, so the unresolved issue is only formatting. "
                                "Check: No missing tool result remains because the final payload is already "
                                "recoverable from the branch trace. "
                                "Self-check: The required final payload artifact is already present in the "
                                "branch trace, and its basic sanity condition is that no further tool result "
                                "is needed to complete the branch output. "
                                "Update: Keep the recovered candidate and stop retrying format-only turns, "
                                "since no rival candidate or new evidence is being added. "
                                "Action: Finish with that recovered payload unchanged."
                            ),
                        )
                        if recoverable_decision is not None:
                            decision = recoverable_decision
                            thought = str(decision.get("thought") or thought)
                            break
                    failure_reason = "decision_parse_exhausted"
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": failure_reason,
                    })
                    raise DecisionOutputExhaustedError(
                        agent_kind="subagent",
                        sub_agent_idx=idx,
                        failure_reason=failure_reason,
                        partial_result=self._failure_partial_result(
                            expected_type=plan["task_type"],
                            task_type=plan["task_type"],
                            execution_log=execution_log,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                            recoverable_final=recoverable_answer or recoverable_final,
                            failure_reason=failure_reason,
                            extra={
                                "slot_id": slot_id,
                                "sub_task_description": task_description,
                                "tools_hint": tools_hint,
                                "branch_metadata": branch_metadata,
                            },
                        ),
                    )

                if not parse_failed:
                    conversation.append({
                        "role": "assistant",
                        "content": self._decision_to_conversation_text(decision),
                    })

                if not self._has_structured_thought(thought):
                    structured_thought_msg = (
                        "Your `thought` must explicitly include `Reason:`, `Check:`, "
                        "`Self-check:`, `Update:`, and `Action:` in that order. Please continue with "
                        "that structure."
                    )
                    if repair_delivery_mode:
                        repair_delivery_feedback_messages = [structured_thought_msg]
                    conversation.append({
                        "role": "user",
                        "content": structured_thought_msg,
                    })
                    skip_generic_turn_prompt_once = True
                    if non_action_self_corrections < MAX_NON_ACTION_SELF_CORRECTIONS_PER_STEP:
                        non_action_self_corrections += 1
                        continue
                    if decision_recovery_exhaustions < MAX_DECISION_RECOVERY_EXHAUSTIONS_PER_STEP:
                        decision_recovery_exhaustions += 1
                        non_action_self_corrections = 0
                        continue
                    if repair_delivery_mode:
                        recoverable_decision = self._decision_from_recoverable_final(
                            self._recoverable_failure_record(
                                expected_type=plan["task_type"],
                                task_type=plan["task_type"],
                                execution_log=execution_log,
                                horizon=plan.get("horizon"),
                                indicator_output_keys=plan.get("indicator_output_keys", []),
                            ),
                            thought=(
                                "Reason: This branch already has an execution-grounded final payload and "
                                "repeated format retries are no longer changing the judgment. "
                                "Check: The existing branch trace already contains the decisive evidence, "
                                "so another retry would not test anything new. "
                                "Self-check: The required final payload artifact is already present, and "
                                "its basic sanity condition is that another retry would not add missing "
                                "evidence or change the branch output. "
                                "Update: Keep the recovered final candidate rather than attempting another "
                                "format-only retry, because no competing candidate or new evidence remains. "
                                "Action: Finish with that recovered payload."
                            ),
                        )
                        if recoverable_decision is not None:
                            decision = recoverable_decision
                            thought = str(decision.get("thought") or thought)
                            break
                    failure_reason = "decision_format_exhausted"
                    trace.agent_step("executor", "subagent_self_correction", {
                        "sub_agent_idx": idx,
                        "step": step,
                        "reason": failure_reason,
                    })
                    raise DecisionOutputExhaustedError(
                        agent_kind="subagent",
                        sub_agent_idx=idx,
                        failure_reason=failure_reason,
                        partial_result=self._failure_partial_result(
                            expected_type=plan["task_type"],
                            task_type=plan["task_type"],
                            execution_log=execution_log,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
                            recoverable_final=recoverable_answer or recoverable_final,
                            failure_reason=failure_reason,
                            extra={
                                "slot_id": slot_id,
                                "sub_task_description": task_description,
                                "tools_hint": tools_hint,
                                "branch_metadata": branch_metadata,
                            },
                        ),
                    )
                break

            if advance_to_next_step:
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
                        domain=plan.get("domain", ""),
                        scope_keys=plan.get("scope_keys", []),
                    )
                    and step < MAX_SUB_REACT_STEPS
                ):
                    _, invalid_reason = self._validate_final_answer(
                        requested_type,
                        answer,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        domain=plan.get("domain", ""),
                        scope_keys=plan.get("scope_keys", []),
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
                    correction_msg = (
                        f"Your current final answer is not a valid '{requested_type}' payload. "
                        f"Reason: {invalid_reason}. "
                        f"Return a proper task-typed result instead of an explanation."
                    )
                    if requested_type == "forecast" and plan.get("horizon") is not None:
                        correction_msg += (
                            f" The forecast must contain exactly {plan['horizon']} time steps "
                            "of finite numeric values, with all requested channels preserved. "
                            "Return the complete tool forecast without appending or omitting values."
                        )
                    conversation.append({"role": "user", "content": correction_msg})
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
                repair_record = (
                    recoverable_answer if isinstance(recoverable_answer, dict) and recoverable_answer
                    else recoverable_final if isinstance(recoverable_final, dict) and recoverable_final
                    else None
                )
                reasoning = self._coerce_reasoning_text(
                    decision.get("reasoning", decision.get("thought", ""))
                )
                execution_context = self._normalize_execution_context(
                    decision.get("execution_context"),
                    execution_log=execution_log,
                )
                if (
                    repair_delivery_mode
                    and isinstance(repair_record, dict)
                    and not any(
                        execution_context.get(key)
                        for key in ("tool_chain", "public_reasoning", "evidence_summary")
                    )
                ):
                    execution_context = self._normalize_execution_context(
                        repair_record.get("execution_context"),
                        execution_log=execution_log,
                    )
                if (
                    repair_delivery_mode
                    and isinstance(repair_record, dict)
                    and not reasoning
                ):
                    reasoning = self._coerce_reasoning_text(
                        repair_record.get("reasoning", decision.get("thought", ""))
                    )
                reasoning = self._normalize_reasoning_for_reuse(
                    reasoning,
                    execution_log,
                    execution_context,
                    task_type=requested_type,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                )
                repair_base_control_context = base_control_context
                if repair_delivery_mode and isinstance(repair_record, dict):
                    repair_base_control_context = self._merge_control_context(
                        base_control_context,
                        repair_record.get("control_context"),
                    )
                control_context = self._merge_control_context(
                    repair_base_control_context,
                    decision.get("control_context"),
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
                    execution_context = self._recoverable_execution_context_from_trace(
                        tool_chain=self._reasoning_substantive_tool_chain(execution_log),
                        answer=answer,
                        reasoning=reasoning,
                        execution_context=execution_context,
                        execution_log=execution_log,
                    )
                    reasoning = self._normalize_reasoning_for_reuse(
                        reasoning,
                        execution_log,
                        execution_context,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                    execution_context_verdict = await self._call_execution_context_verdict(
                        execution_context,
                        execution_log,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                    and not execution_context_verdict.get("passed", False)
                ):
                    repair_delivery_feedback_messages = [
                        self._execution_context_correction_message(
                            execution_log,
                            verdict=execution_context_verdict,
                            task_type=requested_type,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
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
                    reasoning_verdict = await self._call_stepwise_reasoning_verdict(
                        reasoning,
                        execution_log,
                        task_type=requested_type,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                    )
                if (
                    self._should_enforce_stepwise_reasoning(plan.get("_mode"))
                    and not reasoning_verdict.get("passed", False)
                ):
                    repair_delivery_feedback_messages = [
                        self._stepwise_reasoning_correction_message(
                            execution_log,
                            verdict=reasoning_verdict,
                            task_type=requested_type,
                            horizon=plan.get("horizon"),
                            indicator_output_keys=plan.get("indicator_output_keys", []),
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
            selected_data, selected_state, selected_source_ref, source_error = self._resolve_series_source(
                self._tool_series_source_ref(tool_name, tool_args),
                current_data=current_data,
                current_series_state=current_series_state,
                series_sources=series_sources,
            )
            if source_error:
                trace.agent_step("executor", "subagent_self_correction", {
                    "sub_agent_idx": idx,
                    "step": step,
                    "reason": "unknown_source_ref",
                    "tool": tool_name,
                    "source_ref": tool_args.get("source_ref"),
                })
                conversation.append({
                    "role": "user",
                    "content": source_error,
                })
                continue
            runtime_incompatibility_msg = self._tool_runtime_incompatibility_message(
                tool_name,
                current_data=selected_data,
                current_series_state=selected_state,
            )
            if runtime_incompatibility_msg and step < MAX_SUB_REACT_STEPS:
                trace.agent_step("executor", "subagent_self_correction", {
                    "sub_agent_idx": idx,
                    "step": step,
                    "reason": "runtime_incompatible_tool",
                    "tool": tool_name,
                    "source_ref": selected_source_ref,
                })
                conversation.append({
                    "role": "user",
                    "content": runtime_incompatibility_msg,
                })
                continue
            coordinate_misuse_msg = self._tool_argument_coordinate_misuse_message(
                tool_name,
                tool_args=tool_args,
                current_data=selected_data,
                source_ref=selected_source_ref,
            )
            if coordinate_misuse_msg and step < MAX_SUB_REACT_STEPS:
                trace.agent_step("executor", "subagent_self_correction", {
                    "sub_agent_idx": idx,
                    "step": step,
                    "reason": "coordinate_argument_misuse",
                    "tool": tool_name,
                    "source_ref": selected_source_ref,
                })
                conversation.append({
                    "role": "user",
                    "content": coordinate_misuse_msg,
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
            result = await self._execute_tool(tool_name, tool_args, selected_data, text_data, images)
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
                "source_ref": selected_source_ref,
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
                    self._register_series_source(
                        series_sources,
                        ref=self._series_source_ref_for_step(step, tool_name),
                        data=downstream_data,
                        state=current_series_state,
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
                    accessible_series_sources=self._series_source_prompt_payload(
                        series_sources=series_sources,
                        current_series_state=current_series_state,
                    ),
                    task_answer_status=self._task_answer_status(
                        task_type=plan["task_type"],
                        candidate=observation_payload,
                        source_tool=tool_name,
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        scope_keys=plan.get("scope_keys", []),
                    ),
                    branch_task_answer_status=self._latest_branch_task_answer_status(
                        task_type=plan["task_type"],
                        execution_log=execution_log + [log_entry],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
                        scope_keys=plan.get("scope_keys", []),
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
            recovered_terminal = self._recoverable_terminal_record(
                expected_type=plan["task_type"],
                task_type=plan["task_type"],
                execution_log=execution_log,
                horizon=plan.get("horizon"),
                indicator_output_keys=plan.get("indicator_output_keys", []),
            )
            if recovered_terminal is not None and recoverable_final is None:
                recoverable_final = recovered_terminal
            locked_recoverable_final = recoverable_answer or recoverable_final
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
                    scope_keys=plan.get("scope_keys", []),
                    sub_agent_idx=idx,
                    recoverable_final=locked_recoverable_final,
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
                    domain=plan.get("domain", ""),
                    scope_keys=plan.get("scope_keys", []),
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
                elif not self._terminal_answer_preserves_recoverable_answer(
                    expected_type=plan["task_type"],
                    task_type=plan["task_type"],
                    recoverable_final=locked_recoverable_final,
                    candidate_answer=terminal_answer,
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
                ):
                    terminal_valid = False
                    terminal_failure_reason = "changed_recoverable_answer"
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
                terminal_execution_context = self._recoverable_execution_context_from_trace(
                    tool_chain=self._reasoning_substantive_tool_chain(execution_log),
                    answer=terminal_answer,
                    reasoning=terminal_reasoning,
                    execution_context=terminal_execution_context,
                    execution_log=execution_log,
                )
                terminal_reasoning = self._normalize_reasoning_for_reuse(
                    terminal_reasoning,
                    execution_log,
                    terminal_execution_context,
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
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
                        task_type=plan["task_type"],
                        horizon=plan.get("horizon"),
                        indicator_output_keys=plan.get("indicator_output_keys", []),
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
                    task_type=plan["task_type"],
                    horizon=plan.get("horizon"),
                    indicator_output_keys=plan.get("indicator_output_keys", []),
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

        if not self._is_valid_final_answer(
            plan["task_type"], final_result.get("answer"),
            horizon=plan.get("horizon"),
            indicator_output_keys=plan.get("indicator_output_keys", []),
            domain=plan.get("domain", ""), scope_keys=plan.get("scope_keys", []),
        ):
            raise ExecutionContractError("Sub-agent final answer violates the scoped task contract")
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
        if bool(plan.get("_disable_tool_dropout", self._disable_tool_dropout)):
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
        never_drop = hinted_tools | self._required_answer_tools(plan.get("task_type", ""), base_candidates)

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
    def _required_answer_tools(task_type: str, candidates: list[str]) -> set[str]:
        """Tools that provide the task's typed final judgment, before dropout.

        QA/correlation/indicator evidence tools do not generally produce the
        required final answer schema. Their task-specific reasoning tool must
        stay callable under the tool-backed delivery contract. Forecasts have
        multiple interchangeable numeric-output models and keep normal dropout.
        This does not call a tool automatically or relax answer validation.
        """
        if task_type not in {"qa", "correlation", "indicator"}:
            return set()
        tool = ExecutorAgent._task_llm_reasoning_tool(task_type)
        return {tool} if tool in candidates else set()

    @staticmethod
    def _build_learning_summary_contract_section() -> str:
        return (
            '- In training mode, the main agent must finish with `answer_type` exactly `"learning_summary"`.\n'
            "- This learning-summary contract applies only on the final completion turn.\n"
            "- If the current turn is a tool call (including `spawn_subagent` or `evaluate_*`), do NOT include any `learning_summary`, winner selection, or final-answer payload in the same reply.\n"
            "- A placeholder `learning_summary` is invalid. Do not finish with text like 'pending', 'waiting for sub-agent results', 'need evaluation first', or any other not-yet-earned summary.\n"
            "- Only return `learning_summary` after actual candidate outputs already exist.\n"
            "- If at least one candidate was validated, name the winning candidate path plus decisive evidence.\n"
            "- If no candidate was validated, say so in `answer.insight` / `answer.recommendation` as a failure pattern, and keep `execution_context` empty instead of inventing a winner-like evidence card.\n"
            "- `answer` must be a JSON object.\n"
            "- Required keys inside `answer`: `insight`, `recommendation`.\n"
            "- `insight` must be a short non-empty string that names the winning candidate path/tool chain and the decisive evidence that made it stronger than the compared candidates.\n"
            "- `recommendation` must be a short non-empty string stating one reusable next-time lesson tied to the same kind of evidence pattern; if the selected winner is incorrect overall, state a cautionary failure pattern instead.\n"
            "- `recommendation` is not the task prediction, not a forecast array, and not an indicator JSON object.\n"
            "- Do not put winner bookkeeping fields such as `winning_candidate_name`, `winning_tool_chain`, `winning_prediction`, `compared_candidates`, or `selection_basis` inside `answer`; those are controller-derived metadata, not the required user-facing summary fields.\n"
            "- `answer.insight` and `answer.recommendation` are the only place for candidate comparison, winner/loser discussion, cautionary failure patterns, or what GT evaluation revealed.\n"
            "- `reasoning` must be the numbered long-form version of the same execution-side evidence recorded in `execution_context.public_reasoning`; reuse the same literal tool/path names instead of introducing looser aliases.\n"
            "- Write `reasoning` as if it will be shown without any training metadata. It must stand on execution evidence alone, not on branch management or evaluator outcomes.\n"
            "- Final-answer fields must also include `execution_context` and `control_context`.\n"
            "- `execution_context` must contain only execution-side evidence: the substantive `tool_chain`, reusable `public_reasoning`, and a concise `evidence_summary`.\n"
            "- Treat `execution_context` as a reusable evidence card for downstream inference. It must still make sense if the reader never sees any training-only metadata.\n"
            "- If no candidate was validated, `execution_context.tool_chain` should be `[]`, `execution_context.public_reasoning` should be `[]`, and `execution_context.evidence_summary` should be an empty string.\n"
            "- Each `public_reasoning` item should state one concrete observed fact and should literally name the tool or candidate path that produced that fact.\n"
            "- `evidence_summary` must be a short synthesis of those same observed facts; it should summarize what the tools/data showed, not how the training controller reacted.\n"
            "- `control_context` is the only place for completion/comparison mechanics; use an empty string if there is nothing extra to record.\n"
            "- Do not write about framework/orchestration mechanics such as `spawn_subagent`, `evaluate_*`, `execution context`, `assigned path`, `before finishing`, or ground truth revelation.\n"
            "- In `reasoning` and `execution_context`, do not talk about candidates being correct/incorrect because of GT evaluation, branches converging/diverging, winner selection bookkeeping, or the need to continue/stop the training loop.\n"
            "- Do not justify the summary with control-flow phrases such as `task-valid`, `must call a tool`, `latest instruction requires`, `branch comparison`, `learning mode`, or `spawning independent branches`.\n"
            "- Do not write about answer packaging mechanics such as re-wrapping, payload preservation, JSON repair, or returning an unchanged saved answer.\n"
            "- Even if your top-level work was mostly orchestration, summarize the real candidate approaches, their concrete tool chains, the decisive evidence, and why the winner was selected.\n"
            "- The framework will derive winner fields from the actual candidate traces, but you should still summarize which candidate won and why.\n"
            '- Exact legacy JSON finish shape if you are not using a tool call: '
            '`{"thought":"Reason: ... Check: ... Self-check: ... Update: ... Action: finish with learning summary.","done":true,'
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
        scope_keys: list[str] | None = None,
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
                + ExecutorAgent._build_scoped_classification_output_contract_section(
                    scope_keys=scope_keys,
                    include_example=True,
                )
            )
        if task_type == "qa":
            if ExecutorAgent._scope_keys_match_suffix(scope_keys, "_mcqa"):
                return (
                    "- Core task: select exactly one of the supplied options A, B, C, D.\n"
                    "- Task-valid candidate output: the single uppercase option letter.\n"
                    "- Reading or summarizing news and computing forecasts supplies evidence only. "
                    "Finish the option judgment with llm_reasoning_qa before returning.\n"
                )
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

    @staticmethod
    def _scope_keys_match_suffix(
        scope_keys: list[str] | None,
        suffix: str,
    ) -> bool:
        suffix_text = str(suffix or "").strip()
        if not suffix_text:
            return False
        return any(
            str(scope).strip().endswith(suffix_text)
            for scope in (scope_keys or [])
            if str(scope).strip()
        )

    @classmethod
    def _canonicalize_scoped_classification_answer(
        cls,
        answer: Any,
        *,
        domain: str = "",
        scope_keys: list[str] | None = None,
    ) -> str | None:
        normalized = cls._normalize_text_answer(answer)
        if normalized is None:
            return None

        domain_text = str(domain or "").strip().lower()
        normalized_scope_keys = [
            str(scope).strip()
            for scope in (scope_keys or [])
            if str(scope).strip()
        ]
        lowered = normalized.strip().lower()

        is_weather_trend = (
            domain_text == "weather"
            or any(scope.startswith("weather_") for scope in normalized_scope_keys)
        )
        if is_weather_trend:
            alias_map = {
                "up": "increasing",
                "down": "decreasing",
                "flat": "stable",
            }
            if lowered in alias_map:
                return alias_map[lowered]
            if lowered in {"increasing", "decreasing", "stable"}:
                return lowered
            parsed = mtbench_prompts.parse_weather_trend_output(normalized)
            if isinstance(parsed, str):
                parsed_lower = parsed.strip().lower()
                if parsed_lower in {"increasing", "decreasing", "stable"}:
                    return parsed_lower
            return None

        is_finance_trend = (
            any(scope.startswith("finance_aligned_") for scope in normalized_scope_keys)
            and cls._scope_keys_match_suffix(normalized_scope_keys, "_trend")
        )
        if is_finance_trend:
            explicit = {
                "<-4%": "<-4%",
                "-2% ~ -4%": "-2% ~ -4%",
                "-2% ~ +2%": "-2% ~ +2%",
                "+2% ~ +4%": "+2% ~ +4%",
                ">+4%": ">+4%",
            }
            if normalized in explicit:
                return explicit[normalized]
            compact = normalized.replace(" ", "")
            compact_map = {
                "<-4%": "<-4%",
                "-2%~-4%": "-2% ~ -4%",
                "-2%~+2%": "-2% ~ +2%",
                "+2%~+4%": "+2% ~ +4%",
                ">+4%": ">+4%",
            }
            if compact in compact_map:
                return compact_map[compact]
            parsed = mtbench_prompts.parse_finance_trend_output(normalized)
            if isinstance(parsed, str) and parsed.strip():
                parsed_text = parsed.strip()
                parsed_compact = parsed_text.replace(" ", "")
                if parsed_text in explicit:
                    return parsed_text
                if parsed_compact in compact_map:
                    return compact_map[parsed_compact]
            return None

        return normalized

    @classmethod
    def _build_scoped_classification_output_contract_section(
        cls,
        *,
        scope_keys: list[str] | None = None,
        include_example: bool = False,
    ) -> str:
        if (
            any(str(scope).strip().startswith("finance_aligned_") for scope in (scope_keys or []))
            and cls._scope_keys_match_suffix(scope_keys, "_trend")
        ):
            example_line = '- Example: "-2% ~ +2%"\n' if include_example else ""
            return (
                "- Final delivery label must be exactly one of: `<-4%`, `-2% ~ -4%`, `-2% ~ +2%`, `+2% ~ +4%`, `>+4%`.\n"
                "- `up`, `down`, `flat`, `bullish`, `bearish`, or a raw numeric forecast are not valid final answers for this task.\n"
                "- If your intermediate reasoning is directional or numeric, continue until you map it into one of the five official return buckets before finishing.\n"
                + example_line
            )
        if (
            any(str(scope).strip().startswith("weather_aligned_") for scope in (scope_keys or []))
            and (
                cls._scope_keys_match_suffix(scope_keys, "_trend")
                or cls._scope_keys_match_suffix(scope_keys, "_trend_past")
            )
        ):
            example_line = '- Example: "decreasing"\n' if include_example else ""
            return (
                "- Final delivery label must be exactly one of: `increasing`, `decreasing`, `stable`.\n"
                "- `up`, `down`, and `flat` are shorthand only; convert them to the official label before finishing.\n"
                + example_line
            )
        return ""

    @classmethod
    def _build_task_semantic_guidance_section(
        cls,
        *,
        task_type: str,
        scope_keys: list[str] | None = None,
    ) -> str:
        """Return benchmark-specific semantic requirements for judgment, not output shape.

        New definition:
            `Task Semantic Guidance` is a prompt-only section for the few tasks
            whose task meaning is easy to confuse. It constrains what must be
            established before answering, but it does not require extra output
            fields and does not prescribe a fixed tool order.
        """
        if cls._scope_keys_match_suffix(scope_keys, "_correlation"):
            return (
                "- This is a predictive correlation task: infer the most likely next-window price reaction from the target-specific text signal and the recent market context, not from hidden future prices you do not have.\n"
                "- First isolate the target-asset-specific directional signal from the text/news context, then use the visible price regime only as supporting context for what reaction is plausible next.\n"
                "- Do not let the pre-publication trend, background drift, or broader market regime by itself override a clear target-specific directional signal; the label is about the likely reaction after publication, not whether the visible history was already moving the same way.\n"
                "- If a true post-publication tail is visible, use it as extra evidence; if it is missing or only partially visible, do not collapse to `No Correlation` for that reason alone.\n"
                "- Judge only the signal about the target asset; do not treat industry, competitor, or macro commentary as a direct target signal unless the text clearly ties it to the target asset.\n"
                "- Use `No Correlation` only when the target-specific text signal itself is genuinely neutral, mixed, too weak to sign, or clearly unrelated to the likely next-window move.\n"
                "- If the target-specific signal is directional but uncertainty remains, prefer a signed correlation label and use strong vs moderate to express confidence; do not retreat to `No Correlation` just because the evidence is imperfect.\n"
                "- Analyst recap, ranking, and valuation articles are weaker than fresh catalysts, but they can still support a moderate signed label if their directional implication is clear.\n"
                "- Reserve strong correlation labels for clean, clearly attributable reactions or very strong directional signals; weak or noisy but still directional evidence should usually map to a moderate positive or moderate negative label.\n"
            )
        if (
            task_type == "classification"
            and any(str(scope).strip().startswith("finance_aligned_") for scope in (scope_keys or []))
            and cls._scope_keys_match_suffix(scope_keys, "_trend")
        ):
            return (
                "- This label is the return bucket from the latest observed price to the task's target future window, not a name for the recent historical regime.\n"
                "- Estimate the future move from the latest observed price for the target prediction window, then map that move to the required return bucket.\n"
                "- If you already have a numeric endpoint or forecast-window value, explicitly compute the percentage change from the latest observed price before choosing the bucket.\n"
                "- Determine the direction first, then the magnitude bucket; do not let a bearish or bullish historical regime alone decide the final bucket.\n"
                "- Use news/context only as evidence about the likely future move from the latest observed price, not as a standalone bucket label.\n"
                "- Do not default to the central bucket just because one path is flatter or the news is weak; if the implied move is near a bucket boundary, compare the adjacent buckets explicitly before settling on neutral.\n"
                "- When candidate forecast paths disagree, prefer the path that better respects the latest regime and current endpoint context before mapping its implied move into the return bucket.\n"
            )
        if task_type == "forecast":
            return (
                "- If multiple forecast paths are available, first decide which path best respects the latest regime, boundary behavior, and short-horizon continuity; do not average them by prose.\n"
                "- Do not prefer an over-smoothed mean-reverting path or an unjustifiably explosive extension just because it is simpler to narrate.\n"
                "- Use text/context as a catalyst prior only when it is concrete and time-aligned; otherwise let the observed series regime dominate path selection.\n"
            )
        if (
            task_type == "classification"
            and any(str(scope).strip().startswith("weather_aligned_") for scope in (scope_keys or []))
            and cls._scope_keys_match_suffix(scope_keys, "_trend")
            and not cls._scope_keys_match_suffix(scope_keys, "_trend_past")
        ):
            return (
                "- Judge this task from a like-for-like comparison between the required observed reference window and the target future window, not from the whole-series slope.\n"
                "- For hourly weather trend tasks, compare the mean of the last observed 24 hours with the mean of the first forecast-day 24 hours using the task thresholds.\n"
                "- Treat broader warming/cooling narrative, older history, and text context only as supporting evidence for that window-to-window comparison.\n"
                "- Use challenger forecasts only to test the same first-future-day mean comparison; do not let a generic challenger path replace the exact observed reference-window computation.\n"
            )
        if (
            task_type == "classification"
            and cls._scope_keys_match_suffix(scope_keys, "_trend")
            and not cls._scope_keys_match_suffix(scope_keys, "_trend_past")
        ):
            return (
                "- This task asks for the trend of the target future window relative to the required observed reference window, not the overall slope of the full historical input.\n"
                "- Base the label on the task-defined future-vs-reference comparison before using broader history or context as supporting evidence.\n"
            )
        if cls._scope_keys_match_suffix(scope_keys, "_trend_past"):
            return (
                "- This task asks for the trend of the observed input window itself, not a forecast of what happens next.\n"
                "- If text context is useful, use it only to interpret the observed past pattern, not as a future signal.\n"
            )
        if task_type == "indicator" and cls._scope_keys_match_suffix(scope_keys, "_indicator_macd"):
            return (
                "- The target object is the future MACD sequence, not the raw future price series.\n"
                "- Compute that target over the future prediction window itself using MACD = EMA(12) - EMA(26), rather than returning the future price path or another derived series.\n"
                "- Intermediate reasoning may use price behavior, but the final task answer must be MACD values for the required horizon.\n"
            )
        if task_type == "indicator" and cls._scope_keys_match_suffix(scope_keys, "_indicator_upper_bb"):
            return (
                "- The target object is the future upper Bollinger Band sequence, not the raw future price series and not another indicator.\n"
                "- Compute that target over the future prediction window itself as upper_bb = rolling_mean(window=20, min_periods=1) + 2 * rolling_std(window=20, min_periods=1), rather than over a concatenated history+future series.\n"
                "- Intermediate reasoning may use price behavior or volatility, but the final task answer must be upper Bollinger Band values for the required horizon.\n"
            )
        if task_type == "indicator":
            return (
                "- First decide which future price path is most plausible for this sample, then derive the indicator from that path.\n"
                "- Do not let sentiment, regime names, or generic market narrative substitute for the indicator sequence itself.\n"
                "- When candidate price paths disagree, compare which one better respects the latest regime and boundary behavior before converting it into the indicator output.\n"
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
        scope_keys: list[str] | None = None,
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
                    scope_keys=scope_keys,
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
        disallowed_summary_fields = [
            key for key in (
                "winning_candidate_name",
                "winning_tool_chain",
                "winning_prediction",
                "compared_candidates",
                "selection_basis",
            )
            if key in answer
        ]
        if disallowed_summary_fields:
            issues.append(
                "do not put winner bookkeeping fields inside `answer`: "
                + ", ".join(f"`{key}`" for key in disallowed_summary_fields)
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
    def _invalid_json_self_correction_message(
        parse_thought: str,
        *,
        allow_tool_call: bool = True,
        allow_finish: bool = False,
        spawn_available: bool = False,
        ordinary_tool_names: list[str] | None = None,
        evaluate_tool_names: list[str] | None = None,
    ) -> str:
        base = (
            "Self-correction: your reply did not produce exactly one valid decision action. "
            "Emit exactly one action for this turn: either one tool call or one final answer."
        )
        lowered = str(parse_thought or "").strip().lower()
        allowed_ordinary = [
            str(name).strip() for name in (ordinary_tool_names or []) if str(name).strip()
        ]
        allowed_eval = [
            str(name).strip() for name in (evaluate_tool_names or []) if str(name).strip()
        ]
        if lowered == "invalid_decision_wrong_tool_family":
            if not allow_tool_call and allow_finish:
                return (
                    base
                    + " This is a finish-only turn. Do not call any tools. "
                    + "Return exactly one `done=true` final-answer JSON using the already gathered evidence."
                )
            if spawn_available and not allowed_ordinary and not allowed_eval and not allow_finish:
                return (
                    base
                    + " This is a spawn-only turn. The only valid tool for this turn is "
                    + "`spawn_subagent`. Do not call ordinary tools, do not call `evaluate_*`, "
                    + "and do not set `done=true`."
                )
            if allowed_eval and not allowed_ordinary and not spawn_available and not allow_finish:
                joined_eval = ", ".join(f"`{name}`" for name in allowed_eval)
                return (
                    base
                    + " This is an evaluation-only turn. Call exactly one GT evaluation tool from the available list: "
                    + joined_eval
                    + ". Do not call ordinary tools, do not branch, and do not set `done=true`."
                )
            if allow_tool_call and allowed_ordinary and not allowed_eval and not spawn_available:
                joined_ordinary = ", ".join(f"`{name}`" for name in allowed_ordinary[:8])
                return (
                    base
                    + " The previous tool was outside the allowed tool family for this turn. "
                    + "Call exactly one ordinary tool from the available list, such as "
                    + joined_ordinary
                    + "."
                )
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
        scope_keys: list[str] | None = None,
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
        if task_type == "classification":
            return shared + (
                "- Return a single label string or a minimal JSON object containing that label.\n"
                + ExecutorAgent._build_scoped_classification_output_contract_section(
                    scope_keys=scope_keys,
                    include_example=True,
                )
            )
        if task_type == "correlation":
            return shared + (
                "- Return a single label string or a minimal JSON object containing that label.\n"
                '- Example: "Moderate Positive Correlation"\n'
            )
        if task_type == "qa":
            if ExecutorAgent._scope_keys_match_suffix(scope_keys, "_mcqa"):
                return shared + (
                    "- Return exactly one uppercase option letter: A, B, C, or D.\n"
                    "- A copied news paragraph, summary, or forecast is not an option choice. "
                    "Use llm_reasoning_qa to resolve the options from that evidence.\n"
                )
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

    @staticmethod
    def _tool_error_exception(error: str | None) -> RuntimeError:
        return RuntimeError(str(error or "tool execution failed"))

    @classmethod
    def _tool_deferred_retry_key(cls, tool_name: str, tool_args: dict[str, Any]) -> str | None:
        work_item_id = current_work_item_id()
        if not should_defer_retry_to_scheduler() or not work_item_id:
            return None
        payload = {
            "tool_name": str(tool_name or ""),
            "tool_args": tool_args or {},
        }
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return f"{work_item_id}:{signature}"

    @classmethod
    def _tool_deferred_retry_load_state(cls, retry_key: str | None) -> dict[str, int]:
        if not retry_key:
            return {}
        state = _TOOL_DEFERRED_RETRY_STATE.get(retry_key)
        if state is None:
            state = {}
            _TOOL_DEFERRED_RETRY_STATE[retry_key] = state
        return state

    @classmethod
    def _tool_deferred_retry_clear_state(cls, retry_key: str | None) -> None:
        if retry_key:
            _TOOL_DEFERRED_RETRY_STATE.pop(retry_key, None)

    @classmethod
    def _is_tool_supplier_pool_retry_error(cls, error: str | None) -> bool:
        if not str(error or "").strip():
            return False
        return _is_supplier_pool_retry(cls._tool_error_exception(error))

    @classmethod
    def _is_tool_retryable_error(cls, error: str | None) -> bool:
        normalized = str(error or "").strip().lower()
        if not normalized:
            return False
        if "llm returned empty response" in normalized:
            return True
        return _is_retryable(cls._tool_error_exception(error))

    @staticmethod
    def _tool_error_kind(result: ToolResult | None) -> str:
        if result is None or not isinstance(result.metadata, dict):
            return ""
        return str(result.metadata.get("error_kind") or "").strip()

    @staticmethod
    def _log_tool_result(tool_name: str, tool_args: dict[str, Any], result: ToolResult) -> None:
        status = "PASS" if result.success else f"FAIL | {result.error}"
        logger.info(f"[Executor] {tool_name}: {status}")
        trace.tool_exec("executor", tool_name, tool_args, result)

    async def _execute_tool_once(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        data: Any,
        text_data: Any,
        images: Any,
    ) -> ToolResult | None:
        """Execute one tool attempt without outer retry orchestration."""
        tool = None
        try:
            tool = get_tool_instance(tool_name)
        except KeyError:
            tool = None

        force_inprocess = (
            tool_name in self._FORCE_INPROCESS
            or (tool is not None and tool.category.value == "text")
        )
        use_mcp = (
            self._mcp_manager
            and self._mcp_manager.get_tool(tool_name)
            and not force_inprocess
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
            return ToolResult(
                success=r["success"],
                data=r["data"],
                error=r.get("error"),
            )

        # ── In-process fallback ──
        if tool is None:
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
            return result
        except ExecutionContractError:
            raise
        except Exception as e:
            logger.error(f"[Executor] Tool '{tool_name}' crashed: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        data: Any,
        text_data: Any,
        images: Any,
    ) -> ToolResult | None:
        """Execute one tool with executor-level retry for transient failures.

        New definition:
            Tool execution follows the same high-level contract as other LLM
            call sites:
            - supplier-pool pressure may defer without consuming retry budget
            - retryable failures get bounded retries
            - after retries are exhausted, the failure becomes terminal
            - non-retryable failures are terminal immediately
        """
        retry_key = self._tool_deferred_retry_key(tool_name, tool_args)
        deferred_state = self._tool_deferred_retry_load_state(retry_key)
        supplier_retry_count = 0
        retryable_retry_count = int(deferred_state.get("retryable_retry_count", 0) or 0)
        non_retryable_retry_count = 0

        while True:
            result = await self._execute_tool_once(
                tool_name=tool_name,
                tool_args=tool_args,
                data=data,
                text_data=text_data,
                images=images,
            )
            if result is None:
                self._tool_deferred_retry_clear_state(retry_key)
                return None
            if result.success:
                self._tool_deferred_retry_clear_state(retry_key)
                self._log_tool_result(tool_name, tool_args, result)
                return result

            if self._tool_error_kind(result) == "invalid_args":
                self._tool_deferred_retry_clear_state(retry_key)
                logger.warning(
                    f"[Executor] {tool_name} failed with invalid_args: {result.error}"
                )
                self._log_tool_result(tool_name, tool_args, result)
                return result

            error = str(result.error or "").strip()
            if self._is_tool_supplier_pool_retry_error(error):
                supplier_retry_count += 1
                delay = retry_delay_seconds_for_attempt(supplier_retry_count - 1)
                if should_defer_retry_to_scheduler():
                    raise RetryLater(
                        delay_s=delay,
                        reason=error or f"{tool_name} supplier pool busy",
                        layer="tool_supplier_pool",
                    )
                await asyncio.sleep(delay)
                continue

            if self._is_tool_retryable_error(error):
                retryable_retry_count += 1
                if retryable_retry_count < _MAX_TOOL_RETRY_ATTEMPTS:
                    delay = retry_delay_seconds_for_attempt(retryable_retry_count - 1)
                    if should_defer_retry_to_scheduler():
                        self._tool_deferred_retry_load_state(retry_key)["retryable_retry_count"] = (
                            retryable_retry_count
                        )
                        raise RetryLater(
                            delay_s=delay,
                            reason=error or f"{tool_name} retryable failure",
                            layer="tool",
                        )
                    logger.warning(
                        f"[Executor] {tool_name} attempt {retryable_retry_count} failed "
                        f"(retryable): {error}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                self._tool_deferred_retry_clear_state(retry_key)
                logger.warning(
                    f"[Executor] {tool_name} retry budget exhausted after "
                    f"{retryable_retry_count} attempts: {error}"
                )
                self._log_tool_result(tool_name, tool_args, result)
                return result

            non_retryable_retry_count += 1
            self._tool_deferred_retry_clear_state(retry_key)
            logger.warning(
                f"[Executor] {tool_name} failed with terminal error: {error}"
            )
            self._log_tool_result(tool_name, tool_args, result)
            return result

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
        domain: str = "",
        scope_keys: list[str] | None = None,
    ) -> bool:
        valid, _ = ExecutorAgent._validate_final_answer(
            task_type,
            answer,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
            domain=domain,
            scope_keys=scope_keys,
        )
        return valid

    @staticmethod
    def _validate_final_answer(
        task_type: str,
        answer: Any,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
        domain: str = "",
        scope_keys: list[str] | None = None,
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
                arr = np.asarray(candidate, dtype=float)
            except (ValueError, TypeError):
                return False, f"forecast answer cannot be converted to numeric array from {type(candidate).__name__}"
            if arr.size == 0:
                return False, "forecast numeric array is empty"
            if arr.ndim not in (1, 2):
                return False, f"forecast array must have shape (time,) or (time, channels), got {arr.shape}"
            if horizon is not None and arr.shape[0] != horizon:
                return False, f"forecast time length mismatch: got {arr.shape[0]}, need exactly {horizon}"
            if not np.isfinite(arr).all():
                return False, "forecast array must contain only finite numeric values"
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
            candidate = ExecutorAgent._extract_task_answer_candidate(task_type, answer)
            if isinstance(candidate, str):
                if not candidate.strip():
                    return False, "classification/correlation answer is empty string"
                if not ExecutorAgent._is_label_like_text(candidate):
                    return False, "classification/correlation answer must be a short label-like string"
                if task_type == "correlation" and ExecutorAgent._scope_keys_match_suffix(scope_keys, "_correlation"):
                    labels = {
                        "Strong Positive Correlation", "Moderate Positive Correlation", "No Correlation",
                        "Moderate Negative Correlation", "Strong Negative Correlation",
                    }
                    if candidate.strip() not in labels:
                        return False, "correlation answer must be one of the five task labels; trend/sentiment labels are not answers"
                if task_type == "classification":
                    canonical = ExecutorAgent._canonicalize_scoped_classification_answer(
                        candidate,
                        domain=domain,
                        scope_keys=scope_keys,
                    )
                    if canonical is None:
                        return False, "classification answer is outside the task label space"
                return True, ""
            return False, "classification/correlation answer is neither string nor dict"

        if task_type == "qa":
            if ExecutorAgent._scope_keys_match_suffix(scope_keys, "_mcqa"):
                candidate = ExecutorAgent._extract_task_answer_candidate(task_type, answer)
                valid = isinstance(candidate, str) and candidate.strip() in {"A", "B", "C", "D"}
                return valid, "" if valid else "MCQA answer must be a single uppercase option A, B, C, or D"
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
                arr = np.asarray(candidate, dtype=float)
            except (ValueError, TypeError):
                return None
            if arr.size == 0 or arr.ndim not in (1, 2):
                return None
            if horizon is not None and arr.shape[0] != horizon:
                return None
            if not np.isfinite(arr).all():
                return None
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
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
                left_arr = np.asarray(left_norm, dtype=float)
                right_arr = np.asarray(right_norm, dtype=float)
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
                output = entry.get("output") or []
                if isinstance(output, dict):
                    output = output.get("results") or []
                for item in output:
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
        scope_keys: list[str] | None = None,
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
            scope_keys=scope_keys,
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
        scope_keys: list[str] | None = None,
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
                scope_keys=scope_keys,
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
            "description": (
                "Must contain the five labeled parts in this order: "
                "`Reason:` then `Check:` then `Self-check:` then `Update:` then `Action:`."
            ),
            "pattern": r"^Reason:[\s\S]*Check:[\s\S]*Self-check:[\s\S]*Update:[\s\S]*Action:[\s\S]*$",
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
                "audit": {
                    "type": "object",
                    "description": (
                        "Controller-populated audit metadata showing which memory/prompt "
                        "sections were injected. The agent may omit this field."
                    ),
                    "properties": {
                        "memory_injected": {
                            "type": "object",
                            "properties": {
                                "soul_text": {"type": "string"},
                                "memory_text": {"type": "string"},
                                "system_memory_text": {"type": "string"},
                                "skills_text": {"type": "string"},
                                "skills_decision_text": {"type": "string"},
                                "decision_skills_text": {"type": "string"},
                                "tool_notes_text": {"type": "string"},
                                "tool_notes_section": {"type": "string"},
                                "selected_packet_tool_notes": {"type": "string"},
                                "sample_packet_text": {"type": "string"},
                                "matched_skills_text": {"type": "string"},
                                "matched_skill_titles": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "matched_skill_rule_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "decision_skill_titles": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "primary_path_tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "challenger_path_tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "challenger_requirement": {"type": "string"},
                                "tool_prep_tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "tool_note_keys": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "tool_note_count": {"type": "integer"},
                            },
                            "additionalProperties": False,
                        },
                        "prompt_sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "chars": {"type": "integer"},
                                    "nonempty": {"type": "boolean"},
                                },
                                "required": ["name", "chars", "nonempty"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
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
        return sanitize_learning_evidence_text(cleaned, drop_imperatives=True)

    @staticmethod
    def _normalize_control_context(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        normalized = {
            "completion_notes": str(payload.get("completion_notes") or "").strip(),
        }
        audit = ExecutorAgent._normalize_control_context_audit(payload.get("audit"))
        if audit:
            normalized["audit"] = audit
        return normalized

    @staticmethod
    def _coerce_reasoning_text(reasoning: Any) -> str:
        if isinstance(reasoning, str):
            return reasoning.strip()
        if isinstance(reasoning, (list, tuple)):
            lines: list[str] = []
            for idx, item in enumerate(reasoning, start=1):
                text = str(item or "").strip()
                if not text:
                    continue
                if re.match(r"^\d+\.\s+", text):
                    lines.append(text)
                else:
                    lines.append(f"{idx}. {text}")
            return "\n".join(lines).strip()
        return str(reasoning or "").strip()

    @classmethod
    def _coerce_structured_thought_text(cls, thought: Any) -> str:
        if isinstance(thought, str):
            return thought.strip()
        if isinstance(thought, dict):
            reason = ""
            check = ""
            self_check = ""
            update = ""
            action = ""
            for key, value in thought.items():
                key_text = str(key or "").strip().lower()
                if key_text == "reason":
                    reason = cls._coerce_reasoning_text(value)
                elif key_text == "check":
                    check = cls._coerce_reasoning_text(value)
                elif key_text in {"self-check", "self_check", "selfcheck"}:
                    self_check = cls._coerce_reasoning_text(value)
                elif key_text == "update":
                    update = cls._coerce_reasoning_text(value)
                elif key_text == "action":
                    action = cls._coerce_reasoning_text(value)
            if reason and check and self_check and update and action:
                return (
                    f"Reason: {reason} "
                    f"Check: {check} "
                    f"Self-check: {self_check} "
                    f"Update: {update} "
                    f"Action: {action}"
                ).strip()
            partial_parts: list[str] = []
            if reason:
                partial_parts.append(f"Reason: {reason}")
            if check:
                partial_parts.append(f"Check: {check}")
            if self_check:
                partial_parts.append(f"Self-check: {self_check}")
            if update:
                partial_parts.append(f"Update: {update}")
            if action:
                partial_parts.append(f"Action: {action}")
            if partial_parts:
                return " ".join(partial_parts).strip()
        return str(thought or "").strip()

    @staticmethod
    def _complete_legacy_thought_skeleton(thought: str) -> str:
        text = str(thought or "").strip()
        if not text:
            return text
        has_reason = "Reason:" in text
        has_check = "Check:" in text
        has_self_check = "Self-check:" in text
        has_update = "Update:" in text
        has_action = "Action:" in text
        if not (has_reason and has_action):
            return text
        if has_check and has_self_check and has_update:
            return text
        action_part = text.split("Action:", 1)[1].strip() if has_action else ""
        reason_part = text.split("Reason:", 1)[1].strip() if has_reason else text
        if has_action and "Action:" in reason_part:
            reason_part = reason_part.split("Action:", 1)[0].strip()
        completed_parts = [f"Reason: {reason_part}".strip()]
        if not has_check:
            completed_parts.append(
                "Check: The current path must satisfy the output contract using only evidence already present."
            )
        if not has_self_check:
            completed_parts.append(
                "Self-check: No additional tool result is required if the current payload is already grounded and complete."
            )
        if not has_update:
            completed_parts.append(
                "Update: Preserve the current evidence-grounded candidate unless a concrete validation failure requires correction."
            )
        completed_parts.append(f"Action: {action_part}".strip() if action_part else "Action:")
        return " ".join(part for part in completed_parts if part).strip()

    @staticmethod
    def _normalize_control_context_audit(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        normalized: dict[str, Any] = {}

        memory_payload = payload.get("memory_injected")
        if isinstance(memory_payload, dict):
            tool_note_keys = [
                str(item).strip()
                for item in memory_payload.get("tool_note_keys") or []
                if str(item).strip()
            ]
            active_rule_ids = [
                str(item).strip()
                for item in memory_payload.get("active_rule_ids") or []
                if str(item).strip()
            ]
            matched_skill_titles = [
                str(item).strip()
                for item in memory_payload.get("matched_skill_titles") or []
                if str(item).strip()
            ]
            matched_skill_rule_ids = [
                str(item).strip()
                for item in memory_payload.get("matched_skill_rule_ids") or []
                if str(item).strip()
            ]
            decision_skill_titles = [
                str(item).strip()
                for item in memory_payload.get("decision_skill_titles") or []
                if str(item).strip()
            ]
            primary_path_tools = [
                str(item).strip()
                for item in memory_payload.get("primary_path_tools") or []
                if str(item).strip()
            ]
            challenger_path_tools = [
                str(item).strip()
                for item in memory_payload.get("challenger_path_tools") or []
                if str(item).strip()
            ]
            challenger_requirement = ExecutorAgent._normalize_challenger_requirement(
                memory_payload.get("challenger_requirement")
            )
            tool_prep_tools = [
                str(item).strip()
                for item in memory_payload.get("tool_prep_tools") or []
                if str(item).strip()
            ]
            prioritized_tool_names = [
                str(item).strip()
                for item in memory_payload.get("prioritized_tool_names") or []
                if str(item).strip()
            ]
            matched_skill_selected_blocks = (
                memory_payload.get("matched_skill_selected_blocks")
                if isinstance(memory_payload.get("matched_skill_selected_blocks"), list)
                else []
            )
            suppressed_conflict_rule_ids = [
                str(item).strip()
                for item in memory_payload.get("active_suppressed_conflict_rule_ids") or []
                if str(item).strip()
            ]
            active_experience_selected_entries = (
                memory_payload.get("active_experience_selected_entries")
                if isinstance(memory_payload.get("active_experience_selected_entries"), list)
                else []
            )
            memory_injected = {
                "soul_text": str(memory_payload.get("soul_text") or "").strip(),
                "memory_text": str(memory_payload.get("memory_text") or "").strip(),
                "raw_memory_text": str(memory_payload.get("raw_memory_text") or "").strip(),
                "system_memory_text": str(memory_payload.get("system_memory_text") or "").strip(),
                "skills_text": str(memory_payload.get("skills_text") or "").strip(),
                "skills_decision_text": str(memory_payload.get("skills_decision_text") or "").strip(),
                "decision_skills_text": str(memory_payload.get("decision_skills_text") or "").strip(),
                "tool_notes_text": str(memory_payload.get("tool_notes_text") or "").strip(),
                "tool_notes_section": str(memory_payload.get("tool_notes_section") or "").strip(),
                "selected_packet_tool_notes": str(
                    memory_payload.get("selected_packet_tool_notes") or ""
                ).strip(),
                "sample_packet_text": str(memory_payload.get("sample_packet_text") or "").strip(),
                "matched_skills_text": str(memory_payload.get("matched_skills_text") or "").strip(),
                "matched_skill_titles": matched_skill_titles,
                "matched_skill_count": max(
                    len(matched_skill_titles),
                    int(memory_payload.get("matched_skill_count") or 0),
                ),
                "matched_skill_scored_blocks": (
                    memory_payload.get("matched_skill_scored_blocks")
                    if isinstance(memory_payload.get("matched_skill_scored_blocks"), list)
                    else []
                ),
                "matched_skill_selected_blocks": matched_skill_selected_blocks,
                "matched_skill_rule_ids": matched_skill_rule_ids,
                "decision_skill_titles": decision_skill_titles,
                "decision_skill_count": max(
                    len(decision_skill_titles),
                    int(memory_payload.get("decision_skill_count") or 0),
                ),
                "decision_skill_scored_blocks": (
                    memory_payload.get("decision_skill_scored_blocks")
                    if isinstance(memory_payload.get("decision_skill_scored_blocks"), list)
                    else []
                ),
                "decision_skill_selected_blocks": (
                    memory_payload.get("decision_skill_selected_blocks")
                    if isinstance(memory_payload.get("decision_skill_selected_blocks"), list)
                    else []
                ),
                "primary_path_tools": primary_path_tools,
                "challenger_path_tools": challenger_path_tools,
                "challenger_requirement": challenger_requirement,
                "tool_prep_tools": tool_prep_tools,
                "tool_note_keys": tool_note_keys,
                "tool_note_count": max(
                    len(tool_note_keys),
                    int(memory_payload.get("tool_note_count") or 0),
                ),
                "active_experience_text": str(memory_payload.get("active_experience_text") or "").strip(),
                "active_rule_ids": active_rule_ids,
                "active_rule_count": max(
                    len(active_rule_ids),
                    int(memory_payload.get("active_rule_count") or 0),
                ),
                "active_suppressed_conflict_rule_ids": suppressed_conflict_rule_ids,
                "memory_entry_count": max(0, int(memory_payload.get("memory_entry_count") or 0)),
                "active_experience_scored_rules": (
                    memory_payload.get("active_experience_scored_rules")
                    if isinstance(memory_payload.get("active_experience_scored_rules"), list)
                    else []
                ),
                "active_experience_selected_entries": active_experience_selected_entries,
                "experience_selector_guidance": str(
                    memory_payload.get("experience_selector_guidance") or ""
                ).strip(),
                "experience_selector_audit": (
                    memory_payload.get("experience_selector_audit")
                    if isinstance(memory_payload.get("experience_selector_audit"), dict)
                    else {}
                ),
                "prioritized_tool_names": prioritized_tool_names,
            }
            if any(
                memory_injected[key]
                for key in (
                    "soul_text",
                    "memory_text",
                    "raw_memory_text",
                    "system_memory_text",
                    "skills_text",
                    "skills_decision_text",
                    "decision_skills_text",
                    "tool_notes_text",
                    "tool_notes_section",
                    "selected_packet_tool_notes",
                    "sample_packet_text",
                    "matched_skills_text",
                    "matched_skill_titles",
                    "matched_skill_count",
                    "matched_skill_scored_blocks",
                    "matched_skill_selected_blocks",
                    "matched_skill_rule_ids",
                    "decision_skill_titles",
                    "decision_skill_count",
                    "decision_skill_scored_blocks",
                    "decision_skill_selected_blocks",
                    "primary_path_tools",
                    "challenger_path_tools",
                    "tool_prep_tools",
                    "tool_note_keys",
                    "tool_note_count",
                    "active_experience_text",
                    "active_rule_ids",
                    "active_rule_count",
                    "active_suppressed_conflict_rule_ids",
                    "memory_entry_count",
                    "active_experience_scored_rules",
                    "active_experience_selected_entries",
                    "experience_selector_guidance",
                    "experience_selector_audit",
                    "prioritized_tool_names",
                )
            ):
                normalized["memory_injected"] = memory_injected

        section_payload = payload.get("prompt_sections")
        if isinstance(section_payload, list):
            prompt_sections: list[dict[str, Any]] = []
            for item in section_payload:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                chars = max(0, int(item.get("chars") or 0))
                nonempty = bool(item.get("nonempty")) if "nonempty" in item else chars > 0
                prompt_sections.append({
                    "name": name,
                    "chars": chars,
                    "nonempty": nonempty,
                })
            if prompt_sections:
                normalized["prompt_sections"] = prompt_sections

        return normalized

    @classmethod
    def _merge_control_context(
        cls,
        base: Any,
        override: Any,
    ) -> dict[str, Any]:
        base_norm = cls._normalize_control_context(base)
        override_norm = cls._normalize_control_context(override)
        merged = {
            "completion_notes": (
                str(override_norm.get("completion_notes") or "").strip()
                or str(base_norm.get("completion_notes") or "").strip()
            ),
        }
        audit = dict(base_norm.get("audit") or {})
        override_audit = override_norm.get("audit") or {}
        if isinstance(override_audit, dict):
            audit.update(override_audit)
        if audit:
            merged["audit"] = audit
        return merged

    @classmethod
    def _build_control_context_audit(
        cls,
        *,
        memory_injected: dict[str, Any] | None = None,
        prompt_sections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        audit: dict[str, Any] = {}
        normalized_memory = cls._normalize_control_context_audit(
            {"memory_injected": memory_injected or {}}
        )
        if "memory_injected" in normalized_memory:
            audit["memory_injected"] = normalized_memory["memory_injected"]
        normalized_sections = cls._normalize_control_context_audit(
            {"prompt_sections": prompt_sections or []}
        )
        if "prompt_sections" in normalized_sections:
            audit["prompt_sections"] = normalized_sections["prompt_sections"]
        return audit

    @classmethod
    def _build_prompt_section_audit(
        cls,
        *,
        name: str,
        text: Any,
    ) -> dict[str, Any]:
        content = str(text or "")
        return {
            "name": str(name),
            "chars": len(content),
            "nonempty": bool(content.strip()),
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
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[str]:
        candidate_labels = cls._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        substantive_tools = cls._available_substantive_tool_names(execution_log)
        evidence_text = " ".join([*public_reasoning, evidence_summary]).strip()
        path_labels = [item for item in tool_chain if "->" in item]
        plain_tools = [item for item in tool_chain if "->" not in item]
        if path_labels:
            return path_labels

        primary_chain = cls._reasoning_substantive_tool_chain(execution_log)
        if len(primary_chain) >= 2 and all(tool in plain_tools for tool in primary_chain):
            return list(primary_chain)

        mentioned_tools = cls._mentioned_substantive_tools(evidence_text, substantive_tools)
        inferred_candidate_labels: list[str] = []
        if mentioned_tools and candidate_labels:
            plain_tool_set = set(plain_tools)
            for label in candidate_labels:
                segments = [
                    str(segment).strip()
                    for segment in cls._normalize_label_text(label).split(" -> ")
                    if str(segment).strip()
                ]
                if not segments:
                    continue
                if not set(segments).issubset(set(mentioned_tools)):
                    continue
                if plain_tool_set and not plain_tool_set.issubset(set(segments)):
                    continue
                inferred_candidate_labels.append(" -> ".join(segments))
        if inferred_candidate_labels:
            inferred_candidate_labels.sort(
                key=lambda label: len([segment for segment in label.split(" -> ") if segment]),
                reverse=True,
            )
            if (
                len(inferred_candidate_labels) == 1
                or len([
                    label for label in inferred_candidate_labels
                    if len(label.split(" -> ")) == len(inferred_candidate_labels[0].split(" -> "))
                ]) == 1
            ):
                return [inferred_candidate_labels[0]]

        mentioned_labels = cls._mentioned_candidate_path_labels(evidence_text, candidate_labels)
        if mentioned_labels:
            mentioned_labels = [
                cls._normalize_label_text(label)
                for label in mentioned_labels
                if cls._normalize_label_text(label)
            ]
            mentioned_labels.sort(
                key=lambda label: len([segment for segment in label.split(" -> ") if segment]),
                reverse=True,
            )
            if (
                len(mentioned_labels) == 1
                or len([
                    label for label in mentioned_labels
                    if len(label.split(" -> ")) == len(mentioned_labels[0].split(" -> "))
                ]) == 1
            ):
                return [mentioned_labels[0]]

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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> str:
        candidate_labels = cls._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
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
        candidate_labels = cls._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        selected_label = cls._selected_candidate_label_from_context(
            execution_context,
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
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
                    sub_execution_log = result.get("execution_log") or []
                    answer = cls._unwrap_task_terminal_answer(
                        task_type,
                        result.get("answer", result.get("prediction")),
                    )
                    observed_answer = cls._observed_task_answer_from_execution_log(
                        task_type=task_type,
                        execution_log=sub_execution_log,
                        horizon=horizon,
                        indicator_output_keys=indicator_output_keys,
                    )
                    if task_type == "indicator":
                        answer = cls._canonicalize_indicator_answer(
                            answer,
                            indicator_output_keys=indicator_output_keys,
                        )
                        observed_answer = cls._canonicalize_indicator_answer(
                            observed_answer,
                            indicator_output_keys=indicator_output_keys,
                        )
                    if (
                        observed_answer is not None
                        and not cls._is_task_answer_backed_by_tool_output(
                            task_type,
                            answer,
                            sub_execution_log,
                            horizon=horizon,
                            indicator_output_keys=indicator_output_keys,
                        )
                    ):
                        answer = observed_answer
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
        return all(
            label in thought
            for label in ("Reason:", "Check:", "Self-check:", "Update:", "Action:")
        )

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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[list[str]]:
        chains: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for record in cls._reasoning_candidate_records(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
            chain = [
                str(tool).strip()
                for tool in record.get("tool_chain") or []
                if str(tool).strip()
            ]
            if not chain:
                continue
            key = tuple(chain)
            if key in seen:
                continue
            seen.add(key)
            chains.append(chain)
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
            "moirai1_1_r_base_forecast": ["Moirai 1.1 R-large"],
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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[str]:
        seen: set[str] = set()
        hints: list[str] = []
        for chain in cls._reasoning_candidate_tool_chains(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[str]:
        return [
            str(record.get("label") or "").strip()
            for record in cls._reasoning_candidate_records(
                execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
            if str(record.get("label") or "").strip()
        ]

    async def _call_stepwise_reasoning_verdict(
        self,
        reasoning: Any,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            return await self._stepwise_reasoning_verdict(
                reasoning,
                execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return await self._stepwise_reasoning_verdict(reasoning, execution_log)

    async def _call_execution_context_verdict(
        self,
        execution_context: dict[str, Any],
        execution_log: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            return await self._execution_context_verdict(
                execution_context,
                execution_log,
                task_type=task_type,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return await self._execution_context_verdict(execution_context, execution_log)

    async def _stepwise_reasoning_verdict(
        self,
        reasoning: Any,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        text = str(reasoning or "").strip()
        candidate_labels = self._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        substantive_chain = self._reasoning_substantive_tool_chain(execution_log)
        explicit_name_hints = self._reasoning_explicit_name_hints(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
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
        if self._has_stepwise_reasoning(
            text,
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
            return {"passed": True, "missing": [], "explanation": ""}

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
            error_context="stepwise reasoning judge",
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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        candidate_labels = self._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
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
        if self._empty_execution_context_allowed_for_failed_learning_summary(
            execution_context,
            execution_log,
        ):
            return {"passed": True, "missing": [], "explanation": ""}
        if self._has_reusable_execution_context(
            execution_context,
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        ):
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
            error_context="execution_context judge",
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

    @staticmethod
    def _empty_execution_context_allowed_for_failed_learning_summary(
        execution_context: dict[str, Any],
        execution_log: list[dict[str, Any]],
    ) -> bool:
        """Allow empty execution_context only for no-validated-winner summaries.

        New definition:
            In training failure samples where all evaluated candidates are
            explicitly incorrect, `execution_context` should stay empty rather
            than fabricating a reusable winner-like evidence card. This keeps
            executor-time validation aligned with runner-time normalization.
        """

        tool_chain = [
            str(item).strip()
            for item in (execution_context.get("tool_chain") or [])
            if str(item).strip()
        ]
        public_reasoning = [
            str(item).strip()
            for item in (execution_context.get("public_reasoning") or [])
            if str(item).strip()
        ]
        evidence_summary = str(execution_context.get("evidence_summary") or "").strip()
        if tool_chain or public_reasoning or evidence_summary:
            return False

        saw_binary_eval = False
        saw_correct = False
        for entry in execution_log:
            if not isinstance(entry, dict):
                continue
            tool = str(entry.get("tool") or "").strip()
            if tool not in {
                "evaluate_against_gt",
                "evaluate_batch_against_gt",
                "batch_evaluate_against_gt",
            }:
                continue
            observation = entry.get("observation")
            if not isinstance(observation, dict):
                continue
            results = observation.get("results")
            if isinstance(results, list):
                for item in results:
                    metrics = item.get("metrics") if isinstance(item, dict) else None
                    if not isinstance(metrics, dict):
                        continue
                    correct = metrics.get("correct")
                    if isinstance(correct, bool):
                        saw_binary_eval = True
                        saw_correct = saw_correct or correct
            metrics = observation.get("metrics")
            if isinstance(metrics, dict):
                correct = metrics.get("correct")
                if isinstance(correct, bool):
                    saw_binary_eval = True
                    saw_correct = saw_correct or correct

        return saw_binary_eval and not saw_correct

    async def _final_payload_reusability_verdict(
        self,
        *,
        mode: Any,
        reasoning: Any,
        execution_context: dict[str, Any],
        execution_log: list[dict[str, Any]],
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self._should_enforce_stepwise_reasoning(mode):
            return {
                "passed": True,
                "execution_context_verdict": {"passed": True, "missing": [], "explanation": ""},
                "reasoning_verdict": {"passed": True, "missing": [], "explanation": ""},
            }

        execution_context_verdict = await self._call_execution_context_verdict(
            execution_context,
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
        if not execution_context_verdict.get("passed", False):
            return {
                "passed": False,
                "execution_context_verdict": execution_context_verdict,
                "reasoning_verdict": {"passed": True, "missing": [], "explanation": ""},
            }

        reasoning_verdict = await self._call_stepwise_reasoning_verdict(
            reasoning,
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
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
        error_context: str,
    ) -> dict[str, Any]:
        last_failure = "invalid JSON"
        for attempt in range(MAX_JSON_JUDGE_ATTEMPTS):
            response = await self.llm.complete(
                messages,
                response_format="json",
                temperature=0.0,
            )
            parsed = self._json_dict_from_response(response, required_keys={"passed"})
            if isinstance(parsed, dict) and "passed" in parsed:
                return parsed
            if self._json_judge_response_is_empty(response):
                last_failure = "empty response"
            else:
                last_failure = "invalid JSON"
            if attempt + 1 >= MAX_JSON_JUDGE_ATTEMPTS:
                break
        raise RuntimeError(f"{error_context} returned {last_failure}")

    @staticmethod
    def _json_judge_response_is_empty(response: Any) -> bool:
        if response is None:
            return True
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict) and parsed:
            return False
        if isinstance(parsed, list) and parsed:
            return False
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return False
        return True

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
    def _json_dict_from_response(
        response: Any,
        *,
        required_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        required = {str(key).strip() for key in (required_keys or set()) if str(key).strip()}
        raw_parsed = getattr(response, "parsed", None)
        parsed = ExecutorAgent._find_nested_dict_with_keys(raw_parsed, required)
        if not parsed and isinstance(getattr(response, "text", None), str) and response.text.strip():
            try:
                decoded = json.loads(response.text)
            except json.JSONDecodeError:
                try:
                    decoded, _ = json.JSONDecoder().raw_decode(response.text.lstrip())
                except Exception:
                    decoded = {}
            parsed = ExecutorAgent._find_nested_dict_with_keys(decoded, required)
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _find_nested_dict_with_keys(payload: Any, required_keys: set[str] | None = None) -> dict[str, Any]:
        required = {str(key).strip() for key in (required_keys or set()) if str(key).strip()}
        if isinstance(payload, dict):
            if not required or required.issubset({str(key).strip() for key in payload.keys()}):
                return payload
            for value in payload.values():
                nested = ExecutorAgent._find_nested_dict_with_keys(value, required)
                if nested:
                    return nested
            return {}
        if isinstance(payload, list):
            for item in payload:
                nested = ExecutorAgent._find_nested_dict_with_keys(item, required)
                if nested:
                    return nested
        return {}

    @classmethod
    def _terminal_payload_repair_messages(
        cls,
        execution_log: list[dict[str, Any]],
        terminal_reuse_verdict: dict[str, Any] | None,
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
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
                    task_type=task_type,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
                )
            )
        elif not reasoning_verdict.get("passed", False):
            messages.append(
                cls._stepwise_reasoning_correction_message(
                    execution_log,
                    verdict=reasoning_verdict,
                    task_type=task_type,
                    horizon=horizon,
                    indicator_output_keys=indicator_output_keys,
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
            "changed_recoverable_answer": (
                "Repair mode may not change the already recovered answer payload; only reasoning/context fields may be repaired."
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

    @classmethod
    def _terminal_answer_preserves_recoverable_answer(
        cls,
        *,
        expected_type: str,
        task_type: str,
        recoverable_final: dict[str, Any] | None,
        candidate_answer: Any,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        if not isinstance(recoverable_final, dict) or "answer" not in recoverable_final:
            return True
        original_answer = recoverable_final.get("answer")
        if expected_type == "learning_summary":
            return candidate_answer == original_answer
        return cls._answers_match_semantically(
            task_type,
            candidate_answer,
            original_answer,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )

    @staticmethod
    def _has_stepwise_reasoning(
        reasoning: Any,
        execution_log: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> bool:
        text = str(reasoning or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if any(tool in lowered for tool in _ORCHESTRATION_ONLY_TOOLS):
            return False

        candidate_names = ExecutorAgent._reasoning_candidate_names(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
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

        candidate_labels = ExecutorAgent._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
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
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
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

        candidate_labels = ExecutorAgent._reasoning_candidate_labels(
            execution_log,
            task_type=task_type,
            horizon=horizon,
            indicator_output_keys=indicator_output_keys,
        )
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
    def _reasoning_candidate_names(
        execution_log: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        horizon: int | None = None,
        indicator_output_keys: list[str] | None = None,
    ) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        def _append(raw_name: Any) -> None:
            name = str(raw_name or "").strip()
            if not name:
                return
            key = name.lower()
            if key in seen:
                return
            seen.add(key)
            names.append(name)

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
                hinted = ExecutorAgent._hinted_tools(result.get("tools_hint"))
                name = (
                    (sub_chain[0] if sub_chain else "")
                    or str(result.get("candidate_name") or "").strip()
                    or ", ".join(hinted)
                    or str(result.get("sub_task_description") or "").strip()
                    or f"candidate_{idx}"
                )
                _append(name)

        task_type_text = str(task_type or "").strip()
        if task_type_text:
            for label in ExecutorAgent._reasoning_candidate_labels(
                execution_log,
                task_type=task_type_text,
                horizon=horizon,
                indicator_output_keys=indicator_output_keys,
            ):
                _append(label)
        return names

    @staticmethod
    def _reasoning_mentions_candidate(reasoning_lower: str, candidate_name: str) -> bool:
        name = str(candidate_name or "").strip().lower()
        if not name:
            return False
        aliases = {name}
        if "->" in name:
            normalized = ExecutorAgent._normalize_label_text(name)
            aliases.add(normalized.lower())
            segments = [
                segment.strip()
                for segment in normalized.split(" -> ")
                if segment.strip()
            ]
            for segment in segments:
                aliases.add(segment.lower())
                if segment.endswith("_forecast"):
                    base = segment[: -len("_forecast")].strip("_")
                    if base:
                        aliases.add(base)
                        aliases.add(f"{base}-based")
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
    def _decision_response_preview(response: Any) -> str:
        return str(getattr(response, "text", "") or "").strip()[:4000]

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
        nested_parsed = cls._json_dict_from_response(
            response,
            required_keys={"thought", "done"},
        )
        normalized = cls._normalize_legacy_decision_payload(
            nested_parsed,
            allow_tool_call=allow_tool_call,
            allow_finish=allow_finish,
            finish_mode=finish_mode,
        ) if nested_parsed else None
        if normalized is None:
            legacy = cls._coerce_llm_decision(
                getattr(response, "parsed", None),
                getattr(response, "text", None),
                fallback_thought=fallback_thought,
            )
            normalized = cls._normalize_legacy_decision_payload(
                legacy,
                allow_tool_call=allow_tool_call,
                allow_finish=allow_finish,
                finish_mode=finish_mode,
            )
        if normalized is None:
            return {
                "done": False,
                "_parse_failed": True,
                "thought": fallback_thought,
            }
        return cls._validate_json_decision_tool_access(
            normalized,
            allow_tool_call=allow_tool_call,
            ordinary_tool_names=ordinary_tool_names,
            evaluate_tool_names=evaluate_tool_names,
            spawn_available=spawn_available,
        )

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

        thought = cls._coerce_structured_thought_text(decision.get("thought"))
        thought = cls._complete_legacy_thought_skeleton(thought)
        is_done = bool(decision.get("done", False))
        answer = decision.get("answer")
        answer_type = str(decision.get("answer_type") or "").strip()
        confidence = decision.get("confidence")
        reasoning = cls._coerce_reasoning_text(decision.get("reasoning"))

        if (
            not is_done
            and allow_finish
            and not allow_tool_call
            and answer is not None
            and reasoning
            and "tool" not in decision
            and "args" not in decision
        ):
            if finish_mode == "learning_summary" and not answer_type:
                answer_type = "learning_summary"
            if answer_type:
                return {
                    "thought": thought or (
                        "Reason: Repair mode already has enough execution-grounded evidence and the "
                        "remaining issue is only final packaging. "
                        "Check: The repaired final payload is already fully specified from the trace, "
                        "so no additional tool evidence is needed. "
                        "Self-check: The required final payload artifact is already present, and its "
                        "basic sanity condition is that the repaired payload is complete without further "
                        "tool output. "
                        "Update: Keep this repaired final candidate because there is no unresolved rival "
                        "candidate after repair. "
                        "Action: finish with the repaired final payload."
                    ),
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

        if is_done:
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

    @staticmethod
    def _validate_json_decision_tool_access(
        decision: dict[str, Any],
        *,
        allow_tool_call: bool,
        ordinary_tool_names: list[str] | None = None,
        evaluate_tool_names: list[str] | None = None,
        spawn_available: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(decision, dict) or decision.get("_parse_failed"):
            return decision
        if bool(decision.get("done", False)):
            return decision

        tool_name = str(decision.get("tool") or "").strip()
        if not allow_tool_call or not tool_name:
            return {
                "done": False,
                "_parse_failed": True,
                "thought": "invalid_decision_wrong_tool_family",
            }

        ordinary_tool_set = {
            str(name).strip() for name in (ordinary_tool_names or []) if str(name).strip()
        }
        evaluate_tool_set = {
            str(name).strip() for name in (evaluate_tool_names or []) if str(name).strip()
        }

        if tool_name == "spawn_subagent":
            if spawn_available:
                return decision
            return {
                "done": False,
                "_parse_failed": True,
                "thought": "invalid_decision_wrong_tool_family",
            }
        if tool_name in evaluate_tool_set:
            return decision
        if tool_name in ordinary_tool_set:
            return decision
        return {
            "done": False,
            "_parse_failed": True,
            "thought": "invalid_decision_wrong_tool_family",
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
        response = await self.llm.complete(
            conversation,
            response_format="json",
            temperature=0.0,
        )
        decision = self._coerce_structured_decision(
            response,
            allow_tool_call=allow_tool_call,
            allow_finish=allow_finish,
            ordinary_tool_names=ordinary_tool_names,
            evaluate_tool_names=evaluate_tool_names,
            spawn_available=spawn_available,
            finish_mode=finish_mode,
            fallback_thought=fallback_thought,
        )
        if decision.get("_parse_failed"):
            nested_parsed = self._json_dict_from_response(
                response,
                required_keys={"thought", "done"},
            )
            normalized_nested = self._normalize_legacy_decision_payload(
                nested_parsed,
                allow_tool_call=allow_tool_call,
                allow_finish=allow_finish,
                finish_mode=finish_mode,
            ) if nested_parsed else None
            trace.agent_step("executor", "decision_parse_debug", {
                "fallback_thought": fallback_thought,
                "response_text_preview": self._decision_response_preview(response),
                "response_parsed_type": type(getattr(response, "parsed", None)).__name__,
                "nested_parsed_found": bool(nested_parsed),
                "nested_parsed_keys": list((nested_parsed or {}).keys())[:10],
                "normalized_nested_preview": normalized_nested,
                "allow_tool_call": allow_tool_call,
                "allow_finish": allow_finish,
                "spawn_available": spawn_available,
                "ordinary_tool_names_preview": list(ordinary_tool_names or [])[:8],
                "evaluate_tool_names_preview": list(evaluate_tool_names or [])[:8],
            })
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
        scope_keys: list[str] | None = None,
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
                    scope_keys=scope_keys,
                    sub_task_description=sub_task_description,
                    tools_hint=tools_hint,
                    terminal_json=True,
                ),
            },
        ]
        retry_response = await self.llm.complete(
            clean_conversation,
            response_format="json",
            temperature=0.0,
        )
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

    def _selected_tool_note_keys(
        self,
        tool_notes: dict[str, Any],
        candidates: list[str],
        task_type: str,
    ) -> list[str]:
        if not isinstance(tool_notes, dict) or not tool_notes:
            return []
        selected: list[str] = []
        seen: set[str] = set()
        for tool_name in candidates:
            for key in self._tool_note_candidates(tool_name, task_type):
                if key in tool_notes and str(tool_notes[key]).strip():
                    if key not in seen:
                        seen.add(key)
                        selected.append(key)
                    break
        return selected

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
                lines.append(f"- {tool_name}: {matched_note}")
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
        task_semantic_guidance = self._build_task_semantic_guidance_section(
            task_type=plan.get("task_type", ""),
            scope_keys=plan.get("scope_keys", []),
        )
        if task_semantic_guidance:
            parts.append(f"## Task Semantic Guidance\n{task_semantic_guidance}")
        profiling = plan.get("profiling_section", "")
        if profiling:
            parts.append(profiling.strip())
        memory_audit = plan.get("memory_audit") or {}
        active_experience_text = str(memory_audit.get("active_experience_text") or "").strip()
        if active_experience_text:
            parts.append(active_experience_text)
        sample_packet_text = str(plan.get("sample_packet_text") or "").strip()
        if sample_packet_text:
            parts.append(sample_packet_text)
        selected_packet_tool_notes = str(plan.get("selected_packet_tool_notes") or "").strip()
        if selected_packet_tool_notes:
            parts.append(selected_packet_tool_notes)
        trace_lines: list[str] = []
        for idx, entry in enumerate(execution_log, start=1):
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name or tool_name in _ORCHESTRATION_ONLY_TOOLS:
                continue
            if not bool(entry.get("success")):
                continue
            args_text = self._format_prompt_payload(entry.get("args", {}), limit=None)
            result_text = self._format_prompt_payload(entry.get("observation"), limit=None)
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
    def _format_prompt_payload(value: Any, *, limit: int | None = None) -> str:
        try:
            text = json.dumps(
                ExecutorAgent._round_prompt_numbers(value),
                ensure_ascii=False,
                default=str,
            )
        except TypeError:
            text = str(value)
        if limit is None or int(limit) <= 0:
            return text
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
        return text

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
