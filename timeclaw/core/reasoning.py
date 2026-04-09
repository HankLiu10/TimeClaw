"""Evidence iteration reasoning engine for explicit tool execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from timeclaw.core import trace_logger as trace
from timeclaw.core.context_manager import ContextManager
from timeclaw.core.evidence import EvidenceEntry, EvidenceLog

if TYPE_CHECKING:
    from timeclaw.llm.base import LLMBackend


class EvidenceIterator:
    """Inner layer: Think -> Act -> Observe -> Critic loop with evidence logging."""

    def __init__(
        self,
        llm: LLMBackend,
        tools: dict[str, Any],
        max_steps: int = 15,
        context_manager: ContextManager | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.ctx_mgr = context_manager or ContextManager()

    async def iterate(
        self,
        goal: str,
        evidence_log: EvidenceLog,
        context: dict[str, Any],
    ) -> EvidenceLog:
        """Run the evidence iteration loop until sufficient or max steps reached."""
        for step in range(evidence_log.num_steps, self.max_steps):
            thought = await self._think(goal, evidence_log, context)

            if thought.get("done", False):
                logger.info(f"Evidence iteration complete at step {step}")
                trace.reasoning_step("evidence_iterator", step, thought=thought)
                break

            tool_name = thought.get("tool", "")
            tool_args = thought.get("args", {})

            observation = await self._act(tool_name, tool_args, context)

            critic_feedback = await self._critic(thought, observation, evidence_log)

            trace.reasoning_step(
                "evidence_iterator", step,
                thought=thought, observation=observation, critic=critic_feedback,
            )

            entry = EvidenceEntry(
                step=step,
                thought=thought.get("reasoning", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                observation=observation,
                critic_feedback=critic_feedback,
            )
            evidence_log.append(entry)

            covered = thought.get("covers_predicates", [])
            for p in covered:
                evidence_log.mark_covered(p)

            if evidence_log.is_sufficient:
                logger.info(f"All predicates covered at step {step}")
                break

        return evidence_log

    async def _think(
        self, goal: str, evidence_log: EvidenceLog, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate next thought: hypothesis, tool selection, and arguments."""
        tool_descriptions = [
            {"name": name, "description": getattr(t, "description", str(t)),
             "category": getattr(t, "category", "unknown").value
             if hasattr(getattr(t, "category", None), "value") else "unknown"}
            for name, t in self.tools.items()
        ]

        pruned_tools = self.ctx_mgr.prune_tool_descriptions(tool_descriptions)

        modalities = context.get("modalities", ["timeseries"])
        modality_hint = f"Active modalities: {', '.join(modalities)}"
        if "text" in modalities:
            modality_hint += "\nText data is available -- consider using text analysis tools (sentiment, ner, keyword_extract, summarize, text_to_event)."
        if "image" in modalities:
            modality_hint += "\nImage data is available -- consider using image tools (image_load, image_feature, image_ocr, chart_to_data)."

        evidence_summary = evidence_log.summary()
        evidence_summary = self.ctx_mgr.prune_evidence_summary(evidence_summary)

        sections = {
            "system": (
                "You are a multimodal time-series reasoning agent. Given a goal and accumulated "
                "evidence, decide the next analytical step. Choose a tool from the "
                "available set and specify arguments. Return JSON with: reasoning, "
                "tool, args, covers_predicates (list), done (bool).\n\n"
                "For text tools, pass text-related arguments in 'args'. "
                "For image tools, pass image paths in 'args'. "
                "The system will automatically inject multimodal data into the tool."
            ),
            "user": (
                f"Goal: {goal}\n\n"
                f"{modality_hint}\n\n"
                f"Evidence so far:\n{evidence_summary}\n\n"
                f"Available tools: {pruned_tools}\n\n"
                f"Gaps: {evidence_log.gap_predicates}\n\n"
                "What should we do next?"
            ),
        }

        sections = self.ctx_mgr.fit_budget(sections)

        messages = [
            {"role": "system", "content": sections["system"]},
            {"role": "user", "content": sections["user"]},
        ]
        response = await self.llm.complete(messages, response_format="json")
        parsed = response.parsed if isinstance(response.parsed, dict) else None
        if parsed is None:
            raise ValueError("Evidence iterator LLM returned invalid JSON")
        return parsed

    async def _act(
        self, tool_name: str, tool_args: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Execute a tool and return the observation, passing multimodal data."""
        if not tool_name:
            raise ValueError("Evidence iterator did not select a tool")
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' not found")
        ts_data = context.get("data")
        text_data = context.get("text_data")
        images = context.get("images")
        safe_args = {
            k: v for k, v in tool_args.items()
            if k not in ("text", "images", "llm", "ts_data")
        }
        result = await tool.execute(
            ts_data, text=text_data, images=images, llm=self.llm, **safe_args
        )
        status = "PASS" if result.success else f"FAIL | {result.error}"
        logger.info(f"[Reasoning] {tool_name}: {status}")
        trace.tool_exec("reasoning", tool_name, tool_args, result)
        if not result.success:
            raise RuntimeError(f"Tool '{tool_name}' failed: {result.error}")
        return result.data

    async def _critic(
        self,
        thought: dict[str, Any],
        observation: Any,
        evidence_log: EvidenceLog,
    ) -> str:
        """Critic: evaluate tool suitability, output plausibility, evidence sufficiency."""
        obs_str = str(observation)
        if len(obs_str) > self.ctx_mgr.prune_observation_chars * 4:
            obs_str = obs_str[:self.ctx_mgr.prune_observation_chars * 4] + "... [truncated]"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a critic reviewing a time-series analysis step. Evaluate: "
                    "1) Was the tool appropriate? 2) Is the output plausible? "
                    "3) Does this advance toward the goal? Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Thought: {thought.get('reasoning', '')}\n"
                    f"Tool: {thought.get('tool', '')}\n"
                    f"Observation: {obs_str}\n"
                    f"Evidence gaps: {evidence_log.gap_predicates}"
                ),
            },
        ]
        response = await self.llm.complete(messages)
        return response.text or ""

