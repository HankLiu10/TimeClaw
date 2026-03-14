"""Three-layer reasoning engine: Hierarchical Planning + ReflAct + Evidence Iteration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

from timeclaw.core import trace_logger as trace
from timeclaw.core.context_manager import ContextManager
from timeclaw.core.evidence import EvidenceEntry, EvidenceLog, QualityGate, QualityGateResult
from timeclaw.core.task_graph import TaskDAG

if TYPE_CHECKING:
    from timeclaw.llm.base import LLMBackend


class HierarchicalPlanner:
    """Outer layer: parses task intent and generates a Task DAG."""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    async def parse_intent(self, task_description: str) -> dict[str, Any]:
        """Use LLM to parse task intent into structured form."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a time-series task planner. Parse the user's request into a "
                    "structured task specification. Return JSON with fields: task_type "
                    "(forecast/anomaly/classification/imputation/qa), target_variable, "
                    "horizon, constraints, and sub_goals (list of strings)."
                ),
            },
            {"role": "user", "content": task_description},
        ]
        response = await self.llm.complete(messages, response_format="json")
        return response.parsed or {"task_type": "forecast", "sub_goals": [task_description]}

    async def generate_dag(
        self, intent: dict[str, Any], modalities: list[str] | None = None,
    ) -> TaskDAG:
        """Generate a modality-aware task DAG from parsed intent.

        When multiple modalities are present, creates parallel executor branches.
        """
        from timeclaw.core.task_graph import TaskNode

        dag = TaskDAG()
        task_type = intent.get("task_type", "forecast")
        modalities = modalities or ["timeseries"]

        dag.add_node(TaskNode(
            id="analyze",
            name="Multimodal Analysis & Fingerprinting",
            agent_role="analyzer",
            description="Extract data fingerprint, detect modalities, diagnose characteristics",
        ))

        executor_ids: list[str] = []
        if len(modalities) <= 1:
            node_id = "execute_ts"
            dag.add_node(TaskNode(
                id=node_id,
                name=f"Execute {task_type} (timeseries)",
                agent_role="executor",
                description=f"Run {task_type} using matched tools and models",
                params={**intent, "modality": "timeseries"},
            ))
            dag.add_edge("analyze", node_id)
            executor_ids.append(node_id)
        else:
            for modality in modalities:
                node_id = f"execute_{modality}"
                dag.add_node(TaskNode(
                    id=node_id,
                    name=f"Execute {task_type} ({modality})",
                    agent_role="executor",
                    description=f"Run {modality}-specialized analysis for {task_type}",
                    params={**intent, "modality": modality},
                ))
                dag.add_edge("analyze", node_id)
                executor_ids.append(node_id)

        dag.add_node(TaskNode(
            id="fuse_verify",
            name="Fuse & Verify",
            agent_role="reasoner",
            description="Fuse multimodal evidence and run SafeAuto quality gate",
            params={"executor_ids": executor_ids, "modalities": modalities},
        ))
        for eid in executor_ids:
            dag.add_edge(eid, "fuse_verify")

        return dag


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
        return response.parsed or {"done": True}

    async def _act(
        self, tool_name: str, tool_args: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Execute a tool and return the observation, passing multimodal data."""
        if not tool_name:
            return None
        tool = self.tools.get(tool_name)
        if tool is None:
            logger.warning(f"Tool '{tool_name}' not found, returning None")
            return None
        try:
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
            return result
        except Exception as e:
            logger.error(f"[Reasoning] Tool '{tool_name}' failed: {e}")
            return {"error": str(e)}

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


class ReflActLoop:
    """Middle layer: goal-state reflection loop wrapping the evidence iterator."""

    def __init__(
        self,
        llm: LLMBackend,
        evidence_iterator: EvidenceIterator,
        quality_gate: QualityGate,
        max_iterations: int = 5,
        max_gate_retries: int = 3,
    ):
        self.llm = llm
        self.iterator = evidence_iterator
        self.gate = quality_gate
        self.max_iterations = max_iterations
        self.max_gate_retries = max_gate_retries

    async def run(
        self,
        sub_goals: list[str],
        context: dict[str, Any],
        constraints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute the ReflAct loop across sub-goals with quality gate verification."""
        evidence_log = EvidenceLog()
        all_results: dict[str, Any] = {}

        for goal_idx, goal in enumerate(sub_goals):
            logger.info(f"ReflAct: processing sub-goal {goal_idx}: {goal}")

            for iteration in range(self.max_iterations):
                evidence_log = await self.iterator.iterate(goal, evidence_log, context)

                reflection = await self._reflect(goal, evidence_log)
                trace.agent_step("reflact", f"reflect (goal={goal_idx}, iter={iteration})", reflection)

                if reflection.get("goal_achieved", False):
                    all_results[goal] = reflection.get("result")
                    break

                if reflection.get("replan", False):
                    logger.info(f"ReflAct: replanning for goal '{goal}'")
                    continue

            else:
                logger.warning(f"ReflAct: max iterations for goal '{goal}'")
                all_results[goal] = None

        prediction = all_results.get(sub_goals[-1]) if sub_goals else None

        gate_result: QualityGateResult | None = None
        for retry in range(self.max_gate_retries):
            gate_result = self.gate.check(prediction, evidence_log, constraints)
            if gate_result.accepted:
                break
            logger.warning(f"Quality gate rejected (retry {retry}): {gate_result.rejections}")
            trace.agent_step("reflact", f"quality_gate_retry_{retry}", {
                "rejections": gate_result.rejections,
            })
            evidence_log = await self.iterator.iterate(
                f"Fix: {gate_result.rejections}", evidence_log, context
            )

        return {
            "prediction": prediction,
            "evidence_log": evidence_log,
            "quality_gate": gate_result,
            "sub_results": all_results,
        }

    async def _reflect(self, goal: str, evidence_log: EvidenceLog) -> dict[str, Any]:
        """Reflect on current state vs goal."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are reflecting on progress toward a time-series analysis goal. "
                    "Evaluate whether the goal is achieved based on evidence. Return JSON "
                    "with: goal_achieved (bool), result (any), replan (bool), reason (str)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Evidence:\n{evidence_log.summary()}\n\n"
                    "Is the goal achieved?"
                ),
            },
        ]
        response = await self.llm.complete(messages, response_format="json")
        return response.parsed or {"goal_achieved": False, "replan": False}
