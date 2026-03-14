"""ManagerAgent: central planner that parses intent and generates dynamic task DAGs.

Generates modality-aware DAGs: when multimodal data is detected, the DAG fans out
into parallel per-modality executor branches that converge at a fusion+verify node.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from timeclaw.agents.prompts import manager_prompt
from timeclaw.core import trace_logger as trace
from timeclaw.core.agent_base import AgentBase
from timeclaw.core.message import AgentRole
from timeclaw.core.task_graph import TaskDAG, TaskNode
from timeclaw.core.tea_protocol import TaskEnvelope
from timeclaw.multimodal.detector import ModalityDetector


class ManagerAgent(AgentBase):
    role = AgentRole.MANAGER

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._modality_detector = ModalityDetector()

    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        """Extract task description, data summary, and detect active modalities."""
        data_shape = envelope.environment.data_shape
        columns = envelope.environment.column_names

        modality_report = self._modality_detector.detect(
            ts_data=envelope.environment.data,
            text_data=envelope.environment.text_data or None,
            images=envelope.environment.images or None,
        )

        perception = {
            "task_description": envelope.task.description,
            "task_type": envelope.task.task_type,
            "data_shape": data_shape,
            "columns": columns,
            "constraints": envelope.task.constraints,
            "modalities": modality_report["modalities"],
            "is_multimodal": modality_report["is_multimodal"],
            "modality_details": modality_report["details"],
        }
        trace.agent_step("manager", "perceive", perception)
        return perception

    async def plan(self, perception: dict[str, Any]) -> dict[str, Any]:
        """Use LLM to parse intent and generate a modality-aware task DAG."""
        modalities = perception.get("modalities", ["timeseries"])
        modality_ctx = f"\nDetected modalities: {modalities}"
        if perception.get("is_multimodal"):
            modality_ctx += (
                "\nThis is a MULTIMODAL task. Generate sub_goals that cover each modality. "
                "The system will create parallel executor branches automatically."
            )

        messages = [
            {"role": "system", "content": manager_prompt.SYSTEM_PROMPT + modality_ctx},
            {"role": "user", "content": perception["task_description"]},
        ]

        response = await self.llm.complete(messages, response_format="json")
        intent = response.parsed or {
            "task_type": perception.get("task_type", "forecast"),
            "sub_goals": [perception["task_description"]],
        }

        dag = self._build_dynamic_dag(intent, modalities)
        logger.info(
            f"[Manager] Generated dynamic DAG with {len(dag.nodes)} nodes "
            f"for modalities {modalities}"
        )

        trace.agent_step("manager", "plan", {
            "intent": intent,
            "dag_nodes": list(dag.nodes.keys()),
            "dag_edges": dag.edges,
            "modalities": modalities,
        })
        return {"intent": intent, "dag": dag, "modalities": modalities}

    async def act(self, plan: dict[str, Any], envelope: TaskEnvelope) -> dict[str, Any]:
        """The Manager's action is dispatching the DAG (handled by pipeline runner)."""
        return {
            "dag": plan["dag"],
            "intent": plan["intent"],
            "modalities": plan.get("modalities", ["timeseries"]),
        }

    async def reflect(self, action_result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate whether the plan was executed successfully."""
        dag: TaskDAG = action_result.get("dag", TaskDAG())
        return {
            "plan_complete": dag.is_complete,
            "has_failures": dag.has_failures,
        }

    def _build_dynamic_dag(
        self, intent: dict[str, Any], modalities: list[str]
    ) -> TaskDAG:
        """Build a modality-aware DAG with parallel executor branches.

        - Single modality: analyze -> execute -> verify (classic)
        - Multi-modal: analyze -> [ts_exec, text_exec, ...] -> fuse_verify
        """
        dag = TaskDAG()
        task_type = intent.get("task_type", "forecast")

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
                description=f"Run {task_type} with time-series tools",
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

        verify_desc = "SafeAuto quality gate verification"
        if len(modalities) > 1:
            verify_desc = (
                "Fuse multimodal evidence from parallel executors, "
                "then run SafeAuto quality gate verification"
            )

        dag.add_node(TaskNode(
            id="fuse_verify",
            name="Fuse & Verify",
            agent_role="reasoner",
            description=verify_desc,
            params={"executor_ids": executor_ids, "modalities": modalities},
        ))

        for eid in executor_ids:
            dag.add_edge(eid, "fuse_verify")

        return dag
