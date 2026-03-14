"""End-to-end pipeline runner: orchestrates the 6-step TimeClaw inference flow.

Supports dynamic DAG execution with parallel per-modality executor branches.
When multimodal data is present, the pipeline fans out into parallel executors
and converges at the Reasoner for fusion and quality gate verification.

If cross-modal conflicts are detected, the Reasoner can request re-analysis
of specific modalities, triggering a retry loop (max 2 retries).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import numpy as np
from loguru import logger

from timeclaw.agents.analyzer import AnalyzerAgent
from timeclaw.agents.executor import ExecutorAgent
from timeclaw.agents.manager import ManagerAgent
from timeclaw.agents.reasoner import ReasonerAgent
from timeclaw.clawhub.tool_registry import discover_and_register, get_tool_instance, list_tool_instances
from timeclaw.core.config import TimeClawConfig
from timeclaw.core.evidence import EvidenceLog, QualityGate
from timeclaw.core.message import AgentRole, ResultMessage
from timeclaw.core.reasoning import EvidenceIterator, HierarchicalPlanner, ReflActLoop
from timeclaw.core.task_graph import TaskDAG, TaskNode
from timeclaw.core.tea_protocol import (
    AgentState,
    EnvironmentSnapshot,
    TaskEnvelope,
    TaskSpec,
)
from timeclaw.core import trace_logger as trace
from timeclaw.llm.router import ModelRouter
from timeclaw.memory.experience_pool import ExperienceEntry, ExperiencePool

MAX_REANALYSIS_RETRIES = 2


class PipelineRunner:
    """Orchestrates the complete TimeClaw inference pipeline with dynamic DAG execution.

    Timeline:
        T+0    [Plan]    Manager parses intent, generates modality-aware DAG
        T+1    [Diagnose] Analyzer extracts fingerprint + detects modalities
        T+2    [Route]   Re-generate DAG based on detected modalities
        T+3    [Execute] Parallel per-modality executors (DAG-driven)
        T+4    [Verify]  Reasoner fuses multimodal evidence + quality gate
               [Retry]   If cross-modal conflict → LLM decides → re-execute branches
        T+5    [Deliver] Assemble result, store experience
    """

    def __init__(self, config: TimeClawConfig | None = None):
        self.config = config or TimeClawConfig()
        trace.configure(self.config.pipeline.verbose)
        self.router = ModelRouter(self.config.llm)
        self.experience_pool = ExperiencePool(
            storage_dir=f"{self.config.workspace_dir}/experiences"
        )

        manager_llm = self.router.get_backend("manager")
        self._text_backends = self.router.get_text_backends()
        self._manager = ManagerAgent(manager_llm)
        self._analyzer = AnalyzerAgent(
            self.router.get_backend("analyzer"),
            text_backends=self._text_backends,
            experience_pool=self.experience_pool,
        )
        self._reasoner = ReasonerAgent(self.router.get_backend("reasoner"))

        self._planner = HierarchicalPlanner(manager_llm)
        self._quality_gate = QualityGate()

        self._tools_registered = False

    def _ensure_tools(self) -> None:
        if not self._tools_registered:
            discover_and_register()
            self._tools_registered = True

    def _build_tool_dict(self) -> dict[str, Any]:
        """Build a name->tool mapping for the EvidenceIterator."""
        tools = list_tool_instances()
        return {t.name: t for t in tools if not t.is_stub}

    def _create_executor(self) -> ExecutorAgent:
        """Create a fresh ExecutorAgent instance for a DAG branch."""
        return ExecutorAgent(
            self.router.get_backend("executor"),
            text_backends=self._text_backends,
            experience_pool=self.experience_pool,
        )

    async def run(
        self,
        data: Any,
        task_type: str = "forecast",
        description: str = "",
        horizon: int | None = None,
        constraints: list[dict[str, Any]] | None = None,
        column_names: list[str] | None = None,
        text_data: list[dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
        verbose: bool | None = None,
    ) -> dict[str, Any]:
        """Execute the full 6-step pipeline with multimodal support.

        Args:
            verbose: Override config-level verbose setting for this run.
                     ``True`` enables full trace logging of LLM prompts/responses,
                     tool args/results, and agent decisions.
        """
        if verbose is not None:
            trace.configure(verbose)
        self._ensure_tools()
        trace_id = uuid4().hex[:12]
        t_start = time.time()
        logger.info(f"[Pipeline] Starting run {trace_id}: {task_type}")

        envelope = self._build_envelope(
            data=data,
            task_type=task_type,
            description=description or f"{task_type} task",
            horizon=horizon,
            constraints=constraints or [],
            column_names=column_names,
            trace_id=trace_id,
            text_data=text_data,
            images=images,
        )

        timeline: dict[str, dict[str, Any]] = {}

        # ── T+0 [Plan] Parse intent ──
        t0 = time.time()
        trace.stage("T+0 Plan", f"任务: {description or task_type}")
        logger.info("[Pipeline] T+0 [Plan] Parsing intent...")
        manager_result = await self._manager.run(envelope)
        intent = manager_result["result"].get("intent", {"task_type": task_type})
        intent["task_type"] = intent.get("task_type", task_type)
        if horizon:
            intent["horizon"] = horizon
        initial_modalities = manager_result["result"].get("modalities", ["timeseries"])
        timeline["plan"] = {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "intent": intent,
            "initial_modalities": initial_modalities,
        }

        # ── T+1 [Diagnose] Analyzer extracts fingerprint + detects modalities ──
        t1 = time.time()
        trace.stage("T+1 Diagnose")
        logger.info("[Pipeline] T+1 [Diagnose] Analyzer extracting fingerprint...")
        analyzer_result = await self._analyzer.run(envelope)
        fingerprint = analyzer_result["result"].get("fingerprint", {})
        suggested_strategy = analyzer_result["result"].get("suggested_strategy", "arima_forecast")
        detected_modalities = analyzer_result["result"].get("modalities", initial_modalities)
        envelope.update_environment(fingerprint=fingerprint)
        timeline["diagnose"] = {
            "elapsed_ms": int((time.time() - t1) * 1000),
            "fingerprint_keys": list(fingerprint.keys()) if isinstance(fingerprint, dict) else [],
            "suggested_strategy": suggested_strategy,
            "modalities": detected_modalities,
        }
        trace.decision("Analyzer", f"策略: {suggested_strategy}, 模态: {detected_modalities}")

        # ── T+2 [Route] Re-generate DAG based on actual modalities ──
        t2 = time.time()
        trace.stage("T+2 Route")
        dag = self._manager._build_dynamic_dag(intent, detected_modalities)
        dag_path = " → ".join(nid for nid in dag.nodes)
        logger.info(
            f"[Pipeline] T+2 [Route] Dynamic DAG: {len(dag.nodes)} nodes, "
            f"modalities={detected_modalities}"
        )
        trace.decision("Pipeline", f"DAG: {dag_path}")
        timeline["route"] = {
            "elapsed_ms": int((time.time() - t2) * 1000),
            "strategy": suggested_strategy,
            "dag_nodes": [nid for nid in dag.nodes],
            "dag_edges": dag.edges,
            "modalities": detected_modalities,
        }

        # ── T+3 [Execute] + T+4 [Verify] with re-analysis loop ──
        t3 = time.time()
        execute_verify_result = await self._execute_and_verify_loop(
            dag=dag,
            envelope=envelope,
            intent=intent,
            data=data,
            text_data=text_data,
            images=images,
            constraints=constraints,
            detected_modalities=detected_modalities,
        )

        per_modality_results = execute_verify_result["per_modality_results"]
        all_execution_logs = execute_verify_result["all_execution_logs"]
        prediction = execute_verify_result["prediction"]
        final_prediction = execute_verify_result["final_prediction"]
        gate_result = execute_verify_result["gate_result"]
        fusion_report = execute_verify_result["fusion_report"]
        reanalysis_history = execute_verify_result["reanalysis_history"]

        timeline["execute"] = {
            "elapsed_ms": execute_verify_result["execute_ms"],
            "executor_branches": execute_verify_result["n_branches"],
            "per_modality": {
                m: {"tools_run": len(r.get("execution_log", [])),
                     "has_prediction": r.get("prediction") is not None}
                for m, r in per_modality_results.items()
            },
            "has_prediction": prediction is not None,
        }
        timeline["verify"] = {
            "elapsed_ms": execute_verify_result["verify_ms"],
            "accepted": gate_result.accepted if gate_result else False,
            "rejections": gate_result.rejections if gate_result else [],
            "has_fusion": fusion_report is not None,
            "reanalysis_rounds": len(reanalysis_history),
        }

        # ── T+5 [Deliver] Assemble and store ──
        t5 = time.time()
        total_ms = int((time.time() - t_start) * 1000)
        trace.stage("T+5 Deliver", f"总耗时: {total_ms}ms")
        logger.info(f"[Pipeline] T+5 [Deliver] Assembling result ({total_ms}ms total)")

        all_tool_names = [e.get("tool", "") for e in all_execution_logs]
        combined_evidence = EvidenceLog()
        gate_passed = gate_result.accepted if gate_result else False
        n_rejections = len(gate_result.rejections) if gate_result else 0
        quality_score = 1.0 if gate_passed else max(0.0, 1.0 - 0.25 * n_rejections)

        experience = ExperienceEntry(
            trace_id=trace_id,
            task_type=task_type,
            task_description=description,
            fingerprint_summary=str(fingerprint)[:500],
            tool_chain=all_tool_names,
            evidence_steps=combined_evidence.num_steps,
            quality_gate_passed=gate_passed,
            quality_score=quality_score,
            prediction_summary=str(final_prediction)[:200] if final_prediction is not None else "",
            metadata={
                "modalities": detected_modalities,
                "executor_branches": execute_verify_result["n_branches"],
                "reanalysis_rounds": len(reanalysis_history),
            },
        )
        try:
            self.experience_pool.add(experience)
        except Exception as e:
            logger.debug(f"Failed to store experience: {e}")

        await self._maybe_refresh_tool_insights()

        timeline["deliver"] = {"elapsed_ms": int((time.time() - t5) * 1000)}

        result = ResultMessage(
            sender=AgentRole.SYSTEM,
            prediction=final_prediction,
            evidence_summary=combined_evidence.summary(),
            confidence=None,
            metadata={
                "trace_id": trace_id,
                "timeline": timeline,
                "total_ms": total_ms,
                "gate_accepted": gate_result.accepted if gate_result else False,
                "modalities": detected_modalities,
                "fusion_report": fusion_report,
            },
        )

        logger.info(
            f"[Pipeline] Run {trace_id} complete in {total_ms}ms "
            f"(modalities={detected_modalities}, "
            f"gate={'PASS' if result.metadata.get('gate_accepted') else 'FAIL'}, "
            f"reanalysis_rounds={len(reanalysis_history)})"
        )

        return {
            "prediction": final_prediction,
            "trace_id": trace_id,
            "timeline": timeline,
            "total_ms": total_ms,
            "gate_accepted": gate_result.accepted if gate_result else False,
            "evidence_summary": result.evidence_summary,
            "result_message": result,
            "modalities": detected_modalities,
            "per_modality_results": per_modality_results,
            "fusion_report": fusion_report,
            "reanalysis_history": reanalysis_history,
            "dag": {
                "nodes": [nid for nid in dag.nodes],
                "edges": dag.edges,
            },
        }

    _insight_refresh_interval = 10

    async def _maybe_refresh_tool_insights(self) -> None:
        """Periodically ask LLM to summarize tool effectiveness from experience pool."""
        pool = self.experience_pool
        if pool.size < 5 or pool.size % self._insight_refresh_interval != 0:
            return
        prompt_text = pool.build_llm_summary_prompt()
        if not prompt_text:
            return
        try:
            llm = self.router.get_backend("manager")
            messages = [
                {"role": "system", "content": "You are a tool effectiveness analyst."},
                {"role": "user", "content": prompt_text},
            ]
            response = await llm.complete(messages, response_format="json")
            if response.parsed and "tool_insights" in response.parsed:
                pool.save_tool_insights(response.parsed["tool_insights"])
                logger.info("[Pipeline] Refreshed tool insights from experience pool")
        except Exception as e:
            logger.debug(f"[Pipeline] Tool insight refresh failed: {e}")

    async def _execute_and_verify_loop(
        self,
        dag: TaskDAG,
        envelope: TaskEnvelope,
        intent: dict[str, Any],
        data: Any,
        text_data: list[dict[str, Any]] | None,
        images: list[dict[str, Any]] | None,
        constraints: list[dict[str, Any]] | None,
        detected_modalities: list[str],
    ) -> dict[str, Any]:
        """Execute → Verify loop with re-analysis on cross-modal conflicts."""
        reanalysis_history: list[dict[str, Any]] = []
        per_modality_results: dict[str, dict[str, Any]] = {}
        all_execution_logs: list[dict[str, Any]] = []

        executor_nodes = {
            nid: node for nid, node in dag.nodes.items()
            if node.agent_role == "executor"
        }

        t_exec_start = time.time()
        trace.stage("T+3 Execute", f"分支数: {len(executor_nodes)}")

        # Initial execution
        per_modality_results = await self._run_all_executor_branches(
            executor_nodes, envelope, intent, data, text_data, images,
            constraints, detected_modalities,
        )

        execute_ms = int((time.time() - t_exec_start) * 1000)

        for modality, result in per_modality_results.items():
            for log_entry in result.get("execution_log", []):
                all_execution_logs.append({**log_entry, "modality": modality})

        prediction = self._extract_primary_prediction(per_modality_results)

        # Verify + re-analysis loop
        t_verify_start = time.time()
        trace.stage("T+4 Verify")

        for retry in range(MAX_REANALYSIS_RETRIES + 1):
            logger.info(
                f"[Pipeline] T+4 [Verify] Round {retry}: "
                f"Multimodal fusion + SafeAuto quality gate..."
            )
            envelope.task.metadata["prediction"] = prediction
            self._reasoner.evidence_log = EvidenceLog()
            self._reasoner.per_modality_results = per_modality_results
            reasoner_result = await self._reasoner.run(envelope)

            gate_result = reasoner_result["result"].get("gate_result")
            final_prediction = reasoner_result["result"].get("prediction", prediction)
            fusion_report = reasoner_result["result"].get("fusion_report")
            reanalyze_request = reasoner_result["result"].get("reanalyze_request")

            if reanalyze_request and retry < MAX_REANALYSIS_RETRIES:
                modalities_to_redo = reanalyze_request.get(
                    "modalities_to_reanalyze", list(per_modality_results.keys())
                )
                logger.info(
                    f"[Pipeline] Re-analysis requested (round {retry+1}): "
                    f"modalities={modalities_to_redo}, "
                    f"reason={reanalyze_request.get('reasoning', '')[:100]}"
                )
                reanalysis_history.append({
                    "round": retry + 1,
                    "modalities": modalities_to_redo,
                    "reasoning": reanalyze_request.get("reasoning", ""),
                    "suggested_tools": reanalyze_request.get("suggested_tools", []),
                })

                redo_nodes = {
                    nid: node for nid, node in executor_nodes.items()
                    if node.params.get("modality", "timeseries") in modalities_to_redo
                }
                if redo_nodes:
                    redo_results = await self._run_all_executor_branches(
                        redo_nodes, envelope, intent, data, text_data, images,
                        constraints, detected_modalities,
                    )
                    per_modality_results.update(redo_results)

                    for modality, result in redo_results.items():
                        for log_entry in result.get("execution_log", []):
                            all_execution_logs.append({
                                **log_entry, "modality": modality, "reanalysis_round": retry + 1,
                            })

                    prediction = self._extract_primary_prediction(per_modality_results)
                    continue
            break

        verify_ms = int((time.time() - t_verify_start) * 1000)

        return {
            "per_modality_results": per_modality_results,
            "all_execution_logs": all_execution_logs,
            "prediction": prediction,
            "final_prediction": final_prediction,
            "gate_result": gate_result,
            "fusion_report": fusion_report,
            "reanalysis_history": reanalysis_history,
            "execute_ms": execute_ms,
            "verify_ms": verify_ms,
            "n_branches": len(executor_nodes),
        }

    async def _run_all_executor_branches(
        self,
        executor_nodes: dict[str, TaskNode],
        envelope: TaskEnvelope,
        intent: dict[str, Any],
        data: Any,
        text_data: list[dict[str, Any]] | None,
        images: list[dict[str, Any]] | None,
        constraints: list[dict[str, Any]] | None,
        modalities: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Run all executor branches (parallel if >1)."""
        per_modality_results: dict[str, dict[str, Any]] = {}

        if len(executor_nodes) <= 1:
            for nid, node in executor_nodes.items():
                result = await self._run_executor_branch(
                    node, envelope, intent, data, text_data, images,
                    constraints, modalities,
                )
                modality = node.params.get("modality", "timeseries")
                per_modality_results[modality] = result
                node.mark_completed(result)
        else:
            tasks = []
            node_items = list(executor_nodes.items())
            for nid, node in node_items:
                tasks.append(self._run_executor_branch(
                    node, envelope, intent, data, text_data, images,
                    constraints, modalities,
                ))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (nid, node), result in zip(node_items, results):
                modality = node.params.get("modality", "timeseries")
                if isinstance(result, Exception):
                    logger.error(f"[Pipeline] Executor {nid} failed: {result}")
                    per_modality_results[modality] = {
                        "prediction": None, "execution_log": [],
                        "error": str(result),
                    }
                    node.mark_failed(str(result))
                else:
                    per_modality_results[modality] = result
                    node.mark_completed(result)

        return per_modality_results

    @staticmethod
    def _extract_primary_prediction(
        per_modality_results: dict[str, dict[str, Any]]
    ) -> Any:
        """Extract the primary prediction from per-modality results."""
        for modality in ["timeseries", "text", "image"]:
            r = per_modality_results.get(modality, {})
            if r.get("prediction") is not None:
                return r["prediction"]
        return None

    async def _run_executor_branch(
        self,
        node: TaskNode,
        envelope: TaskEnvelope,
        intent: dict[str, Any],
        data: Any,
        text_data: list[dict[str, Any]] | None,
        images: list[dict[str, Any]] | None,
        constraints: list[dict[str, Any]] | None,
        modalities: list[str],
    ) -> dict[str, Any]:
        """Run a single executor branch (one modality)."""
        modality = node.params.get("modality", "timeseries")
        logger.info(f"[Pipeline] Running executor branch: {node.id} ({modality})")

        branch_envelope = envelope.with_agent("executor")
        branch_envelope.task.metadata["modality"] = modality

        executor = self._create_executor()
        executor_result = await executor.run(branch_envelope)
        result = executor_result["result"]

        # Seed the evidence log with what the Executor already did,
        # so the EvidenceIterator doesn't repeat work from scratch.
        evidence_log = result.get("evidence_log") or EvidenceLog()
        if evidence_log.num_steps == 0 and result.get("execution_log"):
            from timeclaw.core.evidence import EvidenceEntry
            for entry in result["execution_log"]:
                evidence_log.append(EvidenceEntry(
                    step=evidence_log.num_steps,
                    thought=f"Executor ran {entry.get('tool', '?')}",
                    tool_name=entry.get("tool", ""),
                    tool_args=entry.get("args", {}),
                    observation=entry.get("result_summary"),
                    critic_feedback="ok" if entry.get("success") else "failed",
                ))

        if result.get("prediction") is not None:
            result["evidence_log"] = evidence_log
            return result

        tool_dict = self._build_tool_dict()
        executor_llm = self.router.get_backend("executor")
        evidence_iterator = EvidenceIterator(
            llm=executor_llm,
            tools=tool_dict,
            max_steps=self.config.reasoning.max_evidence_steps,
        )

        exec_context: dict[str, Any] = {
            "data": data, "task_type": intent.get("task_type", "forecast"),
            "horizon": intent.get("horizon"),
            "modalities": modalities,
        }
        if text_data:
            exec_context["text_data"] = text_data
        if images:
            exec_context["images"] = images

        sub_goals = intent.get("sub_goals", [f"analyze {modality} data"])
        trace.decision("Pipeline", f"Executor 未产出预测，启动 EvidenceIterator 补充推理")
        for goal in sub_goals[:2]:
            evidence_log = await evidence_iterator.iterate(
                f"{goal} (focus on {modality})", evidence_log, exec_context
            )

        result["evidence_log"] = evidence_log
        return result

    @staticmethod
    def _build_envelope(
        data: Any,
        task_type: str,
        description: str,
        horizon: int | None,
        constraints: list[dict[str, Any]],
        column_names: list[str] | None,
        trace_id: str,
        text_data: list[dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
    ) -> TaskEnvelope:
        env = EnvironmentSnapshot()
        env.set_data(data)
        if column_names:
            env.column_names = column_names
        if text_data:
            env.set_text_data(text_data)
        if images:
            env.set_images(images)

        task = TaskSpec(
            task_id=trace_id,
            task_type=task_type,
            description=description,
            horizon=horizon,
            constraints=constraints,
        )

        return TaskEnvelope(
            task=task,
            environment=env,
            agent_state=AgentState(agent_role="system"),
            trace_id=trace_id,
        )
