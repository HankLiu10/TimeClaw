"""End-to-end pipeline runner — simplified two-agent architecture.

Architecture (inspired by OpenClaw's single-agent + tools design):

    1. Preprocess (deterministic, no LLM)
       - Modality detection
       - Fingerprint extraction + profiling tools
       - Strategy recommendation via fingerprint matching

    2. Execute (Executor agent — ReAct loop, the core LLM decision-maker)
       - Receives fingerprint, profiling results, and memory context
       - Runs tools step-by-step, deciding what to do next based on results
       - Per-modality branches run in parallel for multimodal data

    3. Verify (Reasoner agent — only when multimodal)
       - Fuses per-modality results
       - Cross-modal consistency check
       - Quality gate

Memory injection:
    - SOUL (fixed identity) + PRINCIPLES (learned) + MEMORY → system prompt
    - Daily notes → first-turn user message (semantic retrieval)
    - TOOLS.md → Tool guide annotations (per-tool experience notes)
    - PLAYBOOKS/ → first-turn user message (scene-specific SOP)
    - SOUL → Reasoner system prompt (lightweight, only if multimodal)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import numpy as np
from loguru import logger

from timeclaw.agents.executor import ExecutorAgent
from timeclaw.agents.reasoner import ReasonerAgent
from timeclaw.clawhub.tool_registry import discover_and_register, get_tool_instance, list_tool_instances
from timeclaw.core.config import TimeClawConfig
from timeclaw.mcp.manager import MCPManager, MCPServerConfig
from timeclaw.core.evidence import EvidenceLog, QualityGate
from timeclaw.core.message import AgentRole, ResultMessage
from timeclaw.core.reasoning import EvidenceIterator
from timeclaw.core.task_graph import TaskDAG, TaskNode
from timeclaw.core.tea_protocol import (
    AgentState,
    EnvironmentSnapshot,
    TaskEnvelope,
    TaskSpec,
)
from timeclaw.core import trace_logger as trace
from timeclaw.fingerprint.extractor import FingerprintExtractor
from timeclaw.fingerprint.matcher import FingerprintMatcher
from timeclaw.llm.router import ModelRouter
from timeclaw.memory.experience_pool import ExperienceEntry, ExperiencePool
from timeclaw.memory.hierarchical import HierarchicalMemory
from timeclaw.multimodal.detector import ModalityDetector

MAX_REANALYSIS_RETRIES = 2

# Profiling tools run deterministically during preprocessing
_PROFILING_TOOLS = ["basic_stats", "autocorrelation", "stationarity_test"]
_DETECTION_TOOLS = ["detect_trend", "detect_anomaly"]
_TEXT_PROFILING_TOOLS = ["sentiment", "keyword_extract"]
_IMAGE_PROFILING_TOOLS = ["image_load", "image_feature"]


class PipelineRunner:
    """Orchestrates the TimeClaw inference pipeline.

    Simplified two-agent architecture:

        Preprocess (pure code) → Executor (ReAct agent) → Reasoner (multimodal only)

    Memory Architecture (six components):
        - SOUL.md: Fixed agent identity and behavioral guidelines (human-written)
        - Daily Notes: Raw experience log, semantically indexed
        - MEMORY.md: Causal insights distilled from daily notes
        - PRINCIPLES.md: Cross-domain strategies distilled from MEMORY
        - TOOLS.md: Per-tool usage notes distilled from daily notes
        - PLAYBOOKS/: Scene-specific SOPs distilled from MEMORY
        Memory is populated via ``record_outcome()`` after ground truth is available,
        and injected into agent prompts automatically during ``run()``.
    """

    def __init__(self, config: TimeClawConfig | None = None, disable_memory: bool = False):
        self.config = config or TimeClawConfig()
        self._disable_memory = disable_memory
        trace.configure(self.config.pipeline.verbose)
        self.router = ModelRouter(self.config.llm)

        if disable_memory:
            self.experience_pool = None
            self.memory = HierarchicalMemory(enabled=False)
        else:
            self.experience_pool = ExperiencePool(
                storage_dir=f"{self.config.workspace_dir}/experiences"
            )
            self.memory = HierarchicalMemory(enabled=True)

        self._text_backends = self.router.get_text_backends()
        self._reasoner = ReasonerAgent(self.router.get_backend("reasoner"))

        self._fingerprint_extractor = FingerprintExtractor()
        self._fingerprint_matcher = FingerprintMatcher()
        self._modality_detector = ModalityDetector()

        self._quality_gate = QualityGate()
        self._tools_registered = False
        self._mcp_started = False
        self._run_counter = 0
        self._total_train_samples = 0

        # MCP tool servers
        import sys
        mcp_cfg = self.config.mcp
        if mcp_cfg.enabled:
            server_configs = []
            for s in list(mcp_cfg.servers) + list(mcp_cfg.extra_servers):
                if not s.enabled:
                    continue
                cmd = s.command if s.command != "python" else sys.executable
                server_configs.append(MCPServerConfig(
                    name=s.name,
                    command=cmd,
                    args=s.args,
                    env=s.env or None,
                ))
            self._mcp_manager = MCPManager(server_configs)
        else:
            self._mcp_manager = None

    async def _ensure_tools(self) -> None:
        if self._mcp_manager and not self._mcp_started:
            await self._mcp_manager.start()
            self._mcp_started = True
        if not self._tools_registered:
            discover_and_register()
            self._tools_registered = True

    def _build_tool_dict(self) -> dict[str, Any]:
        tools = list_tool_instances()
        return {t.name: t for t in tools if not t.is_stub}

    def _create_executor(self) -> ExecutorAgent:
        return ExecutorAgent(
            self.router.get_backend("executor"),
            text_backends=self._text_backends,
            experience_pool=self.experience_pool if not self._disable_memory else None,
            mcp_manager=self._mcp_manager,
        )

    async def shutdown(self) -> None:
        """Gracefully stop MCP servers."""
        if self._mcp_manager and self._mcp_started:
            await self._mcp_manager.shutdown()
            self._mcp_started = False

    # ── Tool warmup (progressive exposure for training) ────────────

    HOMOGENEOUS_GROUPS: dict[str, list[str]] = {
        "forecast": [
            "timesfm_forecast", "chronos_forecast", "timer_forecast",
            "moirai_forecast", "ets_forecast", "arima_forecast",
            "prophet_forecast",
        ],
    }

    def set_training_size(self, total: int) -> None:
        """Call before training loop to enable progressive tool warmup."""
        self._total_train_samples = total
        self._run_counter = 0

    def _apply_warmup_filter(self, candidates: list[str]) -> list[str]:
        """During training warmup, progressively expose homogeneous tools."""
        import random

        total = self._total_train_samples
        if total <= 0:
            return candidates

        warmup_end = int(total * 0.5)
        idx = self._run_counter

        if warmup_end <= 0 or idx >= warmup_end:
            return candidates

        progress = idx / warmup_end
        filtered = list(candidates)

        for _group_name, group_tools in self.HOMOGENEOUS_GROUPS.items():
            present = [t for t in group_tools if t in filtered]
            if len(present) <= 1:
                continue
            n_visible = max(1, min(len(present), int(1 + progress * len(present))))
            exposed = set(random.sample(present, n_visible))
            filtered = [t for t in filtered if t not in present or t in exposed]

        return filtered

    # ── Main entry point ──────────────────────────────────────────────

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
        full_dump: bool | None = None,
        domain: str = "",
    ) -> dict[str, Any]:
        """Execute the pipeline: Preprocess → Execute → Verify → Deliver."""
        self._run_counter += 1
        if verbose is not None:
            current_full_dump = full_dump if full_dump is not None else trace._full_dump
            trace.configure(verbose, full_dump=current_full_dump)
        await self._ensure_tools()
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

        # ── Step 1: Preprocess (deterministic, no LLM) ──
        t_pre = time.time()
        trace.stage("Preprocess", f"任务: {description or task_type}")
        logger.info("[Pipeline] Preprocessing: fingerprint + modality detection...")
        preprocess = await self._preprocess(envelope, data, text_data, images)

        fingerprint = preprocess["fingerprint"]
        suggested_strategy = preprocess["suggested_strategy"]
        detected_modalities = preprocess["modalities"]
        profiling_results = preprocess["profiling_results"]

        envelope.update_environment(fingerprint=fingerprint)
        envelope.task.metadata["suggested_strategy"] = suggested_strategy
        envelope.task.metadata["profiling_results"] = profiling_results

        timeline["preprocess"] = {
            "elapsed_ms": int((time.time() - t_pre) * 1000),
            "fingerprint_keys": list(fingerprint.keys()) if isinstance(fingerprint, dict) else [],
            "suggested_strategy": suggested_strategy,
            "modalities": detected_modalities,
        }
        trace.decision("Pipeline", f"策略: {suggested_strategy}, 模态: {detected_modalities}")

        # ── Step 2: Memory injection ──
        # SOUL (fixed) + PRINCIPLES (learned) + MEMORY → system prompt
        # Daily notes → first-turn user msg; TOOLS.md → tool guide; PLAYBOOKS → SOP
        if not self._disable_memory:
            fp_summary = str(fingerprint)[:2000] if fingerprint else ""
            query = f"{task_type} {description} {fp_summary}"
            soul_prompt = self.memory.soul_text or ""
            principles_prompt = self.memory.principles.get_prompt_section() if self.memory.principles else ""
            memory_prompt = self.memory.memory.get_prompt_section(domain, task_type) if self.memory.memory else ""
            daily_prompt = self.memory.daily.get_prompt_section(query, domain=domain or None, task=task_type or None) if self.memory.daily else ""
            tool_notes = self.memory.get_tool_notes()
            playbook_prompt = self.memory.get_playbook(domain, task_type)

            envelope.task.metadata["soul_prompt"] = soul_prompt
            envelope.task.metadata["principles_prompt"] = principles_prompt
            envelope.task.metadata["memory_l2"] = memory_prompt
            envelope.task.metadata["memory_l1"] = daily_prompt
            envelope.task.metadata["tool_notes"] = tool_notes
            envelope.task.metadata["playbook_prompt"] = playbook_prompt

        envelope.task.metadata["domain"] = domain

        envelope.task.metadata["_warmup_filter"] = (
            self._apply_warmup_filter if self._total_train_samples > 0 else None
        )

        # ── Step 3: Build DAG and Execute ──
        dag = self._build_dag(task_type, detected_modalities, horizon)
        dag_path = " → ".join(nid for nid in dag.nodes)
        logger.info(f"[Pipeline] DAG: {dag_path} (modalities={detected_modalities})")
        trace.decision("Pipeline", f"DAG: {dag_path}")

        t_exec = time.time()
        execute_verify_result = await self._execute_and_verify_loop(
            dag=dag,
            envelope=envelope,
            task_type=task_type,
            data=data,
            text_data=text_data,
            images=images,
            constraints=constraints,
            detected_modalities=detected_modalities,
            horizon=horizon,
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

        # ── Step 4: Deliver ──
        t_del = time.time()
        total_ms = int((time.time() - t_start) * 1000)
        trace.stage("Deliver", f"总耗时: {total_ms}ms")
        logger.info(f"[Pipeline] Delivering result ({total_ms}ms total)")

        all_tool_names = [e.get("tool", "") for e in all_execution_logs]
        gate_passed = gate_result.accepted if gate_result else False

        timeline["deliver"] = {"elapsed_ms": int((time.time() - t_del) * 1000)}

        result = ResultMessage(
            sender=AgentRole.SYSTEM,
            prediction=final_prediction,
            evidence_summary=self._reasoner.evidence_log.summary(),
            confidence=None,
            metadata={
                "trace_id": trace_id,
                "timeline": timeline,
                "total_ms": total_ms,
                "gate_accepted": gate_passed,
                "modalities": detected_modalities,
                "fusion_report": fusion_report,
            },
        )

        logger.info(
            f"[Pipeline] Run {trace_id} complete in {total_ms}ms "
            f"(modalities={detected_modalities}, "
            f"gate={'PASS' if gate_passed else 'FAIL'}, "
            f"reanalysis_rounds={len(reanalysis_history)})"
        )

        return {
            "prediction": final_prediction,
            "trace_id": trace_id,
            "timeline": timeline,
            "total_ms": total_ms,
            "gate_accepted": gate_passed,
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
            "tool_chain": all_tool_names,
        }

    # ── Preprocess (deterministic, replaces Manager + Analyzer) ───────

    async def _preprocess(
        self,
        envelope: TaskEnvelope,
        data: Any,
        text_data: list[dict[str, Any]] | None,
        images: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Run fingerprint extraction, modality detection, and profiling tools.

        This replaces the old Manager (intent parsing) and Analyzer (LLM diagnosis)
        agents with pure deterministic code — no LLM calls.
        """
        modality_report = self._modality_detector.detect(
            ts_data=data, text_data=text_data, images=images,
        )
        modalities = modality_report["modalities"]

        fingerprint = await self._fingerprint_extractor.extract(data)

        profiling_results: dict[str, Any] = {}
        for tool_name in _PROFILING_TOOLS + _DETECTION_TOOLS:
            await self._run_profiling_tool(tool_name, data, profiling_results)

        if "text" in modalities and text_data:
            for tool_name in _TEXT_PROFILING_TOOLS:
                await self._run_profiling_tool(
                    tool_name, data, profiling_results,
                    text=text_data, llm=self._text_backends or None,
                )

        if "image" in modalities and images:
            for tool_name in _IMAGE_PROFILING_TOOLS:
                await self._run_profiling_tool(
                    tool_name, data, profiling_results,
                    images=images,
                )

        strategy = self._fingerprint_matcher.match(fingerprint)

        fp_dict = fingerprint
        if hasattr(fingerprint, "to_summary"):
            fp_dict = fingerprint.to_summary()
        elif not isinstance(fingerprint, dict):
            fp_dict = {}

        envelope.update_environment(modalities=modalities)

        return {
            "fingerprint": fp_dict,
            "suggested_strategy": strategy,
            "modalities": modalities,
            "profiling_results": profiling_results,
        }

    async def _run_profiling_tool(
        self,
        tool_name: str,
        data: Any,
        results: dict[str, Any],
        **extra_params: Any,
    ) -> None:
        if self._mcp_manager and self._mcp_started and self._mcp_manager.get_tool(tool_name):
            args: dict[str, Any] = {}
            if data is not None:
                if hasattr(data, "tolist"):
                    args["ts_data"] = data.tolist()
                elif isinstance(data, list):
                    args["ts_data"] = data
            for k, v in extra_params.items():
                if k not in ("llm",):
                    args[k] = v
            r = await self._mcp_manager.call_tool(tool_name, args)
            results[tool_name] = r["data"] if r["success"] else r.get("error")
            return
        try:
            tool = get_tool_instance(tool_name)
            result = await tool.execute(data, **extra_params)
            results[tool_name] = result.data if result.success else result.error
            trace.tool_exec("preprocess", tool_name, extra_params, result)
        except KeyError:
            logger.debug(f"[Preprocess] Tool '{tool_name}' not registered, skipping")
        except Exception as e:
            logger.warning(f"[Preprocess] Tool '{tool_name}' error: {e}")

    # ── DAG construction (pure code, replaces Manager) ────────────────

    @staticmethod
    def _build_dag(
        task_type: str,
        modalities: list[str],
        horizon: int | None = None,
    ) -> TaskDAG:
        """Build a modality-aware DAG with parallel executor branches.

        Single modality:  execute_ts → verify
        Multi-modal:      [execute_ts, execute_text, ...] → fuse_verify
        """
        dag = TaskDAG()

        executor_ids: list[str] = []
        if len(modalities) <= 1:
            node_id = "execute_ts"
            dag.add_node(TaskNode(
                id=node_id,
                name=f"Execute {task_type} (timeseries)",
                agent_role="executor",
                description=f"Run {task_type} with time-series tools",
                params={"task_type": task_type, "modality": "timeseries", "horizon": horizon},
            ))
            executor_ids.append(node_id)
        else:
            for modality in modalities:
                node_id = f"execute_{modality}"
                dag.add_node(TaskNode(
                    id=node_id,
                    name=f"Execute {task_type} ({modality})",
                    agent_role="executor",
                    description=f"Run {modality}-specialized analysis for {task_type}",
                    params={"task_type": task_type, "modality": modality, "horizon": horizon},
                ))
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

    # ── Hierarchical Memory Interface ────────────────────────────────

    async def record_outcome(self, **run_context) -> None:
        """Record a pipeline run outcome into the hierarchical memory.

        This is the primary interface for feeding ground-truth signals back
        into the memory system.  The L1 store uses an LLM to summarize the
        run into a natural-language note, indexes it for semantic retrieval,
        and triggers L1→L2→L3 distillation when batch thresholds are reached.
        """
        await self.memory.record(**run_context)

    def flush_memory(self) -> None:
        self.memory.flush()

    async def distill_memory(self) -> None:
        await self.memory.force_distill()

    def memory_summary(self) -> dict:
        return self.memory.summary()

    # ── Execute + Verify loop ─────────────────────────────────────────

    async def _execute_and_verify_loop(
        self,
        dag: TaskDAG,
        envelope: TaskEnvelope,
        task_type: str,
        data: Any,
        text_data: list[dict[str, Any]] | None,
        images: list[dict[str, Any]] | None,
        constraints: list[dict[str, Any]] | None,
        detected_modalities: list[str],
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """Execute → Verify loop with re-analysis on cross-modal conflicts."""
        reanalysis_history: list[dict[str, Any]] = []
        all_execution_logs: list[dict[str, Any]] = []

        executor_nodes = {
            nid: node for nid, node in dag.nodes.items()
            if node.agent_role == "executor"
        }

        t_exec_start = time.time()
        trace.stage("Execute", f"分支数: {len(executor_nodes)}")

        per_modality_results = await self._run_all_executor_branches(
            executor_nodes, envelope, data, text_data, images,
            constraints, detected_modalities,
        )

        execute_ms = int((time.time() - t_exec_start) * 1000)

        for modality, result in per_modality_results.items():
            for log_entry in result.get("execution_log", []):
                all_execution_logs.append({**log_entry, "modality": modality})

        prediction = self._extract_primary_prediction(per_modality_results)

        # Verify + re-analysis loop
        t_verify_start = time.time()
        trace.stage("Verify")

        for retry in range(MAX_REANALYSIS_RETRIES + 1):
            logger.info(
                f"[Pipeline] Verify round {retry}: "
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
                    f"reason={reanalyze_request.get('reasoning', '')[:2000]}"
                )

                cross_modal = reasoner_result["result"].get("checks", [])
                cross_detail = ""
                cross_reasoning = ""
                for chk in cross_modal:
                    if chk.get("name") == "cross_modal_consistency":
                        cross_detail = chk.get("details", "")
                        cross_reasoning = chk.get("reasoning", "")
                        break

                reanalysis_history.append({
                    "round": retry + 1,
                    "modalities": modalities_to_redo,
                    "reasoning": reanalyze_request.get("reasoning", ""),
                    "suggested_tools": reanalyze_request.get("suggested_tools", []),
                })

                prev_logs = []
                for m in modalities_to_redo:
                    for entry in per_modality_results.get(m, {}).get("execution_log", []):
                        prev_logs.append(entry)

                envelope.task.metadata["reanalysis_context"] = {
                    "previous_execution_logs": prev_logs,
                    "conflict_details": cross_detail,
                    "conflict_reasoning": cross_reasoning,
                    "suggested_tools": reanalyze_request.get("suggested_tools", []),
                }

                redo_nodes = {
                    nid: node for nid, node in executor_nodes.items()
                    if node.params.get("modality", "timeseries") in modalities_to_redo
                }
                if redo_nodes:
                    redo_results = await self._run_all_executor_branches(
                        redo_nodes, envelope, data, text_data, images,
                        constraints, detected_modalities,
                    )
                    per_modality_results.update(redo_results)
                    for modality, result in redo_results.items():
                        for log_entry in result.get("execution_log", []):
                            all_execution_logs.append({
                                **log_entry, "modality": modality,
                                "reanalysis_round": retry + 1,
                            })
                    prediction = self._extract_primary_prediction(per_modality_results)

                    envelope.task.metadata.pop("reanalysis_context", None)
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
        data: Any,
        text_data: list[dict[str, Any]] | None,
        images: list[dict[str, Any]] | None,
        constraints: list[dict[str, Any]] | None,
        modalities: list[str],
    ) -> dict[str, dict[str, Any]]:
        per_modality_results: dict[str, dict[str, Any]] = {}

        if len(executor_nodes) <= 1:
            for nid, node in executor_nodes.items():
                result = await self._run_executor_branch(
                    node, envelope, data, text_data, images,
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
                    node, envelope, data, text_data, images,
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
        for modality in ["timeseries", "text", "image"]:
            r = per_modality_results.get(modality, {})
            if r.get("prediction") is not None:
                return r["prediction"]
        return None

    async def _run_executor_branch(
        self,
        node: TaskNode,
        envelope: TaskEnvelope,
        data: Any,
        text_data: list[dict[str, Any]] | None,
        images: list[dict[str, Any]] | None,
        constraints: list[dict[str, Any]] | None,
        modalities: list[str],
    ) -> dict[str, Any]:
        modality = node.params.get("modality", "timeseries")
        logger.info(f"[Pipeline] Running executor branch: {node.id} ({modality})")

        branch_envelope = envelope.with_agent("executor")
        branch_envelope.task.metadata["modality"] = modality

        executor = self._create_executor()
        executor_result = await executor.run(branch_envelope)
        result = executor_result["result"]

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
            "data": data, "task_type": node.params.get("task_type", "forecast"),
            "horizon": node.params.get("horizon"),
            "modalities": modalities,
        }
        if text_data:
            exec_context["text_data"] = text_data
        if images:
            exec_context["images"] = images

        trace.decision("Pipeline", f"Executor 未产出预测，启动 EvidenceIterator 补充推理")
        evidence_log = await evidence_iterator.iterate(
            f"analyze {modality} data for {node.params.get('task_type', 'forecast')}",
            evidence_log, exec_context,
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
