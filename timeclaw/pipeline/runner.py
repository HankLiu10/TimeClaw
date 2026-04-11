"""End-to-end pipeline runner -- single-agent architecture.

Architecture:

    1. Preprocess (deterministic, no LLM)
       - Modality detection
       - Fingerprint extraction + profiling tools

    2. Execute (single Executor agent -- ReAct loop)
       - Receives ALL data (timeseries, text, images) in one context
       - Runs tools step-by-step across any modality
       - Produces structured answer with confidence + reasoning

    3. Post-process (pure code quality gate)
       - Bound check, consistency check
       - Constraint enforcement
       - No LLM calls -- deterministic validation only

Memory injection:
    - SOUL (fixed identity) + MEMORY (per-scope + cross_domain) -> system prompt
    - Notes -> first-turn user message (semantic retrieval)
    - TOOLS.md -> Tool guide annotations (per-tool experience notes)
    - SKILLS/ -> first-turn user message (scene-specific SOP)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any
from uuid import uuid4

import numpy as np
from loguru import logger

from timeclaw.agents.executor import ExecutorAgent, MaxStepsExhaustedError
from timeclaw.clawhub.tool_registry import discover_and_register, get_tool_instance, list_tool_instances
from timeclaw.core.config import TimeClawConfig
from timeclaw.core.learning_text import normalize_learning_clause
from timeclaw.mcp.manager import MCPManager, MCPServerConfig
from timeclaw.core.evidence import EvidenceLog, QualityGate, QualityGateResult
from timeclaw.core.message import AgentRole, ResultMessage
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
from timeclaw.benchmark.mtbench_support import generate_learning_analysis
from timeclaw.memory.hierarchical import HierarchicalMemory
from timeclaw.multimodal.detector import ModalityDetector

# Profiling tools run deterministically during preprocessing
_PROFILING_TOOLS = ["basic_stats", "autocorrelation", "stationarity_test"]
_DETECTION_TOOLS = ["detect_trend", "detect_anomaly"]
_TEXT_PROFILING_TOOLS: list[str] = []
_IMAGE_PROFILING_TOOLS = ["image_load", "image_feature"]
_REQUIRED_PREPROCESS_TOOLS = set(_PROFILING_TOOLS)
_FRAMEWORK_TOOLS = {
    "evaluate_against_gt",
    "evaluate_batch_against_gt",
    "batch_evaluate_against_gt",
    "spawn_subagent",
}


class PipelineRunner:
    """Orchestrates the TimeClaw inference pipeline.

    Single-agent architecture:

        Preprocess (pure code) -> Executor (single ReAct agent) -> Post-process (pure code)

    The Executor handles ALL modalities in one ReAct loop, selecting tools
    autonomously. It produces a structured answer with confidence and reasoning.
    Post-processing applies deterministic quality checks (bounds, constraints).

    Memory Architecture (five components, aligned with OpenClaw):
        - SOUL.md: Fixed agent identity and behavioral guidelines
        - Notes: Raw experience log
        - MEMORY: Causal insights (per-scope + cross_domain)
        - TOOLS.md: Per-tool usage experience notes
        - SKILLS/: Scene-specific SOPs
    """

    def __init__(self, config: TimeClawConfig | None = None, disable_memory: bool = False):
        self.config = config or TimeClawConfig.load()
        self._disable_memory = disable_memory
        trace.configure(self.config.pipeline.verbose)
        self.router = ModelRouter(self.config.llm)

        if disable_memory:
            self.memory = HierarchicalMemory(enabled=False)
        else:
            self.memory = HierarchicalMemory(enabled=True)

        self._text_backends = self.router.get_text_backends()
        self._fingerprint_extractor = FingerprintExtractor()
        self._fingerprint_matcher = FingerprintMatcher()
        self._modality_detector = ModalityDetector()

        self._quality_gate = QualityGate()
        self._tools_registered = False
        self._mcp_started = False

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
            mcp_manager=self._mcp_manager,
        )

    async def shutdown(self) -> None:
        """Gracefully stop MCP servers."""
        if not self._disable_memory:
            try:
                await self.memory.distill_pending()
            except Exception as e:
                logger.warning(f"[Pipeline] Memory distill_pending during shutdown failed: {e}")
        if self._mcp_manager and self._mcp_started:
            await self._mcp_manager.shutdown()
            self._mcp_started = False

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
        scope_keys: list[str] | None = None,
        mode: str = "test",
        ground_truth: Any = None,
        defer_learning_record: bool = False,
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the pipeline: Preprocess -> Execute -> Post-process -> Deliver.

        Args:
            mode: ``"test"`` (default) or ``"train"``.  In train mode the
                  system prompt gets a learning-mode hint, ground truth is
                  stored internally for ``evaluate_against_gt``, and that
                  tool is added to the candidate list.
            ground_truth: The ground-truth value for this sample.  Only used
                          when ``mode="train"``; ignored otherwise.
        """
        if verbose is not None:
            current_full_dump = full_dump if full_dump is not None else trace._full_dump
            trace.configure(verbose, full_dump=current_full_dump)
        await self._ensure_tools()
        trace_id = uuid4().hex[:12]
        t_start = time.time()
        logger.info(f"[Pipeline] Starting run {trace_id}: {task_type}")
        trace.start_run(trace_id, {
            "task_type": task_type,
            "description": description or f"{task_type} task",
            "domain": domain,
            "horizon": horizon,
            "mode": mode,
            "scope_keys": scope_keys or [],
            "source_metadata": source_metadata or {},
        })

        try:
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

            # -- Step 1: Preprocess (deterministic, no LLM) --
            t_pre = time.time()
            trace.stage("Preprocess", f"task: {description or task_type}")
            logger.info("[Pipeline] Preprocessing: fingerprint + modality detection...")
            preprocess = await self._preprocess(envelope, data, text_data, images)

            fingerprint = preprocess["fingerprint"]
            detected_modalities = preprocess["modalities"]
            profiling_results = preprocess["profiling_results"]

            envelope.update_environment(fingerprint=fingerprint)
            envelope.task.metadata["profiling_results"] = profiling_results

            timeline["preprocess"] = {
                "elapsed_ms": int((time.time() - t_pre) * 1000),
                "fingerprint_keys": list(fingerprint.keys()) if isinstance(fingerprint, dict) else [],
                "modalities": detected_modalities,
            }
            trace.decision("Pipeline", f"modalities: {detected_modalities}")

            # -- Step 2: Memory injection --
            if not self._disable_memory:
                soul_prompt = self.memory.soul_text or ""
                if scope_keys:
                    mem_parts = []
                    for scope in scope_keys:
                        if self.memory.memory:
                            section = self.memory.memory.get_prompt_section(scope=scope)
                            if section:
                                mem_parts.append(section)
                    memory_prompt = "\n".join(mem_parts)
                    tool_notes = self.memory.get_tool_notes(scope_keys=scope_keys)
                    skill_prompt = self.memory.get_skill(domain, task_type, scope_keys=scope_keys)
                else:
                    memory_prompt = self.memory.memory.get_prompt_section(domain, task_type) if self.memory.memory else ""
                    tool_notes = self.memory.get_tool_notes(domain=domain, task=task_type)
                    skill_prompt = self.memory.get_skill(domain, task_type)

                envelope.task.metadata["soul_prompt"] = soul_prompt
                envelope.task.metadata["memory_l2"] = memory_prompt
                envelope.task.metadata["memory_l1"] = ""
                envelope.task.metadata["tool_notes"] = tool_notes
                envelope.task.metadata["skill_prompt"] = skill_prompt

            envelope.task.metadata["domain"] = domain
            envelope.task.metadata["modality"] = (
                detected_modalities[0] if len(detected_modalities) == 1 else "multimodal"
            )
            envelope.task.metadata["indicator_output_keys"] = self._indicator_output_keys(
                scope_keys,
                domain,
                task_type,
                ground_truth=ground_truth,
            )

            # -- Training mode injection --
            if mode == "train":
                envelope.task.metadata["_mode"] = "train"
                envelope.task.metadata["_defer_learning_record"] = defer_learning_record
                if ground_truth is not None:
                    envelope.task.metadata["_ground_truth"] = ground_truth
                dropout_scope = self._tool_dropout_scope_key(scope_keys, domain, task_type)
                envelope.task.metadata["_tool_dropout_scope"] = dropout_scope
                envelope.task.metadata["_tool_dropout_counts"] = (
                    self.memory.tools.get_scope_usage(dropout_scope)
                    if not self._disable_memory and self.memory.tools and dropout_scope
                    else {}
                )
                logger.info("[Pipeline] Training mode: GT injected, evaluate_against_gt enabled")

            # -- Step 3: Single Executor (ReAct loop, all modalities) --
            t_exec = time.time()
            trace.stage("Execute", f"single agent, modalities: {detected_modalities}")
            logger.info(f"[Pipeline] Running single Executor agent (modalities={detected_modalities})")

            executor = self._create_executor()
            branch_envelope = envelope.with_agent("executor")
            executor_result = await executor.run(branch_envelope)
            result = executor_result["result"]

            raw_answer = result.get("answer") or result.get("prediction")
            answer_type = result.get("answer_type", "")
            confidence = result.get("confidence")
            reasoning = result.get("reasoning", "")
            execution_log = result.get("execution_log", [])

            if mode == "train" and answer_type == "learning_summary":
                raw_answer = self._normalize_learning_summary_answer(
                    raw_answer,
                    execution_log,
                )

            if mode == "train" and not self._disable_memory:
                self._increment_tool_dropout_usage(scope_keys, domain, task_type, execution_log)

            prediction_source = raw_answer
            if mode == "train" and answer_type == "learning_summary":
                prediction_source = self._select_winner_prediction_from_execution_log(
                    execution_log,
                    raw_answer,
                    raw_answer=raw_answer,
                    allow_missing_metrics=True,
                )
            elif mode == "train" and defer_learning_record:
                prediction_source = self._select_winner_prediction_from_execution_log(
                    execution_log,
                    raw_answer,
                )

            prediction = self._extract_prediction(prediction_source, task_type, horizon)

            # -- Training mode: save learning insights to memory --
            if mode == "train" and not self._disable_memory and not defer_learning_record:
                await self._save_learning_outcome(
                    answer_type=answer_type,
                    raw_answer=raw_answer,
                    prediction=prediction,
                    execution_log=execution_log,
                    task_type=task_type,
                    description=description,
                    domain=domain,
                    scope_keys=scope_keys,
                    reasoning=reasoning,
                    fingerprint=fingerprint,
                    profiling_results=profiling_results,
                    horizon=horizon,
                    source_metadata=source_metadata or {},
                    trace_id=trace_id,
                )

            execute_ms = int((time.time() - t_exec) * 1000)
            timeline["execute"] = {
                "elapsed_ms": execute_ms,
                "tools_run": len(execution_log),
                "tools_passed": sum(1 for e in execution_log if e.get("success")),
                "has_prediction": prediction is not None,
                "answer_type": answer_type,
                "confidence": confidence,
            }
            trace.decision("Pipeline", f"Executor done: answer_type={answer_type}, confidence={confidence}")

            # -- Step 4: Post-process (pure code quality gate) --
            t_post = time.time()
            trace.stage("PostProcess", "deterministic quality gate")

            gate_result = self._quality_gate.check(
                prediction,
                executor.evidence_log,
                constraints or [],
            )

            corrections: list[dict[str, Any]] = []
            numeric_payload_ok = self._has_numeric_postprocess_payload(task_type, prediction)
            if prediction is not None and task_type in {"forecast", "indicator"} and not numeric_payload_ok:
                gate_result = QualityGateResult(
                    accepted=False,
                    rejections=list(gate_result.rejections) + [
                        f"{task_type} task returned non-numeric prediction payload",
                    ],
                )

            if prediction is not None and task_type in {"forecast", "indicator"} and numeric_payload_ok:
                bound_result = self._check_bounds(prediction, constraints or [])
                if not bound_result["passed"]:
                    corrected, correction = self._apply_bound_correction(prediction, constraints or [])
                    if correction:
                        corrections.append(correction)
                        prediction = corrected
                        logger.info(f"[Pipeline] Applied bound correction: {correction}")

                consistency = self._check_consistency(prediction, data)
                if not consistency["passed"]:
                    logger.warning(f"[Pipeline] Consistency check failed: {consistency['details']}")

            gate_passed = gate_result.accepted
            postprocess_ms = int((time.time() - t_post) * 1000)
            timeline["postprocess"] = {
                "elapsed_ms": postprocess_ms,
                "gate_accepted": gate_passed,
                "rejections": gate_result.rejections,
                "corrections": len(corrections),
            }

            # -- Step 5: Deliver --
            total_ms = int((time.time() - t_start) * 1000)
            trace.stage("Deliver", f"total: {total_ms}ms")
            logger.info(f"[Pipeline] Delivering result ({total_ms}ms total)")

            all_tool_names = [e.get("tool", "") for e in execution_log]

            timeline["deliver"] = {"elapsed_ms": 0}

            evidence_summary = executor.evidence_log.summary()
            result_message = ResultMessage(
                sender=AgentRole.SYSTEM,
                prediction=prediction,
                evidence_summary=evidence_summary,
                confidence=confidence,
                metadata={
                    "trace_id": trace_id,
                    "timeline": timeline,
                    "total_ms": total_ms,
                    "gate_accepted": gate_passed,
                    "gate_rejections": list(gate_result.rejections),
                    "modalities": detected_modalities,
                    "answer_type": answer_type,
                },
            )

            logger.info(
                f"[Pipeline] Run {trace_id} complete in {total_ms}ms "
                f"(modalities={detected_modalities}, "
                f"gate={'PASS' if gate_passed else 'FAIL'}, "
                f"answer_type={answer_type})"
            )
            trace.end_run("completed", {
                "trace_id": trace_id,
                "total_ms": total_ms,
                "gate_accepted": gate_passed,
                "gate_rejections": list(gate_result.rejections),
                "answer_type": answer_type,
                "confidence": confidence,
            })

            return {
                "prediction": prediction,
                "answer": prediction,
                "raw_answer": raw_answer,
                "answer_type": answer_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "trace_id": trace_id,
                "timeline": timeline,
                "total_ms": total_ms,
                "gate_accepted": gate_passed,
                "gate_rejections": list(gate_result.rejections),
                "evidence_summary": evidence_summary,
                "result_message": result_message,
                "modalities": detected_modalities,
                "execution_log": execution_log,
                "corrections": corrections,
                "tool_chain": all_tool_names,
            }
        except asyncio.CancelledError as e:
            total_ms = int((time.time() - t_start) * 1000)
            trace.end_run("cancelled", {
                "trace_id": trace_id,
                "total_ms": total_ms,
                "error_type": "CancelledError",
                "failure_reason": "cancelled",
                "error": "pipeline run was cancelled",
            })
            setattr(e, "trace_id", trace_id)
            setattr(e, "total_ms", total_ms)
            setattr(e, "error_type", "CancelledError")
            setattr(e, "failure_reason", "cancelled")
            raise
        except Exception as e:
            total_ms = int((time.time() - t_start) * 1000)
            error_type = type(e).__name__
            failure_reason = "max_steps_exhausted" if isinstance(e, MaxStepsExhaustedError) else error_type
            trace.end_run("failed", {
                "trace_id": trace_id,
                "total_ms": total_ms,
                "error_type": error_type,
                "failure_reason": failure_reason,
                "error": str(e),
            })
            setattr(e, "trace_id", trace_id)
            setattr(e, "total_ms", total_ms)
            setattr(e, "error_type", error_type)
            setattr(e, "failure_reason", failure_reason)
            raise

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
            await self._run_profiling_tool(
                tool_name,
                data,
                profiling_results,
                required=tool_name in _REQUIRED_PREPROCESS_TOOLS,
            )

        if "text" in modalities and text_data:
            text_blob = "\n\n".join(
                (
                    item.get("text", item.get("content", ""))
                    if isinstance(item, dict)
                    else str(item)
                )
                for item in text_data
            ).strip()
            for tool_name in _TEXT_PROFILING_TOOLS:
                await self._run_profiling_tool(
                    tool_name, data, profiling_results,
                    text=text_blob, llm=self._text_backends or None,
                    required=False,
                )

        if "image" in modalities and images:
            for tool_name in _IMAGE_PROFILING_TOOLS:
                await self._run_profiling_tool(
                    tool_name, data, profiling_results,
                    images=images,
                    required=False,
                )

        fp_dict = fingerprint
        if hasattr(fingerprint, "model_dump"):
            fp_dict = fingerprint.model_dump(mode="python")
        elif not isinstance(fingerprint, dict):
            fp_dict = {}

        envelope.update_environment(modalities=modalities)
        profiling_results["modality_report"] = modality_report
        profiling_results["fingerprint_snapshot"] = fp_dict

        return {
            "fingerprint": fp_dict,
            "modalities": modalities,
            "profiling_results": profiling_results,
        }

    async def _run_profiling_tool(
        self,
        tool_name: str,
        data: Any,
        results: dict[str, Any],
        required: bool = False,
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
            if required and not r["success"]:
                raise RuntimeError(
                    f"Required preprocessing tool '{tool_name}' failed: {r.get('error')}"
                )
            results[tool_name] = r["data"] if r["success"] else r.get("error")
            return
        try:
            tool = get_tool_instance(tool_name)
            result = await tool.execute(data, **extra_params)
            if required and not result.success:
                raise RuntimeError(
                    f"Required preprocessing tool '{tool_name}' failed: {result.error}"
                )
            results[tool_name] = result.data if result.success else result.error
            trace.tool_exec("preprocess", tool_name, extra_params, result)
        except KeyError:
            if required:
                raise RuntimeError(
                    f"Required preprocessing tool '{tool_name}' is not registered"
                )
            logger.debug(f"[Preprocess] Tool '{tool_name}' not registered, skipping")
        except Exception as e:
            if required:
                raise RuntimeError(
                    f"Required preprocessing tool '{tool_name}' errored: {e}"
                ) from e
            logger.warning(f"[Preprocess] Tool '{tool_name}' error: {e}")

    # -- Training: save learning insights --------------------------------

    @staticmethod
    def _tool_dropout_scope_key(
        scope_keys: list[str] | None,
        domain: str,
        task_type: str,
    ) -> str:
        if scope_keys:
            for scope in scope_keys:
                if str(scope).strip():
                    return str(scope).strip()
        if domain and task_type:
            return f"{domain}_{task_type}"
        return ""

    @staticmethod
    def _indicator_output_keys(
        scope_keys: list[str] | None,
        domain: str,
        task_type: str,
        ground_truth: Any = None,
    ) -> list[str]:
        if task_type != "indicator":
            return []
        if isinstance(ground_truth, dict):
            keys = [str(key).strip() for key in ground_truth.keys() if str(key).strip()]
            if keys:
                return keys
        for scope in scope_keys or []:
            scope_name = str(scope).strip()
            if scope_name.endswith("_indicator_macd"):
                return ["macd"]
            if scope_name.endswith("_indicator_upper_bb"):
                return ["upper_bb"]
        if domain == "weather":
            return ["max", "min", "diff"]
        return []

    @staticmethod
    def _collect_dropout_tool_usage(execution_log: list[dict[str, Any]]) -> list[str]:
        tools: list[str] = []
        for entry in execution_log:
            tool_name = str(entry.get("tool") or "").strip()
            if not tool_name:
                continue
            if tool_name == "spawn_subagent":
                output = entry.get("output") or []
                for item in output:
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    if isinstance(result, dict):
                        tools.extend(
                            PipelineRunner._collect_dropout_tool_usage(
                                result.get("execution_log") or []
                            )
                        )
                continue
            if tool_name in _FRAMEWORK_TOOLS:
                continue
            tools.append(tool_name)
        return tools

    def _increment_tool_dropout_usage(
        self,
        scope_keys: list[str] | None,
        domain: str,
        task_type: str,
        execution_log: list[dict[str, Any]],
    ) -> None:
        if not self.memory.tools:
            return
        used_tools = self._collect_dropout_tool_usage(execution_log)
        if not used_tools:
            return
        target_scopes = scope_keys or [self._tool_dropout_scope_key(scope_keys, domain, task_type)]
        for scope in target_scopes:
            scope_name = str(scope).strip()
            if scope_name:
                self.memory.tools.increment_scope_usage(scope_name, used_tools)

    async def _save_learning_outcome(
        self,
        answer_type: str,
        raw_answer: Any,
        prediction: Any,
        execution_log: list[dict[str, Any]],
        task_type: str,
        description: str,
        domain: str,
        scope_keys: list[str] | None,
        reasoning: str,
        fingerprint: dict[str, Any] | Any,
        profiling_results: dict[str, Any],
        horizon: int | None,
        source_metadata: dict[str, Any] | None,
        trace_id: str,
    ) -> None:
        """Persist training-mode learning outcomes into hierarchical memory (L1).

        All information — tool chain, metrics from every evaluate_against_gt
        call, agent analysis — is written to a single L1 note.  L2/L3
        are populated exclusively by the DistillationAgent after the note is
        committed, using background single-flight rebuild workers.
        """
        tool_chain = [e.get("tool", "") for e in execution_log if e.get("tool")]
        candidate_traces = self._extract_candidate_traces(execution_log)

        all_evals: list[dict[str, Any]] = []
        eval_gt: Any = None
        for entry in execution_log:
            tool_name = entry.get("tool")
            if tool_name == "evaluate_against_gt":
                obs = entry.get("observation", {})
                if isinstance(obs, dict):
                    all_evals.append(obs.get("metrics", {}))
                    eval_gt = obs.get("ground_truth")
            elif tool_name in {"evaluate_batch_against_gt", "batch_evaluate_against_gt"}:
                obs = entry.get("observation", {})
                if isinstance(obs, dict):
                    eval_gt = obs.get("ground_truth")
                    if isinstance(obs.get("metrics"), dict):
                        all_evals.append(obs.get("metrics", {}))
                    for item in obs.get("results", []):
                        if isinstance(item, dict):
                            all_evals.append(item.get("metrics", {}))

        best_metrics = self._select_best_metrics(all_evals)
        if not best_metrics:
            logger.warning(
                f"[Pipeline] Skip memory record for {scope_keys or [f'{domain}_{task_type}']}: "
                "training finished without GT evaluation metrics"
            )
            return
        analysis = ""
        if answer_type == "learning_summary" and isinstance(raw_answer, dict):
            insight = raw_answer.get("insight", "")
            if not isinstance(insight, str):
                insight = json.dumps(insight, ensure_ascii=False) if insight else ""
            rec = raw_answer.get("recommendation", "")
            if not isinstance(rec, str):
                rec = json.dumps(rec, ensure_ascii=False) if rec else ""
            analysis = f"{insight}\n{rec}".strip()

        sample_stats = self._build_sample_stats(fingerprint, profiling_results, horizon)
        winner_trace, _winner_source = self._resolve_learning_summary_winner_trace(
            raw_answer if answer_type == "learning_summary" else None,
            candidate_traces,
            best_metrics,
        )
        winner_tool_chain = list(winner_trace.get("tool_chain") or []) if winner_trace else self._select_winner_tool_chain(candidate_traces, best_metrics)
        fallback_summary = analysis or reasoning or str(raw_answer)[:3000]
        structured_analysis, summary = await generate_learning_analysis(
            scope_key=(scope_keys[0] if scope_keys else f"{domain}_{task_type}"),
            task_type=task_type,
            domain=domain,
            task_description=description,
            sample_fingerprint=sample_stats,
            candidate_traces=candidate_traces,
            best_metrics=best_metrics,
            selected_winner_trace=winner_trace,
            winner_correct=(
                self._learning_outcome_correctness(dict(winner_trace.get("metrics") or {}))
                if winner_trace and isinstance(winner_trace.get("metrics"), dict) and winner_trace.get("metrics")
                else None
            ),
            fallback_summary=fallback_summary,
        )

        recorded_prediction = None
        if winner_trace is not None:
            recorded_prediction = winner_trace.get("prediction")
        if recorded_prediction is None:
            recorded_prediction = prediction
        if recorded_prediction is None:
            recorded_prediction = raw_answer

        recorded_metrics = dict(winner_trace.get("metrics") or {}) if winner_trace else {}
        recorded_correct = self._learning_outcome_correctness(recorded_metrics)

        await self.memory.record(
            domain=domain,
            task=task_type,
            scope_keys=scope_keys,
            trace_id=trace_id,
            tool_chain=winner_tool_chain or tool_chain,
            pipeline_summary=summary,
            data_fingerprint=sample_stats,
            predicted=str(recorded_prediction) if recorded_prediction is not None else "",
            metrics=recorded_metrics,
            correct=recorded_correct,
            description=description,
            all_eval_metrics=all_evals,
            candidate_traces=candidate_traces,
            analysis_structured=structured_analysis,
            source_metadata=source_metadata or {},
        )

    @staticmethod
    def _build_sample_stats(
        fingerprint: dict[str, Any] | Any,
        profiling_results: dict[str, Any],
        horizon: int | None,
    ) -> str:
        fp = fingerprint if isinstance(fingerprint, dict) else {}
        parts: list[str] = []
        if horizon is not None:
            parts.append(f"horizon={horizon}")
        for key in ("length", "n_points", "mean", "std", "min", "max"):
            if key in fp:
                parts.append(f"{key}={fp[key]}")
        for key in ("trend", "dominant_period", "seasonal_period"):
            if key in fp:
                parts.append(f"{key}={fp[key]}")
        for tool_name in ("detect_seasonality", "autocorrelation", "stationarity_test"):
            val = profiling_results.get(tool_name)
            if val is None:
                continue
            text = str(val)
            if text:
                parts.append(f"{tool_name}={text[:120]}")
        return "; ".join(parts[:8])[:500]

    @staticmethod
    def _extract_candidate_traces(execution_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        batch_results: list[dict[str, Any]] = []
        single_results: list[dict[str, Any]] = []
        top_level_traces: list[dict[str, Any]] = []
        for entry_idx, entry in enumerate(execution_log):
            tool_name = str(entry.get("tool") or "").strip()
            if tool_name == "spawn_subagent":
                output = entry.get("output") or []
                for item in output:
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    if not isinstance(result, dict):
                        continue
                    sub_execution_log = result.get("execution_log", [])
                    tool_chain = [e.get("tool", "") for e in sub_execution_log if e.get("tool")]
                    raw_name = result.get("candidate_name") or result.get("tools_hint") or result.get("sub_task_description")
                    derived_name = (
                        PipelineRunner._primary_candidate_tool_name(tool_chain)
                        or PipelineRunner._tool_name_from_hint(raw_name)
                        or str(raw_name or "").strip()
                        or f"candidate_{len(traces)}"
                    )
                    steps = PipelineRunner._compact_candidate_steps(sub_execution_log)
                    if not steps:
                        hinted_tool = PipelineRunner._tool_name_from_hint(
                            result.get("tools_hint") or result.get("candidate_name") or raw_name or ""
                        )
                        synthetic_step: dict[str, str] = {}
                        if hinted_tool:
                            synthetic_step["tool"] = hinted_tool
                        reasoning = str(result.get("reasoning") or "").strip()
                        if reasoning:
                            synthetic_step["thought"] = reasoning
                        prediction = result.get("answer", result.get("prediction"))
                        if prediction is not None:
                            try:
                                synthetic_step["observation"] = json.dumps(
                                    {"answer": prediction}, ensure_ascii=False, default=str
                                )
                            except TypeError:
                                synthetic_step["observation"] = str(prediction)
                        if synthetic_step:
                            steps = [synthetic_step]
                    traces.append({
                        "name": derived_name,
                        "candidate_key": PipelineRunner._normalise_candidate_key(derived_name, tool_chain, len(traces)),
                        "tool_chain": tool_chain,
                        "prediction": result.get("answer", result.get("prediction")),
                        "reasoning": str(result.get("reasoning") or "").strip(),
                        "steps": steps,
                        "entry_index": entry_idx,
                    })
                continue
            if tool_name in {"evaluate_batch_against_gt", "batch_evaluate_against_gt"}:
                obs = entry.get("observation") or {}
                if isinstance(obs, dict) and isinstance(obs.get("results"), list):
                    batch_results = obs["results"]
                continue
            if tool_name == "evaluate_against_gt":
                obs = entry.get("observation") or {}
                args = entry.get("args") or {}
                if isinstance(obs, dict) and isinstance(obs.get("metrics"), dict) and obs.get("metrics"):
                    single_results.append(
                        {
                            "metrics": obs.get("metrics", {}),
                            "prediction": args.get("prediction"),
                            "entry_index": entry_idx,
                        }
                    )
                continue
            if tool_name in {
                "bound_check",
                "consistency_check",
                "spawn_subagent",
                "evaluate_against_gt",
                "evaluate_batch_against_gt",
                "batch_evaluate_against_gt",
            }:
                continue
            steps = PipelineRunner._compact_candidate_steps([entry])
            top_level_traces.append({
                "name": tool_name,
                "candidate_key": PipelineRunner._normalise_candidate_key(tool_name, [tool_name], len(top_level_traces)),
                "tool_chain": [tool_name],
                "prediction": PipelineRunner._prediction_from_observation(entry.get("observation")),
                "steps": steps,
                "entry_index": entry_idx,
            })

        if not traces and top_level_traces:
            traces = top_level_traces

        trace_by_key = {
            trace.get("candidate_key"): trace for trace in traces if trace.get("candidate_key")
        }
        unmatched_traces = [trace for trace in traces if trace.get("metrics") is None]

        for idx, res in enumerate(batch_results):
            if not isinstance(res, dict):
                continue
            res_name = str(res.get("name") or "").strip()
            res_key = PipelineRunner._normalise_candidate_key(res_name, [], idx)

            matched = trace_by_key.get(res_key)
            if matched is None and idx < len(unmatched_traces):
                matched = unmatched_traces[idx]
            if matched is None:
                continue

            matched["metrics"] = res.get("metrics", {})
            if res.get("prediction") is not None:
                # Batch GT evaluation already serializes predictions into a
                # task-typed payload. Prefer that normalized payload over the
                # raw sub-agent answer, which may still be a JSON string.
                matched["prediction"] = res.get("prediction")
            existing_name = str(matched.get("name") or matched.get("candidate_key") or "").strip()
            if (
                res_name
                and not res_name.startswith("approach_")
                and PipelineRunner._trace_name_is_generic(existing_name)
            ):
                matched["name"] = PipelineRunner._normalise_candidate_key(
                    res_name, matched.get("tool_chain", []), idx,
                )
                matched["candidate_key"] = PipelineRunner._normalise_candidate_key(
                    res_name, matched.get("tool_chain", []), idx,
                )

        for res in single_results:
            matched = None
            res_prediction = res.get("prediction")
            eval_entry_index = res.get("entry_index")
            if res_prediction is not None:
                for trace in unmatched_traces:
                    if trace.get("metrics") is not None:
                        continue
                    if PipelineRunner._predictions_match(trace.get("prediction"), res_prediction):
                        matched = trace
                        break
            if matched is None:
                prior_candidates = [
                    trace
                    for trace in unmatched_traces
                    if trace.get("metrics") is None
                    and trace.get("prediction") is not None
                    and isinstance(trace.get("entry_index"), int)
                    and (
                        not isinstance(eval_entry_index, int)
                        or int(trace["entry_index"]) < int(eval_entry_index)
                    )
                ]
                if prior_candidates:
                    matched = max(prior_candidates, key=lambda trace: int(trace.get("entry_index", -1)))
            if matched is None:
                for trace in unmatched_traces:
                    if trace.get("metrics") is None:
                        matched = trace
                        break
            if matched is None:
                continue
            matched["metrics"] = res.get("metrics", {})
        return traces

    @staticmethod
    def _predictions_match(left: Any, right: Any) -> bool:
        left = PipelineRunner._parse_json_like_payload(left)
        right = PipelineRunner._parse_json_like_payload(right)

        if isinstance(left, dict) and isinstance(right, dict):
            if set(left.keys()) != set(right.keys()):
                return False
            return all(PipelineRunner._predictions_match(left[key], right[key]) for key in left.keys())

        if isinstance(left, (list, tuple, np.ndarray)) and isinstance(right, (list, tuple, np.ndarray)):
            try:
                left_arr = np.asarray(left, dtype=float)
                right_arr = np.asarray(right, dtype=float)
            except (TypeError, ValueError):
                return list(left) == list(right)
            if left_arr.shape != right_arr.shape:
                return False
            return bool(np.allclose(left_arr, right_arr, atol=1e-3, rtol=1e-6))

        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False


    @staticmethod
    def _prediction_from_observation(observation: Any) -> Any:
        if isinstance(observation, dict):
            for key in ("forecast", "point_forecast", "answer", "prediction"):
                value = observation.get(key)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _compact_candidate_steps(execution_log: list[dict[str, Any]]) -> list[dict[str, str]]:
        steps: list[dict[str, str]] = []
        for entry in execution_log:
            tool = str(entry.get("tool") or "").strip()
            if not tool or tool in {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt", "spawn_subagent"}:
                continue
            step: dict[str, str] = {"tool": tool}

            thought = str(entry.get("thought") or "").strip()
            if thought:
                step["thought"] = thought

            observation = entry.get("observation")
            if observation is None and entry.get("output") is not None:
                observation = entry.get("output")
            if observation is not None:
                try:
                    obs_text = json.dumps(observation, ensure_ascii=False, default=str)
                except TypeError:
                    obs_text = str(observation)
                if obs_text:
                    step["observation"] = obs_text

            steps.append(step)
        return steps

    @staticmethod
    def _select_winner_tool_chain(candidate_traces: list[dict[str, Any]], best_metrics: dict[str, Any]) -> list[str]:
        winner_trace = PipelineRunner._select_winner_trace(candidate_traces, best_metrics)
        if winner_trace and winner_trace.get("tool_chain"):
            return winner_trace["tool_chain"]
        for trace in candidate_traces:
            if trace.get("tool_chain"):
                return trace["tool_chain"]
        return []

    @staticmethod
    def _metrics_score(metrics: dict[str, Any]) -> tuple[int, float]:
        if isinstance(metrics.get("accuracy"), (int, float)):
            return (0, -float(metrics["accuracy"]))
        if isinstance(metrics.get("correct"), bool):
            return (0, 0.0 if metrics["correct"] else 1.0)
        for key in ("mae", "avg_mae", "rmse", "mse", "max_mae"):
            if isinstance(metrics.get(key), (int, float)):
                return (1, float(metrics[key]))
        return (9, 0.0)

    @staticmethod
    def _select_winner_trace(
        candidate_traces: list[dict[str, Any]],
        best_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        valid_traces = [
            trace
            for trace in candidate_traces
            if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
        ]
        if not valid_traces:
            return None

        if isinstance(best_metrics, dict) and best_metrics:
            target_score = PipelineRunner._metrics_score(best_metrics)
            scored_matches = [
                trace for trace in valid_traces
                if PipelineRunner._metrics_score(trace["metrics"]) == target_score
            ]
            if scored_matches:
                return min(scored_matches, key=lambda trace: PipelineRunner._metrics_score(trace["metrics"]))

        return min(valid_traces, key=lambda trace: PipelineRunner._metrics_score(trace["metrics"]))

    @staticmethod
    def _normalise_candidate_key(name: str, tool_chain: list[str], index: int) -> str:
        clean_name = str(name or "").strip()
        hinted_name = PipelineRunner._tool_name_from_hint(clean_name)
        if hinted_name:
            return hinted_name
        if clean_name and re.fullmatch(r"(slot|candidate|approach)_\d+", clean_name):
            clean_name = ""
        if clean_name and not clean_name.startswith("approach_"):
            return clean_name
        chain = " → ".join(t for t in tool_chain if t)
        if chain:
            return chain
        return f"candidate_{index}"

    @staticmethod
    def _primary_candidate_tool_name(tool_chain: list[str]) -> str:
        for tool in tool_chain:
            if tool in {
                "bound_check",
                "consistency_check",
                "evaluate_against_gt",
                "evaluate_batch_against_gt",
                "batch_evaluate_against_gt",
                "spawn_subagent",
            }:
                continue
            if tool:
                return tool
        return ""

    @classmethod
    def _display_trace_name(cls, trace: dict[str, Any], fallback: str = "candidate") -> str:
        raw_name = str(trace.get("name") or trace.get("candidate_key") or "").strip()
        tool_chain = list(trace.get("tool_chain") or [])
        chain = cls._tool_chain_summary(tool_chain)
        primary = cls._primary_candidate_tool_name(tool_chain)
        if raw_name and not re.fullmatch(r"(slot|candidate|approach)_\d+", raw_name):
            return raw_name
        if chain:
            return chain
        if primary:
            return primary
        if raw_name:
            return raw_name
        return fallback

    @staticmethod
    def _trace_name_is_generic(name: Any) -> bool:
        text = str(name or "").strip().lower()
        if not text:
            return True
        return bool(re.fullmatch(r"(slot|candidate|approach|top_level)_\d+", text)) or bool(
            re.fullmatch(r"(slot|candidate|approach|top_level)::\d+", text)
        ) or text in {"candidate", "top_level"}

    @staticmethod
    def _tool_name_from_hint(text: Any) -> str:
        if isinstance(text, (list, tuple, set)):
            tokens: list[str] = []
            for item in text:
                if item is None:
                    continue
                tokens.extend(re.split(r"[\s,;:/()\[\]\n]+", str(item)))
        else:
            raw = str(text or "").strip()
            if not raw:
                return ""
            tokens = re.split(r"[\s,;:/()\[\]\n]+", raw)
        if not tokens:
            return ""
        for token in tokens:
            token = token.strip()
            if token.endswith(("_forecast", "_indicator", "_trend", "_classification", "_qa", "_correlation")):
                return token
        return ""

    @staticmethod
    def _winner_aliases(winner_trace: dict[str, Any]) -> list[str]:
        aliases: list[str] = []
        for raw in (
            winner_trace.get("name"),
            winner_trace.get("candidate_key"),
            PipelineRunner._primary_candidate_tool_name(list(winner_trace.get("tool_chain") or [])),
        ):
            text = str(raw or "").strip().lower()
            if not text:
                continue
            aliases.append(text)
            short = re.sub(
                r"_(forecast|qa|classification|indicator|correlation|trend)$",
                "",
                text,
            ).strip()
            if short and short != text:
                aliases.append(short)
        seen: set[str] = set()
        ordered: list[str] = []
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            ordered.append(alias)
        return ordered

    @staticmethod
    def _learning_summary_mentions_winner(
        insight: str,
        recommendation: str,
        winner_trace: dict[str, Any] | None,
    ) -> bool:
        if not winner_trace:
            return False
        combined = f"{insight}\n{recommendation}".strip().lower()
        if not combined:
            return False
        for alias in PipelineRunner._winner_aliases(winner_trace):
            if alias and alias in combined:
                return True
        return False

    @staticmethod
    def _format_metric_summary(metrics: dict[str, Any]) -> str:
        if not isinstance(metrics, dict):
            return ""
        ordered_keys = (
            "accuracy",
            "correct",
            "mae",
            "avg_mae",
            "rmse",
            "mse",
            "max_mae",
            "mape",
        )
        parts: list[str] = []
        for key in ordered_keys:
            value = metrics.get(key)
            if isinstance(value, bool):
                parts.append(f"{key}={str(value).lower()}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.3f}")
        return ", ".join(parts[:4])

    @staticmethod
    def _preview_prediction(prediction: Any, limit: int = 120) -> str:
        if prediction is None:
            return ""
        try:
            text = json.dumps(prediction, ensure_ascii=False, default=str)
        except TypeError:
            text = str(prediction)
        return text if len(text) <= limit else text[:limit] + "..."

    @staticmethod
    def _tool_chain_summary(tool_chain: list[str] | Any) -> str:
        if not isinstance(tool_chain, list):
            return ""
        cleaned = [str(tool).strip() for tool in tool_chain if str(tool).strip()]
        if not cleaned:
            return ""
        compact: list[str] = []
        prev = ""
        repeat = 0
        for tool in cleaned:
            if tool == prev:
                repeat += 1
                continue
            if prev:
                compact.append(f"{prev} x{repeat + 1}" if repeat else prev)
            prev = tool
            repeat = 0
        if prev:
            compact.append(f"{prev} x{repeat + 1}" if repeat else prev)
        return " -> ".join(compact[:4])

    @staticmethod
    def _sanitize_reasoning_clause(text: str) -> str:
        cleaned = str(text).strip()
        if not cleaned:
            return ""

        cleaned = re.sub(r"^\s*\d+[\.\)\-:]\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\breason:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\baction:\s*.*$", "", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = normalize_learning_clause(cleaned)

        framework_patterns = (
            r"(?i)\b(spawn_subagent|evaluate_against_gt|evaluate_batch_against_gt|batch_evaluate_against_gt)\b",
            r"(?i)\b(execution context|assigned(?: exploration)? path|assigned branch|branch(?: instructions?)?|training mode|learning-mode|ground truth)\b.*\b(require|requires|required|prioritiz|mandat|must|before finishing)\b",
            r"(?i)\b(required|prioritized?|mandated)\b[^.]{0,80}\bpath\b",
            r"(?i)\bmust use at least one tool\b",
        )
        if any(re.search(pattern, cleaned) for pattern in framework_patterns):
            return ""

        banned_markers = (
            "assigned path",
            "assigned branch",
            "execution context",
            "branch requires",
            "branch explicitly requires",
            "branch explicitly prioritized",
            "branch explicitly prioritizes",
            "branch goal",
            "assigned exploration path",
            "using at least one tool from",
            "calling at least one tool from",
            "before finishing",
            "final turn",
            "ground truth",
            "gt-selected",
            "evaluate_",
            "spawn_subagent",
            "task-valid candidate",
            "training mode",
            "learning-mode",
            "mandated path",
            "comply with",
            "framework",
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;|-")
        lower = cleaned.lower()
        if any(marker in lower for marker in banned_markers):
            return ""
        if not cleaned or re.fullmatch(r"\d+", cleaned):
            return ""
        if len(cleaned) < 12:
            return ""
        return cleaned

    @classmethod
    def _reasoning_snippets(cls, trace: dict[str, Any]) -> list[str]:
        raw_items: list[str] = []
        reasoning = str(trace.get("reasoning") or "").strip()
        if reasoning:
            raw_items.append(reasoning)
        steps = trace.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                thought = str(step.get("thought") or "").strip()
                if thought:
                    raw_items.append(thought)

        snippets: list[str] = []
        seen: set[str] = set()
        for raw in raw_items:
            parts = re.split(r"(?<=[.!?])\s+|[;\n]+", raw)
            for part in parts:
                cleaned = cls._sanitize_reasoning_clause(part)
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                snippets.append(cleaned)
        return snippets

    @classmethod
    def _compact_candidate_reasoning(cls, trace: dict[str, Any], limit: int = 2) -> str:
        snippets: list[str] = []
        for snippet in cls._reasoning_snippets(trace):
            if len(snippet) > 160:
                snippet = snippet[:160].rstrip(" ,;:") + "..."
            snippet = snippet[0].upper() + snippet[1:] if snippet else snippet
            if not snippet.endswith("."):
                snippet += "."
            snippets.append(snippet)
            if len(snippets) >= limit:
                break
        return " ".join(snippets)

    @classmethod
    def _candidate_comparison_summary(cls, trace: dict[str, Any]) -> str:
        name = cls._display_trace_name(trace)
        chain = cls._tool_chain_summary(list(trace.get("tool_chain") or []))
        metric_text = cls._format_metric_summary(dict(trace.get("metrics") or {}))
        reasoning = cls._compact_candidate_reasoning(trace, limit=1)
        parts = [name]
        if chain and chain != name:
            parts.append(f"used `{chain}`")
        if metric_text:
            parts.append(f"scored {metric_text}")
        if reasoning:
            parts.append(reasoning.rstrip("."))
        return "; ".join(parts).strip()

    @classmethod
    def _best_alternative_trace(
        cls,
        winner_trace: dict[str, Any],
        candidate_traces: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        alternatives = [
            trace for trace in candidate_traces
            if isinstance(trace, dict) and trace is not winner_trace
            and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
        ]
        if not alternatives:
            return None
        return min(alternatives, key=lambda trace: cls._metrics_score(trace.get("metrics") or {}))

    @classmethod
    def _build_winner_aligned_learning_summary(
        cls,
        winner_trace: dict[str, Any] | None,
        candidate_traces: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Build a grounded single-sample learning summary.

        New definition:
            If GT-backed metrics show the selected best explored candidate is
            still incorrect overall, the summary must be failure-first rather
            than presenting that candidate as a successful "winner".
        """
        if not winner_trace:
            return "", ""

        winner_name = cls._display_trace_name(winner_trace)
        winner_prediction = winner_trace.get("prediction")
        prediction_text = ""
        if not isinstance(winner_prediction, (list, tuple, dict)):
            prediction_text = cls._preview_prediction(winner_prediction, limit=80)
        metric_text = cls._format_metric_summary(dict(winner_trace.get("metrics") or {}))
        winner_chain = cls._tool_chain_summary(list(winner_trace.get("tool_chain") or []))
        winner_reason = cls._compact_candidate_reasoning(winner_trace)
        compared_bits: list[str] = []
        for trace in candidate_traces:
            if trace is winner_trace or not isinstance(trace, dict):
                continue
            comparison = cls._candidate_comparison_summary(trace)
            if not comparison:
                continue
            compared_bits.append(comparison)
        correct = cls._learning_outcome_correctness(dict(winner_trace.get("metrics") or {}))
        if correct is False:
            insight = f"No correct winner on this run. Best explored candidate: {winner_name}"
            if winner_chain and winner_chain != winner_name:
                insight += f" using `{winner_chain}`"
            if metric_text:
                insight += f" with {metric_text}"
            if prediction_text:
                insight += f". Best explored output: {prediction_text}"
            if winner_reason:
                insight += f". Why it still looked plausible: {winner_reason}"
            if compared_bits:
                insight += " Alternatives checked: " + " | ".join(compared_bits[:2])
            insight = insight.strip() + "."
        else:
            insight = f"Winner for this run: {winner_name}"
            if winner_chain and winner_chain != winner_name:
                insight += f" using `{winner_chain}`"
            if metric_text:
                insight += f" with {metric_text}"
            if prediction_text:
                insight += f". Winner output: {prediction_text}"
            if winner_reason:
                insight += f". Why it won: {winner_reason}"
            if compared_bits:
                insight += " Alternatives: " + " | ".join(compared_bits[:2])
            insight = insight.strip() + "."

        winner_chain_text = f"`{winner_chain}`" if winner_chain else winner_name
        alt_trace = cls._best_alternative_trace(winner_trace, candidate_traces)
        alt_chain = cls._tool_chain_summary(list((alt_trace or {}).get("tool_chain") or []))
        alt_reason = cls._compact_candidate_reasoning(alt_trace or {}, limit=1)
        alt_chain_text = f"`{alt_chain}`" if alt_chain else "the runner-up path"
        recommendation_reason = winner_reason or (
            f"it achieved the strongest evaluation on this sample"
        )
        if correct is False:
            recommendation = (
                f"Failure pattern: do not trust {winner_chain_text} alone under a similar evidence pattern, because "
                f"{recommendation_reason}"
            )
            if alt_chain:
                recommendation += f" Recheck it against {alt_chain_text}"
                if alt_reason:
                    recommendation += f" when that alternative is motivated by {alt_reason.rstrip('.')}"
                recommendation += "."
        else:
            recommendation = (
                f"Reusable lesson: prefer {winner_chain_text} when similar inputs support the same evidence pattern, because "
                f"{recommendation_reason}"
            )
            if alt_chain:
                recommendation += f" Compare it against {alt_chain_text}"
                if alt_reason:
                    recommendation += f" when that alternative is motivated by {alt_reason.rstrip('.')}"
                recommendation += "."
        return insight, recommendation

    @staticmethod
    def _select_best_metrics(all_evals: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [m for m in all_evals if isinstance(m, dict) and m]
        if not valid:
            return {}
        return min(valid, key=PipelineRunner._metrics_score)

    @staticmethod
    def _select_winner_prediction_from_execution_log(
        execution_log: list[dict[str, Any]],
        fallback_prediction: Any,
        raw_answer: Any | None = None,
        *,
        allow_missing_metrics: bool = False,
    ) -> Any:
        candidate_traces = PipelineRunner._extract_candidate_traces(execution_log)
        evaluated_metrics = [
            trace.get("metrics")
            for trace in candidate_traces
            if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
        ]
        best_metrics = PipelineRunner._select_best_metrics(evaluated_metrics)
        winner_trace, _winner_source = PipelineRunner._resolve_learning_summary_winner_trace(
            raw_answer,
            candidate_traces,
            best_metrics,
        )
        if winner_trace and winner_trace.get("prediction") is not None:
            return winner_trace.get("prediction")
        if not best_metrics:
            if allow_missing_metrics:
                return None
            raise ValueError(
                "Deferred learning record requires evaluated candidate metrics, "
                "but none were found in execution_log."
            )
        raise ValueError(
            "Deferred learning record could not map the winning metrics back to a candidate prediction."
        )

    @classmethod
    def _normalize_learning_summary_answer(
        cls,
        raw_answer: Any,
        execution_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        base = dict(raw_answer) if isinstance(raw_answer, dict) else {}
        insight = base.get("insight", "")
        recommendation = base.get("recommendation", "")
        if not isinstance(insight, str):
            insight = json.dumps(insight, ensure_ascii=False, default=str) if insight else ""
        if not isinstance(recommendation, str):
            recommendation = json.dumps(recommendation, ensure_ascii=False, default=str) if recommendation else ""

        candidate_traces = cls._extract_candidate_traces(execution_log)
        evaluated_metrics = [
            trace.get("metrics")
            for trace in candidate_traces
            if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
        ]
        best_metrics = cls._select_best_metrics(evaluated_metrics)
        winner_trace, winner_source = cls._resolve_learning_summary_winner_trace(
            base,
            candidate_traces,
            best_metrics,
        )

        compared_candidates: list[dict[str, Any]] = []
        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            compared_candidates.append({
                "name": trace.get("name") or trace.get("candidate_key") or "",
                "tool_chain": list(trace.get("tool_chain") or []),
                "prediction": trace.get("prediction"),
                "metrics": dict(trace.get("metrics") or {}),
            })

        normalized = {
            "insight": str(insight).strip(),
            "recommendation": str(recommendation).strip(),
            "winner_candidate": "",
            "winner_prediction": None,
            "winner_tool_chain": [],
            "winner_metrics": {},
            "compared_candidates": compared_candidates,
        }
        if winner_trace:
            normalized["winner_candidate"] = str(
                winner_trace.get("name")
                or winner_trace.get("candidate_key")
                or ""
            ).strip()
            normalized["winner_prediction"] = winner_trace.get("prediction")
            normalized["winner_tool_chain"] = list(winner_trace.get("tool_chain") or [])
            normalized["winner_metrics"] = dict(winner_trace.get("metrics") or {})
            if winner_source != "explicit_summary":
                grounded_insight, grounded_recommendation = cls._build_winner_aligned_learning_summary(
                    winner_trace,
                    candidate_traces,
                )
                if grounded_insight:
                    normalized["insight"] = grounded_insight
                if grounded_recommendation:
                    normalized["recommendation"] = grounded_recommendation
        else:
            normalized["summary_kind"] = "failure_summary"

        for key, value in base.items():
            if key not in normalized:
                normalized[key] = value
        return normalized

    @classmethod
    def _resolve_learning_summary_winner_trace(
        cls,
        raw_answer: Any,
        candidate_traces: list[dict[str, Any]],
        best_metrics: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        explicit = cls._select_learning_summary_winner_trace(raw_answer, candidate_traces)
        if explicit is not None:
            return explicit, "explicit_summary"
        fallback = cls._select_winner_trace(candidate_traces, best_metrics) if best_metrics else None
        if fallback is not None:
            return fallback, "best_metrics"
        return None, "none"

    @classmethod
    def _select_learning_summary_winner_trace(
        cls,
        raw_answer: Any,
        candidate_traces: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not isinstance(raw_answer, dict) or not candidate_traces:
            return None

        winner_candidate = str(raw_answer.get("winner_candidate") or "").strip().lower()
        winner_tool_chain = [str(tool).strip() for tool in raw_answer.get("winner_tool_chain") or [] if str(tool).strip()]
        winner_prediction = raw_answer.get("winner_prediction")

        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            name = str(trace.get("name") or trace.get("candidate_key") or "").strip().lower()
            chain = [str(tool).strip() for tool in trace.get("tool_chain") or [] if str(tool).strip()]
            if winner_candidate and winner_candidate in {name, *cls._winner_aliases(trace)}:
                return trace
            if winner_tool_chain and chain == winner_tool_chain:
                return trace
            if winner_prediction is not None and cls._predictions_match(
                trace.get("prediction"),
                winner_prediction,
            ):
                return trace

        reasoning_text = str(raw_answer.get("reasoning") or "").strip()
        if not reasoning_text:
            return None
        combined = "\n".join(
            str(raw_answer.get(key) or "").strip()
            for key in ("insight", "recommendation", "reasoning")
        ).lower()
        if not combined:
            return None

        positive_markers = (
            "winner",
            "won",
            "winning",
            "best candidate",
            "best branch",
            "selected",
            "was selected",
            "preferred",
        )
        negative_markers = (
            "failed",
            "rejected",
            "underperformed",
            "worse",
            "less plausible",
            "ruled out",
        )

        best_trace: dict[str, Any] | None = None
        best_score = 0
        second_best = 0
        sentences = [part.strip().lower() for part in re.split(r"(?<=[.!?])\s+|\n+", combined) if part.strip()]
        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            aliases = cls._winner_aliases(trace)
            score = 0
            for sentence in sentences:
                if not any(alias and alias in sentence for alias in aliases):
                    continue
                score += 1
                if any(marker in sentence for marker in positive_markers):
                    score += 5
                if any(marker in sentence for marker in negative_markers):
                    score -= 4
            if score > best_score:
                second_best = best_score
                best_score = score
                best_trace = trace
            elif score > second_best:
                second_best = score

        if best_trace is not None and best_score > 0 and best_score > second_best:
            return best_trace
        return None

    @staticmethod
    def _learning_outcome_correctness(metrics: dict[str, Any]) -> bool | None:
        """Infer training-note correctness from MTBench-aligned metrics.

        New definition:
            For training notes, we only emit binary correctness when MTBench
            naturally provides one, i.e. explicit label correctness
            (`correct`) or classification accuracy (`accuracy`).

            For continuous-score tasks such as forecast / indicator, MTBench
            reports graded errors (MAE, RMSE, component MAE, ...), not a
            universal pass/fail threshold. In those cases we return ``None``
            and let the note be marked as scored rather than OK/FAIL.

            Explicit parse/type/tool errors still count as failures.
        """
        if not metrics:
            return None

        if bool(metrics.get("parse_error")) or bool(metrics.get("type_error")):
            return False
        if any(
            bool(v) for k, v in metrics.items()
            if k.endswith("_error") and k not in {"parse_error", "type_error"}
        ):
            return False

        explicit_correct = metrics.get("correct")
        if isinstance(explicit_correct, bool):
            return explicit_correct

        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, (int, float)):
            return float(accuracy) >= 1.0

        return None

    # -- Hierarchical Memory Interface ----------------------------------

    async def record_outcome(self, **run_context) -> None:
        """Record a pipeline run outcome into the hierarchical memory.

        This is the primary interface for feeding ground-truth signals back
        into the memory system.  The L1 store uses an LLM to summarize the
        run into a natural-language note, indexes it for semantic retrieval,
        and triggers L1→L2→L3 distillation when batch thresholds are reached.
        """
        await self.memory.record(**run_context)

    async def flush_memory(self) -> None:
        await self.memory.flush()

    async def distill_memory(self) -> None:
        await self.memory.distill_pending()

    def memory_summary(self) -> dict:
        return self.memory.summary()

    # -- Prediction extraction (generic, no task-specific logic) ----------

    @staticmethod
    def _parse_json_like_payload(raw: Any) -> Any:
        """Parse stringified JSON payloads produced by tool/agent envelopes.

        New definition:
            ``_parse_json_like_payload`` peels 1-2 layers of JSON string
            wrapping when a prediction was serialized into text instead of
            being kept as a native dict/list. Non-JSON strings are returned
            unchanged.
        """
        current = raw
        for _ in range(2):
            if not isinstance(current, str):
                break
            stripped = current.strip()
            if not stripped or stripped[0] not in {'{', '[', '"'}:
                break
            try:
                parsed = json.loads(stripped)
            except Exception:
                break
            if parsed == current:
                break
            current = parsed
        return current

    @staticmethod
    def _extract_prediction(
        raw: Any,
        task_type: str,
        horizon: int | None,
    ) -> Any:
        """Extract the usable prediction from the agent's raw answer.

        For forecast tasks, the agent may wrap the array in a dict with
        various key names. This method finds and extracts numeric arrays
        from nested dicts so that downstream evaluation receives a plain
        list/array.

        For non-forecast tasks, the raw answer is returned as-is since it
        should already be in the format requested by the description.
        """
        if raw is None:
            return raw

        raw = PipelineRunner._parse_json_like_payload(raw)

        if task_type == "forecast" and isinstance(raw, dict):
            # Look for the longest numeric array value in the dict
            best_key, best_arr = None, None
            for k, v in raw.items():
                if isinstance(v, (list, np.ndarray)):
                    try:
                        arr = np.asarray(v, dtype=float)
                        if arr.ndim == 1 and (best_arr is None or len(arr) > len(best_arr)):
                            best_key, best_arr = k, arr
                    except (ValueError, TypeError):
                        continue
            if best_arr is not None:
                logger.debug(f"[Pipeline] Extracted forecast array from key '{best_key}' (len={len(best_arr)})")
                return best_arr.tolist()

        if task_type == "indicator" and isinstance(raw, dict):
            normalized: dict[str, Any] = {}
            changed = False
            for key, value in raw.items():
                parsed_value = PipelineRunner._parse_json_like_payload(value)
                normalized[key] = parsed_value
                changed = changed or parsed_value is not value
            if changed:
                return normalized

        return raw

    # -- Post-process helpers (pure code, no LLM) -----------------------

    @staticmethod
    def _check_bounds(
        prediction: Any,
        constraints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check if prediction values are within specified bounds."""
        if not constraints:
            return {"passed": True, "details": "No constraints"}

        violations = []
        pred_array = None
        if isinstance(prediction, (list, np.ndarray)):
            pred_array = np.asarray(prediction, dtype=float)
        elif isinstance(prediction, dict) and "forecast" in prediction:
            pred_array = np.asarray(prediction["forecast"], dtype=float)

        if pred_array is None:
            return {"passed": True, "details": "Non-numeric prediction, skipping bounds"}

        for c in constraints:
            if c.get("type") == "range":
                lo, hi = c.get("min", -np.inf), c.get("max", np.inf)
                below = pred_array < lo
                above = pred_array > hi
                if below.any() or above.any():
                    violations.append(
                        f"Range [{lo}, {hi}]: {below.sum()} below, {above.sum()} above"
                    )

        if violations:
            return {"passed": False, "details": "; ".join(violations)}
        return {"passed": True, "details": "All bounds satisfied"}

    @staticmethod
    def _apply_bound_correction(
        prediction: Any,
        constraints: list[dict[str, Any]],
    ) -> tuple[Any, dict[str, Any] | None]:
        """Clip prediction to constraint bounds."""
        pred_array = None
        is_list = False
        is_dict = False

        if isinstance(prediction, list):
            pred_array = np.asarray(prediction, dtype=float)
            is_list = True
        elif isinstance(prediction, np.ndarray):
            pred_array = prediction.copy()
        elif isinstance(prediction, dict) and "forecast" in prediction:
            pred_array = np.asarray(prediction["forecast"], dtype=float)
            is_dict = True

        if pred_array is None:
            return prediction, None

        corrected = False
        for c in constraints:
            if c.get("type") == "range":
                lo, hi = c.get("min", -np.inf), c.get("max", np.inf)
                clipped = np.clip(pred_array, lo, hi)
                if not np.array_equal(clipped, pred_array):
                    pred_array = clipped
                    corrected = True

        if not corrected:
            return prediction, None

        correction = {"type": "bound_clip", "applied": True}
        if is_dict:
            prediction = dict(prediction)
            prediction["forecast"] = pred_array.tolist()
            prediction["bound_corrected"] = True
        elif is_list:
            prediction = pred_array.tolist()
        else:
            prediction = pred_array

        return prediction, correction

    @staticmethod
    def _check_consistency(prediction: Any, historical_data: Any) -> dict[str, Any]:
        """Basic consistency check: prediction magnitude vs historical data."""
        try:
            hist = np.asarray(historical_data, dtype=float).flatten()
            hist = hist[~np.isnan(hist)]
            if len(hist) == 0:
                return {"passed": True, "details": "No valid historical data"}

            pred_array = None
            if isinstance(prediction, (list, np.ndarray)):
                pred_array = np.asarray(prediction, dtype=float).flatten()
            elif isinstance(prediction, dict) and "forecast" in prediction:
                pred_array = np.asarray(prediction["forecast"], dtype=float).flatten()

            if pred_array is None or len(pred_array) == 0:
                return {"passed": True, "details": "Non-numeric prediction"}

            hist_range = float(hist.max() - hist.min())
            hist_mean = float(hist.mean())

            if hist_range > 0:
                max_deviation = max(
                    abs(float(pred_array.max()) - hist_mean),
                    abs(float(pred_array.min()) - hist_mean),
                )
                ratio = max_deviation / hist_range
                if ratio > 10:
                    return {
                        "passed": False,
                        "details": f"Prediction deviates {ratio:.1f}x from historical range",
                    }

            return {"passed": True, "details": "Within reasonable range"}
        except (ValueError, TypeError):
            return {"passed": True, "details": "Could not perform consistency check"}

    @staticmethod
    def _has_numeric_postprocess_payload(task_type: str, prediction: Any) -> bool:
        if prediction is None:
            return False

        if task_type == "forecast":
            candidate = prediction
            if isinstance(candidate, dict):
                for key in ("forecast", "point_forecast", "prediction", "values", "answer"):
                    if key in candidate:
                        candidate = candidate[key]
                        break
            try:
                arr = np.asarray(candidate, dtype=float).ravel()
            except (ValueError, TypeError):
                return False
            return arr.size > 0

        if task_type == "indicator":
            if not isinstance(prediction, dict) or not prediction:
                return False
            for value in prediction.values():
                try:
                    float(value)
                    continue
                except (ValueError, TypeError):
                    pass
                try:
                    arr = np.asarray(value, dtype=float).ravel()
                except (ValueError, TypeError):
                    return False
                if arr.size == 0:
                    return False
            return True

        return True

    # -- Envelope construction -------------------------------------------

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
