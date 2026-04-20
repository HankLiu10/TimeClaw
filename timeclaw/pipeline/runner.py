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
    - SOUL (fixed identity) + MEMORY (per-scope) -> system prompt
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
from timeclaw.clawhub.validation.evaluate_against_gt import EvaluateAgainstGTTool
from timeclaw.core.config import TimeClawConfig
from timeclaw.core.confidence import coerce_confidence
from timeclaw.core.learning_text import (
    has_meta_learning_content,
    sanitize_learning_evidence_text,
)
from timeclaw.core.candidate_labels import (
    canonical_candidate_label,
    canonicalize_candidate_mentions,
    is_generic_candidate_label,
    strip_branch_prefix,
)
from timeclaw.core.indicator_answers import canonicalize_indicator_answer
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
from timeclaw.benchmark.mtbench_support import (
    generate_learning_analysis,
    render_learning_analysis_text,
    render_learning_summary_from_analysis,
)
from timeclaw.memory.hierarchical import HierarchicalMemory, Notes
from timeclaw.multimodal.detector import ModalityDetector

# Profiling tools run deterministically during preprocessing
_PROFILING_TOOLS = ["basic_stats", "autocorrelation", "stationarity_test"]
_DETECTION_TOOLS = ["detect_trend", "detect_anomaly"]
_TEXT_PROFILING_TOOLS: list[str] = []
_IMAGE_PROFILING_TOOLS = ["image_load", "image_feature"]
_REQUIRED_PREPROCESS_TOOLS = set(_PROFILING_TOOLS)
_LEARNING_GT_EVALUATOR = EvaluateAgainstGTTool()
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
        - MEMORY: Causal insights (per-scope)
        - TOOLS.md: Per-tool usage experience notes
        - SKILLS/: Scene-specific SOPs
    """

    def __init__(
        self,
        config: TimeClawConfig | None = None,
        disable_memory: bool = False,
        *,
        drain_memory_on_shutdown: bool = True,
    ):
        """Create a pipeline runner.

        New definition:
            `drain_memory_on_shutdown` controls whether `shutdown()` is allowed
            to advance pending Notes→MEMORY / MEMORY→TOOLS / MEMORY→SKILLS
            distillation work.

        Example:
            - Training runner: `drain_memory_on_shutdown=True`
            - Test runner: `drain_memory_on_shutdown=False`
        """
        self.config = config or TimeClawConfig.load()
        self._disable_memory = disable_memory
        self._drain_memory_on_shutdown = bool(drain_memory_on_shutdown)
        trace.configure(self.config.pipeline.verbose)
        self.router = ModelRouter(self.config.llm)

        if disable_memory:
            self.memory = HierarchicalMemory(enabled=False)
        else:
            self.memory = HierarchicalMemory(enabled=True)
        self._touched_memory_scopes: set[str] = set()

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
        if not self._disable_memory and getattr(self, "_drain_memory_on_shutdown", True):
            try:
                touched_scopes = sorted(self._touched_memory_scopes)
                logger.info(
                    f"[Pipeline] Shutdown: memory flush start "
                    f"(touched_scopes={len(touched_scopes)})"
                )
                if touched_scopes:
                    await self.memory.distill_pending_scopes(touched_scopes)
                else:
                    await self.memory.distill_pending()
                logger.info("[Pipeline] Shutdown: memory flush done")
            except Exception as e:
                logger.warning(f"[Pipeline] Memory distill_pending during shutdown failed: {e}")
        try:
            logger.info("[Pipeline] Shutdown: router shutdown start")
            await self.router.shutdown()
            logger.info("[Pipeline] Shutdown: router shutdown done")
        except Exception as e:
            logger.warning(f"[Pipeline] Router shutdown failed: {e}")
        if self._mcp_manager and self._mcp_started:
            logger.info("[Pipeline] Shutdown: MCP shutdown start")
            await self._mcp_manager.shutdown()
            self._mcp_started = False
            logger.info("[Pipeline] Shutdown: MCP shutdown done")

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
            envelope.task.metadata["scope_keys"] = list(scope_keys or [])
            envelope.task.metadata["modality"] = (
                detected_modalities[0] if len(detected_modalities) == 1 else "multimodal"
            )
            indicator_output_keys = self._indicator_output_keys(
                scope_keys,
                domain,
                task_type,
                ground_truth=ground_truth,
            )
            envelope.task.metadata["indicator_output_keys"] = indicator_output_keys

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

            raw_answer = result.get("answer")
            if raw_answer is None:
                raw_answer = result.get("prediction")
            answer_type = result.get("answer_type", "")
            confidence = coerce_confidence(result.get("confidence"))
            reasoning = result.get("reasoning", "")
            execution_context = (
                result.get("execution_context")
                if isinstance(result.get("execution_context"), dict)
                else {}
            )
            control_context = (
                result.get("control_context")
                if isinstance(result.get("control_context"), dict)
                else {}
            )
            execution_log = result.get("execution_log", [])

            prefetched_learning_analysis: dict[str, Any] | None = None
            prefetched_learning_summary_text = ""
            if mode == "train" and answer_type == "learning_summary":
                if isinstance(raw_answer, dict) and execution_context:
                    raw_answer = {
                        **raw_answer,
                        "execution_context": execution_context,
                    }
                raw_answer = self._normalize_learning_summary_answer(
                    raw_answer,
                    execution_log,
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )
                if ground_truth is not None:
                    raw_answer = self._repair_learning_summary_with_ground_truth(
                        raw_answer,
                        execution_log,
                        ground_truth=ground_truth,
                        task_type=task_type,
                        indicator_output_keys=indicator_output_keys,
                    )
                raw_answer, prefetched_learning_analysis, prefetched_learning_summary_text = await self._rewrite_learning_summary_from_analysis(
                    raw_answer,
                    execution_log,
                    task_type=task_type,
                    description=description,
                    domain=domain,
                    scope_keys=scope_keys,
                    fingerprint=fingerprint,
                    profiling_results=profiling_results,
                    horizon=horizon,
                    reasoning=reasoning,
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
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )
            elif mode == "train" and defer_learning_record:
                prediction_source = self._select_winner_prediction_from_execution_log(
                    execution_log,
                    raw_answer,
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )

            prediction = self._extract_prediction(
                prediction_source,
                task_type,
                horizon,
                indicator_output_keys=indicator_output_keys,
            )

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
                    prefetched_analysis_structured=prefetched_learning_analysis,
                    prefetched_pipeline_summary=prefetched_learning_summary_text,
                    ground_truth=ground_truth,
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

            evidence_summary = str(execution_context.get("evidence_summary") or "").strip()
            if not evidence_summary:
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
                "execution_context": execution_context,
                "control_context": control_context,
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
            failure_reason = str(
                getattr(e, "failure_reason", "")
                or ("max_steps_exhausted" if isinstance(e, MaxStepsExhaustedError) else error_type)
            )
            partial_result = getattr(e, "partial_result", None)
            if isinstance(partial_result, dict):
                setattr(e, "partial_result", partial_result)
                setattr(e, "execution_log", partial_result.get("execution_log", []))
                setattr(e, "raw_answer", partial_result.get("answer"))
                setattr(e, "answer_type", partial_result.get("answer_type", ""))
                setattr(e, "confidence", partial_result.get("confidence"))
                setattr(e, "reasoning", partial_result.get("reasoning", ""))
                setattr(e, "execution_context", partial_result.get("execution_context", {}))
                setattr(e, "control_context", partial_result.get("control_context", {}))
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
        prefetched_analysis_structured: dict[str, Any] | None = None,
        prefetched_pipeline_summary: str = "",
        ground_truth: Any = None,
    ) -> None:
        """Persist training-mode learning outcomes into hierarchical memory (L1).

        All information — tool chain, metrics from every evaluate_against_gt
        call, agent analysis — is written to a single L1 note.  L2/L3
        are populated exclusively by the DistillationAgent after the note is
        committed, using background single-flight rebuild workers.
        """
        if answer_type == "learning_summary" and isinstance(raw_answer, dict) and ground_truth is not None:
            repaired_raw_answer = self._repair_learning_summary_with_ground_truth(
                raw_answer,
                execution_log,
                ground_truth=ground_truth,
                task_type=task_type,
                indicator_output_keys=self._indicator_output_keys(scope_keys, domain, task_type),
            )
            if repaired_raw_answer != raw_answer:
                raw_answer = repaired_raw_answer
                prefetched_analysis_structured = None
                prefetched_pipeline_summary = ""

        tool_chain = [e.get("tool", "") for e in execution_log if e.get("tool")]
        indicator_output_keys = self._indicator_output_keys(
            scope_keys,
            domain,
            task_type,
        )
        candidate_traces = self._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
        summary_candidate_traces = self._dedupe_candidate_traces_by_path(candidate_traces)

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
        no_validated_winner = self._all_binary_scored_candidates_failed(summary_candidate_traces)
        winner_tool_chain = (
            []
            if no_validated_winner
            else (
                list(winner_trace.get("tool_chain") or [])
                if winner_trace
                else self._select_winner_tool_chain(candidate_traces, best_metrics)
            )
        )
        fallback_summary = analysis or self._execution_side_learning_summary_fallback(raw_answer)
        structured_analysis = prefetched_analysis_structured
        summary = str(prefetched_pipeline_summary or "").strip()
        if structured_analysis is None:
            structured_analysis, summary = await generate_learning_analysis(
                scope_key=(scope_keys[0] if scope_keys else f"{domain}_{task_type}"),
                scope_keys=scope_keys,
                task_type=task_type,
                domain=domain,
                task_description=description,
                sample_fingerprint=sample_stats,
                candidate_traces=summary_candidate_traces,
                best_metrics=best_metrics,
                selected_winner_trace=None if no_validated_winner else winner_trace,
                winner_correct=(
                    False
                    if no_validated_winner
                    else (
                    self._learning_outcome_correctness(dict(winner_trace.get("metrics") or {}))
                    if winner_trace and isinstance(winner_trace.get("metrics"), dict) and winner_trace.get("metrics")
                    else None
                    )
                ),
                fallback_summary=fallback_summary,
            )
        elif not summary:
            summary = render_learning_analysis_text(structured_analysis)

        recorded_prediction = None
        if winner_trace is not None and not no_validated_winner:
            recorded_prediction = winner_trace.get("prediction")
        if recorded_prediction is None:
            recorded_prediction = None if no_validated_winner else prediction
        if recorded_prediction is None:
            recorded_prediction = None if no_validated_winner else raw_answer
        recorded_prediction = self._canonicalize_task_prediction(
            recorded_prediction,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )

        recorded_metrics = (
            {}
            if no_validated_winner
            else (dict(winner_trace.get("metrics") or {}) if winner_trace else {})
        )
        recorded_correct = False if no_validated_winner else self._learning_outcome_correctness(recorded_metrics)

        await self.memory.record(
            domain=domain,
            task=task_type,
            scope_keys=scope_keys,
            trace_id=trace_id,
            tool_chain=winner_tool_chain,
            pipeline_summary=summary,
            data_fingerprint=sample_stats,
            predicted=str(recorded_prediction) if recorded_prediction is not None else "",
            metrics=recorded_metrics,
            correct=recorded_correct,
            description=description,
            all_eval_metrics=all_evals,
            candidate_traces=summary_candidate_traces,
            analysis_structured=structured_analysis,
            source_metadata=source_metadata or {},
        )
        self._touched_memory_scopes.update(
            scope_keys or [f"{domain}_{task_type}"]
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
    def _extract_candidate_traces(
        execution_log: list[dict[str, Any]],
        *,
        task_type: str = "",
        indicator_output_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        batch_results: list[dict[str, Any]] = []
        single_results: list[dict[str, Any]] = []
        pending_top_level_steps: list[dict[str, Any]] = []
        pending_top_level_start_idx: int | None = None

        def _flush_top_level_trace(end_entry_index: int | None = None) -> None:
            nonlocal pending_top_level_steps, pending_top_level_start_idx
            if not pending_top_level_steps:
                return
            steps = PipelineRunner._compact_candidate_steps(pending_top_level_steps)
            for segment_steps in PipelineRunner._split_candidate_steps_by_answer_boundary(steps):
                tool_chain = PipelineRunner._tool_chain_from_steps(segment_steps)
                prediction = PipelineRunner._prediction_from_steps(segment_steps)
                prediction = PipelineRunner._canonicalize_task_prediction(
                    prediction,
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )
                reasoning = ""
                for step in reversed(segment_steps):
                    thought = str(step.get("thought") or "").strip()
                    if thought:
                        reasoning = thought
                        break
                if not segment_steps or prediction is None:
                    continue
                derived_name = (
                    PipelineRunner._tool_chain_summary(tool_chain)
                    or PipelineRunner._primary_candidate_tool_name(tool_chain)
                    or (tool_chain[0] if tool_chain else "")
                    or f"candidate_{len(traces)}"
                )
                candidate_key = PipelineRunner._candidate_identity_key(
                    derived_name,
                    tool_chain,
                    len(traces),
                )
                trace_payload = {
                    "name": derived_name,
                    "candidate_key": candidate_key,
                    "_stable_key": (
                        f"top_level::{pending_top_level_start_idx}"
                        if isinstance(pending_top_level_start_idx, int)
                        else ""
                    ),
                    "tool_chain": tool_chain,
                    "prediction": prediction,
                    "reasoning": reasoning,
                    "reasoning_source": "step_thought" if reasoning else "",
                    "steps": segment_steps,
                    "entry_index": (
                        pending_top_level_start_idx
                        if isinstance(pending_top_level_start_idx, int)
                        else end_entry_index
                    ),
                    "_match_keys": PipelineRunner._candidate_trace_match_keys(
                        name=derived_name,
                        candidate_key=candidate_key,
                        tool_chain=tool_chain,
                        stable_key=(
                            f"top_level::{pending_top_level_start_idx}"
                            if isinstance(pending_top_level_start_idx, int)
                            else None
                        ),
                        raw_name=" -> ".join(tool_chain),
                    ),
                }
                trace_payload["public_reasoning_snippets"] = PipelineRunner._public_reasoning_snippets(trace_payload)
                traces.append(trace_payload)
            pending_top_level_steps = []
            pending_top_level_start_idx = None

        for entry_idx, entry in enumerate(execution_log):
            tool_name = str(entry.get("tool") or "").strip()
            if tool_name == "spawn_subagent":
                _flush_top_level_trace(end_entry_index=entry_idx)
                output = entry.get("output") or []
                for idx, item in enumerate(output):
                    if not isinstance(item, dict) or not item.get("success"):
                        continue
                    result = item.get("result") or {}
                    if not isinstance(result, dict):
                        continue
                    sub_execution_log = result.get("execution_log", [])
                    raw_name = result.get("candidate_name") or result.get("tools_hint") or result.get("sub_task_description")
                    derived_name = (
                        PipelineRunner._primary_candidate_tool_name(
                            [e.get("tool", "") for e in sub_execution_log if e.get("tool")]
                        )
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
                    tool_chain = PipelineRunner._tool_chain_from_steps(steps)
                    derived_name = (
                        PipelineRunner._tool_chain_summary(tool_chain)
                        or PipelineRunner._primary_candidate_tool_name(tool_chain)
                        or PipelineRunner._tool_name_from_hint(raw_name)
                        or str(raw_name or "").strip()
                        or f"candidate_{len(traces)}"
                    )
                    prediction = PipelineRunner._prediction_from_steps(steps)
                    if prediction is None:
                        prediction = result.get("answer", result.get("prediction"))
                    prediction = PipelineRunner._canonicalize_task_prediction(
                        prediction,
                        task_type=task_type,
                        indicator_output_keys=indicator_output_keys,
                    )
                    stable_key = ExecutorAgent._candidate_branch_id(
                        result, fallback_idx=idx,
                    )
                    candidate_key = PipelineRunner._candidate_identity_key(
                        derived_name,
                        tool_chain,
                        len(traces),
                    )
                    trace_payload = {
                        "name": derived_name,
                        "candidate_key": candidate_key,
                        "_stable_key": stable_key,
                        "tool_chain": tool_chain,
                        "prediction": prediction,
                        "reasoning": str(result.get("reasoning") or "").strip(),
                        "reasoning_source": (
                            "execution_context"
                            if PipelineRunner._execution_context_public_reasoning(result) is not None
                            else ("final_reasoning" if str(result.get("reasoning") or "").strip() else "")
                        ),
                        "steps": steps,
                        "entry_index": entry_idx,
                        "_match_keys": PipelineRunner._candidate_trace_match_keys(
                            name=derived_name,
                            candidate_key=candidate_key,
                            tool_chain=tool_chain,
                            stable_key=stable_key,
                            raw_name=raw_name,
                        ),
                    }
                    explicit_public_reasoning = PipelineRunner._execution_context_public_reasoning(result)
                    trace_payload["public_reasoning_snippets"] = (
                        explicit_public_reasoning
                        if explicit_public_reasoning is not None
                        else PipelineRunner._public_reasoning_snippets(trace_payload)
                    )
                    traces.append(trace_payload)
                continue
            if tool_name in {"evaluate_batch_against_gt", "batch_evaluate_against_gt"}:
                _flush_top_level_trace(end_entry_index=entry_idx)
                obs = entry.get("observation")
                if obs is None:
                    obs = {}
                if isinstance(obs, dict) and isinstance(obs.get("results"), list):
                    batch_results.extend(
                        item for item in obs["results"] if isinstance(item, dict)
                    )
                continue
            if tool_name == "evaluate_against_gt":
                _flush_top_level_trace(end_entry_index=entry_idx)
                obs = entry.get("observation")
                if obs is None:
                    obs = {}
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
            if not tool_name:
                continue
            if not pending_top_level_steps:
                pending_top_level_start_idx = entry_idx
            pending_top_level_steps.append(entry)

        _flush_top_level_trace(end_entry_index=len(execution_log))

        trace_by_key: dict[str, list[dict[str, Any]]] = {}
        for trace in traces:
            for key in trace.get("_match_keys") or []:
                trace_by_key.setdefault(str(key), []).append(trace)
        unmatched_traces = [trace for trace in traces if trace.get("metrics") is None]

        for idx, res in enumerate(batch_results):
            if not isinstance(res, dict):
                continue
            res_name = str(res.get("name") or "").strip()
            res_key = PipelineRunner._normalise_candidate_key(res_name, [], idx)

            matched = None
            matched_candidates: list[dict[str, Any]] = []
            seen_ids: set[int] = set()
            res_match_keys = PipelineRunner._candidate_trace_match_keys(
                name=res_name,
                candidate_key=res_key,
                tool_chain=[],
                stable_key=res_name,
                raw_name=res_name,
            )
            for key in res_match_keys:
                for trace in trace_by_key.get(key, []):
                    if trace.get("metrics") is not None:
                        continue
                    trace_id = id(trace)
                    if trace_id in seen_ids:
                        continue
                    seen_ids.add(trace_id)
                    matched_candidates.append(trace)
            res_prediction = PipelineRunner._normalize_prediction_match_value(
                PipelineRunner._canonicalize_task_prediction(
                    res.get("prediction"),
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )
            )
            if res_prediction is not None and matched_candidates:
                prediction_matches = [
                    trace
                    for trace in matched_candidates
                    if PipelineRunner._predictions_match(
                        trace.get("prediction"), res_prediction,
                    )
                ]
                if len(prediction_matches) == 1:
                    matched = prediction_matches[0]
                elif prediction_matches:
                    matched_candidates = prediction_matches
            if matched is None and len(matched_candidates) == 1:
                matched = matched_candidates[0]
            elif matched is None and matched_candidates:
                matched = matched_candidates[0]
            if matched is None and idx < len(unmatched_traces):
                matched = unmatched_traces[idx]
            if matched is None:
                continue

            res_prediction = PipelineRunner._normalize_prediction_match_value(
                PipelineRunner._canonicalize_task_prediction(
                    res.get("prediction"),
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )
            )
            if (
                res_prediction is not None
                and matched.get("prediction") is not None
                and not PipelineRunner._predictions_match(
                    matched.get("prediction"),
                    res_prediction,
                )
            ):
                global_prediction_matches = [
                    trace
                    for trace in unmatched_traces
                    if trace.get("metrics") is None
                    and PipelineRunner._predictions_match(
                        trace.get("prediction"),
                        res_prediction,
                    )
                ]
                if len(global_prediction_matches) == 1:
                    matched = global_prediction_matches[0]
                else:
                    continue

            matched["metrics"] = res.get("metrics", {})
            if res_prediction is not None:
                # Prefer the evaluation payload only when it agrees with the
                # execution-side observed answer. If they conflict, the result
                # should already have been remapped above; do not overwrite the
                # observed candidate prediction here.
                if matched.get("prediction") is None or PipelineRunner._predictions_match(
                    matched.get("prediction"),
                    res_prediction,
                ):
                    matched["prediction"] = res_prediction
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
            res_prediction = PipelineRunner._normalize_prediction_match_value(
                PipelineRunner._canonicalize_task_prediction(
                    res.get("prediction"),
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )
            )
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
        PipelineRunner._propagate_metrics_across_equivalent_predictions(traces)
        PipelineRunner._disambiguate_duplicate_candidate_keys(traces)
        return traces

    @staticmethod
    def _candidate_trace_match_keys(
        *,
        name: Any,
        candidate_key: Any,
        tool_chain: list[str],
        stable_key: Any | None = None,
        raw_name: Any | None = None,
    ) -> list[str]:
        keys: list[str] = []

        def _add(value: Any) -> None:
            text = str(value or "").strip()
            if not text or text in keys:
                return
            keys.append(text)

        for value in (candidate_key, name, stable_key, raw_name):
            _add(value)

        normalized_name = PipelineRunner._normalise_candidate_key(
            str(name or ""), tool_chain, 0,
        )
        normalized_raw_name = PipelineRunner._normalise_candidate_key(
            str(raw_name or ""), tool_chain, 0,
        )
        _add(normalized_name)
        _add(normalized_raw_name)

        if tool_chain:
            _add(" -> ".join(str(tool).strip() for tool in tool_chain if str(tool).strip()))
            primary = PipelineRunner._primary_candidate_tool_name(tool_chain)
            _add(primary)

        return keys

    @staticmethod
    def _trace_identity_set(traces: list[dict[str, Any]]) -> set[int]:
        return {
            id(trace)
            for trace in traces
            if isinstance(trace, dict)
        }

    @staticmethod
    def _predictions_match(left: Any, right: Any) -> bool:
        left = PipelineRunner._normalize_prediction_match_value(left)
        right = PipelineRunner._normalize_prediction_match_value(right)

        if isinstance(left, dict) and isinstance(right, dict):
            if set(left.keys()) != set(right.keys()):
                return False
            return all(PipelineRunner._predictions_match(left[key], right[key]) for key in left.keys())

        if isinstance(left, (list, tuple, np.ndarray)) and isinstance(right, (list, tuple, np.ndarray)):
            try:
                left_arr = np.asarray(left, dtype=float)
                right_arr = np.asarray(right, dtype=float)
            except (TypeError, ValueError):
                left_seq = left.tolist() if isinstance(left, np.ndarray) else list(left)
                right_seq = right.tolist() if isinstance(right, np.ndarray) else list(right)
                if len(left_seq) != len(right_seq):
                    return False
                return all(
                    PipelineRunner._predictions_match(left_item, right_item)
                    for left_item, right_item in zip(left_seq, right_seq)
                )
            if left_arr.shape != right_arr.shape:
                return False
            return bool(np.allclose(left_arr, right_arr, atol=1e-3, rtol=1e-6))

        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_prediction_match_value(value: Any) -> Any:
        value = PipelineRunner._parse_json_like_payload(value)
        while isinstance(value, dict) and len(value) == 1:
            key = str(next(iter(value.keys()))).strip().lower()
            if key not in {"label", "answer", "prediction"}:
                break
            value = PipelineRunner._parse_json_like_payload(
                value.get(next(iter(value.keys())))
            )
        return value


    @staticmethod
    def _prediction_from_observation(observation: Any) -> Any:
        if isinstance(observation, dict):
            for key in ("forecast", "point_forecast", "answer", "prediction"):
                value = observation.get(key)
                if value is not None:
                    return PipelineRunner._parse_json_like_payload(value)
        return None

    @staticmethod
    def _canonicalize_task_prediction(
        prediction: Any,
        *,
        task_type: str,
        indicator_output_keys: list[str] | None = None,
    ) -> Any:
        if task_type != "indicator":
            return prediction
        return canonicalize_indicator_answer(
            PipelineRunner._parse_json_like_payload(prediction),
            indicator_output_keys=indicator_output_keys,
        )

    @staticmethod
    def _prediction_from_steps(steps: Any) -> Any:
        step_list = steps if isinstance(steps, list) else []
        for step in reversed(step_list):
            if not isinstance(step, dict):
                continue
            observation = PipelineRunner._parse_json_like_payload(step.get("observation"))
            prediction = PipelineRunner._prediction_from_observation(observation)
            if prediction is not None:
                return prediction
        return None

    @staticmethod
    def _trim_candidate_steps_to_answer_prefix(steps: list[dict[str, str]]) -> list[dict[str, str]]:
        # Keep the full linear path here. Candidate segmentation is handled
        # later by answer boundaries, so post-answer analysis/validation steps
        # remain attached to the candidate that produced the answer.
        return steps

    @staticmethod
    def _split_candidate_steps_by_answer_boundary(
        steps: list[dict[str, str]],
    ) -> list[list[dict[str, str]]]:
        segments: list[list[dict[str, str]]] = []
        current: list[dict[str, str]] = []
        current_has_answer = False
        current_has_followup_after_answer = False
        total = len(steps)
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            observation = PipelineRunner._parse_json_like_payload(step.get("observation"))
            produces_answer = PipelineRunner._prediction_from_observation(observation) is not None
            has_future_substantive_step = any(
                isinstance(future_step, dict)
                for future_step in steps[idx + 1:total]
            )
            if (
                produces_answer
                and current
                and current_has_answer
                and (current_has_followup_after_answer or has_future_substantive_step)
            ):
                segments.append(current)
                current = []
                current_has_answer = False
                current_has_followup_after_answer = False
            current.append(step)
            if produces_answer:
                current_has_answer = True
                current_has_followup_after_answer = False
            elif current_has_answer:
                current_has_followup_after_answer = True
        if current:
            segments.append(current)
        return segments

    @staticmethod
    def _tool_chain_from_steps(steps: list[dict[str, Any]]) -> list[str]:
        tool_chain: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "").strip()
            if not tool:
                continue
            observation = PipelineRunner._parse_json_like_payload(step.get("observation"))
            has_prediction = PipelineRunner._prediction_from_observation(observation) is not None
            if step.get("success") is False and not has_prediction:
                continue
            if isinstance(observation, dict) and "error" in observation and not has_prediction:
                continue
            tool_chain.append(tool)
        return tool_chain

    @staticmethod
    def _compact_candidate_steps(execution_log: list[dict[str, Any]]) -> list[dict[str, str]]:
        steps: list[dict[str, str]] = []
        for entry in execution_log:
            tool = str(entry.get("tool") or "").strip()
            if not tool or tool in {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt", "spawn_subagent"}:
                continue
            step: dict[str, str] = {"tool": tool}
            if "success" in entry:
                step["success"] = bool(entry.get("success"))

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
        return PipelineRunner._trim_candidate_steps_to_answer_prefix(steps)

    @staticmethod
    def _select_winner_tool_chain(candidate_traces: list[dict[str, Any]], best_metrics: dict[str, Any]) -> list[str]:
        if PipelineRunner._all_binary_scored_candidates_failed(candidate_traces):
            return []
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
    def _prefer_candidate_trace(
        current: dict[str, Any],
        challenger: dict[str, Any],
    ) -> dict[str, Any]:
        current_metrics = current.get("metrics") if isinstance(current.get("metrics"), dict) else {}
        challenger_metrics = challenger.get("metrics") if isinstance(challenger.get("metrics"), dict) else {}
        if current_metrics and not challenger_metrics:
            return current
        if challenger_metrics and not current_metrics:
            return challenger
        if current_metrics and challenger_metrics:
            current_score = PipelineRunner._metrics_score(current_metrics)
            challenger_score = PipelineRunner._metrics_score(challenger_metrics)
            if challenger_score < current_score:
                return challenger
            if current_score < challenger_score:
                return current
        if len(challenger.get("steps") or []) > len(current.get("steps") or []):
            return challenger
        return current

    @classmethod
    def _dedupe_candidate_traces_by_path(
        cls,
        candidate_traces: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep one representative candidate trace per substantive path.

        New definition:
            Summary-facing candidate displays and counts should track unique
            tool paths, not repeated branch slots that happened to explore the
            same path multiple times. When duplicates share a path, keep the
            strongest representative trace for that path.
        """
        deduped: dict[tuple[str, ...], dict[str, Any]] = {}
        order: list[tuple[str, ...]] = []
        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            tool_chain = [
                str(tool).strip()
                for tool in trace.get("tool_chain") or []
                if str(tool).strip()
            ]
            if tool_chain:
                key = tuple(tool_chain)
            else:
                key = (
                    canonical_candidate_label(
                        candidate_key=trace.get("candidate_key"),
                        name=trace.get("name"),
                        tool_chain=[],
                    )
                    or f"candidate_{len(order)}",
                )
                key = (str(key),)
            if key not in deduped:
                deduped[key] = trace
                order.append(key)
                continue
            deduped[key] = cls._prefer_candidate_trace(deduped[key], trace)
        return [deduped[key] for key in order]

    @classmethod
    def _all_binary_scored_candidates_failed(
        cls,
        candidate_traces: list[dict[str, Any]],
    ) -> bool:
        total_candidates = 0
        correctness: list[bool] = []
        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            if trace.get("prediction") is None:
                continue
            total_candidates += 1
            metrics = trace.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                continue
            resolved = cls._learning_outcome_correctness(dict(metrics))
            if resolved is None:
                continue
            correctness.append(resolved)
        return bool(correctness) and total_candidates == len(correctness) and not any(correctness)

    @staticmethod
    def _dedupe_compared_candidates(
        compared_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduped: dict[tuple[str, ...], dict[str, Any]] = {}
        order: list[tuple[str, ...]] = []
        for item in compared_candidates:
            if not isinstance(item, dict):
                continue
            tool_chain = [
                str(tool).strip()
                for tool in item.get("tool_chain") or []
                if str(tool).strip()
            ]
            key = tuple(tool_chain) if tool_chain else (str(item.get("name") or "").strip(),)
            if key not in deduped:
                deduped[key] = item
                order.append(key)
                continue
            current_metrics = deduped[key].get("metrics") if isinstance(deduped[key].get("metrics"), dict) else {}
            challenger_metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            if challenger_metrics and (
                not current_metrics
                or PipelineRunner._metrics_score(challenger_metrics) < PipelineRunner._metrics_score(current_metrics)
            ):
                deduped[key] = item
        return [deduped[key] for key in order]

    @staticmethod
    def _propagate_metrics_across_equivalent_predictions(
        candidate_traces: list[dict[str, Any]],
    ) -> None:
        """Backfill omitted metrics for paths that emitted the same prediction.

        New definition:
            Evaluation metrics are a property of the sample plus the emitted
            prediction. If two different branches produced the same prediction
            and one branch received GT metrics while the other did not, copy the
            unique scored metrics onto the unscored branch so downstream
            summaries stay semantically aligned.

        Example:
            `llm_reasoning_qa -> "C"` scored with `accuracy=1.0` should allow
            `window_stats -> llm_reasoning_qa -> "C"` to carry the same metrics
            when that branch was not explicitly named in the evaluator output.
        """
        def _prediction_key(value: Any) -> str | None:
            normalized = PipelineRunner._normalize_prediction_match_value(value)
            if normalized is None:
                return None
            return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)

        metrics_by_prediction: dict[str, list[dict[str, Any]]] = {}
        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            prediction_key = _prediction_key(trace.get("prediction"))
            metrics = trace.get("metrics")
            if prediction_key is None or not isinstance(metrics, dict) or not metrics:
                continue
            metrics_by_prediction.setdefault(prediction_key, []).append(dict(metrics))

        unique_metrics_by_prediction: dict[str, dict[str, Any]] = {}
        for prediction_key, metric_list in metrics_by_prediction.items():
            unique_payloads: dict[str, dict[str, Any]] = {}
            for metrics in metric_list:
                payload_key = json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str)
                unique_payloads[payload_key] = metrics
            if len(unique_payloads) == 1:
                unique_metrics_by_prediction[prediction_key] = next(iter(unique_payloads.values()))

        for trace in candidate_traces:
            if not isinstance(trace, dict):
                continue
            metrics = trace.get("metrics")
            if isinstance(metrics, dict) and metrics:
                continue
            prediction_key = _prediction_key(trace.get("prediction"))
            if prediction_key is None:
                continue
            propagated = unique_metrics_by_prediction.get(prediction_key)
            if propagated:
                trace["metrics"] = dict(propagated)

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
    def _candidate_identity_key(name: str, tool_chain: list[str], index: int) -> str:
        clean_chain = [str(tool).strip() for tool in tool_chain if str(tool).strip()]
        if clean_chain:
            return " -> ".join(clean_chain)
        return PipelineRunner._normalise_candidate_key(name, clean_chain, index)

    @classmethod
    def _branch_anchored_candidate_key(cls, trace: dict[str, Any], index: int) -> str:
        path = cls._tool_chain_summary(list(trace.get("tool_chain") or []))
        base = str(trace.get("candidate_key") or trace.get("name") or "").strip()
        stable = str(trace.get("_stable_key") or "").strip()
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
    def _disambiguate_duplicate_candidate_keys(cls, traces: list[dict[str, Any]]) -> None:
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, trace in enumerate(traces):
            if not isinstance(trace, dict):
                continue
            key = str(trace.get("candidate_key") or trace.get("name") or "").strip()
            if not key:
                continue
            groups.setdefault(key, []).append((index, trace))

        for group in groups.values():
            if len(group) <= 1:
                continue
            for index, trace in group:
                trace["candidate_key"] = cls._branch_anchored_candidate_key(trace, index)

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
        tool_chain = list(trace.get("tool_chain") or [])
        display = canonical_candidate_label(
            candidate_key=trace.get("candidate_key"),
            name=trace.get("name"),
            tool_chain=tool_chain,
        )
        if display:
            return display
        primary = cls._primary_candidate_tool_name(tool_chain)
        if primary:
            return primary
        return fallback

    @staticmethod
    def _trace_name_is_generic(name: Any) -> bool:
        return is_generic_candidate_label(name) or str(name or "").strip().lower() in {"candidate", "top_level"}

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
            strip_branch_prefix(winner_trace.get("candidate_key")),
            canonical_candidate_label(
                candidate_key=winner_trace.get("candidate_key"),
                name=winner_trace.get("name"),
                tool_chain=winner_trace.get("tool_chain"),
            ),
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
        cleaned = sanitize_learning_evidence_text(cleaned, drop_imperatives=True)
        if has_meta_learning_content(cleaned):
            return ""
        if not cleaned or re.fullmatch(r"\d+", cleaned):
            return ""
        if len(cleaned) < 12:
            return ""
        return cleaned

    @classmethod
    def _public_reasoning_snippets(cls, trace: dict[str, Any]) -> list[str]:
        if str(trace.get("reasoning_source") or "").strip() != "final_reasoning":
            return []

        reasoning = str(trace.get("reasoning") or "").strip()
        if not reasoning:
            return []

        snippets: list[str] = []
        seen: set[str] = set()
        for part in re.split(r"(?<=[.!?])\s+|[;\n]+", reasoning):
            cleaned = Notes._sanitize_candidate_reasoning(part)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            snippets.append(cleaned)
        return snippets

    @staticmethod
    def _execution_context_public_reasoning(result: dict[str, Any]) -> list[str] | None:
        execution_context = result.get("execution_context")
        if not isinstance(execution_context, dict):
            return None
        if "public_reasoning" not in execution_context:
            return None
        items = execution_context.get("public_reasoning")
        if not isinstance(items, list):
            return []
        cleaned_items: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = Notes._sanitize_candidate_reasoning(str(item))
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned_items.append(cleaned)
        return cleaned_items

    @staticmethod
    def _execution_side_learning_summary_fallback(
        raw_answer: Any,
        execution_context: dict[str, Any] | None = None,
    ) -> str:
        """Build learning-analysis fallback text from execution-side fields only.

        New definition:
            Fallback text sent into learning analysis must come from reusable
            execution evidence (`insight`, `recommendation`,
            `execution_context.public_reasoning`, `execution_context.evidence_summary`)
            rather than search/control-flow-heavy final reasoning text.
        """
        parts: list[str] = []
        if isinstance(raw_answer, dict):
            for key in ("insight", "recommendation"):
                value = raw_answer.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        context = execution_context if isinstance(execution_context, dict) else {}
        public_reasoning = context.get("public_reasoning")
        if isinstance(public_reasoning, list):
            joined_public = "\n".join(
                str(item).strip()
                for item in public_reasoning
                if str(item).strip()
            ).strip()
            if joined_public:
                parts.append(joined_public)
        evidence_summary = str(context.get("evidence_summary") or "").strip()
        if evidence_summary:
            parts.append(evidence_summary)
        return "\n".join(parts).strip()[:3000]

    @classmethod
    def _reasoning_snippets(cls, trace: dict[str, Any]) -> list[str]:
        raw_items: list[str] = []
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
        public_snippets = trace.get("public_reasoning_snippets")
        if isinstance(public_snippets, list):
            for snippet in public_snippets:
                snippet = str(snippet).strip()
                if not snippet:
                    continue
                if len(snippet) > 160:
                    snippet = snippet[:160].rstrip(" ,;:") + "..."
                snippet = snippet[0].upper() + snippet[1:] if snippet else snippet
                if not snippet.endswith("."):
                    snippet += "."
                snippets.append(snippet)
                if len(snippets) >= limit:
                    break
            if snippets:
                return " ".join(snippets)
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
            parts = []
            path_label = f"`{winner_chain}`" if winner_chain else winner_name
            strongest_path = f"Strongest failed path: {path_label}"
            if metric_text:
                strongest_path += f" with {metric_text}"
            parts.append(strongest_path)
            if prediction_text:
                parts.append(f"Observed output: {prediction_text}")
            if winner_reason:
                parts.append(f"Why it still looked plausible: {winner_reason}")
            if compared_bits:
                parts.append("Alternative signals: " + " | ".join(compared_bits[:2]))
            insight = ". ".join(part.strip().rstrip(".") for part in parts if part).strip()
            if insight and not insight.endswith("."):
                insight += "."
        else:
            parts = []
            path_label = f"`{winner_chain}`" if winner_chain else winner_name
            strongest_path = f"Strongest path on this sample: {path_label}"
            if metric_text:
                strongest_path += f" with {metric_text}"
            parts.append(strongest_path)
            if prediction_text:
                parts.append(f"Observed output: {prediction_text}")
            if winner_reason:
                parts.append(f"Why it fit: {winner_reason}")
            if compared_bits:
                parts.append("Alternative signals: " + " | ".join(compared_bits[:2]))
            insight = ". ".join(part.strip().rstrip(".") for part in parts if part).strip()
            if insight and not insight.endswith("."):
                insight += "."

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
        task_type: str = "",
        indicator_output_keys: list[str] | None = None,
    ) -> Any:
        candidate_traces = PipelineRunner._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
        if PipelineRunner._all_binary_scored_candidates_failed(candidate_traces):
            return None
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
        *,
        task_type: str = "",
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        base = dict(raw_answer) if isinstance(raw_answer, dict) else {}
        execution_context = (
            base.get("execution_context")
            if isinstance(base.get("execution_context"), dict)
            else {}
        )
        insight = base.get("insight", "")
        recommendation = base.get("recommendation", "")
        if not isinstance(insight, str):
            insight = json.dumps(insight, ensure_ascii=False, default=str) if insight else ""
        if not isinstance(recommendation, str):
            recommendation = json.dumps(recommendation, ensure_ascii=False, default=str) if recommendation else ""

        candidate_traces = cls._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
        if not any(
            isinstance(trace, dict)
            and (
                trace.get("prediction") is not None
                or (isinstance(trace.get("metrics"), dict) and bool(trace.get("metrics")))
            )
            for trace in candidate_traces
        ):
            return base
        summary_candidate_traces = cls._dedupe_candidate_traces_by_path(candidate_traces)
        no_validated_winner = cls._all_binary_scored_candidates_failed(summary_candidate_traces)
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
        execution_context = cls._normalize_learning_summary_execution_context(
            execution_context,
            winner_trace=None if no_validated_winner else winner_trace,
            candidate_traces=summary_candidate_traces,
            no_validated_winner=no_validated_winner,
        )

        compared_candidates: list[dict[str, Any]] = []
        raw_compared = base.get("compared_candidates")
        if isinstance(raw_compared, list):
            def _prediction_key(value: Any) -> str | None:
                normalized = cls._normalize_prediction_match_value(value)
                if normalized is None:
                    return None
                return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)

            scored_metrics_by_prediction: dict[str, dict[str, Any]] = {}
            for trace in summary_candidate_traces:
                if not isinstance(trace, dict):
                    continue
                metrics = trace.get("metrics")
                if not isinstance(metrics, dict) or not metrics:
                    continue
                prediction_key = _prediction_key(trace.get("prediction"))
                if prediction_key is None:
                    continue
                existing = scored_metrics_by_prediction.get(prediction_key)
                if existing is None:
                    scored_metrics_by_prediction[prediction_key] = dict(metrics)
                elif existing != metrics:
                    scored_metrics_by_prediction[prediction_key] = {}
            for item in raw_compared:
                if not isinstance(item, dict):
                    continue
                item_tool_chain = [
                    str(tool).strip()
                    for tool in item.get("tool_chain") or []
                    if str(tool).strip()
                ]
                item_prediction = cls._canonicalize_task_prediction(
                    item.get("prediction"),
                    task_type=task_type,
                    indicator_output_keys=indicator_output_keys,
                )
                item_metrics = dict(item.get("metrics") or {}) if isinstance(item.get("metrics"), dict) else {}
                if not item_metrics:
                    for trace in summary_candidate_traces:
                        if not isinstance(trace, dict):
                            continue
                        trace_tool_chain = [
                            str(tool).strip()
                            for tool in trace.get("tool_chain") or []
                            if str(tool).strip()
                        ]
                        trace_metrics = trace.get("metrics")
                        if trace_tool_chain == item_tool_chain and isinstance(trace_metrics, dict) and trace_metrics:
                            item_metrics = dict(trace_metrics)
                            break
                if not item_metrics:
                    prediction_key = _prediction_key(item_prediction)
                    propagated_metrics = scored_metrics_by_prediction.get(prediction_key or "")
                    if propagated_metrics:
                        item_metrics = dict(propagated_metrics)
                compared_candidates.append({
                    "name": canonical_candidate_label(
                        name=item.get("name"),
                        tool_chain=item.get("tool_chain"),
                    ),
                    "tool_chain": item_tool_chain,
                    "prediction": item_prediction,
                    "metrics": item_metrics,
                })
        if not compared_candidates:
            for trace in summary_candidate_traces:
                if not isinstance(trace, dict):
                    continue
                compared_candidates.append({
                    "name": canonical_candidate_label(
                        candidate_key=trace.get("candidate_key"),
                        name=trace.get("name"),
                        tool_chain=trace.get("tool_chain"),
                    ),
                    "tool_chain": list(trace.get("tool_chain") or []),
                    "prediction": cls._canonicalize_task_prediction(
                        trace.get("prediction"),
                        task_type=task_type,
                        indicator_output_keys=indicator_output_keys,
                    ),
                    "metrics": dict(trace.get("metrics") or {}),
                })
        compared_candidates = cls._dedupe_compared_candidates(compared_candidates)

        normalized = {
            "insight": str(insight).strip(),
            "recommendation": str(recommendation).strip(),
            "summary_kind": "validated_summary",
            "winner_candidate": canonical_candidate_label(
                candidate_key=base.get("winner_candidate"),
                tool_chain=base.get("winner_tool_chain"),
            ),
            "winner_prediction": cls._canonicalize_task_prediction(
                base.get("winner_prediction"),
                task_type=task_type,
                indicator_output_keys=indicator_output_keys,
            ),
            "winner_tool_chain": [
                str(tool).strip()
                for tool in base.get("winner_tool_chain") or []
                if str(tool).strip()
            ],
            "winner_metrics": (
                dict(base.get("winner_metrics") or {})
                if isinstance(base.get("winner_metrics"), dict)
                else {}
            ),
            "execution_context": execution_context,
            "compared_candidates": compared_candidates,
            "winner_state": "selected",
        }
        if winner_trace and not no_validated_winner:
            normalized["winner_candidate"] = canonical_candidate_label(
                candidate_key=winner_trace.get("candidate_key"),
                name=winner_trace.get("name"),
                tool_chain=winner_trace.get("tool_chain"),
            )
            normalized["winner_prediction"] = cls._canonicalize_task_prediction(
                winner_trace.get("prediction"),
                task_type=task_type,
                indicator_output_keys=indicator_output_keys,
            )
            normalized["winner_tool_chain"] = list(winner_trace.get("tool_chain") or [])
            normalized["winner_metrics"] = dict(winner_trace.get("metrics") or {})
            if (
                winner_source != "explicit_summary"
                or cls._learning_summary_text_conflicts_with_winner(
                    base,
                    winner_trace,
                    candidate_traces,
                )
                or cls._learning_summary_text_is_meta_finalization(base)
            ):
                grounded_insight, grounded_recommendation = cls._build_winner_aligned_learning_summary(
                    winner_trace,
                    candidate_traces,
                )
                if grounded_insight:
                    normalized["insight"] = grounded_insight
                if grounded_recommendation:
                    normalized["recommendation"] = grounded_recommendation
        else:
            normalized["winner_candidate"] = ""
            normalized["winner_prediction"] = None
            normalized["winner_tool_chain"] = []
            normalized["winner_metrics"] = {}
            normalized["summary_kind"] = "failure_summary"
            normalized["winner_state"] = "none"
            # Failure summaries should not expose a winner-like execution path.
            if isinstance(normalized.get("execution_context"), dict):
                normalized["execution_context"]["tool_chain"] = []
                normalized["execution_context"]["public_reasoning"] = []
                normalized["execution_context"]["evidence_summary"] = ""

        normalized["insight"] = canonicalize_candidate_mentions(normalized.get("insight")).strip()
        normalized["recommendation"] = canonicalize_candidate_mentions(
            normalized.get("recommendation")
        ).strip()

        for key, value in base.items():
            if key not in normalized:
                normalized[key] = value
        return normalized

    @classmethod
    def _repair_learning_summary_with_ground_truth(
        cls,
        raw_answer: dict[str, Any],
        execution_log: list[dict[str, Any]],
        *,
        ground_truth: Any,
        task_type: str = "",
        indicator_output_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Repair learning-summary winner selection with sample GT before persistence.

        New definition:
            Training summaries may mention a late challenger explored after the
            last explicit GT-evaluation call. Before persisting any learning
            artifact, we must rescore such late challengers against the sample's
            true GT and then resolve the winner from that repaired evidence, so
            memory and train-results learn the same path.
        """
        if not isinstance(raw_answer, dict) or not isinstance(execution_log, list):
            return raw_answer
        if ground_truth is None:
            return raw_answer

        repaired = cls._normalize_learning_summary_answer(
            raw_answer,
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
        candidate_traces = cls._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=indicator_output_keys,
        )
        if not candidate_traces:
            return repaired

        for trace_payload in candidate_traces:
            if not isinstance(trace_payload, dict):
                continue
            if trace_payload.get("prediction") is None:
                continue
            metrics = trace_payload.get("metrics")
            if isinstance(metrics, dict) and metrics:
                continue
            computed = _LEARNING_GT_EVALUATOR._compute_metrics(
                trace_payload.get("prediction"),
                ground_truth,
                task_type,
            )
            if isinstance(computed, dict) and computed:
                trace_payload["metrics"] = dict(computed)

        cls._propagate_metrics_across_equivalent_predictions(candidate_traces)
        summary_candidate_traces = cls._dedupe_candidate_traces_by_path(candidate_traces)
        no_validated_winner = cls._all_binary_scored_candidates_failed(summary_candidate_traces)
        evaluated_metrics = [
            trace_payload.get("metrics")
            for trace_payload in candidate_traces
            if isinstance(trace_payload, dict)
            and isinstance(trace_payload.get("metrics"), dict)
            and trace_payload.get("metrics")
        ]
        best_metrics = cls._select_best_metrics(evaluated_metrics)
        winner_trace, winner_source = cls._resolve_learning_summary_winner_trace(
            repaired,
            candidate_traces,
            best_metrics,
        )

        execution_context = (
            repaired.get("execution_context")
            if isinstance(repaired.get("execution_context"), dict)
            else {}
        )
        repaired["execution_context"] = cls._normalize_learning_summary_execution_context(
            execution_context,
            winner_trace=None if no_validated_winner else winner_trace,
            candidate_traces=summary_candidate_traces,
            no_validated_winner=no_validated_winner,
        )
        repaired["compared_candidates"] = cls._dedupe_compared_candidates([
            {
                "name": canonical_candidate_label(
                    candidate_key=trace_payload.get("candidate_key"),
                    name=trace_payload.get("name"),
                    tool_chain=trace_payload.get("tool_chain"),
                ),
                "tool_chain": [
                    str(tool).strip()
                    for tool in trace_payload.get("tool_chain") or []
                    if str(tool).strip()
                ],
                "prediction": trace_payload.get("prediction"),
                "metrics": dict(trace_payload.get("metrics") or {}),
            }
            for trace_payload in summary_candidate_traces
            if isinstance(trace_payload, dict)
        ])

        if winner_trace and not no_validated_winner:
            repaired["winner_state"] = "selected"
            repaired["winner_candidate"] = canonical_candidate_label(
                candidate_key=winner_trace.get("candidate_key"),
                name=winner_trace.get("name"),
                tool_chain=winner_trace.get("tool_chain"),
            )
            repaired["winner_prediction"] = winner_trace.get("prediction")
            repaired["winner_tool_chain"] = list(winner_trace.get("tool_chain") or [])
            repaired["winner_metrics"] = dict(winner_trace.get("metrics") or {})
            if (
                winner_source != "explicit_summary"
                or cls._learning_summary_text_conflicts_with_winner(
                    repaired,
                    winner_trace,
                    candidate_traces,
                )
                or cls._learning_summary_text_is_meta_finalization(repaired)
            ):
                grounded_insight, grounded_recommendation = cls._build_winner_aligned_learning_summary(
                    winner_trace,
                    candidate_traces,
                )
                if grounded_insight:
                    repaired["insight"] = grounded_insight
                if grounded_recommendation:
                    repaired["recommendation"] = grounded_recommendation
            return repaired

        repaired["winner_state"] = "none"
        repaired["summary_kind"] = "failure_summary"
        repaired["winner_candidate"] = ""
        repaired["winner_prediction"] = None
        repaired["winner_tool_chain"] = []
        repaired["winner_metrics"] = {}
        if isinstance(repaired.get("execution_context"), dict):
            repaired["execution_context"]["tool_chain"] = []
            repaired["execution_context"]["public_reasoning"] = []
            repaired["execution_context"]["evidence_summary"] = ""
        return repaired

    @classmethod
    def _normalize_learning_summary_execution_context(
        cls,
        execution_context: Any,
        *,
        winner_trace: dict[str, Any] | None,
        candidate_traces: list[dict[str, Any]],
        no_validated_winner: bool,
    ) -> dict[str, Any]:
        """Keep execution_context aligned with the grounded candidate paths.

        New definition:
            `execution_context` is a reusable evidence display layer. It must
            not decide the winner, and it must not drift away from the
            grounded candidate paths chosen from the execution log.

        Examples:
            - Selected winner:
              `tool_chain=["timesfm2_forecast", "llm_reasoning_indicator"]`
            - No validated winner:
              `tool_chain=[]`
        """
        normalized = dict(execution_context) if isinstance(execution_context, dict) else {}

        def _canonical_path_label(trace: dict[str, Any]) -> str:
            if not isinstance(trace, dict):
                return ""
            return canonical_candidate_label(
                candidate_key=trace.get("candidate_key"),
                name=trace.get("name"),
                tool_chain=trace.get("tool_chain"),
            ).strip()

        if winner_trace and not no_validated_winner:
            winner_chain = [
                str(tool).strip()
                for tool in winner_trace.get("tool_chain") or []
                if str(tool).strip()
            ]
            winner_public = winner_trace.get("public_reasoning_snippets")
            public_reasoning = [
                canonicalize_candidate_mentions(str(item)).strip()
                for item in (winner_public if isinstance(winner_public, list) else [])
                if str(item).strip()
            ]
            evidence_summary = canonicalize_candidate_mentions(
                cls._compact_candidate_reasoning(winner_trace, limit=2)
            ).strip()
            normalized["tool_chain"] = winner_chain
            normalized["public_reasoning"] = public_reasoning[:3]
            normalized["evidence_summary"] = evidence_summary
            return normalized

        if no_validated_winner:
            # `execution_context` is the selected evidence view, not a candidate
            # browser. Failure cases already retain explored paths under
            # `compared_candidates`, so leaving candidate labels here creates a
            # fake winner-like path and breaks schema alignment.
            normalized["tool_chain"] = []
            normalized["public_reasoning"] = []
            normalized["evidence_summary"] = ""
            return normalized

        tool_chain_labels: list[str] = []
        seen_labels: set[str] = set()
        for trace in candidate_traces:
            label = _canonical_path_label(trace)
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            tool_chain_labels.append(label)

        public_reasoning = cls._execution_context_public_reasoning(
            {"execution_context": normalized}
        ) or []
        evidence_summary = canonicalize_candidate_mentions(
            str(normalized.get("evidence_summary") or "")
        ).strip()

        normalized["tool_chain"] = tool_chain_labels
        normalized["public_reasoning"] = [
            canonicalize_candidate_mentions(item).strip()
            for item in public_reasoning
            if canonicalize_candidate_mentions(item).strip()
        ]
        normalized["evidence_summary"] = evidence_summary
        return normalized

    async def _rewrite_learning_summary_from_analysis(
        self,
        raw_answer: dict[str, Any],
        execution_log: list[dict[str, Any]],
        *,
        task_type: str,
        description: str,
        domain: str,
        scope_keys: list[str],
        fingerprint: dict[str, Any] | Any,
        profiling_results: dict[str, Any],
        horizon: int | None,
        reasoning: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
        """Rewrite final learning text from structured analysis instead of raw thoughts.

        New definition:
            Training `learning_summary` has two layers:
            1. internal executor reasoning for search/exploration
            2. final experience text for later reuse
            Only the second layer is rewritten here, and it must stay grounded
            in data characteristics, candidate evidence, and tool choice.
        """
        if not isinstance(raw_answer, dict):
            return raw_answer, None, ""

        candidate_traces = self._extract_candidate_traces(
            execution_log,
            task_type=task_type,
            indicator_output_keys=self._indicator_output_keys(scope_keys, domain, task_type),
        )
        summary_candidate_traces = self._dedupe_candidate_traces_by_path(candidate_traces)
        evaluated_metrics = [
            trace.get("metrics")
            for trace in candidate_traces
            if isinstance(trace, dict) and isinstance(trace.get("metrics"), dict) and trace.get("metrics")
        ]
        best_metrics = self._select_best_metrics(evaluated_metrics)
        if not best_metrics:
            return raw_answer, None, ""

        winner_trace, _winner_source = self._resolve_learning_summary_winner_trace(
            raw_answer,
            candidate_traces,
            best_metrics,
        )
        no_validated_winner = self._all_binary_scored_candidates_failed(summary_candidate_traces)
        if not winner_trace:
            if not no_validated_winner:
                return raw_answer, None, ""

        fallback_summary = self._execution_side_learning_summary_fallback(
            raw_answer,
            execution_context=(
                raw_answer.get("execution_context")
                if isinstance(raw_answer.get("execution_context"), dict)
                else None
            ),
        )
        sample_stats = self._build_sample_stats(fingerprint, profiling_results, horizon)
        winner_correct = (
            self._learning_outcome_correctness(dict(winner_trace.get("metrics") or {}))
            if isinstance(winner_trace, dict)
            and isinstance(winner_trace.get("metrics"), dict)
            and winner_trace.get("metrics")
            else None
        )
        structured_analysis, analysis_text = await generate_learning_analysis(
            scope_key=(scope_keys[0] if scope_keys else f"{domain}_{task_type}"),
            scope_keys=scope_keys,
            task_type=task_type,
            domain=domain,
            task_description=description,
            sample_fingerprint=sample_stats,
            candidate_traces=summary_candidate_traces,
            best_metrics=best_metrics,
            selected_winner_trace=None if no_validated_winner else winner_trace,
            winner_correct=False if no_validated_winner else winner_correct,
            fallback_summary=fallback_summary,
        )
        if not structured_analysis:
            return raw_answer, None, analysis_text

        insight, recommendation = render_learning_summary_from_analysis(
            structured_analysis,
            winner_trace=winner_trace,
            winner_correct=False if no_validated_winner else winner_correct,
        )
        rewritten = dict(raw_answer)
        if no_validated_winner:
            rewritten["winner_candidate"] = ""
            rewritten["winner_prediction"] = None
            rewritten["winner_tool_chain"] = []
            rewritten["winner_metrics"] = {}
            rewritten["winner_state"] = "none"
        if insight:
            rewritten["insight"] = insight
        if recommendation:
            rewritten["recommendation"] = recommendation
        return rewritten, structured_analysis, render_learning_analysis_text(structured_analysis)

    @classmethod
    def _learning_summary_text_is_meta_finalization(cls, summary: Any) -> bool:
        """Detect summaries that describe execution mechanics instead of evidence.

        New definition:
            A learning summary is considered meta-execution-heavy when its
            wording focuses on packaging, control-flow requirements, branch
            orchestration, or "already have a valid answer" mechanics rather
            than on the actual tool evidence that justified the winner. Such
            summaries are rebuilt from grounded candidate traces.
        """
        if not isinstance(summary, dict):
            return False
        combined = " ".join(
            str(summary.get(key) or "")
            for key in ("insight", "recommendation", "reasoning")
        )
        if not combined.strip():
            return False
        return has_meta_learning_content(combined)

    @classmethod
    def _resolve_learning_summary_winner_trace(
        cls,
        raw_answer: Any,
        candidate_traces: list[dict[str, Any]],
        best_metrics: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        if cls._all_binary_scored_candidates_failed(candidate_traces):
            return None, "all_failed"
        explicit = cls._select_learning_summary_winner_trace(raw_answer, candidate_traces)
        fallback = cls._select_winner_trace(candidate_traces, best_metrics) if best_metrics else None
        if explicit is not None:
            if fallback is not None and fallback is not explicit:
                explicit_correct = cls._learning_outcome_correctness(
                    dict(explicit.get("metrics") or {})
                )
                fallback_correct = cls._learning_outcome_correctness(
                    dict(fallback.get("metrics") or {})
                )
                explicit_metrics = dict(explicit.get("metrics") or {})
                fallback_metrics = dict(fallback.get("metrics") or {})
                # If the model-authored summary points to an explicitly incorrect
                # candidate while execution-log evaluation has a correct winner,
                # trust the evaluated winner. This preserves free-text winner hints
                # unless they contradict the scored outcome.
                if fallback_correct is True and explicit_correct is not True:
                    return fallback, "best_metrics"
                # Continuous-score tasks do not expose boolean correctness, so an
                # explicit free-text winner must not override a strictly better
                # evaluated candidate when both paths have comparable scored metrics.
                if cls._should_prefer_best_metrics_over_explicit_summary(
                    explicit_metrics,
                    fallback_metrics,
                ):
                    return fallback, "best_metrics"
            return explicit, "explicit_summary"
        if fallback is not None:
            return fallback, "best_metrics"
        return None, "none"

    @classmethod
    def _should_prefer_best_metrics_over_explicit_summary(
        cls,
        explicit_metrics: dict[str, Any],
        fallback_metrics: dict[str, Any],
    ) -> bool:
        """Decide when continuous-score evaluation should override free-text winner.

        New definition:
            For binary/categorical tasks, explicit winner hints may remain if
            they do not contradict correctness. For continuous-score tasks, if
            both explicit and fallback candidates have real scored metrics, the
            lower metric score wins even when the summary text names another
            candidate.
        """
        if not explicit_metrics or not fallback_metrics:
            return False

        explicit_correct = cls._learning_outcome_correctness(explicit_metrics)
        fallback_correct = cls._learning_outcome_correctness(fallback_metrics)
        if explicit_correct is not None or fallback_correct is not None:
            return False

        return cls._metrics_score(fallback_metrics) < cls._metrics_score(explicit_metrics)

    @classmethod
    def _select_learning_summary_winner_trace(
        cls,
        raw_answer: Any,
        candidate_traces: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not isinstance(raw_answer, dict) or not candidate_traces:
            return None

        winner_candidate = str(raw_answer.get("winner_candidate") or "").strip().lower()
        winner_tool_chain = [
            str(tool).strip()
            for tool in (
                raw_answer.get("winner_tool_chain")
                or []
            )
            if str(tool).strip()
        ]
        winner_prediction = raw_answer.get("winner_prediction")

        name_matches: list[dict[str, Any]] = []
        if winner_candidate:
            for trace in candidate_traces:
                if not isinstance(trace, dict):
                    continue
                name = str(trace.get("name") or trace.get("candidate_key") or "").strip().lower()
                if winner_candidate and winner_candidate in {name, *cls._winner_aliases(trace)}:
                    name_matches.append(trace)
            if len(name_matches) == 1:
                return name_matches[0]

        prediction_matches: list[dict[str, Any]] = []
        if winner_prediction is not None:
            prediction_matches = [
                trace
                for trace in candidate_traces
                if isinstance(trace, dict)
                and cls._predictions_match(trace.get("prediction"), winner_prediction)
            ]
            if len(prediction_matches) == 1:
                return prediction_matches[0]
            if name_matches:
                prediction_match_ids = cls._trace_identity_set(prediction_matches)
                narrowed = [
                    trace
                    for trace in name_matches
                    if isinstance(trace, dict) and id(trace) in prediction_match_ids
                ]
                if len(narrowed) == 1:
                    return narrowed[0]

        chain_matches: list[dict[str, Any]] = []
        if winner_tool_chain:
            chain_matches = [
                trace
                for trace in candidate_traces
                if isinstance(trace, dict)
                and [str(tool).strip() for tool in trace.get("tool_chain") or [] if str(tool).strip()] == winner_tool_chain
            ]
            if len(chain_matches) == 1:
                return chain_matches[0]
            if name_matches:
                chain_match_ids = cls._trace_identity_set(chain_matches)
                narrowed = [
                    trace
                    for trace in name_matches
                    if isinstance(trace, dict) and id(trace) in chain_match_ids
                ]
                if len(narrowed) == 1:
                    return narrowed[0]
            if prediction_matches:
                prediction_match_ids = cls._trace_identity_set(prediction_matches)
                for trace in chain_matches:
                    if isinstance(trace, dict) and id(trace) in prediction_match_ids:
                        return trace
        if len(name_matches) == 1:
            return name_matches[0]
        if len(prediction_matches) == 1:
            return prediction_matches[0]

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

    @classmethod
    def _learning_summary_text_conflicts_with_winner(
        cls,
        raw_answer: dict[str, Any],
        winner_trace: dict[str, Any],
        candidate_traces: list[dict[str, Any]],
    ) -> bool:
        winner_prediction = winner_trace.get("prediction")
        explicit_prediction = raw_answer.get("winner_prediction")
        if explicit_prediction is not None and not cls._predictions_match(explicit_prediction, winner_prediction):
            return True

        explicit_chain = [
            str(tool).strip()
            for tool in (
                raw_answer.get("winner_tool_chain")
                or []
            )
            if str(tool).strip()
        ]
        winner_chain = [
            str(tool).strip()
            for tool in winner_trace.get("tool_chain") or []
            if str(tool).strip()
        ]
        if explicit_chain and explicit_chain != winner_chain:
            return True

        combined = "\n".join(
            str(raw_answer.get(key) or "").strip()
            for key in ("insight", "recommendation", "reasoning")
        ).strip()
        if not combined:
            return True

        combined_lower = combined.lower()
        winner_aliases = {
            alias.lower()
            for alias in cls._winner_aliases(winner_trace)
            if isinstance(alias, str) and alias.strip()
        }
        winner_scalar = cls._summary_scalar_prediction_text(winner_prediction).lower()

        other_aliases: set[str] = set()
        other_scalars: set[str] = set()
        for trace in candidate_traces:
            if not isinstance(trace, dict) or trace is winner_trace:
                continue
            other_aliases.update(
                alias.lower()
                for alias in cls._winner_aliases(trace)
                if isinstance(alias, str) and alias.strip()
            )
            scalar = cls._summary_scalar_prediction_text(trace.get("prediction")).lower()
            if scalar and scalar != winner_scalar:
                other_scalars.add(scalar)

        if winner_scalar:
            winner_in_text = winner_scalar in combined_lower
            other_in_text = any(scalar in combined_lower for scalar in other_scalars)
            if other_in_text and not winner_in_text:
                return True

        sentences = [
            part.strip().lower()
            for part in re.split(r"(?<=[.!?])\s+|\n+", combined)
            if part.strip()
        ]
        conflict_markers = (
            "winner output",
            "best explored output",
            "returned the task-valid answer",
            "returned",
            "produced",
            "selected",
            "winner",
            "best candidate",
        )
        for sentence in sentences:
            if not any(marker in sentence for marker in conflict_markers):
                continue
            if winner_scalar and winner_scalar in sentence:
                continue
            if winner_scalar and any(
                token in sentence for token in ("answer", "output", "prediction", "label")
            ):
                return True
            if any(scalar in sentence for scalar in other_scalars):
                return True
            if any(alias in sentence for alias in other_aliases) and not any(
                alias in sentence for alias in winner_aliases
            ):
                return True
        return False

    @classmethod
    def _summary_scalar_prediction_text(cls, prediction: Any) -> str:
        parsed = cls._parse_json_like_payload(prediction)
        if isinstance(parsed, (dict, list, tuple, np.ndarray)):
            return ""
        return cls._preview_prediction(parsed, limit=80).strip()

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
        *,
        indicator_output_keys: list[str] | None = None,
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
            for key, value in raw.items():
                normalized[key] = PipelineRunner._parse_json_like_payload(value)
            return canonicalize_indicator_answer(
                normalized,
                indicator_output_keys=indicator_output_keys,
            )

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
