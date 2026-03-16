"""End-to-end pipeline runner -- single-agent architecture.

Architecture:

    1. Preprocess (deterministic, no LLM)
       - Modality detection
       - Fingerprint extraction + profiling tools
       - Strategy recommendation via fingerprint matching

    2. Execute (single Executor agent -- ReAct loop)
       - Receives ALL data (timeseries, text, images) in one context
       - Runs tools step-by-step across any modality
       - Produces structured answer with confidence + reasoning

    3. Post-process (pure code quality gate)
       - Bound check, consistency check
       - Constraint enforcement
       - No LLM calls -- deterministic validation only

Memory injection:
    - SOUL (fixed identity) + PRINCIPLES (learned) + MEMORY -> system prompt
    - Daily notes -> first-turn user message (semantic retrieval)
    - TOOLS.md -> Tool guide annotations (per-tool experience notes)
    - PLAYBOOKS/ -> first-turn user message (scene-specific SOP)
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import numpy as np
from loguru import logger

from timeclaw.agents.executor import ExecutorAgent
from timeclaw.clawhub.tool_registry import discover_and_register, get_tool_instance, list_tool_instances
from timeclaw.core.config import TimeClawConfig
from timeclaw.mcp.manager import MCPManager, MCPServerConfig
from timeclaw.core.evidence import EvidenceLog, QualityGate
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
from timeclaw.memory.experience_pool import ExperiencePool
from timeclaw.memory.hierarchical import HierarchicalMemory
from timeclaw.multimodal.detector import ModalityDetector

# Profiling tools run deterministically during preprocessing
_PROFILING_TOOLS = ["basic_stats", "autocorrelation", "stationarity_test"]
_DETECTION_TOOLS = ["detect_trend", "detect_anomaly"]
_TEXT_PROFILING_TOOLS = ["sentiment", "keyword_extract"]
_IMAGE_PROFILING_TOOLS = ["image_load", "image_feature"]


class PipelineRunner:
    """Orchestrates the TimeClaw inference pipeline.

    Single-agent architecture:

        Preprocess (pure code) -> Executor (single ReAct agent) -> Post-process (pure code)

    The Executor handles ALL modalities in one ReAct loop, selecting tools
    autonomously. It produces a structured answer with confidence and reasoning.
    Post-processing applies deterministic quality checks (bounds, constraints).

    Memory Architecture (six components):
        - SOUL.md: Fixed agent identity and behavioral guidelines (human-written)
        - Daily Notes: Raw experience log, semantically indexed
        - MEMORY.md: Causal insights distilled from daily notes
        - PRINCIPLES.md: Cross-domain strategies distilled from MEMORY
        - TOOLS.md: Per-tool usage notes distilled from daily notes
        - PLAYBOOKS/: Scene-specific SOPs distilled from MEMORY
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
        """Execute the pipeline: Preprocess -> Execute -> Post-process -> Deliver."""
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

        # -- Step 1: Preprocess (deterministic, no LLM) --
        t_pre = time.time()
        trace.stage("Preprocess", f"task: {description or task_type}")
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
        trace.decision("Pipeline", f"strategy: {suggested_strategy}, modalities: {detected_modalities}")

        # -- Step 2: Memory injection --
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
        envelope.task.metadata["modality"] = (
            detected_modalities[0] if len(detected_modalities) == 1 else "multimodal"
        )

        envelope.task.metadata["_warmup_filter"] = (
            self._apply_warmup_filter if self._total_train_samples > 0 else None
        )

        # -- Step 3: Single Executor (ReAct loop, all modalities) --
        t_exec = time.time()
        trace.stage("Execute", f"single agent, modalities: {detected_modalities}")
        logger.info(f"[Pipeline] Running single Executor agent (modalities={detected_modalities})")

        executor = self._create_executor()
        branch_envelope = envelope.with_agent("executor")
        executor_result = await executor.run(branch_envelope)
        result = executor_result["result"]

        raw_prediction = result.get("answer") or result.get("prediction")
        answer_type = result.get("answer_type", "")
        confidence = result.get("confidence")
        reasoning = result.get("reasoning", "")
        execution_log = result.get("execution_log", [])

        prediction = self._extract_prediction(raw_prediction, task_type, horizon)

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
        if prediction is not None:
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

        return {
            "prediction": prediction,
            "answer": prediction,
            "answer_type": answer_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "trace_id": trace_id,
            "timeline": timeline,
            "total_ms": total_ms,
            "gate_accepted": gate_passed,
            "evidence_summary": evidence_summary,
            "result_message": result_message,
            "modalities": detected_modalities,
            "execution_log": execution_log,
            "corrections": corrections,
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

    # -- Hierarchical Memory Interface ----------------------------------

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

    # -- Prediction extraction (generic, no task-specific logic) ----------

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
