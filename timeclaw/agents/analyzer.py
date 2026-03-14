"""AnalyzerAgent: multimodal perception and diagnosis layer.

Detects active modalities (timeseries, text, image) and runs modality-specific
profiling tools in addition to the standard time-series fingerprinting.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from timeclaw.agents.prompts import analyzer_prompt
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.clawhub.tool_registry import get_tool_instance
from timeclaw.core import trace_logger as trace
from timeclaw.core.agent_base import AgentBase
from timeclaw.core.evidence import EvidenceEntry
from timeclaw.core.message import AgentRole
from timeclaw.core.tea_protocol import TaskEnvelope
from timeclaw.fingerprint.extractor import FingerprintExtractor
from timeclaw.fingerprint.matcher import FingerprintMatcher
from timeclaw.memory.experience_pool import ExperiencePool
from timeclaw.multimodal.detector import ModalityDetector


class AnalyzerAgent(AgentBase):
    role = AgentRole.ANALYZER

    PROFILING_TOOLS = ["basic_stats", "autocorrelation", "stationarity_test"]
    DETECTION_TOOLS = ["detect_trend", "detect_anomaly"]
    TEXT_PROFILING_TOOLS = ["sentiment", "keyword_extract"]
    IMAGE_PROFILING_TOOLS = ["image_load", "image_feature"]

    def __init__(self, *args: Any, experience_pool: ExperiencePool | None = None,
                 text_backends: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.fingerprint_extractor = FingerprintExtractor()
        self.fingerprint_matcher = FingerprintMatcher()
        self._modality_detector = ModalityDetector()
        self._experience_pool = experience_pool
        self._text_backends = text_backends

    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        """Run FingerprintExtractor, detect modalities, and run per-modality profiling."""
        data = envelope.environment.data
        text_data = envelope.environment.text_data or None
        images = envelope.environment.images or None

        existing_modalities = envelope.environment.modalities
        if existing_modalities and len(existing_modalities) > 0:
            modalities = existing_modalities
            modality_details = {}
        else:
            modality_report = self._modality_detector.detect(
                ts_data=data, text_data=text_data, images=images,
            )
            modalities = modality_report["modalities"]
            modality_details = modality_report["details"]

        fingerprint = await self.fingerprint_extractor.extract(data)

        tool_results: dict[str, Any] = {}

        for tool_name in self.PROFILING_TOOLS + self.DETECTION_TOOLS:
            await self._run_tool(tool_name, data, tool_results)

        if "text" in modalities and text_data:
            text_llm = self._text_backends if self._text_backends else self.llm
            for tool_name in self.TEXT_PROFILING_TOOLS:
                await self._run_tool(
                    tool_name, data, tool_results,
                    text=text_data, llm=text_llm,
                )

        if "image" in modalities and images:
            for tool_name in self.IMAGE_PROFILING_TOOLS:
                await self._run_tool(
                    tool_name, data, tool_results,
                    images=images, llm=self.llm,
                )

        strategy = self.fingerprint_matcher.match(fingerprint)

        similar_experiences: list[dict[str, Any]] = []
        if self._experience_pool is not None:
            try:
                query = f"{envelope.task.task_type} {envelope.task.description}"
                past = await self._experience_pool.semantic_search(
                    query=query,
                    limit=3,
                    task_type=envelope.task.task_type or None,
                )
                for exp in past:
                    similar_experiences.append({
                        "trace_id": exp.trace_id,
                        "tool_chain": exp.tool_chain,
                        "quality_gate_passed": exp.quality_gate_passed,
                        "prediction_summary": exp.prediction_summary[:200],
                    })
                if similar_experiences:
                    logger.info(f"[Analyzer] Found {len(similar_experiences)} similar past experiences")
            except Exception as e:
                logger.debug(f"[Analyzer] Semantic search failed: {e}")

        return {
            "fingerprint": fingerprint,
            "tool_results": tool_results,
            "data_shape": envelope.environment.data_shape,
            "suggested_strategy": strategy,
            "modalities": modalities,
            "modality_details": modality_details,
            "similar_experiences": similar_experiences,
        }

    async def _run_tool(
        self,
        tool_name: str,
        data: Any,
        tool_results: dict[str, Any],
        **extra_params: Any,
    ) -> None:
        """Run a single tool, log evidence, and store the result."""
        try:
            tool = get_tool_instance(tool_name)
            result: ToolResult = await tool.execute(data, **extra_params)
            tool_results[tool_name] = result.data if result.success else result.error
            trace.tool_exec("analyzer", tool_name, extra_params, result)

            self.evidence_log.append(
                EvidenceEntry(
                    step=self.evidence_log.num_steps,
                    thought=f"Run {tool_name} for data profiling",
                    tool_name=tool_name,
                    observation=result.data if result.success else None,
                    critic_feedback="ok" if result.success else f"failed: {result.error}",
                )
            )
        except KeyError:
            logger.debug(f"[Analyzer] Tool '{tool_name}' not registered, skipping")
        except Exception as e:
            logger.warning(f"[Analyzer] Tool '{tool_name}' error: {e}")

    async def plan(self, perception: dict[str, Any]) -> dict[str, Any]:
        """Use LLM to synthesize tool results into a diagnostic report."""
        tool_results = perception["tool_results"]
        fingerprint = perception["fingerprint"]
        modalities = perception.get("modalities", ["timeseries"])

        system_prompt = analyzer_prompt.SYSTEM_PROMPT
        knowledge_ctx = self.get_knowledge_context("analysis")
        if knowledge_ctx:
            system_prompt += knowledge_ctx

        similar_exp = perception.get("similar_experiences", [])
        exp_ctx = ""
        if similar_exp:
            exp_ctx = "\n\nHistorical similar experiences:\n"
            for exp in similar_exp:
                exp_ctx += (
                    f"- trace={exp['trace_id']}, tools={exp['tool_chain']}, "
                    f"passed={exp['quality_gate_passed']}, summary={exp['prediction_summary']}\n"
                )

        modality_ctx = f"\nDetected modalities: {modalities}" + exp_ctx
        if "text" in modalities:
            text_tools = {k: v for k, v in tool_results.items()
                          if k in self.TEXT_PROFILING_TOOLS}
            modality_ctx += f"\nText analysis results: {str(text_tools)[:1000]}"
        if "image" in modalities:
            img_tools = {k: v for k, v in tool_results.items()
                         if k in self.IMAGE_PROFILING_TOOLS}
            modality_ctx += f"\nImage analysis results: {str(img_tools)[:1000]}"

        messages = [
            {"role": "system", "content": system_prompt + modality_ctx},
            {
                "role": "user",
                "content": analyzer_prompt.DIAGNOSIS_PROMPT.format(
                    task_type="analysis",
                    data_shape=perception.get("data_shape", "unknown"),
                    columns="unknown",
                    tool_results=str(tool_results)[:3000],
                ),
            },
        ]

        response = await self.llm.complete(messages, response_format="json")
        diagnosis = response.parsed or self._fallback_diagnosis(tool_results)

        plan_result = {
            "diagnosis": diagnosis,
            "fingerprint": fingerprint,
            "tool_results": tool_results,
            "suggested_strategy": perception.get("suggested_strategy", "arima_forecast"),
            "modalities": modalities,
            "similar_experiences": perception.get("similar_experiences", []),
        }
        trace.agent_step("analyzer", "plan", {
            "diagnosis": diagnosis,
            "suggested_strategy": plan_result["suggested_strategy"],
            "modalities": modalities,
            "tool_results_keys": list(tool_results.keys()),
        })
        return plan_result

    async def act(self, plan: dict[str, Any], envelope: TaskEnvelope) -> dict[str, Any]:
        """Update the environment with the structured fingerprint and modality info."""
        fingerprint = plan["fingerprint"]
        if hasattr(fingerprint, "to_summary"):
            fp_dict = fingerprint.to_summary()
        elif isinstance(fingerprint, dict):
            fp_dict = fingerprint
        else:
            fp_dict = {}

        drift_detected = fp_dict.get("drift_detected", False) if isinstance(fp_dict, dict) else False

        modalities = plan.get("modalities", envelope.environment.modalities or ["timeseries"])
        envelope.update_environment(
            fingerprint=fp_dict,
            drift_detected=drift_detected,
            modalities=modalities,
        )
        envelope.task.metadata["suggested_strategy"] = plan.get("suggested_strategy")
        envelope.task.metadata["modalities"] = modalities
        envelope.task.metadata["similar_experiences"] = plan.get("similar_experiences", [])

        return {
            "fingerprint": fp_dict,
            "diagnosis": plan["diagnosis"],
            "suggested_strategy": plan.get("suggested_strategy"),
            "modalities": modalities,
        }

    async def reflect(self, action_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "fingerprint_complete": bool(action_result.get("fingerprint")),
            "evidence_steps": self.evidence_log.num_steps,
            "modalities": action_result.get("modalities", []),
        }

    @staticmethod
    def _fallback_diagnosis(tool_results: dict[str, Any]) -> dict[str, Any]:
        """Build a minimal diagnosis when LLM is unavailable."""
        stats = tool_results.get("basic_stats", {})
        trend = tool_results.get("detect_trend", {})
        stationarity = tool_results.get("stationarity_test", {})

        return {
            "fingerprint": {
                "basic_stats": stats,
                "stationarity": stationarity.get("conclusion", "unknown"),
                "trend": trend.get("direction", "unknown"),
                "drift_detected": False,
                "heterogeneity_score": 0.0,
            },
            "data_quality": {"missing_ratio": 0.0, "outlier_ratio": 0.0, "noise_level": "unknown"},
            "recommendations": ["Use default analysis pipeline"],
        }
