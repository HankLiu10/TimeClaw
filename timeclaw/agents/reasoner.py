"""ReasonerAgent: multimodal fusion, logical arbitration, and SafeAuto quality gate.

When multiple executor branches produce per-modality results, the Reasoner:
1. Fuses them using ModalityFusion (late fusion)
2. Checks cross-modal consistency via LLM arbitration
3. If inconsistent, LLM decides whether to request re-analysis (returns reanalyze_request)
4. Synthesizes a final answer by combining all modality evidence via LLM
5. Runs the quality gate on the synthesized result
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.agents.prompts import reasoner_prompt
from timeclaw.core import trace_logger as trace
from timeclaw.core.agent_base import AgentBase
from timeclaw.core.evidence import EvidenceLog, QualityGate, QualityGateResult
from timeclaw.core.message import AgentRole
from timeclaw.core.tea_protocol import TaskEnvelope
from timeclaw.multimodal.confidence import estimate_all_confidences
from timeclaw.multimodal.fusion import ModalityFusion


class ReasonerAgent(AgentBase):
    role = AgentRole.REASONER

    def __init__(self, *args: Any, fusion_strategy: str = "confidence_weighted", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.quality_gate = QualityGate()
        self._fusion = ModalityFusion(strategy=fusion_strategy, llm=self.llm)
        self._fusion_strategy = fusion_strategy
        self.per_modality_results: dict[str, dict[str, Any]] = {}
        self._last_event_assessment: list[dict[str, Any]] = []

    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        modalities = envelope.environment.modalities or ["timeseries"]
        return {
            "task_type": envelope.task.task_type,
            "task_description": envelope.task.description,
            "constraints": envelope.task.constraints,
            "data": envelope.environment.data,
            "text_data": envelope.environment.text_data,
            "fingerprint": envelope.environment.fingerprint,
            "modalities": modalities,
            "is_multimodal": len(modalities) > 1,
            "per_modality_results": self.per_modality_results,
            "soul_prompt": envelope.task.metadata.get("soul_prompt", ""),
        }

    async def plan(self, perception: dict[str, Any]) -> dict[str, Any]:
        checks = ["quality_gate", "bound_check", "consistency_check"]
        if perception.get("constraints"):
            checks.append("constraint_check")
        if perception.get("is_multimodal"):
            checks.insert(0, "multimodal_fusion")
            checks.insert(1, "cross_modal_consistency")
            checks.append("llm_synthesis")
        plan_result = {
            "checks_to_run": checks,
            "constraints": perception.get("constraints", []),
            "is_multimodal": perception.get("is_multimodal", False),
            "per_modality_results": perception.get("per_modality_results", {}),
            "task_type": perception.get("task_type", "forecast"),
            "task_description": perception.get("task_description", ""),
            "text_data": perception.get("text_data"),
        }
        trace.agent_step("reasoner", "plan", {
            "checks_to_run": checks,
            "is_multimodal": plan_result["is_multimodal"],
            "modalities": list(plan_result["per_modality_results"].keys()),
        })
        return plan_result

    async def act(self, plan: dict[str, Any], envelope: TaskEnvelope) -> dict[str, Any]:
        """Fuse multimodal evidence, check consistency, synthesize answer, run gate."""
        prediction = envelope.task.metadata.get("prediction")
        constraints = plan.get("constraints", [])
        checks_results: list[dict[str, Any]] = []
        corrections: list[dict[str, Any]] = []
        fusion_report: dict[str, Any] | None = None
        reanalyze_request: dict[str, Any] | None = None

        if plan.get("is_multimodal") and plan.get("per_modality_results"):
            pmr = plan["per_modality_results"]
            task_type = plan.get("task_type", "forecast")

            fusion_report = self._fuse_multimodal(pmr, task_type=task_type)
            checks_results.append({
                "name": "multimodal_fusion",
                "passed": True,
                "details": f"Fused {len(pmr)} modalities via {self._fusion_strategy} "
                           f"(weights={fusion_report.get('weights_used', {})})",
            })
            trace.agent_step("reasoner", "fusion", fusion_report)

            # Step 2: Cross-modal consistency check — LLM decides if re-analysis needed
            cross_modal = await self._check_cross_modal_consistency(pmr, envelope)
            checks_results.append(cross_modal)
            trace.agent_step("reasoner", "cross_modal_check", cross_modal)

            if not cross_modal["passed"]:
                reanalyze_decision = await self._decide_reanalysis(
                    pmr, cross_modal, envelope
                )
                if reanalyze_decision.get("should_reanalyze"):
                    reanalyze_request = reanalyze_decision
                    logger.info(
                        f"[Reasoner] LLM requests re-analysis: "
                        f"{reanalyze_decision.get('reasoning', '')[:2000]}"
                    )

            # Step 3: LLM synthesizes final answer from all modality evidence
            #
            # If the executor already produced a structured dict from an
            # LLM-based tool (e.g. llm_reasoning), pass it through directly
            # to avoid degrading precision via re-synthesis.
            ts_pred = pmr.get("timeseries", {}).get("prediction")
            ts_exec_log = pmr.get("timeseries", {}).get("execution_log", [])
            last_tool = ts_exec_log[-1].get("tool", "") if ts_exec_log else ""
            skip_synthesis = (
                isinstance(ts_pred, dict)
                and last_tool.startswith("llm_reasoning_")
                and len(ts_pred) > 0
            )

            if skip_synthesis:
                prediction = ts_pred
                checks_results.append({
                    "name": "llm_synthesis",
                    "passed": True,
                    "details": f"Structured dict from {last_tool} — passthrough (no re-synthesis)",
                })
                fusion_report["synthesized_answer"] = {
                    "answer": ts_pred, "confidence": 0.8,
                    "reasoning": "Direct passthrough of executor indicator dict",
                }
            else:
                synthesized = await self._synthesize_answer(pmr, envelope)
                trace.agent_step("reasoner", "synthesis", synthesized)
                if synthesized.get("answer") is not None:
                    prediction = synthesized["answer"]
                    checks_results.append({
                        "name": "llm_synthesis",
                        "passed": True,
                        "details": f"LLM synthesized answer from {len(pmr)} modalities "
                                   f"(confidence={synthesized.get('confidence', '?')})",
                    })
                    fusion_report["synthesized_answer"] = synthesized
                else:
                    fused_pred = fusion_report.get("prediction")
                    if fused_pred is not None:
                        prediction = fused_pred

        # Step 4: Quality gate checks
        gate_result = self.quality_gate.check(
            prediction, self.evidence_log, constraints
        )
        checks_results.append({
            "name": "quality_gate",
            "passed": gate_result.accepted,
            "details": gate_result.rejection_summary,
        })

        if prediction is not None:
            bound_result = self._check_bounds(prediction, constraints)
            checks_results.append(bound_result)

            if not bound_result["passed"]:
                corrected, correction = self._apply_bound_correction(prediction, constraints)
                if correction:
                    corrections.append(correction)
                    prediction = corrected

            consistency = self._check_consistency(prediction, envelope.environment.data)
            checks_results.append(consistency)

        all_passed = all(c["passed"] for c in checks_results)

        if not all_passed and prediction is not None:
            llm_verdict = await self._llm_verify(prediction, envelope, checks_results)
            checks_results.append({
                "name": "llm_review",
                "passed": llm_verdict.get("accepted", False),
                "details": llm_verdict.get("reasoning", ""),
            })

        final_accepted = all(c["passed"] for c in checks_results)
        final_gate = QualityGateResult(
            accepted=final_accepted,
            rejections=[c["details"] for c in checks_results if not c["passed"]],
        )

        act_result = {
            "gate_result": final_gate,
            "checks": checks_results,
            "corrections": corrections,
            "prediction": prediction,
            "fusion_report": fusion_report,
            "reanalyze_request": reanalyze_request,
        }
        trace.agent_step("reasoner", "act", {
            "gate_accepted": final_accepted,
            "checks": checks_results,
            "corrections": corrections,
            "prediction_preview": str(prediction)[:5000] if prediction is not None else None,
            "reanalyze_requested": reanalyze_request is not None,
        })
        return act_result

    async def reflect(self, action_result: dict[str, Any]) -> dict[str, Any]:
        gate: QualityGateResult = action_result.get("gate_result", QualityGateResult(accepted=False))
        return {
            "accepted": gate.accepted,
            "num_checks": len(action_result.get("checks", [])),
            "num_corrections": len(action_result.get("corrections", [])),
            "has_fusion": action_result.get("fusion_report") is not None,
            "reanalyze_requested": action_result.get("reanalyze_request") is not None,
        }

    # ── Multimodal Fusion ──

    def _fuse_multimodal(
        self,
        per_modality_results: dict[str, dict[str, Any]],
        task_type: str = "forecast",
    ) -> dict[str, Any]:
        """Fuse per-modality executor results using the configured strategy."""
        evidence: dict[str, Any] = {}
        for modality, result in per_modality_results.items():
            entry: dict[str, Any] = {}
            if "prediction" in result and result["prediction"] is not None:
                entry["prediction"] = result["prediction"]
            if "confidence" in result:
                entry["confidence"] = result["confidence"]
            if "label" in result:
                entry["label"] = result["label"]
            if not entry:
                entry["prediction"] = result.get("execution_log", "no data")
            evidence[modality] = entry

        modality_confidences = estimate_all_confidences(per_modality_results)
        logger.info(f"[Reasoner] Modality confidences: {modality_confidences}")

        return self._fusion.fuse(
            evidence,
            task_type=task_type,
            modality_confidences=modality_confidences,
        )

    # ── Cross-modal Consistency ──

    @staticmethod
    def _summarize_ts_tail(data: Any, n: int = 15) -> str:
        """Build a concise summary of the time-series tail for cross-modal checking."""
        try:
            arr = np.asarray(data, dtype=float).flatten()
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return "(no numeric data)"
            tail = arr[-min(n, len(arr)):]
            tail_str = ", ".join(f"{v:.2f}" for v in tail)
            return (
                f"Series length: {len(arr)} steps. "
                f"Overall range: [{float(arr.min()):.2f}, {float(arr.max()):.2f}], "
                f"mean={float(arr.mean()):.2f}.\n"
                f"Last {len(tail)} values (most recent on the right): [{tail_str}]"
            )
        except (ValueError, TypeError):
            return "(non-numeric data)"

    async def _check_cross_modal_consistency(
        self,
        per_modality_results: dict[str, dict[str, Any]],
        envelope: TaskEnvelope,
    ) -> dict[str, Any]:
        """Cross-modal consistency check with event-vs-data verification.

        The LLM receives both modality summaries AND the raw time-series tail,
        so it can autonomously judge whether text-described events are already
        reflected in the numerical data.
        """
        summaries = self._build_modality_summaries(per_modality_results)
        ts_tail = self._summarize_ts_tail(envelope.environment.data)

        text_data = envelope.environment.text_data
        text_block = ""
        if text_data:
            for td in text_data[:10]:
                src = td.get("source", "?")
                ts = td.get("timestamp", "?")
                content = td.get("content", td.get("text", ""))
                text_block += f"  [{src}, timestamp={ts}]: {str(content)[:5000]}\n"

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a cross-modal consistency checker for time-series forecasting.\n\n"
                        "You will receive:\n"
                        "1. Per-modality analysis results (timeseries forecast, text events, etc.)\n"
                        "2. The actual time-series data tail (recent values)\n"
                        "3. The original text data with timestamps\n\n"
                        "Your job is to:\n"
                        "A. For each event described in the text, examine the time-series data "
                        "to determine whether the event's impact is ALREADY VISIBLE in the data. "
                        "If the text describes an upcoming change and the recent data already "
                        "shows that change, then the event has already occurred and is captured "
                        "in the data.\n"
                        "B. Assess whether the modality conclusions are consistent.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "event_assessment": [\n'
                        '    {"event": "description", "already_in_data": bool, "evidence": "why you think so"}\n'
                        "  ],\n"
                        '  "consistent": bool,\n'
                        '  "conflicts": ["list of real conflicts if any"],\n'
                        '  "reasoning": "overall assessment"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {envelope.task.task_type}\n"
                        f"Task description: {envelope.task.description}\n\n"
                        f"=== TIME-SERIES DATA (recent values) ===\n{ts_tail}\n\n"
                        f"=== ORIGINAL TEXT DATA ===\n{text_block}\n"
                        f"=== PER-MODALITY RESULTS ===\n" + "\n".join(summaries)
                    ),
                },
            ]
            response = await self.llm.complete(messages, response_format="json")
            parsed = response.parsed or {}
            consistent = parsed.get("consistent", True)
            conflicts = parsed.get("conflicts", [])

            self._last_event_assessment = parsed.get("event_assessment", [])

            return {
                "name": "cross_modal_consistency",
                "passed": consistent,
                "details": "; ".join(conflicts) if conflicts else "modalities consistent",
                "reasoning": parsed.get("reasoning", ""),
                "event_assessment": self._last_event_assessment,
            }
        except Exception as e:
            logger.warning(f"[Reasoner] Cross-modal check failed: {e}")
            self._last_event_assessment = []
            return {
                "name": "cross_modal_consistency",
                "passed": True,
                "details": f"check skipped: {e}",
            }

    # ── Re-analysis Decision ──

    async def _decide_reanalysis(
        self,
        per_modality_results: dict[str, dict[str, Any]],
        cross_modal_result: dict[str, Any],
        envelope: TaskEnvelope,
    ) -> dict[str, Any]:
        """LLM decides whether cross-modal conflicts warrant re-analysis."""
        summaries = self._build_modality_summaries(per_modality_results)

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a decision agent. Cross-modal analysis found conflicts between "
                        "modality results. You must decide whether to request re-analysis of "
                        "specific modalities, or proceed with the current results.\n\n"
                        "Consider:\n"
                        "- Is the conflict severe enough to invalidate the conclusion?\n"
                        "- Could re-analysis with different tools resolve the conflict?\n"
                        "- Is one modality clearly more reliable for this task?\n\n"
                        "Return JSON:\n"
                        '{"should_reanalyze": bool, '
                        '"modalities_to_reanalyze": [str], '
                        '"suggested_tools": [str], '
                        '"reasoning": str}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {envelope.task.task_type}\n"
                        f"Description: {envelope.task.description}\n\n"
                        f"Conflicts found:\n{cross_modal_result.get('details', '')}\n"
                        f"Conflict reasoning: {cross_modal_result.get('reasoning', '')}\n\n"
                        f"Per-modality results:\n" + "\n".join(summaries)
                    ),
                },
            ]
            response = await self.llm.complete(messages, response_format="json")
            return response.parsed or {"should_reanalyze": False, "reasoning": "parse failed"}
        except Exception as e:
            logger.warning(f"[Reasoner] Re-analysis decision failed: {e}")
            return {"should_reanalyze": False, "reasoning": f"decision failed: {e}"}

    # ── LLM Synthesis ──

    @staticmethod
    def _fmt_raw_ts(data: Any) -> str:
        """Format raw time-series data for injection into synthesis prompt."""
        try:
            arr = np.asarray(data, dtype=np.float64).ravel()
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return "(no numeric data)"
            vals = ", ".join(f"{v:.2f}" for v in arr)
            return (
                f"Length: {len(arr)} steps, range: [{float(arr.min()):.2f}, "
                f"{float(arr.max()):.2f}], mean: {float(arr.mean()):.2f}\n"
                f"Values: [{vals}]"
            )
        except (ValueError, TypeError):
            return "(non-numeric data)"

    async def _synthesize_answer(
        self,
        per_modality_results: dict[str, dict[str, Any]],
        envelope: TaskEnvelope,
    ) -> dict[str, Any]:
        """LLM synthesizes a final answer by combining all modality evidence.

        Receives the complete raw time-series data, domain context, full text,
        and the event_assessment from cross-modal check so the synthesis LLM
        can make well-informed decisions.
        """
        summaries = self._build_modality_summaries(per_modality_results, detailed=True)

        task_type = envelope.task.task_type
        description = envelope.task.description
        text_data = envelope.environment.text_data
        domain = envelope.task.metadata.get("domain", "")

        raw_ts_block = ""
        if envelope.environment.data is not None:
            raw_ts_block = (
                "\n=== RAW TIME-SERIES DATA ===\n"
                + self._fmt_raw_ts(envelope.environment.data)
                + "\n"
            )

        domain_block = ""

        text_context = ""
        if text_data:
            for td in text_data[:10]:
                content = td.get("content", td.get("text", ""))
                source = td.get("source", "unknown")
                text_context += f"\n[{source}]: {str(content)[:10000]}\n"

        assessment_block = ""
        if self._last_event_assessment:
            assessment_block = "\n=== CROSS-MODAL EVENT VERIFICATION (from previous step) ===\n"
            for ea in self._last_event_assessment:
                status = "ALREADY IN DATA" if ea.get("already_in_data") else "NOT YET IN DATA"
                assessment_block += (
                    f"  - {ea.get('event', '?')} → {status}\n"
                    f"    Evidence: {ea.get('evidence', 'N/A')}\n"
                )
            assessment_block += (
                "\nThe above assessment was produced by examining the actual time-series "
                "values. Trust it when deciding whether to adjust the forecast.\n"
            )

        soul_block_text = envelope.task.metadata.get("soul_prompt", "")
        l3_block = f"\n\n{soul_block_text}" if soul_block_text else ""

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the final synthesis agent of TimeClaw. Your job is to produce "
                        "the definitive answer by combining evidence from all modality-specialized "
                        "executor branches.\n\n"
                        "You have access to:\n"
                        "- The COMPLETE raw time-series data (all values)\n"
                        "- Domain-specific context\n"
                        "- Time-series analysis results (statistics, trends, forecasts, anomalies)\n"
                        "- Text analysis results (sentiment, keywords, events, summaries)\n"
                        "- The original text data (news, reports, questions)\n"
                        "- Cross-modal event verification: tells you which text events are "
                        "already reflected in the time-series data and which are not\n\n"
                        "IMPORTANT: If the event verification says an event is 'ALREADY IN DATA', "
                        "do NOT apply its impact again — the forecast already accounts for it. "
                        "Only adjust the forecast for events marked 'NOT YET IN DATA'.\n\n"
                        "Read the task description carefully to understand what output is expected, "
                        "then produce the most accurate answer based on all available evidence.\n\n"
                        "Return JSON:\n"
                        '{"answer": <your answer>, "confidence": float 0-1, '
                        '"reasoning": "step-by-step explanation of how you combined evidence"}'
                        + l3_block
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task type: {task_type}\n"
                        f"Task description: {description}\n"
                        f"{domain_block}"
                        f"{raw_ts_block}\n"
                        f"=== MODALITY EVIDENCE ===\n" + "\n".join(summaries) + "\n\n"
                        f"=== ORIGINAL TEXT DATA ==={text_context}\n"
                        f"{assessment_block}\n"
                        f"Based on ALL the above evidence, what is your final answer?"
                    ),
                },
            ]
            response = await self.llm.complete(messages, response_format="json")
            parsed = response.parsed or {}
            logger.info(
                f"[Reasoner] LLM synthesis: answer={str(parsed.get('answer', ''))[:2000]}, "
                f"confidence={parsed.get('confidence', '?')}"
            )
            return parsed
        except Exception as e:
            logger.warning(f"[Reasoner] LLM synthesis failed: {e}")
            return {"answer": None, "reasoning": f"synthesis failed: {e}"}

    def _build_modality_summaries(
        self,
        per_modality_results: dict[str, dict[str, Any]],
        detailed: bool = False,
    ) -> list[str]:
        """Build human-readable summaries of per-modality results.

        Uses generous limits so the synthesis LLM receives enough
        information to make well-informed decisions.
        """
        summaries = []
        for modality, result in per_modality_results.items():
            pred = result.get("prediction")
            log = result.get("execution_log", [])
            method = result.get("method_used", "")

            lines = [f"[{modality.upper()}]"]
            lines.append(f"  Tools executed: {len(log)}")

            for entry in log:
                tool = entry.get("tool", "?")
                ok = entry.get("success", False)
                thought = (entry.get("thought") or "")[:1000]
                obs = entry.get("observation")
                if obs is None and entry.get("output") is not None:
                    obs = entry.get("output")
                try:
                    obs_text = json.dumps(obs, ensure_ascii=False, default=str) if obs is not None else ""
                except TypeError:
                    obs_text = str(obs) if obs is not None else ""
                obs_text = obs_text[:50000]
                line = f"  - {tool}: {'OK' if ok else 'FAIL'}"
                if thought:
                    line += f" | thought={thought}"
                if obs_text:
                    line += f" | observation={obs_text}"
                lines.append(line)

            if detailed:
                pred_str = self._format_prediction(pred)
                lines.append(f"  Prediction: {pred_str}")
                if method:
                    lines.append(f"  Method: {method[:5000]}")

            summaries.append("\n".join(lines))
        return summaries

    @staticmethod
    def _format_prediction(pred: Any) -> str:
        """Format a prediction for the synthesis prompt without aggressive truncation."""
        if pred is None:
            return "None"
        if isinstance(pred, np.ndarray):
            flat = pred.ravel()
            return "[" + ", ".join(f"{v:.4f}" for v in flat) + "]"
        if isinstance(pred, list):
            try:
                arr = np.asarray(pred, dtype=np.float64).ravel()
                return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"
            except (ValueError, TypeError):
                pass
        s = str(pred)
        return s[:50000] if len(s) > 50000 else s

    # ── Standard Checks ──

    @staticmethod
    def _check_bounds(prediction: Any, constraints: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            arr = np.asarray(prediction, dtype=float)
            for c in constraints:
                if c.get("type") == "bound":
                    lower = c.get("lower")
                    upper = c.get("upper")
                    if lower is not None and np.any(arr < lower):
                        return {
                            "name": "bound_check",
                            "passed": False,
                            "details": f"values below lower bound {lower}",
                        }
                    if upper is not None and np.any(arr > upper):
                        return {
                            "name": "bound_check",
                            "passed": False,
                            "details": f"values above upper bound {upper}",
                        }
        except (ValueError, TypeError):
            pass
        return {"name": "bound_check", "passed": True, "details": "within bounds"}

    @staticmethod
    def _apply_bound_correction(
        prediction: Any, constraints: list[dict[str, Any]]
    ) -> tuple[Any, dict[str, Any] | None]:
        try:
            arr = np.asarray(prediction, dtype=float)
            clipped = arr.copy()
            for c in constraints:
                if c.get("type") == "bound":
                    clipped = np.clip(clipped, c.get("lower"), c.get("upper"))
            if not np.array_equal(arr, clipped):
                return clipped, {"type": "bound_clip", "details": "clipped to constraint bounds"}
        except (ValueError, TypeError):
            pass
        return prediction, None

    @staticmethod
    def _check_consistency(prediction: Any, historical_data: Any) -> dict[str, Any]:
        try:
            pred_arr = np.asarray(prediction, dtype=float).flatten()
            hist_arr = np.asarray(historical_data, dtype=float).flatten()
            hist_arr = hist_arr[~np.isnan(hist_arr)]

            if len(hist_arr) == 0 or len(pred_arr) == 0:
                return {"name": "consistency_check", "passed": True, "details": "insufficient data"}

            hist_std = np.std(hist_arr)
            hist_mean = np.mean(hist_arr)
            pred_mean = np.mean(pred_arr)

            if hist_std > 0 and abs(pred_mean - hist_mean) > 5 * hist_std:
                return {
                    "name": "consistency_check",
                    "passed": False,
                    "details": f"prediction mean {pred_mean:.2f} deviates >5 sigma from history {hist_mean:.2f}",
                }
        except (ValueError, TypeError):
            pass
        return {"name": "consistency_check", "passed": True, "details": "consistent with history"}

    async def _llm_verify(
        self,
        prediction: Any,
        envelope: TaskEnvelope,
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            pred_str = str(prediction)[:5000]
            hist_data = envelope.environment.data
            hist_summary = ""
            try:
                hist_arr = np.asarray(hist_data, dtype=float).flatten()
                hist_arr = hist_arr[~np.isnan(hist_arr)]
                if len(hist_arr) > 0:
                    hist_summary = (
                        f"Historical range: [{float(np.nanmin(hist_arr)):.2f}, "
                        f"{float(np.nanmax(hist_arr)):.2f}], mean={float(np.mean(hist_arr)):.2f}"
                    )
            except (ValueError, TypeError):
                hist_summary = "Historical data not numeric"

            messages = [
                {"role": "system", "content": reasoner_prompt.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task: {envelope.task.task_type}\n"
                        f"Prediction: {pred_str}\n"
                        f"{hist_summary}\n"
                        f"Check results: {str(checks)[:10000]}\n"
                        f"Constraints: {str(envelope.task.constraints)}\n\n"
                        "Is this prediction acceptable? Return JSON with 'accepted' (bool) and 'reasoning'."
                    ),
                },
            ]
            response = await self.llm.complete(messages, response_format="json")
            return response.parsed or {"accepted": True, "reasoning": "LLM review passed"}
        except Exception as e:
            logger.warning(f"[Reasoner] LLM verify failed: {e}")
            return {"accepted": True, "reasoning": f"LLM verify skipped: {e}"}
