"""Test evaluation on MTBench using the **official test sample list**
(``test_files.json`` under ``data/splits/`` by default; see ``load_split``).

For **forecast** tasks the pipeline prediction is used directly.

For **non-forecast** tasks (MCQA, trend, indicator, correlation) the
pipeline is run first to collect tool-based analysis (statistics, trend
detection, feature extraction …).  Then a *final reasoning LLM call*
receives the **complete original context** (full time-series summary,
text, question/options) together with the pipeline's tool analysis, and
produces the structured answer.  This keeps TimeClaw's tool-augmented
analysis while preventing information loss across multi-agent hops.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.mtbench_adapter import (
    DATASET_REGISTRY,
    MTBenchSample,
    datasets_with_benchmark_tasks,
)
from timeclaw.benchmark.mtbench_support import (
    build_task_description,
    expand_dataset_tasks,
    generate_learning_analysis,
    load_sample_indices,
    scope_keys_for_dataset_task,
    select_samples_with_original_indices,
)
from timeclaw.benchmark.baseline_eval import (
    _forecast_metrics,
    _parse_json_from_text,
    _weather_trend_3way_label,
    _weather_trend_past_3way_label,
)
from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.core.config import TimeClawConfig
from timeclaw.core import trace_logger as trace
from timeclaw.llm.router import ModelRouter
from timeclaw.pipeline.runner import PipelineRunner

RESULTS_DIR = Path(".timeclaw/eval_results")


def _result_artifact_fields(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": result.get("trace_id"),
        "tool_chain": _extract_tool_chain(result),
        "gate": "PASS" if result.get("gate_accepted") else "FAIL",
        "gate_accepted": result.get("gate_accepted"),
        "gate_rejections": result.get("gate_rejections") or [],
        "total_ms": result.get("total_ms"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return value


def _result_status(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and metrics.get("error"):
        return "error"
    return "completed"


def _artifact_trace_path(trace_id: Any) -> str | None:
    if not trace_id:
        return None
    return str((Path(".timeclaw/traces") / f"{trace_id}.jsonl").as_posix())


def _result_row_for_artifact(
    row: dict[str, Any], ds_name: str, label: str,
) -> dict[str, Any]:
    json_path = RESULTS_DIR / f"{label}_{ds_name}.json"
    md_path = RESULTS_DIR / f"{label}_{ds_name}.md"
    normalized = dict(row)
    normalized["sample_index"] = row.get("idx")
    normalized["status"] = _result_status(row)
    normalized["prediction"] = row.get("predicted")
    normalized["answer"] = row.get("predicted")
    normalized["trace_path"] = _artifact_trace_path(row.get("trace_id"))
    normalized["artifact_path"] = str(json_path.as_posix())
    normalized["markdown_path"] = str(md_path.as_posix())
    return _json_safe(normalized)


def _load_eval_checkpoint(ds_name: str, label: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"{label}_{ds_name}.json"
    if not path.exists() or path.stat().st_size <= 10:
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [r for r in data if isinstance(r, dict)]


def _eval_done_keys(rows: list[dict[str, Any]]) -> set[tuple[int, str]]:
    out: set[tuple[int, str]] = set()
    for r in rows:
        if "idx" in r and "task" in r:
            try:
                out.add((int(r["idx"]), str(r["task"])))
            except (TypeError, ValueError):
                continue
    return out


def _merge_eval_rows(
    existing: list[dict[str, Any]], new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for r in existing:
        if "idx" not in r or "task" not in r:
            continue
        try:
            k = (int(r["idx"]), str(r["task"]))
        except (TypeError, ValueError):
            continue
        by_key[k] = r
    for r in new_rows:
        if "idx" not in r or "task" not in r:
            continue
        try:
            k = (int(r["idx"]), str(r["task"]))
        except (TypeError, ValueError):
            continue
        by_key[k] = r
    merged = list(by_key.values())
    merged.sort(key=lambda x: (int(x.get("idx", 0)), str(x.get("task", ""))))
    return merged

# ── Final-reasoning LLM (lazy singleton) ──────────────────────────────

_final_llm = None
_pipeline_ref = None
_record_enabled = False


def _train_kwargs(sample: "MTBenchSample", task: str) -> dict[str, Any]:
    """Pipeline-eval always runs the executor in test mode.

    New definition:
        - Even when ``split="train"`` is used to collect memory records, the
          pipeline run itself remains a **test-mode single-path analysis**.
        - Learning records are generated externally via ``record_outcome()``
          after task metrics are computed, instead of exposing GT-evaluation
          tools or training-mode multi-candidate exploration to the executor.
    """
    del sample, task
    return {}


def _get_final_llm():
    global _final_llm
    if _final_llm is None:
        cfg = TimeClawConfig.load()
        router = ModelRouter(cfg.llm)
        _final_llm = router.get_backend("executor")
    return _final_llm



def _extract_tool_chain(result: dict[str, Any]) -> list[str]:
    """Extract the list of tools used from pipeline result."""
    chain = result.get("tool_chain")
    if chain:
        return [t for t in chain if t]
    tools = []
    for entry in result.get("execution_log", []):
        t = entry.get("tool", "")
        if t and entry.get("success"):
            tools.append(t)
    return tools


def _build_pipeline_analysis_summary(result: dict[str, Any]) -> str:
    """Extract a comprehensive summary of what the pipeline's tools found.

    Reads the full structured ``observation`` / ``output`` field from
    each tool log entry. Uses generous character limits so the final-
    reasoning LLM receives enough detail to make good decisions.
    """
    parts: list[str] = []

    exec_log = result.get("execution_log", [])
    if exec_log:
        tool_lines = []
        for entry in exec_log:
            tool_name = entry.get("tool", "unknown")
            success = entry.get("success", False)
            raw_output = entry.get("observation")
            if raw_output is None:
                raw_output = entry.get("output")
            if success:
                if raw_output is not None:
                    out_str = json.dumps(raw_output, default=str) if not isinstance(raw_output, str) else raw_output
                else:
                    out_str = "(no output)"
                if len(out_str) > 50000:
                    out_str = out_str[:50000] + "..."
                tool_lines.append(f"  - {tool_name}: {out_str}")
        if tool_lines:
            parts.append("[Tool execution log]")
            parts.extend(tool_lines)

    ev = result.get("evidence_summary")
    if ev and "0 steps" not in str(ev):
        ev_str = str(ev)[:50000]
        parts.append(f"[Evidence summary] {ev_str}")

    pred = result.get("prediction")
    if pred is not None:
        if isinstance(pred, (list, np.ndarray)):
            try:
                arr = np.asarray(pred, dtype=np.float64).ravel()
                pred_str = "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"
            except (ValueError, TypeError):
                pred_str = str(pred)[:50000]
        else:
            pred_str = json.dumps(pred, default=str) if not isinstance(pred, str) else pred
            if len(pred_str) > 50000:
                pred_str = pred_str[:50000] + "..."
        parts.append(f"[Pipeline prediction] {pred_str}")

    return "\n".join(parts) if parts else "(no tool analysis available)"


async def _build_record_learning_analysis(
    *,
    sample: MTBenchSample,
    task_type: str,
    task_description: str,
    result: dict[str, Any],
    best_metrics: dict[str, Any],
    sample_fingerprint: str = "",
) -> tuple[dict[str, list[str]] | None, str]:
    candidate_traces = PipelineRunner._extract_candidate_traces(result.get("execution_log", []))
    fallback_summary = _build_pipeline_analysis_summary(result)
    return await generate_learning_analysis(
        scope_key=(scope_keys_for_dataset_task(getattr(sample, "dataset", ""), task_type) or [""])[0],
        task_type=task_type,
        domain=sample.domain,
        task_description=task_description,
        sample_fingerprint=sample_fingerprint,
        candidate_traces=candidate_traces,
        best_metrics=best_metrics,
        fallback_summary=fallback_summary,
    )


def _record_candidate_traces(result: dict[str, Any]) -> list[dict[str, Any]]:
    return PipelineRunner._extract_candidate_traces(result.get("execution_log", []))


def _record_correctness(task: str, metrics: dict[str, Any]) -> bool | None:
    """Return binary correctness only for tasks with natural binary labels.

    New definition:
        - forecast / indicator-style regression tasks do not synthesize a
          boolean ``correct`` from arbitrary thresholds. They should be
          recorded as scored outcomes and rendered as ``SCORED`` by the
          notes layer.
        - classification / QA style tasks may still provide explicit binary
          correctness from the evaluator.
    """
    if task in {"forecast", "indicator", "stock_indicator"}:
        return None
    if isinstance(metrics.get("correct"), bool):
        return metrics["correct"]
    if isinstance(metrics.get("accuracy"), (int, float)):
        return bool(metrics["accuracy"] >= 1.0)
    return None


def _record_tool_chain(result: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    """Prefer the evaluated winner candidate chain over top-level orchestration."""
    candidate_traces = _record_candidate_traces(result)
    winner_chain = PipelineRunner._select_winner_tool_chain(candidate_traces, metrics)
    if winner_chain:
        return winner_chain
    return _extract_tool_chain(result)


async def _final_llm_call(
    system_prompt: str,
    user_prompt: str,
) -> dict | None:
    """Single LLM call for final structured answer (self-correction)."""
    llm = _get_final_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = await llm.complete(messages, response_format="json")
        trace.llm_call("FinalLLM-SelfCorrection", messages, resp)
        if resp.parsed:
            return resp.parsed
        if resp.text:
            return _parse_json_from_text(resp.text)
    except Exception as e:
        logger.debug(f"[FinalLLM] failed: {e}")
    return None


async def _final_llm_text_call(
    system_prompt: str,
    user_prompt: str,
) -> str:
    llm = _get_final_llm()
    messages = prompts.prompt_messages(system_prompt, user_prompt)
    try:
        resp = await llm.complete(messages)
        trace.llm_call("FinalLLM-OfficialPrompt", messages, resp)
        return resp.text or ""
    except Exception as e:
        logger.debug(f"[FinalLLM-Text] failed: {e}")
        return ""


def _fmt_ts_compact(arr: np.ndarray, max_points: int = 200) -> str:
    """Format time series for LLM prompt (compact)."""
    a = np.asarray(arr).ravel()
    if len(a) > max_points:
        a = a[:max_points]
    return ", ".join(f"{v:.2f}" for v in a)


# ── Lightweight extractors (tried BEFORE self-correction) ────────────

def _extract_forecast_array(result_data: Any) -> np.ndarray | None:
    """Extract a numeric forecast only from structured payloads."""
    if result_data is None:
        return None
    if isinstance(result_data, np.ndarray):
        return result_data
    if isinstance(result_data, list):
        try:
            return np.asarray(result_data, dtype=np.float64)
        except (ValueError, TypeError):
            return None
    if isinstance(result_data, (int, float)):
        return np.array([float(result_data)])
    if isinstance(result_data, str):
        try:
            return _extract_forecast_array(json.loads(result_data))
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(result_data, dict):
        for key in ("point_forecast", "forecast", "prediction", "values",
                     "temperatures", "predictions", "data"):
            if key in result_data:
                try:
                    return np.asarray(result_data[key], dtype=np.float64)
                except (ValueError, TypeError):
                    continue
        if "answer" in result_data and isinstance(result_data["answer"], (list, dict)):
            sub = _extract_forecast_array(result_data["answer"])
            if sub is not None and len(sub) > 1:
                return sub
    return None


def _extract_indicator_dict(
    prediction: Any, keys: list[str],
) -> dict[str, float] | None:
    """Try to pull named numeric fields from a prediction."""
    if prediction is None:
        return None
    if isinstance(prediction, dict):
        result = {}
        for k in keys:
            if k in prediction:
                try:
                    result[k] = float(prediction[k])
                except (ValueError, TypeError):
                    pass
        if result:
            return result
    if isinstance(prediction, str):
        try:
            parsed = json.loads(prediction)
            if isinstance(parsed, dict):
                return _extract_indicator_dict(parsed, keys)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _extract_indicator_array(prediction: Any, key: str) -> np.ndarray | None:
    if prediction is None:
        return None
    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(prediction, dict):
        if key in prediction:
            try:
                return np.asarray(prediction[key], dtype=np.float64).ravel()
            except (ValueError, TypeError):
                return None
        if "answer" in prediction and isinstance(prediction["answer"], dict):
            return _extract_indicator_array(prediction["answer"], key)
    return None


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _quick_extract_letter(pred: str) -> str:
    """Fast attempt to pull a single A/B/C/D from text."""
    s = pred.strip()
    if s.upper() in ("A", "B", "C", "D"):
        return s.upper()
    for pat in [
        r"(?:answer|option|choice)\s*(?:is\s*)?[:\s]*([A-D])\b",
        r"\(([A-D])\)",
        r"^([A-D])[.)\s:]",
        r"\b([A-D])\b",
    ]:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def _quick_match_label(pred: str, valid_labels: set[str]) -> str | None:
    """Return the matching canonical label if pred contains it."""
    pred_l = pred.lower().strip()
    if pred_l in valid_labels:
        return pred_l
    for label in valid_labels:
        if label in pred_l:
            return label
    return None


# ── Per-task handlers ─────────────────────────────────────────────────

async def _eval_forecast(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    if sample.output_window is None:
        return {"metrics": {"error": "no output_window"}}

    horizon = len(sample.output_window)
    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    desc = build_task_description(sample, "forecast")

    result = await pipeline.run(
        data=sample.input_window,
        task_type="forecast",
        description=desc,
        horizon=horizon,
        text_data=text_data,
        verbose=verbose,
        domain=sample.domain,
        scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "forecast"),
        **_train_kwargs(sample, "forecast"),
    )

    prediction = result.get("prediction")
    pred_arr = _extract_forecast_array(prediction)

    if pred_arr is None and prediction is not None:
        corrected = await _final_llm_call(
            "You are a structured-output corrector. Extract or fix the forecast "
            "array from the raw model output. Return ONLY valid JSON.",
            (
                f"Domain: {sample.domain}\n"
                f"Task: time-series forecast, horizon={horizon} steps\n"
                f"Last 5 input values: {list(sample.input_window[-5:])}\n\n"
                f"Raw output: {json.dumps(prediction, default=str)[:50000]}\n\n"
                f'Target: {{"forecast": [<array of {horizon} numbers>]}}'
            ),
        )
        pred_arr = _extract_forecast_array(corrected)

    if pred_arr is not None:
        metrics = _forecast_metrics(sample.output_window, pred_arr)
        if _pipeline_ref and _record_enabled:
            analysis_structured, analysis_text = await _build_record_learning_analysis(
                sample=sample,
                task_type="forecast",
                task_description=desc,
                result=result,
                best_metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                sample_fingerprint=f"horizon={horizon}",
            )
            await _pipeline_ref.record_outcome(
                domain=sample.domain, dataset=getattr(sample, "dataset", ""),
                scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "forecast"),
                task="forecast", tool_chain=_record_tool_chain(result, metrics),
                predicted=json.dumps([float(x) for x in pred_arr[:50]]),
                ground_truth=json.dumps([float(x) for x in sample.output_window[:50]]),
                correct=_record_correctness("forecast", metrics),
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                pipeline_summary=analysis_text,
                analysis_structured=analysis_structured,
                data_fingerprint=f"horizon={horizon}",
                description=desc,
                candidate_traces=_record_candidate_traces(result),
            )
        return {
            "metrics": metrics,
            "predicted": pred_arr.tolist(),
            "ground_truth": np.asarray(sample.output_window, dtype=np.float64).ravel().tolist(),
            **_result_artifact_fields(result),
        }
    return {"metrics": {"error": "no prediction"}}


async def _eval_mcqa(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    if not sample.question or not sample.answer:
        return {"metrics": {"error": "no question/answer"}}

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    mcqa_desc = build_task_description(sample, "mcqa")

    result = await pipeline.run(
        data=sample.input_window,
        task_type="qa",
        description=mcqa_desc,
        text_data=text_data,
        verbose=verbose,
        domain=sample.domain,
        scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "mcqa"),
        **_train_kwargs(sample, "mcqa"),
    )

    analysis = _build_pipeline_analysis_summary(result)
    sys_prompt, base_user_prompt = prompts.build_prompt(sample, "mcqa")
    user_prompt = prompts.append_tool_analysis(base_user_prompt, analysis)
    raw = await _final_llm_text_call(sys_prompt, user_prompt)
    answer_char = prompts.extract_mcqa_letter(raw)

    if not answer_char:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            c = str(parsed.get("answer", "")).strip().upper()[:1]
            if c in ("A", "B", "C", "D"):
                answer_char = c

    if not answer_char:
        pred = str(result.get("prediction", ""))
        answer_char = _quick_extract_letter(pred)

    gt = sample.answer.strip().upper()[:1]
    correct = answer_char == gt

    if _pipeline_ref and _record_enabled:
            analysis_structured, analysis_text = await _build_record_learning_analysis(
                sample=sample,
                task_type="mcqa",
                task_description=mcqa_desc,
                result=result,
                best_metrics={"accuracy": 1.0 if correct else 0.0},
            )
            await _pipeline_ref.record_outcome(
                domain=sample.domain, dataset=getattr(sample, "dataset", ""),
                scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "mcqa"),
                task="mcqa", tool_chain=_extract_tool_chain(result),
                predicted=answer_char, ground_truth=gt, correct=correct,
                pipeline_summary=analysis_text,
                analysis_structured=analysis_structured,
                description=mcqa_desc,
                candidate_traces=_record_candidate_traces(result),
            )

    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": answer_char,
        "ground_truth": gt,
        **_result_artifact_fields(result),
    }


async def _eval_trend(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    trend_desc = build_task_description(sample, "trend")

    result = await pipeline.run(
        data=sample.input_window,
        task_type="classification",
        description=trend_desc,
        text_data=text_data,
        verbose=verbose,
        domain=sample.domain,
        scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "trend"),
        **_train_kwargs(sample, "trend"),
    )

    analysis = _build_pipeline_analysis_summary(result)
    tool_chain = _extract_tool_chain(result)
    sys_prompt, base_user_prompt = prompts.build_prompt(sample, "trend")
    user_prompt = prompts.append_tool_analysis(base_user_prompt, analysis)
    raw = await _final_llm_text_call(sys_prompt, user_prompt)

    if sample.domain == "weather":
        gt_label = _weather_trend_3way_label(sample)
        matched = prompts.parse_weather_trend_output(raw)
        valid = {"increasing", "decreasing", "stable"}
        if matched is None:
            parsed = _parse_json_from_text(raw)
            if isinstance(parsed, dict) and "trend" in parsed:
                matched = _quick_match_label(str(parsed["trend"]).lower(), valid)
        if matched is None:
            matched = "stable"

        if _pipeline_ref and _record_enabled:
            analysis_structured, analysis_text = await _build_record_learning_analysis(
                sample=sample,
                task_type="trend",
                task_description=trend_desc,
                result=result,
                best_metrics={"accuracy": 1.0 if matched == gt_label else 0.0},
            )
            await _pipeline_ref.record_outcome(
                domain="weather", dataset=getattr(sample, "dataset", ""),
                scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "trend"),
                task="trend", tool_chain=tool_chain,
                predicted=matched, ground_truth=gt_label,
                correct=(matched == gt_label),
                pipeline_summary=analysis_text,
                analysis_structured=analysis_structured,
                description=trend_desc,
                candidate_traces=_record_candidate_traces(result),
            )
        return {
            "metrics": {"accuracy": 1.0 if matched == gt_label else 0.0},
            "predicted": matched,
            "ground_truth": gt_label,
            **_result_artifact_fields(result),
        }
    else:
        gt_label = sample.labels.get("trend_label", "")
        if not gt_label:
            return {"metrics": {"error": "no_trend_label"}}
        valid_5 = {"<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%"}
        matched = prompts.parse_finance_trend_output(raw)
        if matched is None:
            parsed = _parse_json_from_text(raw)
            if isinstance(parsed, dict) and "trend" in parsed:
                matched = _quick_match_label(str(parsed["trend"]).strip(), valid_5)
        if matched is None:
            matched = "-2% ~ +2%"

        if _pipeline_ref and _record_enabled:
            analysis_structured, analysis_text = await _build_record_learning_analysis(
                sample=sample,
                task_type="trend",
                task_description=trend_desc,
                result=result,
                best_metrics={"accuracy": 1.0 if matched == gt_label else 0.0},
            )
            await _pipeline_ref.record_outcome(
                domain="finance", dataset=getattr(sample, "dataset", ""),
                scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "trend"),
                task="trend", tool_chain=tool_chain,
                predicted=matched, ground_truth=gt_label,
                correct=(matched == gt_label),
                pipeline_summary=analysis_text,
                analysis_structured=analysis_structured,
                description=trend_desc,
                candidate_traces=_record_candidate_traces(result),
            )
        return {
            "metrics": {"accuracy": 1.0 if matched == gt_label else 0.0},
            "predicted": matched,
            "ground_truth": gt_label,
            **_result_artifact_fields(result),
        }


async def _eval_trend_past(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    if sample.domain != "weather":
        return {"metrics": {"error": "trend_past_only_supported_for_weather"}}

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    input_days = max(1, len(sample.input_window) // 24)
    trend_desc = build_task_description(sample, "trend_past")

    result = await pipeline.run(
        data=sample.input_window,
        task_type="classification",
        description=trend_desc,
        text_data=text_data,
        verbose=verbose,
        domain=sample.domain,
        scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "trend_past"),
        **_train_kwargs(sample, "trend_past"),
    )

    analysis = _build_pipeline_analysis_summary(result)
    tool_chain = _extract_tool_chain(result)
    gt_label = _weather_trend_past_3way_label(sample)
    sys_prompt, base_user_prompt = prompts.build_prompt(sample, "trend_past")
    user_prompt = prompts.append_tool_analysis(base_user_prompt, analysis)
    raw = await _final_llm_text_call(sys_prompt, user_prompt)
    matched = prompts.parse_weather_trend_output(raw)
    valid = {"increasing", "decreasing", "stable"}
    if matched is None:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict) and "trend" in parsed:
            matched = _quick_match_label(str(parsed["trend"]).lower(), valid)
    if matched is None:
        matched = "stable"

    if _pipeline_ref and _record_enabled:
        analysis_structured, analysis_text = await _build_record_learning_analysis(
            sample=sample,
            task_type="trend_past",
            task_description=trend_desc,
            result=result,
            best_metrics={"accuracy": 1.0 if matched == gt_label else 0.0},
        )
        await _pipeline_ref.record_outcome(
            domain="weather", dataset=getattr(sample, "dataset", ""),
            scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "trend_past"),
            task="trend_past", tool_chain=tool_chain,
            predicted=matched, ground_truth=gt_label,
            correct=(matched == gt_label),
            pipeline_summary=analysis_text,
            analysis_structured=analysis_structured,
            description=trend_desc,
            candidate_traces=_record_candidate_traces(result),
        )
    return {
        "metrics": {"accuracy": 1.0 if matched == gt_label else 0.0},
        "predicted": matched,
        "ground_truth": gt_label,
        **_result_artifact_fields(result),
    }


async def _eval_indicator(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    if sample.domain == "weather":
        return await _eval_weather_indicator(sample, pipeline, verbose)
    else:
        return await _eval_stock_indicator(sample, pipeline, verbose)


async def _eval_weather_indicator(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    if sample.output_window is None:
        return {"metrics": {"error": "no output_window"}}

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    gt = prompts.weather_indicator_ground_truth(sample.output_window)
    gt_max = gt["max"]
    gt_min = gt["min"]
    gt_diff = gt["diff"]

    if gt_diff < 0.01:
        return {"metrics": {"skipped": "flat_gt"}}

    n_hours = prompts.weather_indicator_horizon_hours(sample.output_window)
    result = await pipeline.run(
        data=sample.input_window,
        task_type="indicator",
        description=build_task_description(sample, "indicator"),
        horizon=n_hours,
        text_data=text_data,
        verbose=verbose,
        domain="weather",
        scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "indicator"),
        **_train_kwargs(sample, "indicator"),
    )

    prediction = result.get("prediction")
    pred: dict[str, float] | None = None

    if isinstance(prediction, dict):
        pred = _extract_indicator_dict(prediction, ["max", "min", "diff"])

    if pred is None:
        pred_arr = _extract_forecast_array(prediction)
        if pred_arr is not None and len(pred_arr) >= 1:
            pred_arr = pred_arr[:n_hours]
            pred = {
                "max": float(np.max(pred_arr)),
                "min": float(np.min(pred_arr)),
                "diff": float(np.max(pred_arr) - np.min(pred_arr)),
            }

    if pred is None:
        pred = _extract_indicator_dict(prediction, ["max", "min", "diff"])

    if pred is None:
        analysis = _build_pipeline_analysis_summary(result)
        sys_prompt, base_user_prompt = prompts.build_prompt(sample, "indicator")
        user_prompt = prompts.append_tool_analysis(base_user_prompt, analysis)
        raw = await _final_llm_text_call(sys_prompt, user_prompt)
        pred = prompts.parse_weather_indicator_output(raw)
        if pred is None:
            parsed = _parse_json_from_text(raw)
            pred = _extract_indicator_dict(parsed, ["max", "min", "diff"])

    if pred:
        metrics = {f"{k}_mae": abs(gt[k] - pred.get(k, 0.0)) for k in gt}
        if _pipeline_ref and _record_enabled:
            hist_arr = np.asarray(sample.input_window).ravel()
            recent_24 = hist_arr[-24:] if len(hist_arr) >= 24 else hist_arr
            recent_range = float(np.max(recent_24) - np.min(recent_24))
            analysis_structured, analysis_text = await _build_record_learning_analysis(
                sample=sample,
                task_type="indicator",
                task_description=build_task_description(sample, "indicator"),
                result=result,
                best_metrics=metrics,
                sample_fingerprint=f"recent_range={recent_range:.1f}",
            )
            await _pipeline_ref.record_outcome(
                domain="weather", dataset=getattr(sample, "dataset", ""),
                scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "indicator"),
                task="indicator", tool_chain=_record_tool_chain(result, metrics),
                predicted=json.dumps(pred),
                ground_truth=json.dumps(gt),
                correct=_record_correctness("indicator", metrics),
                metrics=metrics,
                pipeline_summary=analysis_text,
                analysis_structured=analysis_structured,
                data_fingerprint=f"recent_range={recent_range:.1f}",
                description=build_task_description(sample, "indicator"),
                candidate_traces=_record_candidate_traces(result),
            )
        return {
            "metrics": metrics,
            "predicted": pred,
            "ground_truth": gt,
            **_result_artifact_fields(result),
        }
    return {"metrics": {"error": "no prediction"}}


async def _eval_stock_indicator(
    sample: MTBenchSample,
    pipeline: PipelineRunner,
    verbose: bool,
    indicator_field: str | None = None,
) -> dict[str, Any]:
    gt_macd = sample.labels.get("out_macd")
    gt_bb = sample.labels.get("out_upper_bb")
    if gt_macd is None and gt_bb is None:
        return {"metrics": {"error": "no_indicator_labels"}}

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    indicator_names = []
    if indicator_field in (None, "macd") and gt_macd is not None:
        indicator_names.append("MACD")
    if indicator_field in (None, "upper_bb") and gt_bb is not None:
        indicator_names.append("upper Bollinger Band")
    if not indicator_names:
        return {"metrics": {"error": "indicator_label_missing"}}

    task_name = "indicator" if indicator_field is None else f"indicator_{indicator_field}"
    target_horizon = 0
    if indicator_field == "macd" and gt_macd is not None:
        target_horizon = len(np.asarray(gt_macd, dtype=np.float64).ravel())
    elif indicator_field == "upper_bb" and gt_bb is not None:
        target_horizon = len(np.asarray(gt_bb, dtype=np.float64).ravel())
    else:
        if gt_macd is not None:
            target_horizon = max(target_horizon, len(np.asarray(gt_macd, dtype=np.float64).ravel()))
        if gt_bb is not None:
            target_horizon = max(target_horizon, len(np.asarray(gt_bb, dtype=np.float64).ravel()))

    result = await pipeline.run(
        data=sample.input_window,
        task_type="indicator",
        description=build_task_description(sample, task_name),
        horizon=target_horizon or (len(sample.output_window) if sample.output_window is not None else 78),
        text_data=text_data,
        verbose=verbose,
        domain="finance",
        scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), task_name),
        **_train_kwargs(sample, task_name),
    )

    analysis = _build_pipeline_analysis_summary(result)
    sys_prompt, base_user_prompt = prompts.build_prompt(sample, task_name)
    user_prompt = prompts.append_tool_analysis(base_user_prompt, analysis)
    raw = await _final_llm_text_call(sys_prompt, user_prompt)
    metrics: dict[str, float] = {}
    parsed_sequence = prompts.parse_finance_value_output(raw)
    if parsed_sequence is not None:
        if indicator_field == "macd" and gt_macd is not None:
            metrics.update(_prefixed_metrics("macd", _forecast_metrics(gt_macd, parsed_sequence)))
        elif indicator_field == "upper_bb" and gt_bb is not None:
            metrics.update(_prefixed_metrics("upper_bb", _forecast_metrics(gt_bb, parsed_sequence)))

    if not metrics:
        prediction = result.get("prediction")
        if indicator_field in (None, "macd") and gt_macd is not None:
            pred_macd = _extract_indicator_array(prediction, "macd")
            if pred_macd is not None and pred_macd.size > 0:
                metrics.update(_prefixed_metrics("macd", _forecast_metrics(gt_macd, pred_macd)))
        if indicator_field in (None, "upper_bb") and gt_bb is not None:
            pred_bb = _extract_indicator_array(prediction, "upper_bb")
            if pred_bb is not None and pred_bb.size > 0:
                metrics.update(_prefixed_metrics("upper_bb", _forecast_metrics(gt_bb, pred_bb)))

    if not metrics:
        return {"metrics": {"error": "no prediction or labels"}}

    if _pipeline_ref and _record_enabled:
        analysis_structured, analysis_text = await _build_record_learning_analysis(
            sample=sample,
            task_type=task_name,
            task_description=build_task_description(sample, task_name),
            result=result,
            best_metrics=metrics,
        )
        predicted_fields = {"macd", "upper_bb"} if indicator_field is None else {indicator_field}
        gt_payload: dict[str, Any] = {}
        if "macd" in predicted_fields:
            gt_payload["macd"] = gt_macd
        if "upper_bb" in predicted_fields:
            gt_payload["upper_bb"] = gt_bb
        await _pipeline_ref.record_outcome(
            domain="finance", dataset=getattr(sample, "dataset", ""),
            scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), task_name),
            task="stock_indicator" if indicator_field is None else task_name,
            tool_chain=_record_tool_chain(result, metrics),
            predicted=json.dumps(parsed_sequence if parsed_sequence is not None else result.get("prediction"), default=str),
            ground_truth=json.dumps(gt_payload),
            correct=_record_correctness("stock_indicator", metrics),
            metrics=metrics,
            pipeline_summary=analysis_text,
            analysis_structured=analysis_structured,
            description=build_task_description(sample, task_name),
            candidate_traces=_record_candidate_traces(result),
        )

    predicted_payload: dict[str, Any] = {}
    ground_truth_payload: dict[str, Any] = {}
    if indicator_field in (None, "macd") and gt_macd is not None:
        ground_truth_payload["macd"] = np.asarray(gt_macd, dtype=np.float64).ravel().tolist()
    if indicator_field in (None, "upper_bb") and gt_bb is not None:
        ground_truth_payload["upper_bb"] = np.asarray(gt_bb, dtype=np.float64).ravel().tolist()
    if parsed_sequence is not None:
        series_key = "values" if indicator_field is not None else "parsed_sequence"
        predicted_payload[series_key] = np.asarray(parsed_sequence, dtype=np.float64).ravel().tolist()
    elif isinstance(result.get("prediction"), dict):
        predicted_payload = result.get("prediction") or {}

    return {
        "metrics": metrics,
        "predicted": predicted_payload or None,
        "ground_truth": ground_truth_payload or None,
        **_result_artifact_fields(result),
    }


async def _eval_indicator_macd(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    return await _eval_stock_indicator(sample, pipeline, verbose, indicator_field="macd")


async def _eval_indicator_upper_bb(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    return await _eval_stock_indicator(sample, pipeline, verbose, indicator_field="upper_bb")


_CORR_CANONICAL = {
    "Strong Positive Correlation": "Strong Positive Correlation",
    "Moderate Positive Correlation": "Moderate Positive Correlation",
    "No Correlation": "No Correlation",
    "Moderate Negative Correlation": "Moderate Negative Correlation",
    "Strong Negative Correlation": "Strong Negative Correlation",
}
_CORR_LOWER_MAP = {k.lower(): k for k in _CORR_CANONICAL}


def _match_correlation(pred_str: str) -> str | None:
    """Match a prediction string to a canonical 5-way correlation label."""
    p = pred_str.strip().lower()
    if p in _CORR_LOWER_MAP:
        return _CORR_LOWER_MAP[p]
    for short, full in [
        ("strong positive", "Strong Positive Correlation"),
        ("strong negative", "Strong Negative Correlation"),
        ("moderate positive", "Moderate Positive Correlation"),
        ("moderate negative", "Moderate Negative Correlation"),
        ("no correlation", "No Correlation"),
        ("weak", "No Correlation"),
        ("none", "No Correlation"),
    ]:
        if short in p:
            return full
    return None


async def _eval_correlation(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    gt_corr = sample.labels.get("correlation", "")
    if not gt_corr:
        return {"metrics": {"error": "no correlation label"}}

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    result = await pipeline.run(
        data=sample.input_window,
        task_type="qa",
        description=build_task_description(sample, "correlation"),
        text_data=text_data,
        verbose=verbose,
        domain="finance",
        scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "correlation"),
        **_train_kwargs(sample, "correlation"),
    )

    analysis = _build_pipeline_analysis_summary(result)
    sys_prompt, base_user_prompt = prompts.build_prompt(sample, "correlation")
    user_prompt = prompts.append_tool_analysis(base_user_prompt, analysis)
    raw = await _final_llm_text_call(sys_prompt, user_prompt)
    matched = _match_correlation(raw)
    if matched is None:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict) and "correlation" in parsed:
            matched = _match_correlation(str(parsed["correlation"]))

    if matched is None:
        pred = str(result.get("prediction", ""))
        matched = _match_correlation(pred)

    if matched is None:
        matched = "No Correlation"

    correct = matched.lower().strip() == gt_corr.lower().strip()

    if _pipeline_ref and _record_enabled:
        analysis_structured, analysis_text = await _build_record_learning_analysis(
            sample=sample,
            task_type="correlation",
            task_description=build_task_description(sample, "correlation"),
            result=result,
            best_metrics={"accuracy": 1.0 if correct else 0.0},
        )
        await _pipeline_ref.record_outcome(
            domain="finance", dataset=getattr(sample, "dataset", ""),
            scope_keys=scope_keys_for_dataset_task(getattr(sample, "dataset", ""), "correlation"),
            task="correlation", tool_chain=_extract_tool_chain(result),
            predicted=matched, ground_truth=gt_corr, correct=correct,
            pipeline_summary=analysis_text,
            analysis_structured=analysis_structured,
            description=build_task_description(sample, "correlation"),
            candidate_traces=_record_candidate_traces(result),
        )

    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": matched,
        "ground_truth": gt_corr,
        **_result_artifact_fields(result),
    }


# ── Task dispatch ────────────────────────────────────────────────────

EVAL_HANDLERS = {
    "forecast": _eval_forecast,
    "trend": _eval_trend,
    "trend_past": _eval_trend_past,
    "indicator": _eval_indicator,
    "indicator_macd": _eval_indicator_macd,
    "indicator_upper_bb": _eval_indicator_upper_bb,
    "mcqa": _eval_mcqa,
    "correlation": _eval_correlation,
}


async def _eval_one_task(
    ds_name: str,
    idx: int,
    sample: MTBenchSample,
    task: str,
    pipeline: PipelineRunner,
    verbose: bool,
    sem: asyncio.Semaphore,
    task_sem: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    if task_sem is not None:
        await task_sem.acquire()
    try:
        async with sem:
            handler = EVAL_HANDLERS.get(task)
            if not handler:
                return {"idx": idx, "file": sample.file, "task": task,
                        "domain": sample.domain,
                        "metrics": {"error": "no handler"}}
            try:
                result = await asyncio.wait_for(
                    handler(sample, pipeline, verbose), timeout=1200,
                )
            except asyncio.TimeoutError:
                result = {"metrics": {"error": "timeout"}}
            except Exception as e:
                logger.error(f"[PipelineEval] {ds_name}[{idx}] {task}: {e}")
                result = {"metrics": {"error": str(e)}}

            return {
                "idx": idx,
                "file": sample.file,
                "task": task,
                "domain": sample.domain,
                **result,
            }
    finally:
        if task_sem is not None:
            task_sem.release()


# ── Main evaluation loop ─────────────────────────────────────────────

async def run_pipeline_eval(
    dataset_name: str | None = None,
    task_filter: str | None = None,
    config: TimeClawConfig | None = None,
    max_samples: int | None = None,
    sample_indices: dict[str, list[int]] | str | Path | None = None,
    use_skills: bool = True,
    verbose: bool = False,
    label: str = "timeclaw",
    concurrency: int = 80,
    per_task_concurrency: int | None = None,
    indicator_concurrency: int | None = None,
    full_dataset: bool = False,
    use_strategy: bool = True,
    split: str = "test",
    merge_checkpoint: bool = True,
) -> dict[str, Any]:
    """Run TimeClaw pipeline on dataset(s) and evaluate results.

    This is the single entry point for both memory-recording runs and eval:
      - ``split="train"``: load **official** ``train_files.json`` subset (see
        ``dataset_split``) and record outcomes into hierarchical memory.
      - ``split="test"``: load **official** ``test_files.json`` subset (default
        benchmark eval).  This is **not** “the same pool as independent training
        split in half”—independent training data lives under
        ``data_preparation/independent/`` unless you rewire loaders.

    Args:
        split: Which **named JSON list** to use: ``"train"`` or ``"test"``
               (filenames under ``data/splits/<dataset>/``).
               Ignored when full_dataset=True.
        use_skills: If False (ablation), disables all memory components.
        full_dataset: If True, loads the entire dataset (ignores split).
        use_strategy: Enables online task strategy learning.
        merge_checkpoint: If True (default), reuse ``{label}_{ds}.json`` under
            ``.timeclaw/eval_results`` and only evaluate (idx, task) pairs
            not already stored; then merge and save.
    """
    cfg = config or TimeClawConfig.load()
    # Disable MCP for benchmark: in-process calls are faster under high concurrency
    # because MCP stdio transport is single-threaded per server.
    cfg.mcp.enabled = False
    trace.configure(verbose)

    global _pipeline_ref

    try:
        disable_mem = not use_skills and not use_strategy
        pipeline = PipelineRunner(cfg, disable_memory=disable_mem)
        _pipeline_ref = pipeline
        sample_indices_map = (
            load_sample_indices(sample_indices)
            if isinstance(sample_indices, (str, Path))
            else sample_indices
        )

        try:
            from timeclaw.sbm.registry import get_global_registry, auto_register
            reg = get_global_registry()
            if not reg.list_available():
                auto_register()
            for name in reg.list_available():
                adapter = reg.get(name)
                if not adapter.is_loaded:
                    try:
                        adapter.load()
                    except Exception as e:
                        logger.warning(f"[Model pre-load] {name} failed: {e}")
            logger.info(f"[Model pre-load] {len(reg.list_available())} models ready")
        except Exception as e:
            logger.warning(f"[Model pre-load] skipped: {e}")

        global _record_enabled
        _record_enabled = (split == "train")
        sem = asyncio.Semaphore(concurrency)
        datasets_to_run = (
            [dataset_name]
            if dataset_name
            else datasets_with_benchmark_tasks()
        )
        all_results: dict[str, Any] = {}
        task_sems: dict[tuple[str, str], asyncio.Semaphore] = {}

        def _get_task_sem(ds_name: str, task_name: str) -> asyncio.Semaphore | None:
            if per_task_concurrency is None and indicator_concurrency is None:
                return None
            key = (ds_name, task_name)
            if key in task_sems:
                return task_sems[key]
            limit = (
                indicator_concurrency
                if task_name.startswith("indicator") and indicator_concurrency is not None
                else per_task_concurrency
            )
            if limit is None:
                return None
            task_sems[key] = asyncio.Semaphore(limit)
            return task_sems[key]

        from timeclaw.benchmark.mtbench_adapter import load_dataset

        for ds_name in datasets_to_run:
            info = DATASET_REGISTRY.get(ds_name)
            if not info:
                continue

            try:
                if full_dataset:
                    samples = load_dataset(ds_name, config=cfg)
                else:
                    samples = load_split(ds_name, split, config=cfg)
            except FileNotFoundError:
                logger.warning(f"[PipelineEval] No data for {ds_name}")
                continue

            selected_samples = select_samples_with_original_indices(samples, ds_name, sample_indices_map)
            if max_samples:
                selected_samples = selected_samples[:max_samples]

            supported_tasks = expand_dataset_tasks(ds_name, list(info["tasks"]))
            if task_filter:
                supported_tasks = [t for t in supported_tasks if t == task_filter]
            if not supported_tasks:
                continue

            existing_eval = _load_eval_checkpoint(ds_name, label) if merge_checkpoint else []
            done_eval = _eval_done_keys(existing_eval)

            work_specs: list[tuple[int, MTBenchSample, str]] = []
            for idx, sample in selected_samples:
                for task in supported_tasks:
                    if merge_checkpoint and (idx, task) in done_eval:
                        continue
                    work_specs.append((idx, sample, task))

            total_tasks = len(selected_samples) * len(supported_tasks)
            ptc_str = f", per_task_concurrency={per_task_concurrency}" if per_task_concurrency else ""
            ic_str = f", indicator_concurrency={indicator_concurrency}" if indicator_concurrency else ""
            logger.info(
                f"[PipelineEval:{label}] {ds_name}: {len(selected_samples)} samples "
                f"× {len(supported_tasks)} tasks = {total_tasks} evals "
                f"({len(work_specs)} pending, {len(done_eval)} in checkpoint), "
                f"concurrency={concurrency}{ptc_str}{ic_str}"
            )
            t0 = time.time()

            if work_specs:
                coros = [
                    _eval_one_task(
                        ds_name, idx, sample, task, pipeline, verbose, sem,
                        _get_task_sem(ds_name, task),
                    )
                    for idx, sample, task in work_specs
                ]
                flat_results = await asyncio.gather(*coros, return_exceptions=True)
                new_rows: list[dict[str, Any]] = []
                for r in flat_results:
                    if isinstance(r, Exception):
                        logger.error(f"[PipelineEval] {ds_name}: task error: {r}")
                        continue
                    new_rows.append(r)
                ds_results = _merge_eval_rows(existing_eval, new_rows)
            else:
                ds_results = list(existing_eval)

            elapsed = time.time() - t0
            _save_results(ds_results, ds_name, label)
            summary = _summarize(ds_results)
            all_results[ds_name] = {
                "total": len(ds_results),
                "elapsed_s": _result_total_seconds(ds_results),
                "run_elapsed_s": round(elapsed, 1),
                "summary": summary,
            }
            logger.info(
                f"[PipelineEval:{label}] {ds_name} done in {elapsed:.1f}s: "
                f"{json.dumps(summary, default=str)}"
            )

        if _pipeline_ref is not None and _record_enabled:
            await _pipeline_ref.flush_memory()
            try:
                await _pipeline_ref.distill_memory()
                mem_summary = _pipeline_ref.memory_summary()
                logger.info(f"[PipelineEval] Memory summary after training: {mem_summary}")
            except Exception as e:
                logger.debug(f"[PipelineEval] Memory distill error: {e}")

        _save_readable_summary(all_results, label)
        return {"label": label, "results": all_results}

    finally:
        if _pipeline_ref is not None:
            try:
                await _pipeline_ref.shutdown()
            except Exception:
                pass


def _save_results(
    results: list[dict[str, Any]], ds_name: str, label: str,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}_{ds_name}.json"
    artifact_rows = [_result_row_for_artifact(row, ds_name, label) for row in results]
    path.write_text(
        json.dumps(artifact_rows, indent=2, ensure_ascii=False, allow_nan=False, default=str),
    )
    _save_readable_results(results, ds_name, label)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, list[float]]] = {}
    for r in results:
        task = r.get("task", "unknown")
        metrics = r.get("metrics", {})
        if task not in by_task:
            by_task[task] = {}
        for k, v in metrics.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                by_task[task].setdefault(k, []).append(float(v))

    summary: dict[str, Any] = {}
    for task, metric_lists in by_task.items():
        task_summary: dict[str, Any] = {}
        for k, vals in metric_lists.items():
            task_summary[f"avg_{k}"] = round(float(np.mean(vals)), 3)
            task_summary["n"] = len(vals)
        summary[task] = task_summary
    return summary


def _result_total_seconds(results: list[dict[str, Any]]) -> float:
    total_ms = 0.0
    for row in results:
        value = row.get("total_ms")
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            total_ms += float(value)
    return round(total_ms / 1000.0, 3)


def _round_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, (int, float)):
        if not np.isfinite(float(value)):
            return None
        return round(float(value), 3)
    return value


def _round_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _round_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_nested(v) for v in value]
    if isinstance(value, tuple):
        return [_round_nested(v) for v in value]
    return _round_number(value)


def _compact_value(value: Any, *, max_chars: int = 300) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(_round_nested(value), ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[:max_chars] + " ..."
    return text


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _summary_table(summary: dict[str, Any]) -> str:
    if not summary:
        return "_No numeric metrics available._\n"

    metric_keys: list[str] = []
    seen: set[str] = set()
    for task_summary in summary.values():
        if not isinstance(task_summary, dict):
            continue
        for key in task_summary.keys():
            if key not in seen:
                metric_keys.append(key)
                seen.add(key)

    headers = ["task"] + metric_keys
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for task in sorted(summary.keys()):
        task_summary = summary.get(task, {})
        row = [task]
        for key in metric_keys:
            row.append(_format_metric_value(task_summary.get(key, "-")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _save_readable_results(
    results: list[dict[str, Any]], ds_name: str, label: str,
) -> None:
    summary = _summarize(results)
    lines = [
        f"# Evaluation Results: {label} / {ds_name}",
        "",
        "## Test Flow",
        "- Executor runs in `test` mode.",
        "- `spawn_subagent` and all `evaluate_*` tools are hidden in test mode.",
        "- `forecast` uses the pipeline prediction directly.",
        "- Non-forecast tasks use TimeClaw tool analysis first, then an official MTBench-style final reasoning prompt with appended tool analysis.",
        "",
        "## Dataset Summary",
        f"- Total evaluated rows: {len(results)}",
        "",
        "## Task Summary",
        _summary_table(summary).rstrip(),
        "",
        "## Sample Details",
    ]

    for row in sorted(results, key=lambda item: (int(item.get("idx", 0)), str(item.get("task", "")))):
        idx = row.get("idx", "?")
        task = row.get("task", "unknown")
        file_name = row.get("file", "-")
        lines.extend([
            "",
            f"### idx={idx} task={task}",
            f"- file: `{file_name}`",
            f"- domain: `{row.get('domain', '-')}`",
            f"- trace_id: `{row.get('trace_id') or '-'}`",
            f"- total_ms: `{row.get('total_ms') if row.get('total_ms') is not None else '-'}`",
            f"- gate_accepted: `{row.get('gate_accepted')}`",
            f"- gate_rejections: {_compact_value(row.get('gate_rejections') or [])}",
            f"- tool_chain: {_compact_value(row.get('tool_chain') or [])}",
            f"- metrics: {_compact_value({k: _round_number(v) for k, v in (row.get('metrics') or {}).items()}, max_chars=400)}",
        ])
        if "predicted" in row:
            lines.append(f"- predicted: {_compact_value(row.get('predicted'), max_chars=600)}")
        if "ground_truth" in row:
            lines.append(f"- ground_truth: {_compact_value(row.get('ground_truth'), max_chars=600)}")

    path = RESULTS_DIR / f"{label}_{ds_name}.md"
    path.write_text("\n".join(lines) + "\n")


def _save_readable_summary(all_results: dict[str, Any], label: str) -> None:
    lines = [
        f"# Evaluation Summary: {label}",
        "",
        "## Test Flow",
        "- Test-time executor keeps the same core prompt skeleton as training, but visible tools are reduced.",
        "- Test mode hides `spawn_subagent` and all GT-based `evaluate_*` tools, so execution is single-path.",
        "- For non-forecast tasks, the final answer comes from an official MTBench-style reasoning prompt plus appended TimeClaw tool analysis.",
        "- For forecast tasks, the final answer is taken directly from the pipeline prediction.",
        "",
        "## Datasets",
    ]
    if not all_results:
        lines.append("_No datasets were evaluated._")
    else:
        for ds_name in sorted(all_results.keys()):
            ds_info = all_results.get(ds_name, {})
            lines.extend([
                "",
                f"### {ds_name}",
                f"- total rows: {ds_info.get('total', 0)}",
                f"- elapsed_s: {_format_metric_value(ds_info.get('elapsed_s', '-'))}",
                f"- run_elapsed_s: {_format_metric_value(ds_info.get('run_elapsed_s', '-'))}",
            ])
            summary = ds_info.get("summary", {})
            if summary:
                lines.append("")
                lines.append(_summary_table(summary).rstrip())

    path = RESULTS_DIR / f"{label}_summary.md"
    path.write_text("\n".join(lines) + "\n")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TimeClaw Test Evaluation")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-index-file", type=str, default=None)
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--concurrency", type=int, default=80)
    parser.add_argument("--per-task-concurrency", type=int, default=None)
    parser.add_argument("--indicator-concurrency", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-merge-checkpoint",
        action="store_true",
        help="Ignore existing eval_results/{label}_*.json when resuming.",
    )
    args = parser.parse_args()

    result = await run_pipeline_eval(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        sample_indices=args.sample_index_file,
        merge_checkpoint=not args.no_merge_checkpoint,
        use_skills=not args.no_skills,
        verbose=args.verbose,
        label="timeclaw_no_skills" if args.no_skills else "timeclaw",
        concurrency=args.concurrency,
        per_task_concurrency=args.per_task_concurrency,
        indicator_concurrency=args.indicator_concurrency,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
