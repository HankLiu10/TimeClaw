"""Test Evaluation: evaluate TimeClaw on all MTBench test splits.

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
from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY, MTBenchSample
from timeclaw.benchmark.baseline_eval import (
    _forecast_metrics,
    _parse_json_from_text,
    _weather_trend_3way_label,
)
from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.core.config import TimeClawConfig
from timeclaw.core import trace_logger as trace
from timeclaw.llm.router import ModelRouter
from timeclaw.pipeline.runner import PipelineRunner

RESULTS_DIR = Path(".timeclaw/eval_results")

# ── Final-reasoning LLM (lazy singleton) ──────────────────────────────

_final_llm = None
_pipeline_ref = None
_record_enabled = False


def _train_kwargs(sample: "MTBenchSample", task: str) -> dict[str, Any]:
    """Build extra kwargs for pipeline.run() when in training mode."""
    if not _record_enabled:
        return {}
    gt: Any = None
    if task == "forecast" and sample.output_window is not None:
        gt = list(sample.output_window)
    elif task == "mcqa":
        gt = sample.answer
    elif task == "trend":
        if sample.domain == "weather":
            gt = _weather_trend_3way_label(sample)
        else:
            gt = sample.labels.get("trend_label", "")
    elif task == "indicator":
        if sample.domain == "weather" and sample.output_window is not None:
            gt = {"max": float(np.max(sample.output_window)),
                  "min": float(np.min(sample.output_window)),
                  "diff": float(np.max(sample.output_window) - np.min(sample.output_window))}
        else:
            gt_parts: dict[str, Any] = {}
            if sample.labels.get("out_macd") is not None:
                gt_parts["macd"] = float(sample.labels["out_macd"])
            if sample.labels.get("out_upper_bb") is not None:
                gt_parts["upper_bb"] = float(sample.labels["out_upper_bb"])
            gt = gt_parts or None
    elif task == "correlation":
        gt = sample.labels.get("correlation", "")
    return {"mode": "train", "ground_truth": gt}


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

    Reads the ``output`` field (full tool data) when available, falling
    back to ``result_summary``.  Uses generous character limits so the
    final-reasoning LLM receives enough detail to make good decisions.
    """
    parts: list[str] = []

    exec_log = result.get("execution_log", [])
    if exec_log:
        tool_lines = []
        for entry in exec_log:
            tool_name = entry.get("tool", "unknown")
            success = entry.get("success", False)
            raw_output = entry.get("output")
            summary = entry.get("result_summary", "")
            if success:
                if raw_output is not None:
                    out_str = json.dumps(raw_output, default=str) if not isinstance(raw_output, str) else raw_output
                elif summary:
                    out_str = summary
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


def _fmt_ts_compact(arr: np.ndarray, max_points: int = 200) -> str:
    """Format time series for LLM prompt (compact)."""
    a = np.asarray(arr).ravel()
    if len(a) > max_points:
        a = a[:max_points]
    return ", ".join(f"{v:.2f}" for v in a)


# ── Lightweight extractors (tried BEFORE self-correction) ────────────

def _extract_forecast_array(result_data: Any) -> np.ndarray | None:
    """Best-effort extraction of a numeric array from various formats."""
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
            pass
        numbers = re.findall(r"-?\d+\.?\d*", result_data)
        if numbers:
            try:
                return np.asarray([float(n) for n in numbers], dtype=np.float64)
            except (ValueError, TypeError):
                pass
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
        list_vals = []
        for v in result_data.values():
            if isinstance(v, (list, np.ndarray)):
                try:
                    arr = np.asarray(v, dtype=np.float64).ravel()
                    if len(arr) >= 2:
                        list_vals.extend(arr.tolist())
                except (ValueError, TypeError):
                    pass
        if list_vals:
            return np.asarray(list_vals, dtype=np.float64)
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
            pass
        result = {}
        for k in keys:
            m = re.search(rf'["\']?{k}["\']?\s*[:=]\s*(-?\d+\.?\d*)', prediction)
            if m:
                result[k] = float(m.group(1))
        if result:
            return result
    return None


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

    input_days = len(sample.input_window) // 24
    horizon_days = horizon // 24
    ds = getattr(sample, "dataset", "")

    if sample.domain == "weather":
        desc = (
            f"The input time-series represents temperature readings from a location "
            f"in the United States, where summers are hot and winters are cold. "
            f"Weather conditions can also be affected by storms, heavy rain, and cold fronts. "
            f"The daytime is usually warmer than the nighttime. "
            f"Every 24 temperature readings represent a full day from 00:00 to 23:00. "
            f"Predict the next {horizon_days} day(s) of hourly temperature readings "
            f"({horizon} values). Return a JSON array of {horizon} numbers."
        )
    else:
        input_desc = "7 days at 5-minute intervals" if "short" in ds else "30 days of daily prices"
        output_desc = f"next {'1 day' if horizon_days <= 1 else f'{horizon_days} days'}"
        desc = (
            f"Time Series Stock Price Data recorded at {input_desc}. "
            f"Predict the stock prices for the {output_desc} ({horizon} values). "
            f"Return a JSON array of {horizon} numbers."
        )

    result = await pipeline.run(
        data=sample.input_window,
        task_type="forecast",
        description=desc,
        horizon=horizon,
        text_data=text_data,
        verbose=verbose,
        domain=sample.domain,
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
            mae = metrics.get("mae", 99)
            await _pipeline_ref.record_outcome(
                domain=sample.domain, dataset=getattr(sample, "dataset", ""),
                task="forecast", tool_chain=_extract_tool_chain(result),
                predicted=json.dumps([float(x) for x in pred_arr[:50]]),
                ground_truth=json.dumps([float(x) for x in sample.output_window[:50]]),
                correct=(mae < 5.0),
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                pipeline_summary=str(result.get("evidence_summary", ""))[:2000],
                data_fingerprint=f"horizon={horizon},last5={[float(x) for x in sample.input_window[-5:]]}",
            )
        return {"metrics": metrics, "trace_id": result.get("trace_id")}
    return {"metrics": {"error": "no prediction"}}


async def _eval_mcqa(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    if not sample.question or not sample.answer:
        return {"metrics": {"error": "no question/answer"}}

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    mcqa_desc = (
        f"{sample.question[:5000]}\n\n"
        f"Answer the multiple-choice question by choosing exactly one of A, B, C, or D. "
        'Return ONLY JSON: {"answer": "X"}'
    )

    result = await pipeline.run(
        data=sample.input_window,
        task_type="qa",
        description=mcqa_desc,
        text_data=text_data,
        verbose=verbose,
        domain=sample.domain,
        **_train_kwargs(sample, "mcqa"),
    )

    analysis = _build_pipeline_analysis_summary(result)

    ts_str = _fmt_ts_compact(sample.input_window)
    text_section = f"\nContextual text:\n{sample.text[:10000]}\n" if sample.text else ""

    sys_prompt = (
        f"You are an expert {sample.domain} analyst. "
        "Answer the multiple-choice question by choosing exactly one of A, B, C, or D.\n"
        'Return ONLY JSON: {{"answer": "X"}}'
    )
    user_prompt = (
        f"Time series data:\n[{ts_str}]\n"
        f"{text_section}\n"
        f"Question:\n{sample.question}\n\n"
        f"=== Tool Analysis ===\n{analysis}\n\n"
        'Choose the best answer. Return: {{"answer": "X"}}'
    )

    parsed = await _final_llm_call(sys_prompt, user_prompt)
    answer_char = ""
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
        await _pipeline_ref.record_outcome(
            domain=sample.domain, dataset=getattr(sample, "dataset", ""),
            task="mcqa", tool_chain=_extract_tool_chain(result),
            predicted=answer_char, ground_truth=gt, correct=correct,
            pipeline_summary=analysis[:2000],
        )

    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": answer_char,
        "ground_truth": gt,
    }


async def _eval_trend(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    if sample.domain == "weather":
        trend_desc = (
            "The input time-series represents temperature readings from a US location. "
            "Every 24 readings represent a full day from 00:00 to 23:00. "
            "Predict the temperature trend for the next 1 day. "
            "Calculate the mean temperature of the last 24-hour period and compare it "
            "with the mean temperature of the first predicted day. "
            "If the difference is >= 0.5, classify as 'increasing'. "
            "If the difference is <= -0.5, classify as 'decreasing'. "
            "Otherwise, classify as 'stable'. "
            'Return ONLY JSON: {"trend": "increasing" or "decreasing" or "stable"}'
        )
    else:
        trend_desc = (
            "Time Series Stock Price Data. "
            "Predict the stock trend and categorize the future percentage change into "
            'one of: "<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%". '
            "Analyze the provided time-series data and news to identify future trends. "
            'Return ONLY JSON: {"trend": "<label>"}'
        )

    result = await pipeline.run(
        data=sample.input_window,
        task_type="classification",
        description=trend_desc,
        text_data=text_data,
        verbose=verbose,
        domain=sample.domain,
        **_train_kwargs(sample, "trend"),
    )

    analysis = _build_pipeline_analysis_summary(result)
    tool_chain = _extract_tool_chain(result)
    ts_str = _fmt_ts_compact(sample.input_window)
    text_section = f"\nContextual text:\n{sample.text[:10000]}\n" if sample.text else ""

    if sample.domain == "weather":
        gt_label = _weather_trend_3way_label(sample)
        sys_prompt = (
            "You are a weather forecasting AI. "
            "The input time-series represents temperature readings from a US location. "
            "Every 24 readings represent a full day from 00:00 to 23:00.\n"
            "Predict the temperature trend for the next 1 day. "
            "Compare the mean temperature of the last 24-hour period with the predicted "
            "next-day mean: if difference >= 0.5, 'increasing'; if <= -0.5, 'decreasing'; "
            "otherwise 'stable'.\n"
            'Return ONLY JSON: {"trend": "<increasing|decreasing|stable>"}'
        )
        user_prompt = (
            f"Temperature time series (past 7 days, hourly, °C):\n[{ts_str}]\n"
            f"{text_section}\n"
            f"=== Tool Analysis ===\n{analysis}\n\n"
            "What is the future temperature trend? "
            '{"trend": "..."}'
        )
        parsed = await _final_llm_call(sys_prompt, user_prompt)
        matched = None
        valid = {"increasing", "decreasing", "stable"}
        if isinstance(parsed, dict) and "trend" in parsed:
            matched = _quick_match_label(str(parsed["trend"]).lower(), valid)
        if matched is None:
            matched = "stable"

        if _pipeline_ref and _record_enabled:
            await _pipeline_ref.record_outcome(
                domain="weather", dataset=getattr(sample, "dataset", ""),
                task="trend", tool_chain=tool_chain,
                predicted=matched, ground_truth=gt_label,
                correct=(matched == gt_label),
                pipeline_summary=analysis[:2000],
            )
        return {
            "metrics": {"accuracy": 1.0 if matched == gt_label else 0.0},
            "predicted": matched,
            "ground_truth": gt_label,
        }
    else:
        gt_label = sample.labels.get("trend_label", "")
        if not gt_label:
            return {"metrics": {"error": "no_trend_label"}}
        valid_5 = {"<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%"}
        sys_prompt = (
            "You are a financial prediction expert with knowledge of advanced "
            "machine learning models and time-series analysis.\n"
            "Categorize the future stock price percentage change into one of: "
            '"<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%".\n'
            "Think step-by-step. Analyze the time-series data and news to identify "
            "future trends.\n"
            'Return ONLY JSON: {"trend": "<label>"}'
        )
        user_prompt = (
            f"Stock price time-series:\n[{ts_str}]\n"
            f"{text_section}\n"
            f"=== Tool Analysis ===\n{analysis}\n\n"
            "Classify the future price change. "
            '{"trend": "..."}'
        )
        parsed = await _final_llm_call(sys_prompt, user_prompt)
        matched = None
        if isinstance(parsed, dict) and "trend" in parsed:
            matched = _quick_match_label(str(parsed["trend"]).strip(), valid_5)
        if matched is None:
            matched = "-2% ~ +2%"

        if _pipeline_ref and _record_enabled:
            await _pipeline_ref.record_outcome(
                domain="finance", dataset=getattr(sample, "dataset", ""),
                task="trend", tool_chain=tool_chain,
                predicted=matched, ground_truth=gt_label,
                correct=(matched == gt_label),
                pipeline_summary=analysis[:2000],
            )
        return {
            "metrics": {"accuracy": 1.0 if matched == gt_label else 0.0},
            "predicted": matched,
            "ground_truth": gt_label,
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

    gt_max = float(np.max(sample.output_window))
    gt_min = float(np.min(sample.output_window))
    gt_diff = gt_max - gt_min

    if gt_diff < 0.01:
        return {"metrics": {"skipped": "flat_gt"}}

    gt = {"max": gt_max, "min": gt_min, "diff": gt_diff}

    n_hours = len(sample.output_window)
    n_days = max(1, n_hours // 24)
    result = await pipeline.run(
        data=sample.input_window,
        task_type="indicator",
        description=(
            f"The input time-series represents temperature readings from a location "
            f"in the United States, where summers are hot and winters are cold. "
            f"Weather conditions can also be affected by storms, heavy rain, and cold fronts. "
            f"The daytime is usually warmer than the nighttime. "
            f"Every 24 temperature readings represent a full day from 00:00 to 23:00. "
            f"Analyze the past 7 days' temperature trend and predict the next {n_days} day(s) "
            f"highest temperature, lowest temperature, and the temperature difference "
            f"between them. "
            f'Return ONLY JSON: {{"max": number, "min": number, "diff": number}}'
        ),
        horizon=n_hours,
        text_data=text_data,
        verbose=verbose,
        domain="weather",
        **_train_kwargs(sample, "indicator"),
    )

    prediction = result.get("prediction")
    pred: dict[str, float] | None = None

    if isinstance(prediction, dict):
        pred = _extract_indicator_dict(prediction, ["max", "min", "diff"])

    if pred is None:
        pred_arr = _extract_forecast_array(prediction)
        if pred_arr is not None and len(pred_arr) >= 1:
            pred = {
                "max": float(np.max(pred_arr)),
                "min": float(np.min(pred_arr)),
                "diff": float(np.max(pred_arr) - np.min(pred_arr)),
            }

    if pred is None:
        pred = _extract_indicator_dict(prediction, ["max", "min", "diff"])

    if pred is None:
        analysis = _build_pipeline_analysis_summary(result)
        ts_str = _fmt_ts_compact(sample.input_window)
        text_section = f"\nWeather report:\n{sample.text[:10000]}\n" if sample.text else ""
        sys_prompt = (
            "You are a weather forecasting AI. "
            "The input time-series represents temperature readings from a location in "
            "the United States. Every 24 readings represent a full day from 00:00 to 23:00.\n"
            f"Predict the next {n_days} day(s) highest temperature, lowest temperature, and the "
            "difference between them.\n"
            'Return ONLY JSON: {{"max": <number>, "min": <number>, "diff": <number>}}'
        )
        user_prompt = (
            f"Temperature time series (past 7 days, hourly, °C):\n[{ts_str}]\n"
            f"{text_section}\n"
            f"=== Tool Analysis ===\n{analysis}\n\n"
            'Predict: {{"max": ..., "min": ..., "diff": ...}}'
        )
        parsed = await _final_llm_call(sys_prompt, user_prompt)
        pred = _extract_indicator_dict(parsed, ["max", "min", "diff"])

    if pred:
        metrics = {f"{k}_mae": abs(gt[k] - pred.get(k, 0.0)) for k in gt}
        if _pipeline_ref and _record_enabled:
            hist_arr = np.asarray(sample.input_window).ravel()
            recent_24 = hist_arr[-24:] if len(hist_arr) >= 24 else hist_arr
            recent_range = float(np.max(recent_24) - np.min(recent_24))
            summary = _build_pipeline_analysis_summary(result)
            await _pipeline_ref.record_outcome(
                domain="weather", dataset=getattr(sample, "dataset", ""),
                task="indicator", tool_chain=_extract_tool_chain(result),
                predicted=json.dumps(pred),
                ground_truth=json.dumps(gt),
                correct=(metrics.get("diff_mae", 99) < 3.0),
                metrics=metrics,
                pipeline_summary=summary[:2000],
                data_fingerprint=f"recent_range={recent_range:.1f}",
            )
        return {"metrics": metrics, "predicted": pred, "ground_truth": gt}
    return {"metrics": {"error": "no prediction"}}


async def _eval_stock_indicator(
    sample: MTBenchSample, pipeline: PipelineRunner, verbose: bool,
) -> dict[str, Any]:
    gt_macd = sample.labels.get("out_macd")
    gt_bb = sample.labels.get("out_upper_bb")
    if gt_macd is None and gt_bb is None:
        return {"metrics": {"error": "no_indicator_labels"}}

    text_data = None
    if sample.text:
        text_data = [{"content": sample.text, "source": sample.domain, "timestamp": ""}]

    indicator_names = []
    if gt_macd is not None:
        indicator_names.append("MACD")
    if gt_bb is not None:
        indicator_names.append("upper Bollinger Band")

    ds = getattr(sample, "dataset", "")
    input_desc = "7 days at 5-minute intervals" if "short" in ds else "30 days of daily prices"
    schema_parts = []
    if gt_macd is not None:
        schema_parts.append('"macd": <number>')
    if gt_bb is not None:
        schema_parts.append('"upper_bb": <number>')
    target_schema = "{" + ", ".join(schema_parts) + "}"

    result = await pipeline.run(
        data=sample.input_window,
        task_type="indicator",
        description=(
            f"Stock price data recorded at {input_desc}. "
            f"Predict the following technical indicators for the NEXT trading day: "
            f"{', '.join(indicator_names)}. "
            f"MACD (Moving Average Convergence Divergence) = EMA(12) - EMA(26) of closing price. "
            f"Upper Bollinger Band = SMA(20) + 2 * StdDev(20). "
            f"Return ONLY JSON: {target_schema}"
        ),
        horizon=len(sample.output_window) if sample.output_window is not None else 78,
        text_data=text_data,
        verbose=verbose,
        domain="finance",
        **_train_kwargs(sample, "indicator"),
    )

    analysis = _build_pipeline_analysis_summary(result)
    ts_str = _fmt_ts_compact(sample.input_window)
    text_section = f"\nFinancial analysis:\n{sample.text[:10000]}\n" if sample.text else ""

    ds = sample.dataset
    input_desc = "7 days at 5-minute intervals" if "short" in ds else "30 days of daily prices"

    schema_parts = []
    if gt_macd is not None:
        schema_parts.append('"macd": <number>')
    if gt_bb is not None:
        schema_parts.append('"upper_bb": <number>')
    target_schema = "{" + ", ".join(schema_parts) + "}"

    sys_prompt = (
        "You are a quantitative finance analyst. "
        f"Given {input_desc} of stock price data, predict the following technical "
        f"indicators for the NEXT trading day: {', '.join(indicator_names)}.\n"
        "MACD = EMA(12) - EMA(26) of closing price.\n"
        "Upper Bollinger Band = SMA(20) + 2 * StdDev(20).\n"
        f"Return ONLY JSON: {target_schema}"
    )
    user_prompt = (
        f"Historical stock price time-series:\n[{ts_str}]\n"
        f"{text_section}\n"
        f"=== Tool Analysis ===\n{analysis}\n\n"
        f"Predict the next-day indicators: {', '.join(indicator_names)}.\n"
        f"Return: {target_schema}"
    )

    parsed = await _final_llm_call(sys_prompt, user_prompt)
    metrics: dict[str, float] = {}
    if isinstance(parsed, dict):
        if gt_macd is not None and "macd" in parsed:
            try:
                metrics["macd_mae"] = abs(float(gt_macd) - float(parsed["macd"]))
            except (ValueError, TypeError):
                pass
        if gt_bb is not None and "upper_bb" in parsed:
            try:
                metrics["upper_bb_mae"] = abs(float(gt_bb) - float(parsed["upper_bb"]))
            except (ValueError, TypeError):
                pass

    if not metrics:
        prediction = result.get("prediction")
        pred_dict = _extract_indicator_dict(prediction, ["macd", "upper_bb"])
        if pred_dict:
            if gt_macd is not None and "macd" in pred_dict:
                metrics["macd_mae"] = abs(float(gt_macd) - float(pred_dict["macd"]))
            if gt_bb is not None and "upper_bb" in pred_dict:
                metrics["upper_bb_mae"] = abs(float(gt_bb) - float(pred_dict["upper_bb"]))

    if not metrics:
        return {"metrics": {"error": "no prediction or labels"}}

    if _pipeline_ref and _record_enabled:
        await _pipeline_ref.record_outcome(
            domain="finance", dataset=getattr(sample, "dataset", ""),
            task="stock_indicator", tool_chain=_extract_tool_chain(result),
            predicted=json.dumps({k: v for k, v in (parsed if isinstance(parsed, dict) else {}).items() if k in ("macd", "upper_bb")}),
            ground_truth=json.dumps({"macd": gt_macd, "upper_bb": gt_bb}),
            correct=(metrics.get("macd_mae", 99) < 1.0 and metrics.get("upper_bb_mae", 99) < 5.0),
            metrics=metrics,
            pipeline_summary=analysis[:2000],
        )

    return {"metrics": metrics}


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
        description=(
            "Predict the correlation between the stock's price fluctuations in the "
            "next 7 days and the analysis sentiment. Positive correlation indicates "
            "that positive analysis leads to price increase and negative analysis "
            "leads to price decrease. Take into account external factors or market "
            "conditions that might affect stock price movement. "
            "Choose one of: Strong Positive Correlation, Moderate Positive Correlation, "
            "No Correlation, Moderate Negative Correlation, Strong Negative Correlation. "
            'Return ONLY JSON: {"correlation": "<label>"}'
        ),
        text_data=text_data,
        verbose=verbose,
        domain="finance",
        **_train_kwargs(sample, "correlation"),
    )

    analysis = _build_pipeline_analysis_summary(result)
    ts_str = _fmt_ts_compact(sample.input_window)
    text_section = f"\nFinancial news:\n{sample.text[:10000]}\n" if sample.text else ""

    sys_prompt = (
        "You are an expert in finance and stock market analysis.\n"
        "Predict the correlation between the stock's price fluctuations in the "
        "next 7 days and the analysis sentiment. Positive correlation indicates "
        "that positive analysis leads to price increase and negative analysis "
        "leads to price decrease.\n"
        "Choose one of: Strong Positive Correlation, Moderate Positive Correlation, "
        "No Correlation, Moderate Negative Correlation, Strong Negative Correlation.\n"
        'Return ONLY JSON: {"correlation": "<label>"}'
    )
    user_prompt = (
        f"Stock price time-series:\n[{ts_str}]\n"
        f"{text_section}\n"
        f"=== Tool Analysis ===\n{analysis}\n\n"
        "Classify the news-price correlation. "
        '{"correlation": "..."}'
    )

    parsed = await _final_llm_call(sys_prompt, user_prompt)
    matched = None
    if isinstance(parsed, dict) and "correlation" in parsed:
        matched = _match_correlation(str(parsed["correlation"]))

    if matched is None:
        pred = str(result.get("prediction", ""))
        matched = _match_correlation(pred)

    if matched is None:
        matched = "No Correlation"

    correct = matched.lower().strip() == gt_corr.lower().strip()

    if _pipeline_ref and _record_enabled:
        await _pipeline_ref.record_outcome(
            domain="finance", dataset=getattr(sample, "dataset", ""),
            task="correlation", tool_chain=_extract_tool_chain(result),
            predicted=matched, ground_truth=gt_corr, correct=correct,
            pipeline_summary=analysis[:2000],
        )

    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": matched,
        "ground_truth": gt_corr,
    }


# ── Task dispatch ────────────────────────────────────────────────────

EVAL_HANDLERS = {
    "forecast": _eval_forecast,
    "trend": _eval_trend,
    "indicator": _eval_indicator,
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
) -> dict[str, Any]:
    async with sem:
        handler = EVAL_HANDLERS.get(task)
        if not handler:
            return {"idx": idx, "file": sample.file, "task": task,
                    "domain": sample.domain,
                    "metrics": {"error": "no handler"}}
        try:
            result = await asyncio.wait_for(
                handler(sample, pipeline, verbose), timeout=600,
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


# ── Main evaluation loop ─────────────────────────────────────────────

async def run_pipeline_eval(
    dataset_name: str | None = None,
    task_filter: str | None = None,
    config: TimeClawConfig | None = None,
    max_samples: int | None = None,
    use_skills: bool = True,
    verbose: bool = False,
    label: str = "timeclaw",
    concurrency: int = 80,
    full_dataset: bool = False,
    use_strategy: bool = True,
    split: str = "test",
) -> dict[str, Any]:
    """Run TimeClaw pipeline on dataset(s) and evaluate results.

    This is the single entry point for both training and testing:
      - Training: split="train" — runs pipeline on train set, records outcomes
        into hierarchical memory (daily notes → distillation).
      - Testing: split="test" — runs pipeline on test set with learned memory.

    Args:
        split: Which data split to load ("train" or "test").
               Ignored when full_dataset=True.
        use_skills: If False (ablation), disables all memory components.
        full_dataset: If True, loads the entire dataset (ignores split).
        use_strategy: Enables online task strategy learning.
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
                        logger.warning(f"[SBM pre-load] {name} failed: {e}")
            logger.info(f"[SBM pre-load] {len(reg.list_available())} models ready")
        except Exception as e:
            logger.warning(f"[SBM pre-load] skipped: {e}")

        global _record_enabled
        _record_enabled = (split == "train")
        sem = asyncio.Semaphore(concurrency)
        datasets_to_run = (
            [dataset_name] if dataset_name else list(DATASET_REGISTRY.keys())
        )
        all_results: dict[str, Any] = {}

        from timeclaw.benchmark.mtbench_adapter import load_dataset

        for ds_name in datasets_to_run:
            info = DATASET_REGISTRY.get(ds_name)
            if not info:
                continue

            try:
                if full_dataset:
                    samples = load_dataset(ds_name)
                else:
                    samples = load_split(ds_name, split)
            except FileNotFoundError:
                logger.warning(f"[PipelineEval] No data for {ds_name}")
                continue

            if max_samples:
                samples = samples[:max_samples]

            supported_tasks = info["tasks"]
            if task_filter:
                supported_tasks = [t for t in supported_tasks if t == task_filter]
            if not supported_tasks:
                continue

            total_tasks = len(samples) * len(supported_tasks)
            logger.info(
                f"[PipelineEval:{label}] {ds_name}: {len(samples)} samples "
                f"× {len(supported_tasks)} tasks = {total_tasks} evals, "
                f"concurrency={concurrency}"
            )
            t0 = time.time()

            coros = [
                _eval_one_task(
                    ds_name, idx, sample, task, pipeline, verbose, sem
                )
                for idx, sample in enumerate(samples)
                for task in supported_tasks
            ]
            flat_results = await asyncio.gather(*coros, return_exceptions=True)

            ds_results: list[dict[str, Any]] = []
            for r in flat_results:
                if isinstance(r, Exception):
                    logger.error(f"[PipelineEval] {ds_name}: task error: {r}")
                    continue
                ds_results.append(r)

            elapsed = time.time() - t0
            _save_results(ds_results, ds_name, label)
            summary = _summarize(ds_results)
            all_results[ds_name] = {
                "total": len(ds_results),
                "elapsed_s": round(elapsed, 1),
                "summary": summary,
            }
            logger.info(
                f"[PipelineEval:{label}] {ds_name} done in {elapsed:.1f}s: "
                f"{json.dumps(summary, default=str)}"
            )

        if _pipeline_ref is not None and _record_enabled:
            _pipeline_ref.flush_memory()
            try:
                await _pipeline_ref.distill_memory()
                mem_summary = _pipeline_ref.memory_summary()
                logger.info(f"[PipelineEval] Memory summary after training: {mem_summary}")
            except Exception as e:
                logger.debug(f"[PipelineEval] Memory distill error: {e}")

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
    path.write_text(json.dumps(results, indent=2, default=str))


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, list[float]]] = {}
    for r in results:
        task = r.get("task", "unknown")
        metrics = r.get("metrics", {})
        if task not in by_task:
            by_task[task] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                by_task[task].setdefault(k, []).append(v)

    summary: dict[str, Any] = {}
    for task, metric_lists in by_task.items():
        task_summary: dict[str, Any] = {}
        for k, vals in metric_lists.items():
            task_summary[f"avg_{k}"] = round(float(np.mean(vals)), 4)
            task_summary["n"] = len(vals)
        summary[task] = task_summary
    return summary


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TimeClaw Test Evaluation")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--concurrency", type=int, default=80)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = await run_pipeline_eval(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        use_skills=not args.no_skills,
        verbose=args.verbose,
        label="timeclaw_no_skills" if args.no_skills else "timeclaw",
        concurrency=args.concurrency,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
