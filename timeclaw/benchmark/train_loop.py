"""Train Loop: agent-driven exploration on MTBench train splits.

Supports multiple task types across all MTBench datasets:
- forecast: compare forecast tools against ground truth output_window
- trend: agent predicts trend direction, compared to ground truth label
- indicator: agent predicts max/min/diff, compared to ground truth
- mcqa: agent answers multiple-choice question, compared to ground truth
- correlation: agent predicts news-price correlation label
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.metrics import mae, rmse, mape, smape
from timeclaw.benchmark.mtbench_adapter import (
    DATASET_REGISTRY,
    MTBenchSample,
)
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.clawhub.tool_registry import discover_and_register, get_tool_instance
from timeclaw.core.config import TimeClawConfig
from timeclaw.core import trace_logger as trace
from timeclaw.llm.router import ModelRouter
from timeclaw.memory.experience_pool import ExperienceEntry, ExperiencePool

FORECAST_TOOLS = [
    "timesfm_forecast", "chronos_forecast", "timer_forecast", "moirai_forecast",
    "ets_forecast", "arima_forecast", "prophet_forecast",
]
ANALYSIS_TOOLS = [
    "detect_trend", "detect_seasonality", "detect_anomaly", "basic_stats",
]
TEXT_TOOLS = [
    "sentiment", "keyword_extract", "summarize", "text_to_event",
]

# Homogeneous tool groups: tools with identical input/output formats.
# During warmup, we progressively expose more tools from each group.
HOMOGENEOUS_GROUPS: dict[str, list[str]] = {
    "forecast": FORECAST_TOOLS,
}


def _progressive_expose(
    group: list[str],
    sample_idx: int,
    total_samples: int,
    warmup_ratio: float = 0.5,
) -> list[str]:
    """Progressively expose more tools from a homogeneous group.

    During the warmup phase (first warmup_ratio of samples), the number
    of visible tools ramps from 1 to len(group).  After warmup, all
    tools are visible.
    """
    import random
    n_tools = len(group)
    warmup_end = int(total_samples * warmup_ratio)

    if warmup_end <= 0 or sample_idx >= warmup_end:
        return list(group)

    progress = sample_idx / warmup_end
    n_visible = max(1, min(n_tools, int(1 + progress * n_tools)))
    exposed = random.sample(group, n_visible)
    return exposed

TRAIN_RESULTS_DIR = Path(".timeclaw/train_results")


# ── Metric helpers ────────────────────────────────────────────────────

def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def _accuracy_metric(pred: str, gt: str) -> dict[str, float]:
    pred_clean = pred.strip().upper()[:1]
    gt_clean = gt.strip().upper()[:1]
    return {"accuracy": 1.0 if pred_clean == gt_clean else 0.0}


def _indicator_metrics(pred: dict, gt: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("max", "min", "diff"):
        if key in gt and key in pred:
            metrics[f"{key}_mae"] = abs(float(gt[key]) - float(pred[key]))
    return metrics


# ── Tool execution ────────────────────────────────────────────────────

def _extract_forecast_array(result_data: Any) -> np.ndarray | None:
    if result_data is None:
        return None
    if isinstance(result_data, np.ndarray):
        return result_data
    if isinstance(result_data, list):
        try:
            return np.asarray(result_data, dtype=np.float64)
        except (ValueError, TypeError):
            return None
    if isinstance(result_data, dict):
        for key in ("point_forecast", "forecast", "prediction"):
            if key in result_data:
                try:
                    return np.asarray(result_data[key], dtype=np.float64)
                except (ValueError, TypeError):
                    continue
    return None


async def _run_forecast_tool(
    tool_name: str, ts_data: np.ndarray, horizon: int,
) -> tuple[np.ndarray | None, str]:
    try:
        tool = get_tool_instance(tool_name)
    except KeyError:
        return None, f"tool '{tool_name}' not found"
    try:
        result: ToolResult = await tool.execute(ts_data, horizon=horizon)
        if not result.success:
            return None, result.error or "unknown error"
        arr = _extract_forecast_array(result.data)
        if arr is None:
            return None, "could not extract numeric forecast"
        return arr, ""
    except Exception as e:
        return None, str(e)


async def _run_analysis_tool(
    tool_name: str, ts_data: np.ndarray,
) -> tuple[Any, str]:
    try:
        tool = get_tool_instance(tool_name)
    except KeyError:
        return None, f"tool '{tool_name}' not found"
    try:
        result: ToolResult = await tool.execute(ts_data)
        if not result.success:
            return None, result.error or "unknown error"
        return result.data, ""
    except Exception as e:
        return None, str(e)


# ── Memory-driven insight refresh ────────────────────────────────────

def _refresh_insights_from_pool(pool: ExperiencePool) -> dict[str, str]:
    """Build tool insights directly from experience pool metrics.

    For each forecast tool, compute avg MAE from ground_truth_metrics
    stored in the pool, then format as natural language insights.
    """
    tool_metrics: dict[str, list[float]] = defaultdict(list)
    tool_wins: dict[str, int] = defaultdict(int)
    tool_total: dict[str, int] = defaultdict(int)

    for entry in pool._entries:
        if entry.task_type != "forecast":
            continue
        gt = entry.ground_truth_metrics
        best = entry.metadata.get("best_tool")
        for tool_name, metrics in gt.items():
            if isinstance(metrics, dict) and "mae" in metrics:
                tool_metrics[tool_name].append(metrics["mae"])
                tool_total[tool_name] += 1
                if tool_name == best:
                    tool_wins[tool_name] += 1

    if not tool_metrics:
        return {}

    insights: dict[str, str] = {}
    ranking = []
    for tool, maes in tool_metrics.items():
        avg_mae = sum(maes) / len(maes)
        wins = tool_wins.get(tool, 0)
        total = tool_total.get(tool, 0)
        win_rate = wins / total if total > 0 else 0
        ranking.append((tool, avg_mae, win_rate, total))

    ranking.sort(key=lambda x: x[1])

    for rank, (tool, avg_mae, win_rate, total) in enumerate(ranking, 1):
        insights[tool] = (
            f"Rank {rank}/{len(ranking)} by accuracy. "
            f"Avg MAE={avg_mae:.4f}, win_rate={win_rate:.1%} over {total} samples."
        )

    return insights


# ── LLM helpers ───────────────────────────────────────────────────────

async def _ask_llm_pick_tool(
    llm: Any, tool_list: list[str], fingerprint: str,
    purpose: str, tool_insights: dict[str, str],
) -> str:
    insight_text = "\n".join(
        f"- {t}: {tool_insights[t]}" for t in tool_list if t in tool_insights
    )
    prompt = (
        f"Pick the best tool for {purpose}.\n"
        f"Data: {fingerprint}\n"
        f"Tools: {tool_list}\n"
    )
    if insight_text:
        prompt += f"Insights:\n{insight_text}\n"
    prompt += 'Return JSON: {"tool": "name"}'

    messages = [
        {"role": "system", "content": "You select tools."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = await llm.complete(messages, response_format="json")
        if resp.parsed and resp.parsed.get("tool") in tool_list:
            return resp.parsed["tool"]
    except Exception:
        pass
    import random
    return random.choice(tool_list)


async def _ask_llm_trend(
    llm: Any, ts_data: np.ndarray, text: str, domain: str,
) -> str:
    ts_str = ", ".join(f"{v:.2f}" for v in ts_data)
    if domain == "weather":
        prompt = (
            "The input time-series represents temperature readings. "
            "This data is from a location in the United States, where summers are hot and winters are cold. "
            "Every 24 temperature readings represent a full day from 00:00 to 23:00.\n\n"
            f"Temperature time series (hourly):\n[{ts_str}]\n\n"
        )
        if text:
            prompt += f"Weather report:\n{text}\n\n"
        prompt += (
            "Based on the temperature data and any available context, predict the overall "
            "temperature trend for the next period. "
            'Return JSON: {"trend": "one of: up, stable, down"}'
        )
        sys_msg = "You are a weather forecasting AI."
    else:
        prompt = (
            f"Stock price time series:\n[{ts_str}]\n\n"
        )
        if text:
            prompt += f"News:\n{text[:10000]}\n\n"
        prompt += (
            "Based on the price data and any available context, predict the overall "
            "price trend for the next period. "
            'Return JSON: {"trend": "one of: strong_up, up, stable, down, strong_down"}'
        )
        sys_msg = "You are a financial analysis AI."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = await llm.complete(messages, response_format="json")
        if resp.parsed and "trend" in resp.parsed:
            return resp.parsed["trend"]
    except Exception:
        pass
    return "stable"


async def _ask_llm_indicator(
    llm: Any, ts_data: np.ndarray, text: str, domain: str = "weather",
) -> dict[str, float]:
    ts_str = ", ".join(f"{v:.2f}" for v in ts_data)
    if domain == "weather":
        prompt = (
            "The input time-series represents temperature readings. "
            "This data is from a location in the United States, where summers are hot and winters are cold. "
            "Weather conditions can also be affected by storms, heavy rain, and cold fronts. "
            "The daytime is usually warmer than the nighttime. "
            "Every 24 temperature readings represent a full day from 00:00 to 23:00.\n\n"
            f"Temperature time series (hourly):\n[{ts_str}]\n\n"
        )
        if text:
            prompt += f"Weather report:\n{text}\n\n"
        prompt += (
            "Your task is to analyze the temperature trend and predict "
            "the next 1 day's highest temperature and lowest temperature "
            "as well as the temperature difference between the highest and lowest temperature.\n\n"
            'Return ONLY a JSON object: {"max": number, "min": number, "diff": number}'
        )
        sys_msg = "You are a weather forecasting AI."
    else:
        prompt = (
            f"Stock price time series:\n[{ts_str}]\n\n"
        )
        if text:
            prompt += f"News:\n{text[:10000]}\n\n"
        prompt += (
            "Predict the next period's MACD value and upper Bollinger Band value.\n\n"
            'Return ONLY a JSON object: {"macd": number, "upper_bb": number}'
        )
        sys_msg = "You are a financial analysis AI."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = await llm.complete(messages, response_format="json")
        if resp.parsed:
            if domain == "weather":
                return {k: float(resp.parsed.get(k, 0)) for k in ("max", "min", "diff")}
            else:
                return {k: float(resp.parsed.get(k, 0)) for k in ("macd", "upper_bb") if k in resp.parsed}
    except Exception:
        pass
    if domain == "weather":
        recent = ts_data[-24:] if len(ts_data) >= 24 else ts_data
        r_max, r_min = float(np.max(recent)), float(np.min(recent))
        return {"max": r_max, "min": r_min, "diff": r_max - r_min}
    return {"macd": 0.0, "upper_bb": float(ts_data[-1])}


async def _ask_llm_mcqa(
    llm: Any, question: str, ts_data: np.ndarray, text: str,
    domain: str = "weather",
) -> str:
    ts_str = ", ".join(f"{v:.2f}" for v in ts_data)
    if domain == "weather":
        prompt = (
            f"Temperature time series (hourly):\n[{ts_str}]\n\n"
        )
        if text:
            prompt += f"Weather report:\n{text}\n\n"
        sys_msg = "You are a weather analysis expert."
    else:
        prompt = (
            f"Stock price time series:\n[{ts_str}]\n\n"
        )
        if text:
            prompt += f"Financial analysis:\n{text[:10000]}\n\n"
        sys_msg = "You are a financial analysis expert."
    prompt += (
        f"Question: {question}\n\n"
        "Return your answer ONLY as a single letter (A, B, C, or D) in JSON: "
        '{"answer": "X"}'
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = await llm.complete(messages, response_format="json")
        if resp.parsed and "answer" in resp.parsed:
            return str(resp.parsed["answer"]).strip().upper()[:1]
    except Exception:
        pass
    return "A"


async def _ask_llm_correlation(
    llm: Any, ts_data: np.ndarray, text: str,
) -> str:
    ts_str = ", ".join(f"{v:.2f}" for v in ts_data)
    prompt = (
        f"Historical stock price time-series:\n[{ts_str}]\n\n"
        f"Financial analysis published at the last timestamp:\n{text[:10000]}\n\n"
        "Your task is to predict the correlation between the stock's price fluctuations "
        "and the financial analysis. "
        'Return JSON: {"correlation": "one of: Strong Positive Correlation, '
        'Weak Positive Correlation, No Correlation, Weak Negative Correlation, '
        'Strong Negative Correlation"}'
    )
    messages = [
        {"role": "system", "content": "You analyze news-price correlations."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = await llm.complete(messages, response_format="json")
        if resp.parsed and "correlation" in resp.parsed:
            return resp.parsed["correlation"]
    except Exception:
        pass
    return "No Correlation"


# ── Fingerprint ───────────────────────────────────────────────────────

def _build_fingerprint(sample: MTBenchSample) -> str:
    ts = sample.input_window
    n = len(ts)
    parts = [
        f"domain={sample.domain}",
        f"length={n}",
        f"mean={ts.mean():.2f}",
        f"std={ts.std():.2f}",
        f"min={ts.min():.2f}",
        f"max={ts.max():.2f}",
    ]
    if n > 24:
        try:
            from scipy.signal import find_peaks
            acf = np.correlate(ts - ts.mean(), ts - ts.mean(), mode="full")
            acf = acf[n - 1:]
            acf = acf / (acf[0] + 1e-12)
            peaks, _ = find_peaks(acf[1:], height=0.3)
            if len(peaks) > 0:
                parts.append(f"period={peaks[0]+1}")
        except Exception:
            pass
    trend = (ts[-1] - ts[0]) / n if n > 1 else 0
    parts.append(f"trend={trend:.4f}")
    if sample.text:
        parts.append("has_text=true")
    return ", ".join(parts)


# ── Per-task train handlers ───────────────────────────────────────────

async def _train_forecast(
    sample: MTBenchSample, llm: Any, tool_insights: dict,
    sample_idx: int = 0, total_samples: int = 100,
) -> dict[str, Any]:
    """Train on a forecast sample with progressive tool exposure + forced competition.

    Phase logic:
    - visible == 1: that tool is primary, explore is random from the rest
    - visible >= 2: LLM picks primary from visible tools (using memory/insights),
      then one random competitor is forced from the remaining visible tools
    """
    import random

    fp = _build_fingerprint(sample)
    horizon = len(sample.output_window) if sample.output_window is not None else 24

    visible_tools = _progressive_expose(
        FORECAST_TOOLS, sample_idx, total_samples,
    )

    EPSILON = 0.25

    if len(visible_tools) == 1:
        primary = visible_tools[0]
        others = [t for t in FORECAST_TOOLS if t != primary]
        explore = random.choice(others)
    elif random.random() < EPSILON:
        primary = random.choice(visible_tools)
        others = [t for t in visible_tools if t != primary]
        if not others:
            others = [t for t in FORECAST_TOOLS if t != primary]
        explore = random.choice(others)
    else:
        primary = await _ask_llm_pick_tool(
            llm, visible_tools, fp, "forecasting", tool_insights,
        )
        others = [t for t in visible_tools if t != primary]
        if not others:
            others = [t for t in FORECAST_TOOLS if t != primary]
        explore = random.choice(others)

    gt_metrics: dict[str, dict[str, float]] = {}

    p_pred, p_err = await _run_forecast_tool(primary, sample.input_window, horizon)
    if p_pred is not None and sample.output_window is not None:
        gt_metrics[primary] = _forecast_metrics(sample.output_window, p_pred)
    else:
        gt_metrics[primary] = {"error": p_err}

    e_pred, e_err = await _run_forecast_tool(explore, sample.input_window, horizon)
    if e_pred is not None and sample.output_window is not None:
        gt_metrics[explore] = _forecast_metrics(sample.output_window, e_pred)
    else:
        gt_metrics[explore] = {"error": e_err}

    best_tool, best_mae = None, float("inf")
    for t, m in gt_metrics.items():
        if "mae" in m and m["mae"] < best_mae:
            best_mae, best_tool = m["mae"], t

    return {
        "task": "forecast",
        "primary_tool": primary,
        "explore_tool": explore,
        "best_tool": best_tool,
        "visible_tools": visible_tools,
        "gt_metrics": gt_metrics,
        "fingerprint": fp,
        "quality_score": 1.0 / (1.0 + best_mae) if best_tool else 0.0,
    }


async def _train_trend(
    sample: MTBenchSample, llm: Any, tool_insights: dict,
) -> dict[str, Any]:
    """Train on a trend classification sample."""
    pred = await _ask_llm_trend(
        llm, sample.input_window, sample.text, sample.domain,
    )
    gt_label = sample.labels.get("trend_label", "")
    if not gt_label and sample.output_window is not None:
        diff = float(sample.output_window[-1] - sample.output_window[0])
        pct = diff / (abs(float(sample.output_window[0])) + 1e-8) * 100
        if pct > 4:
            gt_label = "strong_up"
        elif pct > 1:
            gt_label = "up"
        elif pct < -4:
            gt_label = "strong_down"
        elif pct < -1:
            gt_label = "down"
        else:
            gt_label = "stable"

    correct = _trend_match(pred, gt_label)
    return {
        "task": "trend",
        "predicted": pred,
        "ground_truth": gt_label,
        "correct": correct,
        "gt_metrics": {"accuracy": 1.0 if correct else 0.0},
        "fingerprint": _build_fingerprint(sample),
        "quality_score": 1.0 if correct else 0.0,
    }


def _trend_match(pred: str, gt: str) -> bool:
    """Fuzzy match trend predictions."""
    pos = {"strong_up", "up", ">+4%", "+2% ~ +4%", "positive"}
    neg = {"strong_down", "down", "<-4%", "-2% ~ -4%", "negative", "negtive"}
    p_lower = pred.lower().strip()
    g_lower = gt.lower().strip()
    if p_lower == g_lower:
        return True
    if p_lower in pos and g_lower in pos:
        return True
    if p_lower in neg and g_lower in neg:
        return True
    return False


async def _train_indicator(
    sample: MTBenchSample, llm: Any, tool_insights: dict,
) -> dict[str, Any]:
    """Train on an indicator prediction sample."""
    pred = await _ask_llm_indicator(llm, sample.input_window, sample.text, domain=sample.domain)

    gt: dict[str, float] = {}
    if sample.output_window is not None:
        gt = {
            "max": float(np.max(sample.output_window)),
            "min": float(np.min(sample.output_window)),
            "diff": float(np.max(sample.output_window) - np.min(sample.output_window)),
        }

    metrics = _indicator_metrics(pred, gt) if gt else {}
    total_mae = np.mean(list(metrics.values())) if metrics else float("inf")

    return {
        "task": "indicator",
        "predicted": pred,
        "ground_truth": gt,
        "gt_metrics": metrics,
        "fingerprint": _build_fingerprint(sample),
        "quality_score": 1.0 / (1.0 + total_mae),
    }


async def _train_mcqa(
    sample: MTBenchSample, llm: Any, tool_insights: dict,
) -> dict[str, Any]:
    """Train on an MCQA sample."""
    if not sample.question or not sample.answer:
        return {"task": "mcqa", "gt_metrics": {}, "quality_score": 0.0, "skipped": True}

    pred = await _ask_llm_mcqa(
        llm, sample.question, sample.input_window, sample.text,
        domain=sample.domain,
    )
    metrics = _accuracy_metric(pred, sample.answer)

    return {
        "task": "mcqa",
        "predicted": pred,
        "ground_truth": sample.answer,
        "gt_metrics": metrics,
        "fingerprint": _build_fingerprint(sample),
        "quality_score": metrics["accuracy"],
    }


async def _train_correlation(
    sample: MTBenchSample, llm: Any, tool_insights: dict,
) -> dict[str, Any]:
    """Train on a correlation prediction sample."""
    gt_corr = sample.labels.get("correlation", "")
    if not gt_corr:
        return {"task": "correlation", "gt_metrics": {}, "quality_score": 0.0, "skipped": True}

    pred = await _ask_llm_correlation(llm, sample.input_window, sample.text)
    correct = pred.lower().strip() == gt_corr.lower().strip()

    return {
        "task": "correlation",
        "predicted": pred,
        "ground_truth": gt_corr,
        "gt_metrics": {"accuracy": 1.0 if correct else 0.0},
        "fingerprint": _build_fingerprint(sample),
        "quality_score": 1.0 if correct else 0.0,
    }


TASK_HANDLERS = {
    "forecast": _train_forecast,
    "trend": _train_trend,
    "indicator": _train_indicator,
    "mcqa": _train_mcqa,
    "correlation": _train_correlation,
}


# ── Main train loop ──────────────────────────────────────────────────

async def _process_one_sample(
    ds_name: str,
    sample_idx: int,
    sample: MTBenchSample,
    supported_tasks: list[str],
    llm: Any,
    tool_insights: dict,
    pool: ExperiencePool,
    sem: asyncio.Semaphore,
    total_samples: int = 100,
) -> list[dict[str, Any]]:
    """Process a single sample across all its tasks (semaphore-gated)."""
    async with sem:
        results: list[dict[str, Any]] = []
        for task in supported_tasks:
            handler = TASK_HANDLERS.get(task)
            if not handler:
                continue
            try:
                if task == "forecast":
                    result = await asyncio.wait_for(
                        handler(sample, llm, tool_insights,
                                sample_idx=sample_idx, total_samples=total_samples),
                        timeout=120,
                    )
                else:
                    result = await asyncio.wait_for(
                        handler(sample, llm, tool_insights), timeout=120,
                    )
            except asyncio.TimeoutError:
                logger.warning(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: timeout 120s")
                result = {"task": task, "gt_metrics": {}, "quality_score": 0.0, "error": "timeout"}
            except Exception as e:
                logger.error(f"[TrainLoop] {ds_name}[{sample_idx}] {task}: {e}")
                result = {"task": task, "gt_metrics": {}, "quality_score": 0.0, "error": str(e)}

            if result.get("skipped"):
                continue

            entry = ExperienceEntry(
                trace_id=f"train_{ds_name}_{sample_idx:04d}_{task}",
                task_type=task,
                task_description=f"{ds_name} {task} sample {sample_idx}",
                fingerprint_summary=result.get("fingerprint", ""),
                tool_chain=[result.get("primary_tool", ""), result.get("explore_tool", "")]
                if task == "forecast"
                else [task],
                evidence_steps=1,
                quality_gate_passed=result.get("quality_score", 0) > 0.5,
                quality_score=result.get("quality_score", 0.0),
                prediction_summary=str(result.get("predicted", ""))[:2000],
                ground_truth_metrics=result.get("gt_metrics", {}),
                metadata={
                    "dataset": ds_name,
                    "domain": sample.domain,
                    "task": task,
                    "file": sample.file,
                    **{k: v for k, v in result.items()
                       if k in ("best_tool", "primary_tool", "explore_tool",
                                "predicted", "ground_truth", "correct")},
                },
            )
            pool.add(entry)

            results.append({
                "idx": sample_idx,
                "file": sample.file,
                "task": task,
                **{k: v for k, v in result.items() if k != "fingerprint"},
            })
        return results


async def run_train_loop(
    dataset_name: str | None = None,
    task_filter: str | None = None,
    config: TimeClawConfig | None = None,
    max_samples: int | None = None,
    resume_from: int = 0,
    verbose: bool = False,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Run training loop over one or all MTBench datasets.

    Args:
        dataset_name: Specific dataset or None for all.
        task_filter: Specific task type or None for all supported.
        max_samples: Limit samples per dataset.
        resume_from: Skip first N samples.
        verbose: Enable trace logging.
        concurrency: Max parallel samples (API calls).
    """
    cfg = config or TimeClawConfig.load()
    trace.configure(verbose)
    discover_and_register()

    router = ModelRouter(cfg.llm)
    llm = router.get_backend("executor")
    pool = ExperiencePool(storage_dir=f"{cfg.workspace_dir}/experiences")
    tool_insights = pool.load_tool_insights()
    sem = asyncio.Semaphore(concurrency)

    datasets_to_run = [dataset_name] if dataset_name else list(DATASET_REGISTRY.keys())
    all_results: dict[str, Any] = {}

    for ds_name in datasets_to_run:
        info = DATASET_REGISTRY.get(ds_name)
        if not info:
            logger.warning(f"[TrainLoop] Unknown dataset: {ds_name}")
            continue

        ckpt = TRAIN_RESULTS_DIR / f"train_{ds_name}.json"
        if ckpt.exists() and ckpt.stat().st_size > 10 and resume_from == 0:
            logger.info(f"[TrainLoop] {ds_name}: checkpoint exists ({ckpt.stat().st_size/1024:.0f}KB), skipping")
            continue

        try:
            samples = load_split(ds_name, "train")
        except FileNotFoundError:
            logger.warning(f"[TrainLoop] No split for {ds_name}, skipping")
            continue

        if max_samples:
            samples = samples[:max_samples]
        if resume_from > 0:
            samples = samples[resume_from:]

        supported_tasks = info["tasks"]
        if task_filter:
            supported_tasks = [t for t in supported_tasks if t == task_filter]
        if not supported_tasks:
            continue

        total = len(samples)
        batch_size = max(concurrency, 10)
        logger.info(
            f"[TrainLoop] {ds_name}: {total} samples, tasks={supported_tasks}, "
            f"concurrency={concurrency}, batch_size={batch_size}"
        )

        ds_results: list[dict[str, Any]] = []
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_samples = samples[batch_start:batch_end]

            coros = [
                _process_one_sample(
                    ds_name, batch_start + i + resume_from, sample, supported_tasks,
                    llm, tool_insights, pool, sem,
                    total_samples=total,
                )
                for i, sample in enumerate(batch_samples)
            ]

            batch_results = await asyncio.gather(*coros, return_exceptions=True)

            for batch in batch_results:
                if isinstance(batch, Exception):
                    logger.error(f"[TrainLoop] {ds_name}: batch error: {batch}")
                    continue
                ds_results.extend(batch)

            tool_insights = _refresh_insights_from_pool(pool)
            logger.info(
                f"[TrainLoop] {ds_name}: batch {batch_start}-{batch_end} done, "
                f"insights updated for {len(tool_insights)} tools"
            )

        _save_checkpoint(ds_results, ds_name)
        summary = _summarize_results(ds_results)
        all_results[ds_name] = {
            "total": len(ds_results),
            "summary": summary,
        }
        logger.info(f"[TrainLoop] {ds_name} done: {len(ds_results)} results")

    return all_results


def _save_checkpoint(results: list[dict[str, Any]], ds_name: str) -> None:
    TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAIN_RESULTS_DIR / f"train_{ds_name}.json"
    path.write_text(json.dumps(results, indent=2, default=str))


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}

    for r in results:
        task = r.get("task", "unknown")
        if task not in by_task:
            by_task[task] = {"count": 0, "metrics": {}}
        bt = by_task[task]
        bt["count"] += 1

        gt = r.get("gt_metrics", {})
        for k, v in gt.items():
            if isinstance(v, (int, float)):
                bt["metrics"].setdefault(k, []).append(v)

    summary = {}
    for task, data in by_task.items():
        task_summary = {"count": data["count"]}
        for k, vals in data["metrics"].items():
            task_summary[f"avg_{k}"] = float(np.mean(vals))
        summary[task] = task_summary

    if "forecast" in by_task:
        tool_wins: dict[str, int] = {}
        for r in results:
            if r.get("task") == "forecast" and r.get("best_tool"):
                tool_wins[r["best_tool"]] = tool_wins.get(r["best_tool"], 0) + 1
        summary["forecast"]["tool_wins"] = tool_wins

    return summary


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="TimeClaw Multi-Task Train Loop")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = await run_train_loop(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        resume_from=args.resume_from,
        verbose=args.verbose,
        concurrency=args.concurrency,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
