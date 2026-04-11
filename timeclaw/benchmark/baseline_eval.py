"""Direct-LLM baseline evaluator for MTBench.

Sends the same input data to the LLM using official MTBench prompt
templates (from mtbench_prompts.py) and parses the response.
No agent pipeline, no tools — pure LLM reasoning.
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
    expand_dataset_tasks,
    load_sample_indices,
    select_samples_with_original_indices,
)
from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.core.config import TimeClawConfig
from timeclaw.llm.router import ModelRouter

RESULTS_DIR = Path(".timeclaw/test_results")


def _quick_match_label(pred: str, valid_labels: set[str]) -> str | None:
    """Return the matching canonical label if pred contains it."""
    pred_l = pred.lower().strip()
    if pred_l in valid_labels:
        return pred_l
    for label in valid_labels:
        if label in pred_l:
            return label
    return None


# ── Response parsers ──────────────────────────────────────────────────

def _parse_json_from_text(text: str) -> dict | list | None:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass
    for start, end in [("{", "}"), ("[", "]")]:
        idx = text.find(start)
        if idx == -1:
            continue
        depth = 0
        for i in range(idx, len(text)):
            if text[i] == start:
                depth += 1
            elif text[i] == end:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[idx: i + 1])
                except (json.JSONDecodeError, TypeError):
                    break
    return None


def _extract_number_array(parsed: Any, expected_len: int, raw_text: str = "") -> np.ndarray | None:
    if isinstance(parsed, list):
        try:
            arr = np.asarray(parsed, dtype=np.float64).ravel()
            if len(arr) >= 1:
                return arr
        except (ValueError, TypeError):
            pass
    if isinstance(parsed, dict):
        for key in ("forecast", "prediction", "values", "temperatures", "data",
                     "hourly_temperatures", "temperature_predictions", "predictions"):
            if key in parsed:
                try:
                    arr = np.asarray(parsed[key], dtype=np.float64).ravel()
                    if len(arr) >= 1:
                        return arr
                except (ValueError, TypeError):
                    continue
        all_vals = []
        for v in parsed.values():
            if isinstance(v, list):
                try:
                    arr = np.asarray(v, dtype=np.float64).ravel()
                    if len(arr) >= expected_len // 2:
                        all_vals.extend(arr.tolist())
                except (ValueError, TypeError):
                    pass
        if all_vals:
            return np.asarray(all_vals, dtype=np.float64)
    if raw_text:
        numbers = re.findall(r"-?\d+\.?\d*", raw_text)
        if len(numbers) >= expected_len // 2:
            try:
                arr = np.asarray([float(n) for n in numbers[-expected_len * 2:]], dtype=np.float64)
                if len(arr) >= expected_len // 2:
                    return arr[:expected_len]
            except (ValueError, TypeError):
                pass
    return None


# ── Metric helpers ────────────────────────────────────────────────────

def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mse": 999.0, "mae": 999.0, "rmse": 999.0, "mape": 999.0}
    if len(y_true) != len(y_pred) and len(y_pred) > 1:
        y_pred = np.interp(
            np.linspace(0, 1, len(y_true)),
            np.linspace(0, 1, len(y_pred)),
            y_pred,
        )
    else:
        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"mse": 999.0, "mae": 999.0, "rmse": 999.0, "mape": 999.0}
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    denom = np.abs(y_true)
    valid = denom > 0.01
    mape = float(np.mean(np.abs(y_true[valid] - y_pred[valid]) / denom[valid]) * 100) if valid.any() else 999.0
    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}


def _prompt_messages(sample: MTBenchSample, task: str) -> list[dict[str, str]]:
    sys_p, user_p = prompts.build_prompt(sample, task)
    return prompts.prompt_messages(sys_p, user_p)


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


# ── Per-task baseline handlers ────────────────────────────────────────

async def _baseline_weather_forecast(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    if sample.output_window is None:
        return {"error": "no output_window"}
    resp = await llm.complete(_prompt_messages(sample, "forecast"))
    raw = resp.text or ""
    pred = prompts.parse_weather_forecast_output(raw)
    if pred is None:
        parsed = _parse_json_from_text(raw)
        pred_arr = _extract_number_array(parsed, len(sample.output_window), raw)
        pred = pred_arr.tolist() if pred_arr is not None else None
    if pred is not None:
        metrics = _forecast_metrics(sample.output_window, pred)
        return {"metrics": metrics, "pred_len": len(pred)}
    return {"metrics": {"error": "parse_fail"}}


async def _baseline_weather_trend(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    resp = await llm.complete(_prompt_messages(sample, "trend"))
    raw = resp.text or ""
    pred_label = prompts.parse_weather_trend_output(raw)
    if pred_label is None:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            pred_label = _quick_match_label(str(parsed.get("trend", "")).strip().lower(), {"increasing", "decreasing", "stable"})
    pred_label = pred_label or ""

    gt_label = _weather_trend_3way_label(sample)
    correct = pred_label == gt_label
    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": pred_label,
        "ground_truth": gt_label,
    }


async def _baseline_weather_indicator(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    if sample.output_window is None:
        return {"metrics": {"error": "no output_window"}}
    gt = prompts.weather_indicator_ground_truth(sample.output_window)
    gt_max = gt["max"]
    gt_min = gt["min"]
    gt_diff = gt["diff"]
    if gt_diff < 0.01:
        return {"metrics": {"skipped": "flat_gt"}}
    resp = await llm.complete(_prompt_messages(sample, "indicator"))
    raw = resp.text or ""
    pred = prompts.parse_weather_indicator_output(raw)
    if pred is None:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            pred = {k: float(parsed.get(k, 0)) for k in ("max", "min", "diff") if k in parsed}
    if pred:
        metrics = {f"{k}_mae": abs(gt[k] - pred.get(k, 0)) for k in gt}
        return {"metrics": metrics, "predicted": pred, "ground_truth": gt}
    return {"metrics": {"error": "parse_fail"}}


async def _baseline_weather_mcqa(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    if not sample.question or not sample.answer:
        return {"metrics": {"error": "no_qa"}}
    resp = await llm.complete(_prompt_messages(sample, "mcqa"))
    raw = resp.text or ""
    pred = prompts.extract_mcqa_letter(raw)
    if not pred:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            pred = str(parsed.get("answer", "")).strip().upper()[:1]
    gt = sample.answer.strip().upper()[:1]
    return {
        "metrics": {"accuracy": 1.0 if pred == gt else 0.0},
        "predicted": pred,
        "ground_truth": gt,
    }


async def _baseline_stock_forecast(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    if sample.output_window is None:
        return {"metrics": {"error": "no output_window"}}
    horizon = len(sample.output_window)
    resp = await llm.complete(_prompt_messages(sample, "forecast"))
    raw = resp.text or ""
    pred = prompts.parse_finance_value_output(raw)
    if pred is None:
        parsed = _parse_json_from_text(raw)
        pred_arr = _extract_number_array(parsed, horizon, raw)
        pred = pred_arr.tolist() if pred_arr is not None else None
    if pred is not None:
        metrics = _forecast_metrics(sample.output_window, pred)
        return {"metrics": metrics, "pred_len": len(pred)}
    return {"metrics": {"error": "parse_fail"}}


async def _baseline_stock_trend(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    gt_label = sample.labels.get("trend_label", "")
    if not gt_label:
        return {"metrics": {"error": "no_trend_label"}}
    resp = await llm.complete(_prompt_messages(sample, "trend"))
    raw = resp.text or ""
    pred_label = prompts.parse_finance_trend_output(raw)
    if pred_label is None:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            pred_label = str(parsed.get("trend", "")).strip()
    pred_label = pred_label or ""
    correct = pred_label == gt_label
    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": pred_label,
        "ground_truth": gt_label,
    }


async def _baseline_stock_indicator(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    gt_macd = sample.labels.get("out_macd")
    gt_bb = sample.labels.get("out_upper_bb")
    if gt_macd is None and gt_bb is None:
        return {"metrics": {"error": "no_indicator_labels"}}

    metrics: dict[str, float] = {}

    if gt_macd is not None:
        resp = await llm.complete(_prompt_messages(sample, "indicator_macd"))
        raw = resp.text or ""
        pred_macd = prompts.parse_finance_value_output(raw)
        if pred_macd is not None:
            metrics.update(_prefixed_metrics("macd", _forecast_metrics(gt_macd, pred_macd)))

    if gt_bb is not None:
        resp = await llm.complete(_prompt_messages(sample, "indicator_upper_bb"))
        raw = resp.text or ""
        pred_bb = prompts.parse_finance_value_output(raw)
        if pred_bb is not None:
            metrics.update(_prefixed_metrics("upper_bb", _forecast_metrics(gt_bb, pred_bb)))

    if not metrics:
        return {"metrics": {"error": "parse_fail"}}
    return {"metrics": metrics}


async def _baseline_stock_correlation(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    gt_corr = sample.labels.get("correlation", "")
    if not gt_corr:
        return {"metrics": {"error": "no_correlation_label"}}
    resp = await llm.complete(_prompt_messages(sample, "correlation"))
    raw = resp.text or ""
    pred = raw.strip()
    if not pred:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            pred = str(parsed.get("correlation", "")).strip()
    correct = pred.lower() == gt_corr.lower()
    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": pred,
        "ground_truth": gt_corr,
    }


async def _baseline_stock_mcqa(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    if not sample.question or not sample.answer:
        return {"metrics": {"error": "no_qa"}}
    resp = await llm.complete(_prompt_messages(sample, "mcqa"))
    raw = resp.text or ""
    pred = prompts.extract_mcqa_letter(raw)
    if not pred:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            pred = str(parsed.get("answer", "")).strip().upper()[:1]
    gt = sample.answer.strip().upper()[:1]
    return {
        "metrics": {"accuracy": 1.0 if pred == gt else 0.0},
        "predicted": pred,
        "ground_truth": gt,
    }


# ── Ground truth label helpers ────────────────────────────────────────

def _weather_trend_3way_label(sample: MTBenchSample) -> str:
    """3-way weather trend: increasing / decreasing / stable.
    
    Based on mean temperature difference between last input day
    and first output day, threshold ±0.5.
    """
    if sample.output_window is None or len(sample.input_window) < 24:
        return "stable"
    last_day_mean = float(np.mean(sample.input_window[-24:]))
    first_pred_day_mean = float(np.mean(sample.output_window[:24]))
    diff = first_pred_day_mean - last_day_mean
    if diff >= 0.5:
        return "increasing"
    elif diff <= -0.5:
        return "decreasing"
    return "stable"


def _weather_trend_past_3way_label(sample: MTBenchSample) -> str:
    """Official MTBench past-trend label for weather.

    Aligns with the official protocol: compute daily means from the
    input window, fit a linear trend, then threshold the slope at
    +/-0.25.
    """
    if len(sample.input_window) < 24:
        return "stable"
    arr = np.asarray(sample.input_window, dtype=np.float64).ravel()
    usable = (len(arr) // 24) * 24
    if usable < 24:
        return "stable"
    daily_mean = arr[:usable].reshape(usable // 24, 24).mean(axis=1)
    if len(daily_mean) < 2:
        return "stable"
    x = np.arange(len(daily_mean), dtype=np.float64)
    slope = float(np.polyfit(x, daily_mean, 1)[0])
    if slope >= 0.25:
        return "increasing"
    elif slope <= -0.25:
        return "decreasing"
    return "stable"


async def _baseline_weather_trend_past(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    resp = await llm.complete(_prompt_messages(sample, "trend_past"))
    raw = resp.text or ""
    pred_label = prompts.parse_weather_trend_output(raw)
    if pred_label is None:
        parsed = _parse_json_from_text(raw)
        if isinstance(parsed, dict):
            pred_label = _quick_match_label(str(parsed.get("trend", "")).strip().lower(), {"increasing", "decreasing", "stable"})
    pred_label = pred_label or ""

    gt_label = _weather_trend_past_3way_label(sample)
    correct = pred_label == gt_label
    return {
        "metrics": {"accuracy": 1.0 if correct else 0.0},
        "predicted": pred_label,
        "ground_truth": gt_label,
    }


# ── Task dispatch ─────────────────────────────────────────────────────

BASELINE_HANDLERS: dict[str, dict[str, Any]] = {
    "weather": {
        "forecast": _baseline_weather_forecast,
        "trend": _baseline_weather_trend,
        "trend_past": _baseline_weather_trend_past,
        "indicator": _baseline_weather_indicator,
        "mcqa": _baseline_weather_mcqa,
    },
    "finance": {
        "forecast": _baseline_stock_forecast,
        "trend": _baseline_stock_trend,
        "indicator": _baseline_stock_indicator,
        "indicator_macd": _baseline_stock_indicator,
        "indicator_upper_bb": _baseline_stock_indicator,
        "mcqa": _baseline_stock_mcqa,
        "correlation": _baseline_stock_correlation,
    },
}


async def _eval_one_sample(
    ds_name: str,
    idx: int,
    sample: MTBenchSample,
    tasks: list[str],
    llm: Any,
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    async with sem:
        results = []
        domain_handlers = BASELINE_HANDLERS.get(sample.domain, {})
        for task in tasks:
            handler = domain_handlers.get(task)
            if not handler:
                continue
            try:
                result = await asyncio.wait_for(handler(sample, llm), timeout=180)
            except asyncio.TimeoutError:
                result = {"metrics": {"error": "timeout"}}
            except Exception as e:
                logger.error(f"[Baseline] {ds_name}[{idx}] {task}: {e}")
                result = {"metrics": {"error": str(e)}}
            results.append({
                "idx": idx,
                "file": sample.file,
                "task": task,
                "domain": sample.domain,
                **result,
            })
        return results


async def run_baseline(
    dataset_name: str | None = None,
    task_filter: str | None = None,
    config: TimeClawConfig | None = None,
    max_samples: int | None = None,
    sample_indices: dict[str, list[int]] | str | Path | None = None,
    concurrency: int = 80,
    full_dataset: bool = False,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Run direct-LLM baseline on the official **test list** or full dataset.

    If ``reuse_existing`` is True (default) and ``baseline_<ds>.json`` already
    exists under ``.timeclaw/test_results`` with non-trivial size, skip that
    dataset (baseline only needs to be computed once for a fixed sample list).
    """
    cfg = config or TimeClawConfig.load()
    router = ModelRouter(cfg.llm)
    llm = router.get_backend("executor")
    sem = asyncio.Semaphore(concurrency)
    sample_indices_map = (
        load_sample_indices(sample_indices)
        if isinstance(sample_indices, (str, Path))
        else sample_indices
    )

    from timeclaw.benchmark.mtbench_adapter import load_dataset

    datasets = (
        [dataset_name] if dataset_name else datasets_with_benchmark_tasks()
    )
    all_results: dict[str, Any] = {}

    for ds_name in datasets:
        info = DATASET_REGISTRY.get(ds_name)
        if not info:
            continue
        bl_path = RESULTS_DIR / f"baseline_{ds_name}.json"
        if reuse_existing and bl_path.exists() and bl_path.stat().st_size > 10:
            try:
                prev = json.loads(bl_path.read_text())
                summary = _summarize(prev if isinstance(prev, list) else [])
                all_results[ds_name] = {
                    "total": len(prev) if isinstance(prev, list) else 0,
                    "elapsed_s": 0.0,
                    "summary": summary,
                    "reused": True,
                }
                logger.info(f"[Baseline] {ds_name}: reuse_existing, skipping ({bl_path})")
                continue
            except (json.JSONDecodeError, OSError):
                logger.warning(f"[Baseline] {ds_name}: corrupt baseline file, will re-run")

        try:
            if full_dataset:
                samples = load_dataset(ds_name, config=cfg)
            else:
                samples = load_split(ds_name, "test", config=cfg)
        except FileNotFoundError:
            logger.warning(f"[Baseline] No data for {ds_name}")
            continue

        selected_samples = select_samples_with_original_indices(samples, ds_name, sample_indices_map)
        if max_samples:
            selected_samples = selected_samples[:max_samples]

        tasks = expand_dataset_tasks(ds_name, list(info["tasks"]))
        if task_filter:
            tasks = [t for t in tasks if t == task_filter]
        if not tasks:
            continue

        logger.info(f"[Baseline] {ds_name}: {len(selected_samples)} samples, tasks={tasks}, concurrency={concurrency}")
        t0 = time.time()

        coros = [
            _eval_one_sample(ds_name, idx, s, tasks, llm, sem)
            for idx, s in selected_samples
        ]
        batch = await asyncio.gather(*coros, return_exceptions=True)

        ds_results: list[dict[str, Any]] = []
        for b in batch:
            if isinstance(b, Exception):
                logger.error(f"[Baseline] {ds_name}: {b}")
                continue
            ds_results.extend(b)

        elapsed = time.time() - t0
        _save_results(ds_results, ds_name)
        summary = _summarize(ds_results)
        all_results[ds_name] = {"total": len(ds_results), "elapsed_s": round(elapsed, 1), "summary": summary}
        logger.info(f"[Baseline] {ds_name} done in {elapsed:.1f}s: {json.dumps(summary, default=str)}")

    return all_results


# ── LLM Reasoning baseline (uses llm_reasoning tool for all tasks) ──


async def _llm_reasoning_call(llm: Any, sample: MTBenchSample, question: str, output_schema: str) -> Any:
    """Direct call to llm_reasoning logic without MCP overhead."""
    from timeclaw.clawhub.generic_llm.llm_reasoning import _fmt_array
    ts_section = ""
    if sample.input_window is not None:
        arr = np.asarray(sample.input_window, dtype=np.float64).ravel()
        ts_section = f"Time-series data ({len(arr)} data points):\n[{_fmt_array(arr)}]\n\n"
    domain_section = f"Domain context:\n{sample.domain}\n\n" if sample.domain else ""
    text_section = f"Contextual information:\n{sample.text}\n\n" if sample.text else ""
    system_prompt = (
        "You are a data analysis expert. Analyze the provided data "
        "and any contextual information to answer the question.\n"
        f"Return ONLY a JSON object matching this schema: {output_schema}"
    )
    user_prompt = f"{domain_section}{ts_section}{text_section}\nQuestion: {question}"
    resp = await llm.complete(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format="json",
    )
    return _parse_json_from_text(resp.text or "")


async def _llmr_forecast(sample: MTBenchSample, llm: Any) -> dict[str, Any]:
    if sample.output_window is None:
        return {"metrics": {"error": "no output_window"}}
    horizon = len(sample.output_window)
    horizon_days = max(1, horizon // 24)
    ds = getattr(sample, "dataset", "")
    if sample.domain == "weather":
        question = (
            f"Predict the next {horizon_days} day(s) of hourly temperature readings "
            f"({horizon} values). Return a JSON array of {horizon} numbers."
        )
    else:
        question = (
            f"Predict the next {horizon} stock price values. "
            f"Return a JSON array of {horizon} numbers."
        )
    parsed = await _llm_reasoning_call(llm, sample, question, f'{{"forecast": [array of {horizon} numbers]}}')
    pred = _extract_number_array(parsed, horizon)
    if pred is not None:
        return {"metrics": _forecast_metrics(sample.output_window, pred)}
    return {"metrics": {"error": "parse_fail"}}


async def _llmr_trend(sample: MTBenchSample, llm: Any) -> dict[str, Any]:
    if sample.domain == "weather":
        gt_label = _weather_trend_3way_label(sample)
        question = (
            "Every 24 readings represent a full day. Predict the temperature trend "
            "for the next 1 day: compare the mean of the last 24h with the predicted "
            "next-day mean. If diff >= 0.5 → 'increasing', <= -0.5 → 'decreasing', "
            "else 'stable'."
        )
        parsed = await _llm_reasoning_call(llm, sample, question, '{"trend": "<increasing|decreasing|stable>"}')
        pred = ""
        valid = {"increasing", "decreasing", "stable"}
        if isinstance(parsed, dict) and "trend" in parsed:
            pred = _quick_match_label(str(parsed["trend"]).lower(), valid) or "stable"
        else:
            pred = "stable"
        return {"metrics": {"accuracy": 1.0 if pred == gt_label else 0.0}, "predicted": pred, "ground_truth": gt_label}
    else:
        gt_label = sample.labels.get("trend_label", "")
        if not gt_label:
            return {"metrics": {"error": "no_trend_label"}}
        valid_5 = {"<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%"}
        question = (
            "Categorize the future stock price percentage change into one of: "
            '"<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%".'
        )
        parsed = await _llm_reasoning_call(llm, sample, question, '{"trend": "<label>"}')
        pred = ""
        if isinstance(parsed, dict) and "trend" in parsed:
            pred = _quick_match_label(str(parsed["trend"]).strip(), valid_5) or "-2% ~ +2%"
        else:
            pred = "-2% ~ +2%"
        return {"metrics": {"accuracy": 1.0 if pred == gt_label else 0.0}, "predicted": pred, "ground_truth": gt_label}


async def _llmr_indicator(sample: MTBenchSample, llm: Any) -> dict[str, Any]:
    if sample.output_window is None:
        return {"metrics": {"error": "no output_window"}}
    if sample.domain == "weather":
        gt = prompts.weather_indicator_ground_truth(sample.output_window)
        n_days = 1
        gt_max = gt["max"]
        gt_min = gt["min"]
        gt_diff = gt["diff"]
        if gt_diff < 0.01:
            return {"metrics": {"skipped": "flat_gt"}}
        question = (
            f"Every 24 readings represent a full day. Predict the next {n_days} day(s) "
            f"highest temperature, lowest temperature, and the temperature difference."
        )
        parsed = await _llm_reasoning_call(llm, sample, question, '{"max": number, "min": number, "diff": number}')
        if isinstance(parsed, dict):
            p_max = float(parsed.get("max", 0))
            p_min = float(parsed.get("min", 0))
            p_diff = float(parsed.get("diff", 0))
            return {"metrics": {
                "max_mae": abs(p_max - gt_max),
                "min_mae": abs(p_min - gt_min),
                "diff_mae": abs(p_diff - gt_diff),
            }}
        return {"metrics": {"error": "parse_fail"}}
    else:
        gt_macd = sample.labels.get("out_macd")
        gt_upper_bb = sample.labels.get("out_upper_bb")
        if gt_macd is None and gt_upper_bb is None:
            return {"metrics": {"error": "no indicator labels"}}
        question = (
            "Predict the future MACD values and upper Bollinger Band values for the next time steps."
        )
        parsed = await _llm_reasoning_call(
            llm,
            sample,
            question,
            '{"macd": [array of numbers], "upper_bb": [array of numbers]}',
        )
        metrics: dict[str, float] = {}
        if isinstance(parsed, dict):
            if gt_macd is not None and "macd" in parsed:
                metrics.update(_prefixed_metrics("macd", _forecast_metrics(gt_macd, parsed["macd"])))
            if gt_upper_bb is not None and "upper_bb" in parsed:
                metrics.update(_prefixed_metrics("upper_bb", _forecast_metrics(gt_upper_bb, parsed["upper_bb"])))
        if not metrics:
            return {"metrics": {"error": "parse_fail"}}
        return {"metrics": metrics}


async def _llmr_mcqa(sample: MTBenchSample, llm: Any) -> dict[str, Any]:
    if not sample.question or not sample.answer:
        return {"metrics": {"error": "no question/answer"}}
    question = f"{sample.question}\nAnswer with exactly one of A, B, C, or D."
    parsed = await _llm_reasoning_call(llm, sample, question, '{"answer": "X"}')
    pred = ""
    if isinstance(parsed, dict) and "answer" in parsed:
        pred = str(parsed["answer"]).strip().upper()[:1]
    gt = str(sample.answer).strip().upper()[:1]
    return {"metrics": {"accuracy": 1.0 if pred == gt else 0.0}, "predicted": pred, "ground_truth": gt}


async def _llmr_correlation(sample: MTBenchSample, llm: Any) -> dict[str, Any]:
    gt = sample.labels.get("correlation", "")
    if not gt:
        return {"metrics": {"error": "no_label"}}
    question = (
        "Determine whether the two stocks are positively correlated, "
        "negatively correlated, or not correlated."
    )
    parsed = await _llm_reasoning_call(llm, sample, question, '{"correlation": "positive|negative|none"}')
    pred = ""
    if isinstance(parsed, dict) and "correlation" in parsed:
        pred = str(parsed["correlation"]).strip().lower()
    correct = pred == gt.lower()
    return {"metrics": {"accuracy": 1.0 if correct else 0.0}, "predicted": pred, "ground_truth": gt}


LLMR_HANDLERS: dict[str, Any] = {
    "forecast": _llmr_forecast,
    "trend": _llmr_trend,
    "indicator": _llmr_indicator,
    "mcqa": _llmr_mcqa,
    "correlation": _llmr_correlation,
}


async def run_llm_reasoning_baseline(
    dataset_name: str | None = None,
    task_filter: str | None = None,
    config: TimeClawConfig | None = None,
    max_samples: int | None = None,
    concurrency: int = 80,
    full_dataset: bool = False,
) -> dict[str, Any]:
    """Run llm_reasoning-only baseline on all tasks."""
    cfg = config or TimeClawConfig.load()
    router = ModelRouter(cfg.llm)
    llm = router.get_backend("executor")
    sem = asyncio.Semaphore(concurrency)

    from timeclaw.benchmark.mtbench_adapter import load_dataset

    datasets = (
        [dataset_name] if dataset_name else datasets_with_benchmark_tasks()
    )
    all_results: dict[str, Any] = {}

    for ds_name in datasets:
        info = DATASET_REGISTRY.get(ds_name)
        if not info:
            continue
        try:
            if full_dataset:
                samples = load_dataset(ds_name, config=cfg)
            else:
                samples = load_split(ds_name, "test", config=cfg)
        except FileNotFoundError:
            logger.warning(f"[LLM-Reasoning] No data for {ds_name}")
            continue
        if max_samples:
            samples = samples[:max_samples]
        tasks = expand_dataset_tasks(ds_name, list(info["tasks"]))
        if task_filter:
            tasks = [t for t in tasks if t == task_filter]
        if not tasks:
            continue

        logger.info(f"[LLM-Reasoning] {ds_name}: {len(samples)} samples, tasks={tasks}")
        t0 = time.time()

        async def _eval_llmr_one(idx: int, s: MTBenchSample) -> list[dict[str, Any]]:
            async with sem:
                results = []
                for task in tasks:
                    handler = LLMR_HANDLERS.get(task)
                    if not handler:
                        continue
                    try:
                        result = await asyncio.wait_for(handler(s, llm), timeout=180)
                    except asyncio.TimeoutError:
                        result = {"metrics": {"error": "timeout"}}
                    except Exception as e:
                        logger.error(f"[LLM-Reasoning] {ds_name}[{idx}] {task}: {e}")
                        result = {"metrics": {"error": str(e)}}
                    results.append({"idx": idx, "file": s.file, "task": task, "domain": s.domain, **result})
                return results

        coros = [_eval_llmr_one(idx, s) for idx, s in enumerate(samples)]
        batch = await asyncio.gather(*coros, return_exceptions=True)

        ds_results: list[dict[str, Any]] = []
        for b in batch:
            if isinstance(b, Exception):
                logger.error(f"[LLM-Reasoning] {ds_name}: {b}")
                continue
            ds_results.extend(b)

        elapsed = time.time() - t0
        summary = _summarize(ds_results)
        all_results[ds_name] = {"total": len(ds_results), "elapsed_s": round(elapsed, 1), "summary": summary}
        logger.info(f"[LLM-Reasoning] {ds_name} done in {elapsed:.1f}s: {json.dumps(summary, default=str)}")

    return all_results


def _save_results(results: list[dict[str, Any]], ds_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"baseline_{ds_name}.json"
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
            task_summary[f"n"] = len(vals)
        summary[task] = task_summary
    return summary


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MTBench Direct-LLM Baseline")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-index-file", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=80)
    args = parser.parse_args()

    result = await run_baseline(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        sample_indices=args.sample_index_file,
        concurrency=args.concurrency,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
