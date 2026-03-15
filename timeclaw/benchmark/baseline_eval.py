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
from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY, MTBenchSample
from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.core.config import TimeClawConfig
from timeclaw.llm.router import ModelRouter

RESULTS_DIR = Path(".timeclaw/test_results")


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
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return {"mse": 999.0, "mae": 999.0, "rmse": 999.0, "mape": 999.0}
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


# ── Per-task baseline handlers ────────────────────────────────────────

async def _baseline_weather_forecast(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    if sample.output_window is None:
        return {"error": "no output_window"}
    horizon_days = len(sample.output_window) // 24
    if horizon_days < 1:
        horizon_days = 1
    sys_p, user_p = prompts.weather_forecast(
        sample.input_window, sample.timestamps, sample.text, horizon_days,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
    )
    raw = resp.text or ""
    parsed = _parse_json_from_text(raw)
    pred = _extract_number_array(parsed, len(sample.output_window), raw)
    if pred is not None:
        metrics = _forecast_metrics(sample.output_window, pred)
        return {"metrics": metrics, "pred_len": len(pred)}
    return {"metrics": {"error": "parse_fail"}}


async def _baseline_weather_trend(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    sys_p, user_p = prompts.weather_trend_future(
        sample.input_window, sample.timestamps, sample.text,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        response_format="json",
    )
    parsed = _parse_json_from_text(resp.text or "")
    pred_label = ""
    if isinstance(parsed, dict):
        pred_label = str(parsed.get("trend", "")).strip().lower()

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
    gt_max = float(np.max(sample.output_window))
    gt_min = float(np.min(sample.output_window))
    gt_diff = gt_max - gt_min
    if gt_diff < 0.01:
        return {"metrics": {"skipped": "flat_gt"}}
    sys_p, user_p = prompts.weather_indicator(
        sample.input_window, sample.timestamps, sample.text,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        response_format="json",
    )
    parsed = _parse_json_from_text(resp.text or "")
    gt = {"max": gt_max, "min": gt_min, "diff": gt_diff}
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
    sys_p, user_p = prompts.weather_mcqa(
        sample.input_window, sample.timestamps, sample.text, sample.question,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        response_format="json",
    )
    parsed = _parse_json_from_text(resp.text or "")
    pred = ""
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
    ds = sample.dataset
    if "short" in ds:
        input_desc, output_desc = "7 days at 5-minute intervals", "next 1 day"
    else:
        input_desc, output_desc = "30 days at 1-hour intervals", "next 7 days"
    sys_p, user_p = prompts.stock_forecast(
        sample.input_window, sample.timestamps, sample.text,
        horizon, input_desc, output_desc,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
    )
    raw = resp.text or ""
    parsed = _parse_json_from_text(raw)
    pred = _extract_number_array(parsed, horizon, raw)
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
    ds = sample.dataset
    if "short" in ds:
        input_desc = "5-minute intervals over the last week"
    else:
        input_desc = "1-hour intervals over the last month"
    sys_p, user_p = prompts.stock_trend_5way(
        sample.input_window, sample.timestamps, sample.text, input_desc,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        response_format="json",
    )
    parsed = _parse_json_from_text(resp.text or "")
    pred_label = ""
    if isinstance(parsed, dict):
        pred_label = str(parsed.get("trend", "")).strip()
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

    ds = sample.dataset
    input_desc = "7 days" if "short" in ds else "30 days"
    metrics: dict[str, float] = {}

    if gt_macd is not None:
        sys_p, user_p = prompts.stock_indicator_macd(
            sample.input_window, sample.timestamps, sample.text, input_desc,
        )
        resp = await llm.complete(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            response_format="json",
        )
        parsed = _parse_json_from_text(resp.text or "")
        if isinstance(parsed, dict) and "macd" in parsed:
            metrics["macd_mae"] = abs(float(gt_macd) - float(parsed["macd"]))

    if gt_bb is not None:
        sys_p, user_p = prompts.stock_indicator_bb(
            sample.input_window, sample.timestamps, sample.text, input_desc,
        )
        resp = await llm.complete(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            response_format="json",
        )
        parsed = _parse_json_from_text(resp.text or "")
        if isinstance(parsed, dict) and "upper_bb" in parsed:
            metrics["upper_bb_mae"] = abs(float(gt_bb) - float(parsed["upper_bb"]))

    if not metrics:
        return {"metrics": {"error": "parse_fail"}}
    return {"metrics": metrics}


async def _baseline_stock_correlation(
    sample: MTBenchSample, llm: Any,
) -> dict[str, Any]:
    gt_corr = sample.labels.get("correlation", "")
    if not gt_corr:
        return {"metrics": {"error": "no_correlation_label"}}
    sys_p, user_p = prompts.stock_correlation_5way(
        sample.input_window, sample.timestamps, sample.text,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        response_format="json",
    )
    parsed = _parse_json_from_text(resp.text or "")
    pred = ""
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
    sys_p, user_p = prompts.stock_mcqa(
        sample.input_window, sample.timestamps, sample.text, sample.question,
    )
    resp = await llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        response_format="json",
    )
    parsed = _parse_json_from_text(resp.text or "")
    pred = ""
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


# ── Task dispatch ─────────────────────────────────────────────────────

BASELINE_HANDLERS: dict[str, dict[str, Any]] = {
    "weather": {
        "forecast": _baseline_weather_forecast,
        "trend": _baseline_weather_trend,
        "indicator": _baseline_weather_indicator,
        "mcqa": _baseline_weather_mcqa,
    },
    "finance": {
        "forecast": _baseline_stock_forecast,
        "trend": _baseline_stock_trend,
        "indicator": _baseline_stock_indicator,
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
    concurrency: int = 80,
    full_dataset: bool = False,
) -> dict[str, Any]:
    """Run direct-LLM baseline on test splits or full dataset."""
    cfg = config or TimeClawConfig.load()
    router = ModelRouter(cfg.llm)
    llm = router.get_backend("executor")
    sem = asyncio.Semaphore(concurrency)

    from timeclaw.benchmark.mtbench_adapter import load_dataset

    datasets = [dataset_name] if dataset_name else list(DATASET_REGISTRY.keys())
    all_results: dict[str, Any] = {}

    for ds_name in datasets:
        info = DATASET_REGISTRY.get(ds_name)
        if not info:
            continue
        try:
            if full_dataset:
                samples = load_dataset(ds_name)
            else:
                samples = load_split(ds_name, "test")
        except FileNotFoundError:
            logger.warning(f"[Baseline] No data for {ds_name}")
            continue

        if max_samples:
            samples = samples[:max_samples]

        tasks = info["tasks"]
        if task_filter:
            tasks = [t for t in tasks if t == task_filter]
        if not tasks:
            continue

        logger.info(f"[Baseline] {ds_name}: {len(samples)} samples, tasks={tasks}, concurrency={concurrency}")
        t0 = time.time()

        coros = [
            _eval_one_sample(ds_name, idx, s, tasks, llm, sem)
            for idx, s in enumerate(samples)
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
    parser.add_argument("--concurrency", type=int, default=80)
    args = parser.parse_args()

    result = await run_baseline(
        dataset_name=args.dataset,
        task_filter=args.task,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
