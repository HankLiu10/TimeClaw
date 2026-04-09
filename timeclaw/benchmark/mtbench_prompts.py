"""MTBench prompt bridge.

New definition:
    This module no longer maintains a small set of rewritten prompt
    templates. It dispatches to dataset/task-specific prompt builders that
    mirror the official MTBench repository implementation as closely as
    possible, including some quirks in the upstream scripts.
"""

from __future__ import annotations

from datetime import datetime
import importlib.util
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from timeclaw.benchmark.mtbench_adapter import MTBenchSample


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, module_name: str) -> Any:
    path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MTBench module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_WEATHER_META = _load_module(
    "mtbench_official/evaluation/weather/meta_prompt.py",
    "timeclaw_mtbench_weather_meta_prompt",
)
_FINANCE_META = _load_module(
    "mtbench_official/evaluation/finance/meta_prompt.py",
    "timeclaw_mtbench_finance_meta_prompt",
)
_EVAL_UTILS = _load_module(
    "mtbench_official/evaluation/utils.py",
    "timeclaw_mtbench_eval_utils",
)


def _weather_ts_string(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def _finance_ts_object(epoch: float) -> datetime:
    # Official finance scripts use datetime.fromtimestamp(...) directly.
    return datetime.fromtimestamp(epoch)


def _finance_ts_string(epoch: float) -> str:
    return _finance_ts_object(epoch).strftime("%Y-%m-%d %H:%M:%S")


def _input_prices(sample: "MTBenchSample") -> list[float]:
    return np.asarray(sample.input_window, dtype=np.float64).ravel().tolist()


def _output_prices(sample: "MTBenchSample") -> list[float]:
    if sample.output_window is None:
        return []
    return np.asarray(sample.output_window, dtype=np.float64).ravel().tolist()


def _indicator_series(sample: "MTBenchSample", key: str) -> list[float]:
    raw = sample.labels.get(key)
    if raw is None:
        return []
    return np.asarray(raw, dtype=np.float64).ravel().tolist()


def _output_ts_string(sample: "MTBenchSample") -> str:
    output_timestamps = sample.metadata.get("output_timestamps") or []
    if output_timestamps:
        return _finance_ts_string(float(output_timestamps[-1]))
    output_window = _output_prices(sample)
    return str(output_window[-1]) if output_window else ""


def _finance_granularity(sample: "MTBenchSample") -> str:
    if len(sample.timestamps) < 2:
        return "unknown"
    arr = np.asarray(sample.timestamps, dtype=np.float64).ravel()
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return "unknown"
    return _EVAL_UTILS.format_time_difference(float(np.median(diffs)))


def _prompt(raw_prompt: str) -> tuple[str, str]:
    return "", raw_prompt


def weather_aligned_short_forecast_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_forecast_metaprompt_generation(
            text=sample.text,
            temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            prediction_length=len(_output_prices(sample)),
            mode="combined",
        )
    )


def weather_aligned_long_forecast_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_forecast_metaprompt_generation(
            text=sample.text,
            temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            prediction_length=len(_output_prices(sample)),
            mode="combined",
            next_days=3,
        )
    )


def weather_aligned_short_trend_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_trend_metaprompt_generation(
            text=sample.text,
            past_temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            past_days=7,
            next_days=1,
            mode="combined",
            past_future="future",
        )
    )


def weather_aligned_long_trend_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_trend_metaprompt_generation(
            text=sample.text,
            past_temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            past_days=14,
            next_days=3,
            mode="combined",
            past_future="future",
        )
    )


def weather_aligned_short_trend_past_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_trend_metaprompt_generation(
            text=sample.text,
            past_temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            past_days=7,
            next_days=1,
            mode="combined",
            past_future="past",
        )
    )


def weather_aligned_long_trend_past_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_trend_metaprompt_generation(
            text=sample.text,
            past_temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            past_days=14,
            next_days=3,
            mode="combined",
            past_future="past",
        )
    )


def weather_aligned_short_indicator_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_indicator_metaprompt_generation(
            text=sample.text,
            past_temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            past_days=7,
            mode="combined",
        )
    )


def weather_aligned_long_indicator_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_indicator_metaprompt_generation(
            text=sample.text,
            past_temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            past_days=14,
            mode="combined",
        )
    )


def weather_qa_short_mcqa_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.weather_mcqa_metaprompt_generation(
            past_days=7,
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            past_temperatures=_input_prices(sample),
            news=sample.text,
            question=sample.question or "",
        )
    )


def weather_qa_long_forecast_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_forecast_metaprompt_generation(
            text=sample.text,
            temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            prediction_length=len(_output_prices(sample)),
            mode="combined",
            next_days=3,
        )
    )


def weather_qa_long_mcqa_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.weather_mcqa_metaprompt_generation(
            past_days=14,
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            past_temperatures=_input_prices(sample),
            news=sample.text,
            question=sample.question or "",
        )
    )


def finance_aligned_short_forecast_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _FINANCE_META.finance_mse_metaprompt_generation(
            text=sample.text,
            prices=_input_prices(sample),
            start_datetime=_finance_ts_string(sample.timestamps[0]),
            end_datetime=_finance_ts_string(sample.timestamps[-1]),
            pred_end_datetime=_output_ts_string(sample),
            granularity=_finance_granularity(sample),
            prediction_length=len(_output_prices(sample)),
            mode="combined",
        )
    )


def finance_aligned_long_forecast_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _FINANCE_META.finance_mse_metaprompt_generation(
            text=sample.text,
            prices=_input_prices(sample),
            start_datetime=_finance_ts_string(sample.timestamps[0]),
            end_datetime=_finance_ts_string(sample.timestamps[-1]),
            pred_end_datetime=_output_ts_string(sample),
            granularity=_finance_granularity(sample),
            prediction_length=len(_output_prices(sample)),
            mode="combined",
        )
    )


def finance_aligned_short_trend_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    timestamp_strings = [_finance_ts_string(ts) for ts in sample.timestamps]
    prompt = _FINANCE_META.finance_classification_metaprompt_generation(
        text=sample.text,
        timestamps=timestamp_strings,
        prices=_input_prices(sample),
        mode="combined",
    )
    prompt = prompt.replace(
        "1-hour intervals over the last month",
        "5-minute intervals over the last week",
    )
    return _prompt(prompt)


def finance_aligned_long_trend_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    timestamp_strings = [_finance_ts_string(ts) for ts in sample.timestamps]
    return _prompt(
        _FINANCE_META.finance_classification_metaprompt_generation(
            text=sample.text,
            timestamps=timestamp_strings,
            prices=_input_prices(sample),
            mode="combined",
        )
    )


def finance_aligned_short_indicator_macd_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    out_macd = _indicator_series(sample, "out_macd")
    return _prompt(
        _FINANCE_META.finance_macd_metaprompt_generation(
            text=sample.text,
            prices=_input_prices(sample),
            start_datetime=_finance_ts_string(sample.timestamps[0]),
            end_datetime=_finance_ts_string(sample.timestamps[-1]),
            pred_end_datetime=_output_ts_string(sample),
            granularity=_finance_granularity(sample),
            prediction_length=len(out_macd),
            mode="combined",
        )
    )


def finance_aligned_short_indicator_upper_bb_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    out_bb = _indicator_series(sample, "out_upper_bb")
    return _prompt(
        _FINANCE_META.finance_bb_metaprompt_generation(
            text=sample.text,
            prices=_input_prices(sample),
            start_datetime=_finance_ts_string(sample.timestamps[0]),
            end_datetime=_finance_ts_string(sample.timestamps[-1]),
            pred_end_datetime=_output_ts_string(sample),
            granularity=_finance_granularity(sample),
            prediction_length=len(out_bb),
            mode="combined",
        )
    )


def finance_aligned_long_indicator_macd_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    out_macd = _indicator_series(sample, "out_macd")
    return _prompt(
        _FINANCE_META.finance_macd_metaprompt_generation(
            text=sample.text,
            prices=_input_prices(sample),
            start_datetime=_finance_ts_string(sample.timestamps[0]),
            end_datetime=_finance_ts_string(sample.timestamps[-1]),
            pred_end_datetime=_output_ts_string(sample),
            granularity=_finance_granularity(sample),
            prediction_length=len(out_macd),
            mode="combined",
        )
    )


def finance_aligned_long_indicator_upper_bb_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    out_bb = _indicator_series(sample, "out_upper_bb")
    return _prompt(
        _FINANCE_META.finance_bb_metaprompt_generation(
            text=sample.text,
            prices=_input_prices(sample),
            start_datetime=_finance_ts_string(sample.timestamps[0]),
            end_datetime=_finance_ts_string(sample.timestamps[-1]),
            pred_end_datetime=_output_ts_string(sample),
            granularity=_finance_granularity(sample),
            prediction_length=len(out_bb),
            mode="combined",
        )
    )


def finance_qa_short_mcqa_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _FINANCE_META.finance_mcqa_metaprompt_generation(
            setting="short",
            sticker=str(sample.metadata.get("sticker", "")),
            time1=_finance_ts_object(sample.timestamps[0]),
            time2=_finance_ts_object(sample.timestamps[-1]),
            in_price=_input_prices(sample),
            news=sample.text,
            time_news=sample.metadata.get("published_utc"),
            question=sample.question or "",
        )
    )


def finance_qa_long_mcqa_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _FINANCE_META.finance_mcqa_metaprompt_generation(
            setting="long",
            sticker=str(sample.metadata.get("sticker", "")),
            time1=_finance_ts_object(sample.timestamps[0]),
            time2=_finance_ts_object(sample.timestamps[-1]),
            in_price=_input_prices(sample),
            news=sample.text,
            time_news=sample.metadata.get("published_utc"),
            question=sample.question or "",
        )
    )


def finance_qa_short_correlation_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _FINANCE_META.finance_correlation_metaprompt_generation(
            setting="short",
            sticker=str(sample.metadata.get("sticker", "")),
            time1=_finance_ts_object(sample.timestamps[0]),
            time2=_finance_ts_object(sample.timestamps[-1]),
            in_price=_input_prices(sample),
            news=sample.text,
            time_news=sample.metadata.get("published_utc"),
        )
    )


def finance_qa_long_correlation_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _FINANCE_META.finance_correlation_metaprompt_generation(
            setting="long",
            sticker=str(sample.metadata.get("sticker", "")),
            time1=_finance_ts_object(sample.timestamps[0]),
            time2=_finance_ts_object(sample.timestamps[-1]),
            in_price=_input_prices(sample),
            news=sample.text,
            time_news=sample.metadata.get("published_utc"),
        )
    )


_PROMPT_BUILDERS: dict[tuple[str, str], Any] = {
    ("weather_aligned_short", "forecast"): weather_aligned_short_forecast_prompt,
    ("weather_aligned_long", "forecast"): weather_aligned_long_forecast_prompt,
    ("weather_aligned_short", "trend"): weather_aligned_short_trend_prompt,
    ("weather_aligned_long", "trend"): weather_aligned_long_trend_prompt,
    ("weather_aligned_short", "trend_past"): weather_aligned_short_trend_past_prompt,
    ("weather_aligned_long", "trend_past"): weather_aligned_long_trend_past_prompt,
    ("weather_aligned_short", "indicator"): weather_aligned_short_indicator_prompt,
    ("weather_aligned_long", "indicator"): weather_aligned_long_indicator_prompt,
    ("weather_qa_short", "mcqa"): weather_qa_short_mcqa_prompt,
    ("weather_qa_long", "forecast"): weather_qa_long_forecast_prompt,
    ("weather_qa_long", "mcqa"): weather_qa_long_mcqa_prompt,
    ("finance_aligned_short", "forecast"): finance_aligned_short_forecast_prompt,
    ("finance_aligned_long", "forecast"): finance_aligned_long_forecast_prompt,
    ("finance_aligned_short", "trend"): finance_aligned_short_trend_prompt,
    ("finance_aligned_long", "trend"): finance_aligned_long_trend_prompt,
    ("finance_aligned_short", "indicator_macd"): finance_aligned_short_indicator_macd_prompt,
    ("finance_aligned_short", "indicator_upper_bb"): finance_aligned_short_indicator_upper_bb_prompt,
    ("finance_aligned_long", "indicator_macd"): finance_aligned_long_indicator_macd_prompt,
    ("finance_aligned_long", "indicator_upper_bb"): finance_aligned_long_indicator_upper_bb_prompt,
    ("finance_qa_short", "mcqa"): finance_qa_short_mcqa_prompt,
    ("finance_qa_long", "mcqa"): finance_qa_long_mcqa_prompt,
    ("finance_qa_short", "correlation"): finance_qa_short_correlation_prompt,
    ("finance_qa_long", "correlation"): finance_qa_long_correlation_prompt,
}


def build_prompt(sample: "MTBenchSample", task: str) -> tuple[str, str]:
    if sample.domain == "finance" and task == "indicator":
        parts: list[str] = []
        if sample.labels.get("out_macd") is not None:
            parts.append(build_prompt(sample, "indicator_macd")[1])
        if sample.labels.get("out_upper_bb") is not None:
            parts.append(build_prompt(sample, "indicator_upper_bb")[1])
        if parts:
            return "", "\n\n---\n\n".join(parts)

    builder = _PROMPT_BUILDERS.get((sample.dataset, task))
    if builder is None:
        raise KeyError(f"Unsupported MTBench prompt mapping: dataset={sample.dataset}, task={task}")
    return builder(sample)


def append_tool_analysis(user_prompt: str, analysis: str) -> str:
    if not analysis.strip():
        return user_prompt
    return (
        f"{user_prompt}\n\n"
        "Additional TimeClaw tool analysis (not part of the original MTBench prompt):\n"
        f"{analysis}\n\n"
        "Follow the original MTBench task instructions and output format exactly."
    )


def prompt_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    if system_prompt.strip():
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return [{"role": "user", "content": user_prompt}]


def parse_weather_forecast_output(text: str) -> list[float] | None:
    try:
        return list(_WEATHER_META.decode_temperature_forecast(text))
    except Exception:
        return None


def parse_weather_trend_output(text: str) -> str | None:
    try:
        return str(_WEATHER_META.decode_temperature_trend_prediction(text))
    except Exception:
        return None


def parse_weather_indicator_output(text: str) -> dict[str, float] | None:
    try:
        max_temp, min_temp, diff_temp = _WEATHER_META.decode_temperature_indicator(text)
    except Exception:
        return None
    return {"max": float(max_temp), "min": float(min_temp), "diff": float(diff_temp)}


def parse_finance_value_output(text: str) -> list[float] | None:
    try:
        values = _FINANCE_META.parse_val_prediction_response(text)
    except Exception:
        return None
    return list(values) if values is not None else None


def parse_finance_trend_output(text: str) -> str | None:
    try:
        return str(_FINANCE_META.parse_cls_response(text))
    except Exception:
        return None


def extract_mcqa_letter(text: str) -> str:
    stripped = text.strip()
    if stripped[:1].upper() in {"A", "B", "C", "D"}:
        return stripped[:1].upper()
    match = re.search(r"\b([A-D])\b", stripped, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def weather_indicator_horizon_hours(output_window: np.ndarray | list[float]) -> int:
    arr = np.asarray(output_window, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0
    return min(arr.size, 24)


def weather_indicator_ground_truth(output_window: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(output_window, dtype=np.float64).ravel()
    horizon = weather_indicator_horizon_hours(arr)
    if horizon <= 0:
        return {"max": 0.0, "min": 0.0, "diff": 0.0}
    return _EVAL_UTILS.get_temperature_diff_max_min(arr[:horizon].tolist())
