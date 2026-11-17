"""MTBench prompt bridge.

New definition:
    This module no longer maintains a small set of rewritten prompt
    templates. It dispatches to dataset/task-specific prompt builders that
    mirror the official MTBench repository implementation as closely as
    possible, including some quirks in the upstream scripts.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import replace
from numbers import Integral
import importlib.util
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from timeclaw.data_integrity import DataIntegrityError
from timeclaw.text.task_pruning import task_text_view

if TYPE_CHECKING:
    from timeclaw.benchmark.mtbench_adapter import MTBenchSample


_MTBENCH_ROOT = Path(os.environ.get("TIMECLAW_MTBENCH_ROOT", "mtbench_official")).resolve()


def _load_module(rel_path: str, module_name: str) -> Any:
    path = _MTBENCH_ROOT / rel_path
    if not path.is_file():
        raise ImportError(
            f"MTBench dependency not found: {path}. Run python scripts/setup_mtbench.py "
            "from the release root, or set TIMECLAW_MTBENCH_ROOT to its checkout."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MTBench module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_WEATHER_META = _load_module(
    "evaluation/weather/meta_prompt.py",
    "timeclaw_mtbench_weather_meta_prompt",
)
_FINANCE_META = _load_module(
    "evaluation/finance/meta_prompt.py",
    "timeclaw_mtbench_finance_meta_prompt",
)
_EVAL_UTILS = _load_module(
    "evaluation/utils.py",
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


def resolve_forecast_horizon(sample: "MTBenchSample") -> int:
    """Resolve the public forecast length without reading future values.

    An explicit ``forecast_horizon`` is authoritative and must be a positive
    integer. Only legacy benchmark samples without that field use the target
    array's time-axis length; the target values themselves are not inspected.
    """
    horizon = getattr(sample, "forecast_horizon", None)
    if horizon is None:
        if sample.output_window is None:
            raise DataIntegrityError(
                "Forecast input requires an explicit positive forecast_horizon."
            )
        horizon = len(sample.output_window)
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
        raise DataIntegrityError("forecast_horizon must be a positive integer.")
    return int(horizon)


def _indicator_series(sample: "MTBenchSample", key: str) -> list[float]:
    raw = sample.labels.get(key)
    if raw is None:
        return []
    return np.asarray(raw, dtype=np.float64).ravel().tolist()


def _output_ts_string(sample: "MTBenchSample") -> str:
    output_timestamps = sample.metadata.get("output_timestamps") or []
    if output_timestamps:
        return _finance_ts_string(float(output_timestamps[-1]))
    raise DataIntegrityError(
        "Finance input requires metadata.output_timestamps describing the known "
        "prediction time range; future prices cannot supply a timestamp."
    )


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


def _append_weather_indicator_consistency_rule(raw_prompt: str) -> str:
    return (
        f"{raw_prompt}\n\n"
        "Consistency requirement: the reported temperature difference must equal "
        "the reported highest temperature minus the reported lowest temperature."
    )


def _append_weather_indicator_target_window_rule(raw_prompt: str, *, next_days: int) -> str:
    """Clarify which forecast window defines the weather-indicator target."""
    if int(next_days) <= 1:
        return (
            f"{raw_prompt}\n\n"
            "Target window clarification: report the highest temperature, lowest "
            "temperature, and their difference for the next forecast day, i.e. "
            "the next 24 hourly readings."
        )
    return (
        f"{raw_prompt}\n\n"
        "Target window clarification: even though the forecast horizon spans "
        f"{next_days} days, this indicator task is defined by the first forecast "
        "day only. Report the highest temperature, lowest temperature, and "
        "their difference from the first 24 predicted hourly readings, not "
        "from the full multi-day horizon."
    )


def _append_weather_mcqa_answer_contract(raw_prompt: str) -> str:
    """Clarify answer extraction when options are embedded inline."""
    return (
        f"{raw_prompt}\n\n"
        "Answer contract: choose exactly one option from the question and "
        "return only the single uppercase letter `A`, `B`, `C`, or `D`. "
        "The options may appear as `A) ... D)`, `A. ... D.`, or inline "
        "lowercase `a) ... d)` text; map your chosen option back to the "
        "uppercase answer letter only."
    )


def weather_aligned_short_forecast_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _WEATHER_META.temperature_forecast_metaprompt_generation(
            text=sample.text,
            temperatures=_input_prices(sample),
            start_datetime=_weather_ts_string(sample.timestamps[0]),
            end_datetime=_weather_ts_string(sample.timestamps[-1]),
            granularity="hourly",
            prediction_length=resolve_forecast_horizon(sample),
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
            prediction_length=resolve_forecast_horizon(sample),
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
        _append_weather_indicator_target_window_rule(
            _append_weather_indicator_consistency_rule(
                _WEATHER_META.temperature_indicator_metaprompt_generation(
                    text=sample.text,
                    past_temperatures=_input_prices(sample),
                    start_datetime=_weather_ts_string(sample.timestamps[0]),
                    end_datetime=_weather_ts_string(sample.timestamps[-1]),
                    granularity="hourly",
                    past_days=7,
                    mode="combined",
                )
            ),
            next_days=1,
        )
    )


def weather_aligned_long_indicator_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _append_weather_indicator_target_window_rule(
            _append_weather_indicator_consistency_rule(
                _WEATHER_META.temperature_indicator_metaprompt_generation(
                    text=sample.text,
                    past_temperatures=_input_prices(sample),
                    start_datetime=_weather_ts_string(sample.timestamps[0]),
                    end_datetime=_weather_ts_string(sample.timestamps[-1]),
                    granularity="hourly",
                    past_days=14,
                    mode="combined",
                )
            ),
            next_days=3,
        )
    )


def weather_qa_short_mcqa_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _append_weather_mcqa_answer_contract(
            _WEATHER_META.weather_mcqa_metaprompt_generation(
                past_days=7,
                start_datetime=_weather_ts_string(sample.timestamps[0]),
                end_datetime=_weather_ts_string(sample.timestamps[-1]),
                past_temperatures=_input_prices(sample),
                news=sample.text,
                question=sample.question or "",
            )
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
            prediction_length=resolve_forecast_horizon(sample),
            mode="combined",
            next_days=3,
        )
    )


def weather_qa_long_mcqa_prompt(sample: "MTBenchSample") -> tuple[str, str]:
    return _prompt(
        _append_weather_mcqa_answer_contract(
            _WEATHER_META.weather_mcqa_metaprompt_generation(
                past_days=14,
                start_datetime=_weather_ts_string(sample.timestamps[0]),
                end_datetime=_weather_ts_string(sample.timestamps[-1]),
                past_temperatures=_input_prices(sample),
                news=sample.text,
                question=sample.question or "",
            )
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
            prediction_length=resolve_forecast_horizon(sample),
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
            prediction_length=resolve_forecast_horizon(sample),
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
    task_view = task_text_view(sample, task)
    cleaned_text = str(task_view.get("cleaned_text") or "").strip()
    if cleaned_text:
        sample = replace(sample, text=cleaned_text)

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
    """Read one distinct standalone option letter; word prefixes are not labels.

    For example, ``Answer: B`` means B, not the A at the start of Answer.
    Multiple different option letters are ambiguous and are rejected.
    """
    matches = {value.upper() for value in re.findall(r"\b([A-D])\b", str(text).strip(), re.IGNORECASE)}
    return matches.pop() if len(matches) == 1 else ""


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
