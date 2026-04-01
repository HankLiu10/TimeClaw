from __future__ import annotations

from typing import Any

from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.benchmark.mtbench_adapter import MTBenchSample


def _join_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"System prompt:\n{system_prompt}\n\nUser prompt:\n{user_prompt}"


def build_task_description(sample: MTBenchSample, task: str) -> str:
    text = sample.text or ""
    timestamps = sample.timestamps
    input_window = sample.input_window
    dataset = sample.dataset or ""

    if task == "forecast":
        if sample.domain == "weather":
            horizon_days = max(1, len(sample.output_window) // 24) if sample.output_window is not None else 1
            sys_p, user_p = prompts.weather_forecast(
                input_window, timestamps, text, horizon_days,
            )
            return _join_prompt(sys_p, user_p)
        horizon = len(sample.output_window) if sample.output_window is not None else 24
        if "short" in dataset:
            input_desc, output_desc = "7 days at 5-minute intervals", "next 1 day"
        else:
            input_desc, output_desc = "30 days at 1-hour intervals", "next 7 days"
        sys_p, user_p = prompts.stock_forecast(
            input_window, timestamps, text, horizon, input_desc, output_desc,
        )
        return _join_prompt(sys_p, user_p)

    if task == "trend":
        if sample.domain == "weather":
            horizon_days = max(1, len(sample.output_window) // 24) if sample.output_window is not None else 1
            sys_p, user_p = prompts.weather_trend_future(
                input_window, timestamps, text, horizon_days,
            )
            return _join_prompt(sys_p, user_p)
        input_desc = (
            "5-minute intervals over the last week"
            if "short" in dataset
            else "1-hour intervals over the last month"
        )
        sys_p, user_p = prompts.stock_trend_5way(
            input_window, timestamps, text, input_desc,
        )
        return _join_prompt(sys_p, user_p)

    if task == "trend_past":
        sys_p, user_p = prompts.weather_trend_past(
            input_window, timestamps, text,
        )
        return _join_prompt(sys_p, user_p)

    if task == "indicator":
        if sample.domain == "weather":
            horizon_days = max(1, len(sample.output_window) // 24) if sample.output_window is not None else 1
            sys_p, user_p = prompts.weather_indicator(
                input_window, timestamps, text, horizon_days,
            )
            return _join_prompt(sys_p, user_p)
        input_desc = "7 days" if "short" in dataset else "30 days"
        sections: list[str] = []
        if sample.labels.get("out_macd") is not None:
            sys_p, user_p = prompts.stock_indicator_macd(
                input_window, timestamps, text, input_desc,
            )
            sections.append(_join_prompt(sys_p, user_p))
        if sample.labels.get("out_upper_bb") is not None:
            sys_p, user_p = prompts.stock_indicator_bb(
                input_window, timestamps, text, input_desc,
            )
            sections.append(_join_prompt(sys_p, user_p))
        if sections:
            return "\n\n---\n\n".join(sections)

    if task == "indicator_macd":
        input_desc = "7 days" if "short" in dataset else "30 days"
        sys_p, user_p = prompts.stock_indicator_macd(
            input_window, timestamps, text, input_desc,
        )
        return _join_prompt(sys_p, user_p)

    if task == "indicator_upper_bb":
        input_desc = "7 days" if "short" in dataset else "30 days"
        sys_p, user_p = prompts.stock_indicator_bb(
            input_window, timestamps, text, input_desc,
        )
        return _join_prompt(sys_p, user_p)

    if task == "mcqa":
        question = sample.question or ""
        if sample.domain == "weather":
            sys_p, user_p = prompts.weather_mcqa(
                input_window, timestamps, text, question,
            )
            return _join_prompt(sys_p, user_p)
        sys_p, user_p = prompts.stock_mcqa(
            input_window, timestamps, text, question,
        )
        return _join_prompt(sys_p, user_p)

    if task == "correlation":
        sys_p, user_p = prompts.stock_correlation_5way(
            input_window, timestamps, text,
        )
        return _join_prompt(sys_p, user_p)

    return sample.metadata.get("description") or f"{task} task on {sample.domain} data."
