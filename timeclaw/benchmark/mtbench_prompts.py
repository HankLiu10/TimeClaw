"""Official MTBench prompt templates — aligned with the paper (Appendix C/D).

Each function returns (system_prompt, user_prompt) ready for LLM.complete().
All prompts use TS+Text setting (the primary evaluation mode).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _fmt_ts(values: np.ndarray | list, max_points: int = 0) -> str:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if max_points > 0 and len(arr) > max_points:
        arr = arr[:max_points]
    return ", ".join(f"{v:.2f}" for v in arr)


def _fmt_timestamps(timestamps: list, max_points: int = 0) -> str:
    ts = timestamps[:max_points] if max_points > 0 else timestamps
    if not ts:
        return ""
    return f"{ts[0]} to {ts[-1]}"


# ═══════════════════════════════════════════════════════════════════════
# WEATHER TASKS
# ═══════════════════════════════════════════════════════════════════════

def weather_forecast(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
    horizon_days: int = 1,
) -> tuple[str, str]:
    input_days = len(ts_data) // 24
    sys = "You are a weather forecasting AI."
    user = (
        f"The input time-series represents temperature readings. "
        f"This data is from a location in the United States, where summers are hot and winters are cold. "
        f"Weather conditions can also be affected by storms, heavy rain, and cold fronts. "
        f"The daytime is usually warmer than the nighttime. "
        f"Every 24 temperature readings represent a full day from 00:00 to 23:00.\n\n"
        f"Temperature time series (past {input_days} days, hourly):\n[{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"Weather report:\n{text}\n\n"
    user += (
        f"Your task is to predict the next {horizon_days} day(s) of hourly temperature readings "
        f"({horizon_days * 24} values).\n"
        f"Return ONLY a JSON array of {horizon_days * 24} numbers, nothing else."
    )
    return sys, user


def weather_trend_future(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
) -> tuple[str, str]:
    sys = "You are a weather forecasting AI."
    user = (
        f"The input time-series represents temperature readings from a US location. "
        f"Every 24 readings represent a full day from 00:00 to 23:00.\n\n"
        f"Temperature time series (past 7 days, hourly):\n[{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"Weather report:\n{text}\n\n"
    user += (
        "Based on the given information, predict the temperature trend for the next 1 day. "
        "Calculate the mean temperature of the last 24-hour period (i.e., the most recent day in the input) "
        "and compare it with the mean temperature of the first predicted day. "
        "If the difference is greater or equal than 0.5, classify the trend as 'increasing'. "
        "If the difference is less or equal than -0.5, classify the trend as 'decreasing'. "
        "Otherwise, classify it as 'stable'.\n\n"
        'Return ONLY a JSON object: {"trend": "increasing" or "decreasing" or "stable"}'
    )
    return sys, user


def weather_indicator(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
) -> tuple[str, str]:
    sys = "You are a weather forecasting AI."
    user = (
        f"The input time-series represents temperature readings. "
        f"This data is from a location in the United States, where summers are hot and winters are cold. "
        f"Weather conditions can also be affected by storms, heavy rain, and cold fronts. "
        f"The daytime is usually warmer than the nighttime. "
        f"Every 24 temperature readings represent a full day from 00:00 to 23:00.\n\n"
        f"Temperature time series (past 7 days, hourly):\n[{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"Weather report:\n{text}\n\n"
    user += (
        "Your task is to analyze the past 7 days' temperature trend and predict "
        "the next 1 day's highest temperature and lowest temperature "
        "as well as the temperature difference between the highest and lowest temperature.\n\n"
        'Return ONLY a JSON object: {"max": number, "min": number, "diff": number}'
    )
    return sys, user


def weather_mcqa(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
    question: str,
) -> tuple[str, str]:
    sys = "You are a weather analysis expert."
    user = (
        f"Temperature time series (hourly):\n[{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"Weather report:\n{text}\n\n"
    user += (
        f"Question: {question}\n\n"
        "Return your answer ONLY as a single letter (A, B, C, or D) in JSON: "
        '{"answer": "X"}'
    )
    return sys, user


# ═══════════════════════════════════════════════════════════════════════
# FINANCE TASKS
# ═══════════════════════════════════════════════════════════════════════

def stock_forecast(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
    horizon_points: int,
    input_desc: str = "7 days at 5-minute intervals",
    output_desc: str = "next 1 day",
) -> tuple[str, str]:
    sys = (
        "You are a financial prediction expert with knowledge of advanced "
        "machine learning models and time-series analysis."
    )
    user = (
        f"Time Series Stock Price Data: This data includes stock prices recorded at "
        f"{input_desc}.\n"
        f"Data: [{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"News Data:\n{text[:10000]}\n\n"
    user += (
        f"Your task is to predict the stock prices for the {output_desc} "
        f"({horizon_points} values).\n"
        f"Return ONLY a JSON array of {horizon_points} numbers, nothing else."
    )
    return sys, user


def stock_trend_5way(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
    input_desc: str = "1-hour intervals over the last week",
) -> tuple[str, str]:
    sys = (
        "You are a financial prediction expert with knowledge of advanced "
        "machine learning models and time-series analysis."
    )
    user = (
        f"Time Series Stock Price Data: This data includes stock prices recorded at "
        f"{input_desc}.\n"
        f"Data: [{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"News Data:\n{text[:10000]}\n\n"
    user += (
        "Output: Provide a prediction for the stock trend and categorize it into "
        'one of the following labels: "<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%".\n\n'
        "Task: Please think step-by-step. Analyze the provided time-series data and news "
        "to identify future trends of the stock performance.\n"
        'Return ONLY a JSON object: {"trend": "<label>"}'
    )
    return sys, user


def stock_indicator_macd(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
    input_desc: str = "7 days",
) -> tuple[str, str]:
    sys = "You are a financial technical analysis expert."
    user = (
        f"Stock price data ({input_desc}):\n[{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"News:\n{text[:10000]}\n\n"
    user += (
        "MACD (Moving Average Convergence Divergence) is calculated as the difference "
        "between the 12-period and 26-period exponential moving averages.\n"
        "Predict the average MACD value for the forecast period.\n"
        'Return ONLY a JSON object: {"macd": number}'
    )
    return sys, user


def stock_indicator_bb(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
    input_desc: str = "7 days",
) -> tuple[str, str]:
    sys = "You are a financial technical analysis expert."
    user = (
        f"Stock price data ({input_desc}):\n[{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"News:\n{text[:10000]}\n\n"
    user += (
        "Bollinger Bands: Upper Band = SMA + 2*sigma. "
        "Predict the average Upper Bollinger Band value for the forecast period.\n"
        'Return ONLY a JSON object: {"upper_bb": number}'
    )
    return sys, user


def stock_correlation_5way(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
) -> tuple[str, str]:
    sys = "You are an expert in finance and stock market analysis."
    user = (
        f"Historical stock price time-series:\n[{_fmt_ts(ts_data)}]\n\n"
        f"Financial analysis published at the last timestamp:\n{text[:10000]}\n\n"
        "Your task is to predict the correlation between the stock's price fluctuations "
        "in the next 7 days and the analysis sentiment. "
        "Positive correlation indicates that positive analysis leads to price increase "
        "and negative analysis leads to price decrease.\n"
        "Take into account external factors or market conditions that might affect stock price movement.\n\n"
        "Choose one of: "
        '"Strong Positive Correlation", "Moderate Positive Correlation", "No Correlation", '
        '"Moderate Negative Correlation", "Strong Negative Correlation".\n'
        'Return ONLY a JSON object: {"correlation": "<label>"}'
    )
    return sys, user


def stock_mcqa(
    ts_data: np.ndarray,
    timestamps: list,
    text: str,
    question: str,
) -> tuple[str, str]:
    sys = "You are an expert in finance and stock market analysis."
    user = (
        f"Historical stock price time-series:\n[{_fmt_ts(ts_data)}]\n\n"
    )
    if text:
        user += f"Financial analysis:\n{text[:10000]}\n\n"
    user += (
        f"Question: {question}\n\n"
        "Return your answer ONLY as a single letter (A, B, C, or D) in JSON: "
        '{"answer": "X"}'
    )
    return sys, user
