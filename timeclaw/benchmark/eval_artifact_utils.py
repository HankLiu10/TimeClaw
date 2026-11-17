from __future__ import annotations

import json
from typing import Any

import numpy as np

from timeclaw.pipeline.runner import PipelineRunner


def _train_kwargs(sample: Any, task: str) -> dict[str, Any]:
    """Return the runner kwargs used by external scored-eval helpers.

    New definition:
        This helper captures the old external-eval contract that kept runner
        execution in test mode even when metrics were computed outside the
        runner. It is preserved here only for active helper tests and artifact
        utilities, not as a benchmark entrypoint.
    """
    del sample, task
    return {}


def _extract_tool_chain(result: dict[str, Any]) -> list[str]:
    chain = result.get("tool_chain")
    if chain:
        return [tool for tool in chain if tool]
    tools: list[str] = []
    for entry in result.get("execution_log", []):
        tool = entry.get("tool", "")
        if tool and entry.get("success"):
            tools.append(tool)
    return tools


def _record_candidate_traces(
    result: dict[str, Any],
    *,
    task_type: str = "",
    domain: str = "",
    scope_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    indicator_output_keys = PipelineRunner._indicator_output_keys(
        scope_keys,
        domain,
        task_type,
    )
    return PipelineRunner._extract_candidate_traces(
        result.get("execution_log", []),
        task_type=task_type,
        indicator_output_keys=indicator_output_keys,
    )


def _record_correctness(task: str, metrics: dict[str, Any]) -> bool | None:
    if task in {"forecast", "indicator", "stock_indicator"}:
        return None
    if isinstance(metrics.get("correct"), bool):
        return metrics["correct"]
    if isinstance(metrics.get("accuracy"), (int, float)):
        return bool(metrics["accuracy"] >= 1.0)
    return None


def _record_tool_chain(
    result: dict[str, Any],
    metrics: dict[str, Any],
    *,
    task_type: str = "",
    domain: str = "",
    scope_keys: list[str] | None = None,
) -> list[str]:
    candidate_traces = _record_candidate_traces(
        result,
        task_type=task_type,
        domain=domain,
        scope_keys=scope_keys,
    )
    winner_chain = PipelineRunner._select_winner_tool_chain(candidate_traces, metrics)
    if winner_chain:
        return winner_chain
    return _extract_tool_chain(result)


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
    if isinstance(result_data, (int, float)):
        return np.array([float(result_data)])
    if isinstance(result_data, str):
        try:
            return _extract_forecast_array(json.loads(result_data))
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(result_data, dict):
        for key in (
            "point_forecast",
            "forecast",
            "prediction",
            "values",
            "temperatures",
            "predictions",
            "data",
        ):
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
    prediction: Any,
    keys: list[str],
) -> dict[str, float] | None:
    if prediction is None:
        return None
    if isinstance(prediction, dict):
        result: dict[str, float] = {}
        for key in keys:
            if key in prediction:
                try:
                    result[key] = float(prediction[key])
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
