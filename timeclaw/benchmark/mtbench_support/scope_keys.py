"""Canonical scope-key helpers for benchmark train/test memory routing."""

from __future__ import annotations


def expand_dataset_tasks(dataset_name: str, tasks: list[str]) -> list[str]:
    """Expand dataset tasks into canonical per-scope task names."""
    expanded: list[str] = []
    for task in tasks:
        if dataset_name.startswith("finance_aligned") and task == "indicator":
            expanded.extend(["indicator_macd", "indicator_upper_bb"])
        else:
            expanded.append(task)
    return expanded


def scope_keys_for_dataset_task(dataset_name: str, task: str) -> list[str]:
    """Return canonical scope key(s) for one dataset/task pair."""
    if not dataset_name:
        return [task] if task else []

    if dataset_name.startswith("finance_aligned"):
        if task == "indicator":
            return [
                f"{dataset_name}_indicator_macd",
                f"{dataset_name}_indicator_upper_bb",
            ]
        if task in {"indicator_macd", "indicator_upper_bb"}:
            return [f"{dataset_name}_{task}"]

    return [f"{dataset_name}_{task}"]
