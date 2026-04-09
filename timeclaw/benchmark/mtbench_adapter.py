"""Unified adapter for all MTBench datasets.

Normalises weather (JSON) and finance (JSON/Parquet) samples into a
common ``MTBenchSample`` dataclass so that downstream code (sample-list
loading, train-loop, evaluation) never has to worry about format differences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.core.config import TimeClawConfig

# ── Dataset registry ──────────────────────────────────────────────────
#
# Train/eval use datasets with non-empty ``tasks``.  ``indicator`` on finance
# aligned sets expands to ``indicator_macd`` + ``indicator_upper_bb``.
# Weather QA sets intentionally have no tasks (TimeART-style 18-scope subset).
# Expected expanded (dataset, task) count = 18.

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # Weather
    "weather_aligned_short": {
        "domain": "weather",
        "path": "data/processed/weather/aligned_in7days_out1days",
        "format": "json",
        "tasks": [
            "forecast",
            "trend",
            "trend_past",
            # Excluded to match the TimeART MTBench evaluation subset.
            # "indicator",
        ],
    },
    "weather_aligned_long": {
        "domain": "weather",
        "path": "data/processed/weather/aligned_in14days_out3days",
        "format": "json",
        "tasks": [
            "forecast",
            "trend",
            "trend_past",
            # Excluded to match the TimeART MTBench evaluation subset.
            # "indicator",
        ],
    },
    "weather_qa_short": {
        "domain": "weather",
        "path": "data/processed/weather/QAshort",
        "format": "json",
        "tasks": [
            # Excluded to match the TimeART MTBench evaluation subset.
            # "mcqa",
        ],
    },
    "weather_qa_long": {
        "domain": "weather",
        "path": "data/processed/weather/QAlong",
        "format": "json",
        "tasks": [
            # Excluded to match the TimeART MTBench evaluation subset.
            # "forecast",
            # "mcqa",
        ],
    },
    # Finance
    "finance_aligned_short": {
        "domain": "finance",
        "path": "data/processed/finance/aligned_in7days_out1days",
        "format": "parquet",
        "tasks": ["forecast", "trend", "indicator"],
    },
    "finance_aligned_long": {
        "domain": "finance",
        "path": "data/processed/finance/aligned_in30days_out7days",
        "format": "parquet",
        "tasks": ["forecast", "trend", "indicator"],
    },
    "finance_qa_short": {
        "domain": "finance",
        "path": "data/processed/finance/QAshort",
        "format": "json",
        "tasks": ["mcqa", "correlation"],
    },
    "finance_qa_long": {
        "domain": "finance",
        "path": "data/processed/finance/QAlong",
        "format": "json",
        "tasks": ["mcqa", "correlation"],
    },
}

BENCHMARK_EXPANDED_TASK_TARGET = 18


def _coerce_benchmark_config(config: TimeClawConfig | None) -> TimeClawConfig:
    return config if config is not None else TimeClawConfig()


def benchmark_repo_root(config: TimeClawConfig | None = None) -> Path:
    return Path(_coerce_benchmark_config(config).benchmark.repo_root)


def benchmark_splits_dir(config: TimeClawConfig | None = None) -> Path:
    cfg = _coerce_benchmark_config(config)
    root = Path(cfg.benchmark.repo_root)
    sdir = Path(cfg.benchmark.splits_dir)
    return sdir if sdir.is_absolute() else (root / sdir)


def datasets_with_benchmark_tasks() -> list[str]:
    """Registry names that run in train/eval (non-empty ``tasks``)."""
    return [name for name, meta in DATASET_REGISTRY.items() if meta.get("tasks")]


def expanded_benchmark_task_specs() -> list[tuple[str, str]]:
    """(dataset, scope_task) pairs; length should match ``BENCHMARK_EXPANDED_TASK_TARGET``."""
    from timeclaw.benchmark.mtbench_support.scope_keys import expand_dataset_tasks

    specs: list[tuple[str, str]] = []
    for name in datasets_with_benchmark_tasks():
        for task in expand_dataset_tasks(name, list(DATASET_REGISTRY[name]["tasks"])):
            specs.append((name, task))
    return specs


@dataclass
class MTBenchSample:
    """Unified sample representation across all MTBench datasets."""

    file: str
    dataset: str
    domain: str
    task_types: list[str]
    input_window: np.ndarray
    output_window: np.ndarray | None
    text: str
    timestamps: list[float]
    start_year: int
    question: str | None = None
    answer: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Loaders ───────────────────────────────────────────────────────────

def _parse_timestamp_to_epoch(ts: Any) -> float:
    """Convert a timestamp (ISO string or unix float) to unix epoch."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(ts, fmt).timestamp()
            except ValueError:
                continue
    return 0.0


def _epoch_to_year(epoch: float) -> int:
    try:
        return datetime.utcfromtimestamp(epoch).year
    except (OSError, ValueError):
        return 0


def _extract_text(raw_text: Any) -> str:
    """Extract plain text from weather string or finance dict/string."""
    if isinstance(raw_text, str):
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed.get("content", str(parsed))
        except (json.JSONDecodeError, TypeError):
            pass
        return raw_text
    if isinstance(raw_text, dict):
        return raw_text.get("content", str(raw_text))
    return str(raw_text) if raw_text else ""


def _load_weather_json(path: Path, dataset_name: str, info: dict) -> list[MTBenchSample]:
    samples = []
    for f in sorted(path.glob("*.json")):
        try:
            raw = json.loads(f.read_text())
        except Exception:
            continue

        ts_raw = raw.get("input_timestamps", [])
        timestamps = [_parse_timestamp_to_epoch(t) for t in ts_raw]
        out_ts_raw = raw.get("output_timestamps", [])
        output_timestamps = [_parse_timestamp_to_epoch(t) for t in out_ts_raw]
        start_year = _epoch_to_year(timestamps[0]) if timestamps else 0

        input_win = np.asarray(raw.get("input_window", []), dtype=np.float64)
        out_raw = raw.get("output_window")
        output_win = np.asarray(out_raw, dtype=np.float64) if out_raw else None

        text = _extract_text(raw.get("text", ""))
        question = raw.get("question")
        answer = raw.get("answer")

        labels: dict[str, Any] = {}
        if raw.get("hasrain") is not None:
            labels["hasrain"] = raw["hasrain"]

        samples.append(MTBenchSample(
            file=f.name,
            dataset=dataset_name,
            domain="weather",
            task_types=list(info["tasks"]),
            input_window=input_win,
            output_window=output_win,
            text=text,
            timestamps=timestamps,
            start_year=start_year,
            question=question,
            answer=answer,
            labels=labels,
            metadata={
                "station": f.stem.rsplit("_", 1)[0],
                "output_timestamps": output_timestamps,
            },
        ))
    return samples


def _load_finance_json(path: Path, dataset_name: str, info: dict) -> list[MTBenchSample]:
    samples = []
    for f in sorted(path.glob("*.json")):
        try:
            raw = json.loads(f.read_text())
        except Exception:
            continue

        ts_raw = raw.get("input_timestamps", [])
        timestamps = [_parse_timestamp_to_epoch(t) for t in ts_raw]
        out_ts_raw = raw.get("output_timestamps", [])
        output_timestamps = [_parse_timestamp_to_epoch(t) for t in out_ts_raw]
        start_year = _epoch_to_year(timestamps[0]) if timestamps else 0

        input_win = np.asarray(raw.get("input_window", []), dtype=np.float64)
        out_raw = raw.get("output_window")
        output_win = np.asarray(out_raw, dtype=np.float64) if out_raw else None

        text = _extract_text(raw.get("text", ""))

        mcqa = raw.get("MCQA", {})
        question = mcqa.get("question") if isinstance(mcqa, dict) else None
        answer = mcqa.get("answer") if isinstance(mcqa, dict) else None

        labels: dict[str, Any] = {}
        if raw.get("news_price_correlation"):
            labels["correlation"] = raw["news_price_correlation"]

        sticker = f.stem.split("_")[1] if "_" in f.stem else f.stem

        samples.append(MTBenchSample(
            file=f.name,
            dataset=dataset_name,
            domain="finance",
            task_types=list(info["tasks"]),
            input_window=input_win,
            output_window=output_win,
            text=text,
            timestamps=timestamps,
            start_year=start_year,
            question=question,
            answer=answer,
            labels=labels,
            metadata={
                "sticker": sticker,
                "published_utc": raw.get("published_utc"),
                "output_timestamps": output_timestamps,
            },
        ))
    return samples


def _load_finance_parquet(path: Path, dataset_name: str, info: dict) -> list[MTBenchSample]:
    import pandas as pd

    parquet_files = list(path.rglob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files in {path}")
        return []

    samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for idx, row in df.iterrows():
            ts_raw = row.get("input_timestamps")
            if ts_raw is not None:
                timestamps = [float(t) for t in np.asarray(ts_raw).ravel()]
            else:
                timestamps = []
            out_ts_raw = row.get("output_timestamps")
            if out_ts_raw is not None:
                output_timestamps = [float(t) for t in np.asarray(out_ts_raw).ravel()]
            else:
                output_timestamps = []
            start_year = _epoch_to_year(timestamps[0]) if timestamps else 0

            input_win = np.asarray(row.get("input_window", []), dtype=np.float64).ravel()
            out_raw = row.get("output_window")
            output_win = np.asarray(out_raw, dtype=np.float64).ravel() if out_raw is not None else None

            text = _extract_text(row.get("text", ""))

            labels: dict[str, Any] = {}
            trend_raw = row.get("trend")
            if trend_raw is not None:
                if isinstance(trend_raw, str):
                    try:
                        trend_raw = json.loads(trend_raw)
                    except Exception:
                        trend_raw = {}
                if isinstance(trend_raw, dict):
                    labels["trend_label"] = trend_raw.get("output_bin_label")
                    labels["trend_pct"] = trend_raw.get("output_percentage_change")

            tech_raw = row.get("technical")
            if tech_raw is not None:
                if isinstance(tech_raw, str):
                    try:
                        tech_raw = json.loads(tech_raw)
                    except Exception:
                        tech_raw = {}
                if isinstance(tech_raw, dict):
                    out_macd = tech_raw.get("out_macd")
                    if isinstance(out_macd, list):
                        labels["out_macd"] = np.asarray(out_macd, dtype=np.float64).ravel().tolist()
                    out_bb = tech_raw.get("out_upper_bb")
                    if isinstance(out_bb, list):
                        labels["out_upper_bb"] = np.asarray(out_bb, dtype=np.float64).ravel().tolist()

            fname = f"{idx:05d}.parquet_row"

            samples.append(MTBenchSample(
                file=fname,
                dataset=dataset_name,
                domain="finance",
                task_types=list(info["tasks"]),
                input_window=input_win,
                output_window=output_win,
                text=text,
                timestamps=timestamps,
                start_year=start_year,
                labels=labels,
                metadata={
                    "parquet_file": pf.name,
                    "row_idx": int(idx),
                    "output_timestamps": output_timestamps,
                },
            ))
    return samples


# ── Public API ────────────────────────────────────────────────────────

def load_dataset(
    dataset_name: str,
    *,
    config: TimeClawConfig | None = None,
) -> list[MTBenchSample]:
    """Load all samples from a named MTBench dataset.

    Paths in the registry are resolved under ``config.benchmark.repo_root``
    (default ``.``, i.e. repository root when cwd is the repo).
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    info = DATASET_REGISTRY[dataset_name]
    cfg = _coerce_benchmark_config(config)
    path = Path(cfg.benchmark.repo_root) / info["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. "
            f"Run `python scripts/download_mtbench.py` first."
        )

    fmt = info["format"]
    domain = info["domain"]

    if fmt == "parquet":
        samples = _load_finance_parquet(path, dataset_name, info)
    elif domain == "finance":
        samples = _load_finance_json(path, dataset_name, info)
    else:
        samples = _load_weather_json(path, dataset_name, info)

    logger.info(f"[MTBenchAdapter] Loaded {len(samples)} samples from {dataset_name}")
    return samples


def load_all_datasets(
    *,
    config: TimeClawConfig | None = None,
) -> dict[str, list[MTBenchSample]]:
    """Load all available MTBench datasets."""
    result = {}
    for name in DATASET_REGISTRY:
        try:
            result[name] = load_dataset(name, config=config)
        except FileNotFoundError:
            logger.warning(f"[MTBenchAdapter] Dataset '{name}' not found, skipping")
    return result


def list_datasets() -> list[str]:
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    return DATASET_REGISTRY[dataset_name]
