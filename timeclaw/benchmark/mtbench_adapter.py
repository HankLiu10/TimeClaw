"""Unified adapter for all MTBench datasets.

Normalises weather (JSON) and finance (JSON/Parquet) samples into a
common ``MTBenchSample`` dataclass so that downstream code (sample-list
loading, train-loop, evaluation) never has to worry about format differences.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from timeclaw.data_integrity import DataIntegrityError
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.core.config import TimeClawConfig
from timeclaw.data_integrity import (
    QUARANTINE_FILE,
    assert_not_quarantined,
    assert_weather_record_admissible,
)

# ── Dataset registry ──────────────────────────────────────────────────
#
# Dataset layouts live here; the only task roster is task_registry.py.
from timeclaw.benchmark.task_registry import BENCHMARK_TASKS, TASK_SET, benchmark_tasks, tasks_for_dataset

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # Weather
    "weather_aligned_short": {
        "domain": "weather",
        "path": "data/processed/weather/aligned_in7days_out1days",
        "format": "json",
        "tasks": tasks_for_dataset('weather_aligned_short'),
    },
    "weather_aligned_long": {
        "domain": "weather",
        "path": "data/processed/weather/aligned_in14days_out3days",
        "format": "json",
        "tasks": tasks_for_dataset('weather_aligned_long'),
    },
    "weather_qa_short": {
        "domain": "weather",
        "path": "data/processed/weather/QAshort",
        "format": "json",
        "tasks": tasks_for_dataset('weather_qa_short'),
    },
    "weather_qa_long": {
        "domain": "weather",
        "path": "data/processed/weather/QAlong",
        "format": "json",
        "tasks": tasks_for_dataset('weather_qa_long'),
    },
    # Finance
    "finance_aligned_short": {
        "domain": "finance",
        "path": "data/processed/finance/aligned_in7days_out1days",
        "format": "parquet",
        "tasks": tasks_for_dataset('finance_aligned_short'),
    },
    "finance_aligned_long": {
        "domain": "finance",
        "path": "data/processed/finance/aligned_in30days_out7days",
        "format": "parquet",
        "tasks": tasks_for_dataset('finance_aligned_long'),
    },
    "finance_qa_short": {
        "domain": "finance",
        "path": "data/processed/finance/QAshort",
        "format": "json",
        "tasks": tasks_for_dataset('finance_qa_short'),
    },
    "finance_qa_long": {
        "domain": "finance",
        "path": "data/processed/finance/QAlong",
        "format": "json",
        "tasks": tasks_for_dataset('finance_qa_long'),
    },
}

BENCHMARK_EXPANDED_TASK_TARGET = len(BENCHMARK_TASKS)


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


def expanded_benchmark_task_specs(task_set: str = TASK_SET) -> list[tuple[str, str]]:
    """(dataset, scope_task) pairs; length should match ``BENCHMARK_EXPANDED_TASK_TARGET``."""
    return [(spec.dataset_name, spec.task_name) for spec in benchmark_tasks(task_set)]


@dataclass
class MTBenchSample:
    """Unified sample representation across all MTBench datasets.

    ``forecast_horizon`` is the requested number of future time steps, known
    before inference. It lets visible-only requests specify their output
    contract without providing any future observations or evaluation labels.
    Legacy benchmark samples may omit it and use their target window length.
    """

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
    forecast_horizon: int | None = None


# ── Loaders ───────────────────────────────────────────────────────────

def _parse_timestamp_to_epoch(ts: Any) -> float:
    """Convert a timestamp (ISO string or unix float) to unix epoch."""
    if isinstance(ts, (int, float)) and not isinstance(ts, bool) and np.isfinite(ts):
        return float(ts)
    if isinstance(ts, str):
        try:
            # Keep explicit offsets. A failed parse must not invent an epoch
            # date and collapse an entire hourly/minute grid to zero.
            return datetime.fromisoformat(ts.strip().replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise DataIntegrityError(f"Invalid timestamp: {ts!r}") from exc
    raise DataIntegrityError(f"Invalid timestamp: {ts!r}")


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


def _natural_sort_key(text: str) -> list[Any]:
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


_DATASET_METADATA_JSON_NAMES = {
    QUARANTINE_FILE,
    "all_files.json",
    "split_meta.json",
    "test_files.json",
    "train_files.json",
}


def _iter_sample_json_files(path: Path) -> list[Path]:
    """Return data records only, excluding split manifests stored beside them."""
    return [
        item
        for item in sorted(path.glob("*.json"), key=lambda file: _natural_sort_key(file.name))
        if item.name not in _DATASET_METADATA_JSON_NAMES
    ]


def _read_sample_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"Invalid sample JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise DataIntegrityError(f"Invalid sample JSON (expected an object): {path}")
    return raw


def _load_weather_json(path: Path, dataset_name: str, info: dict) -> list[MTBenchSample]:
    assert_not_quarantined(path)
    samples = []
    for f in _iter_sample_json_files(path):
        raw = _read_sample_json(f)

        assert_weather_record_admissible(raw, source=str(f))
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
                "source_group": raw.get("source_group"),
                "text_source_id": raw.get("text_source_id"),
                "output_timestamps": output_timestamps,
            },
        ))
    return samples


def _load_finance_json(path: Path, dataset_name: str, info: dict) -> list[MTBenchSample]:
    assert_not_quarantined(path)
    samples = []
    for f in _iter_sample_json_files(path):
        raw = _read_sample_json(f)

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
                "source_group": raw.get("source_group"),
                "text_source_id": raw.get("text_source_id"),
                "published_utc": raw.get("published_utc"),
                "timestamp_ms": raw.get("timestamp_ms"),
                "output_timestamps": output_timestamps,
            },
        ))
    return samples


def _load_finance_parquet(path: Path, dataset_name: str, info: dict) -> list[MTBenchSample]:
    import pandas as pd

    parquet_files = sorted(path.rglob("*.parquet"))
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
                    "source_group": row.get("source_group"),
                    "text_source_id": row.get("text_source_id"),
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
    info = {**info, "tasks": tasks_for_dataset(dataset_name, cfg.benchmark.task_set)}
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

    if cfg.benchmark.observation_time_evidence is not None:
        from timeclaw.benchmark.observation_time_evidence import apply_observation_time_evidence
        sources = sorted(path.rglob("*.parquet")) if fmt == "parquet" else list(_iter_sample_json_files(path))
        apply_observation_time_evidence(
            samples, dataset=dataset_name, directory=path, source_files=sources,
            evidence_file=Path(cfg.benchmark.repo_root) / cfg.benchmark.observation_time_evidence,
        )

    logger.info(f"[MTBenchAdapter] Loaded {len(samples)} samples from {dataset_name}")
    return samples


def _assert_independent_paths(split_dir: Path, config: TimeClawConfig) -> None:
    """Reject directories or individual links that resolve into evaluation roots."""
    protected = [(Path(config.benchmark.repo_root) / info["path"]).resolve()
                 for info in DATASET_REGISTRY.values()]
    location = split_dir.resolve()
    for evaluation in protected:
        if location.is_relative_to(evaluation) or evaluation.is_relative_to(location):
            raise DataIntegrityError(f"Independent training points at official evaluation data: {split_dir}")
    for source in split_dir.rglob("*"):
        if source.is_file() and any(source.resolve().is_relative_to(pool) for pool in protected):
            raise DataIntegrityError(f"Independent training file points at official evaluation data: {source}")


def observed_input_identity(sample: MTBenchSample) -> str:
    """Identity of an observed source window, independent of names and targets.

    Numeric source windows are identified by timestamps and values,
    ignoring question/news/answer changes that do not create a new source
    window. Text-only inputs use their text and question. This detects exact
    overlap, not partial time overlap or undisclosed shared source ancestry.
    """
    values = np.asarray(sample.input_window, dtype=np.float64).copy()
    values[values == 0] = 0  # Canonicalize signed zero, without changing samples.
    if values.size:
        payload = [list(sample.timestamps), list(values.shape), values.tolist()]
    else:
        payload = [sample.text, sample.question]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _assert_independent_samples(samples: list[MTBenchSample], config: TimeClawConfig) -> None:
    from timeclaw.benchmark.source_admission import admit_independent_samples
    admit_independent_samples(samples, config)


def load_independent_dataset(
    dataset_name: str,
    *,
    config: TimeClawConfig | None = None,
) -> list[MTBenchSample]:
    """Load the disjoint project training corpus.

    This loader intentionally has a separate entry point from
    :func:`load_dataset`, which is reserved for the official MTBench pool.
    Records live under ``training_root/splits/<dataset>``. Aligned finance
    manifests index the task's materialized ``data`` table: the generator
    resets row indices after selection and constructs long windows separately.
    They must never index the upstream ``finance/processed`` source pool.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    info = DATASET_REGISTRY[dataset_name]
    cfg = _coerce_benchmark_config(config)
    info = {**info, "tasks": tasks_for_dataset(dataset_name, cfg.benchmark.task_set)}
    root = Path(cfg.benchmark.repo_root) / cfg.benchmark.training_root
    split_dir = root / "splits" / dataset_name
    _assert_independent_paths(split_dir, cfg)
    assert_not_quarantined(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Independent dataset directory not found: {split_dir}")

    if info["domain"] == "finance" and info["format"] == "parquet":
        parquet_root = split_dir / "data"
        assert_not_quarantined(parquet_root)
        parquet_files = list(parquet_root.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Independent task parquet not found: {parquet_root}")
        if len(parquet_files) != 1:
            raise ValueError("Independent row-only manifests require exactly one task parquet")
        samples = _load_finance_parquet(parquet_root, dataset_name, info)
        manifest_path = split_dir / "train_files.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Independent training manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        selected = set(manifest)
        if len(selected) != len(manifest):
            raise ValueError(f"Duplicate independent training IDs: {manifest_path}")
        available = {sample.file for sample in samples}
        if selected - available:
            raise ValueError(f"Independent training IDs missing from task parquet: {sorted(selected - available)[:5]}")
        samples = [sample for sample in samples if sample.file in selected]
    elif info["domain"] == "finance":
        samples = _load_finance_json(split_dir, dataset_name, info)
    else:
        samples = _load_weather_json(split_dir, dataset_name, info)

    if not samples:
        raise ValueError(f"Independent training corpus is empty for {dataset_name}")
    _assert_independent_samples(samples, cfg)
    for sample in samples:
        sample.metadata["training_source_admission"]["training_directory"] = str(split_dir.resolve())
    logger.info(f"[IndependentAdapter] Loaded {len(samples)} samples from {split_dir}")
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
