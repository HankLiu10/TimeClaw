#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import re
from contextlib import contextmanager
import hashlib
import asyncio
import json
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the canonical script runnable directly from a clean checkout, without
# requiring an editable install just to resolve the local ``timeclaw`` package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _mtbench_run_utils import build_logic_fingerprint
from timeclaw.benchmark.concurrency import (
    interleave_items_round_robin,
    iter_completed_tasks_with_retry_queue,
    resolve_active_work_item_limit,
)
from timeclaw.benchmark.eval_artifact_utils import _record_tool_chain
from timeclaw.benchmark.mtbench_adapter import MTBenchSample, get_dataset_info, load_dataset
from timeclaw.benchmark.mtbench_support import (
    apply_official_metric_policy_to_row,
    backfill_scored_metrics,
    summarize_official_inference_rows,
)
from timeclaw.benchmark.mtbench_support.runtime_context import (
    pipeline_task_for_benchmark_task,
)
from timeclaw.benchmark.mtbench_support.scope_keys import scope_keys_for_dataset_task
from timeclaw.benchmark.mtbench_support.task_scoring import (
    score_mtbench_prediction,
    validate_mtbench_sample_for_scoring,
)
from timeclaw.benchmark.train_loop import _extract_ground_truth
from timeclaw.benchmark.source_admission import assert_inference_lineage, workspace_access
from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner


# Canonical inference defaults to the currently validated provider config.
# Archived proxy configurations remain available only when selected explicitly.
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".timeclaw" / "inference_runs"
LOGIC_FINGERPRINT_FILES = [
    *sorted((REPO_ROOT / "timeclaw").rglob("*.py")),
    *sorted((REPO_ROOT / "scripts").glob("*.py")),
    REPO_ROOT / "configs" / "datasets.lock.json",
    REPO_ROOT / "requirements-py310.lock",
]


from timeclaw.benchmark.task_registry import (
    BENCHMARK_TASKS, TASK_SET, TASK_SETS, benchmark_tasks, BenchmarkTask as InferenceTaskSpec,
)


@dataclass(frozen=True)
class SelectedInferenceSample:
    sample_idx: int
    sample: MTBenchSample
    selection_entry: dict[str, Any] | None = None


def _task_specs(task_set: str, task_filters: list[str] | None) -> list[InferenceTaskSpec]:
    specs = list(benchmark_tasks(task_set))
    if not task_filters:
        return specs
    wanted = {str(item).strip() for item in task_filters if str(item).strip()}
    selected = [spec for spec in specs if spec.key in wanted]
    missing = sorted(wanted - {spec.key for spec in selected})
    if missing:
        raise SystemExit(f"Unknown task keys: {', '.join(missing)}")
    return selected


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return value.dict()
        except Exception:
            return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _compute_logic_fingerprint(config_path: Path, *, disable_memory: bool) -> dict[str, Any]:
    return build_logic_fingerprint(
        [*LOGIC_FINGERPRINT_FILES, config_path.resolve()],
        extra={
            "runner": "timeclaw_mtbench_inference",
            "mode": "test",
            "disable_memory": bool(disable_memory),
        },
        root=REPO_ROOT,
    )


def _selection_manifest_files(selection_manifests_dir: Path | None) -> list[Path]:
    if selection_manifests_dir is None:
        return []
    if not selection_manifests_dir.exists():
        raise FileNotFoundError(f"Selection manifests dir does not exist: {selection_manifests_dir}")
    return sorted(selection_manifests_dir.glob("*.samples_manifest.json"))


def _compute_selection_aware_logic_fingerprint(
    config_path: Path,
    selection_manifests_dir: Path | None,
    *,
    disable_memory: bool,
    request_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = [*LOGIC_FINGERPRINT_FILES, config_path.resolve(), *_selection_manifest_files(selection_manifests_dir)]
    extra: dict[str, Any] = {
        "runner": "timeclaw_mtbench_inference",
        "mode": "test",
        "disable_memory": bool(disable_memory),
    }
    if request_bindings is not None:
        extra["request_bindings"] = request_bindings
    if selection_manifests_dir is not None:
        extra["selection_manifests_dir"] = str(selection_manifests_dir.resolve())
    return build_logic_fingerprint(files, extra=extra, root=REPO_ROOT)


def _request_bindings(cfg: TimeClawConfig, datasets: dict[str, list[MTBenchSample]],
                      specs: list[InferenceTaskSpec], samples_per_task: int) -> dict[str, Any]:
    """Bind evaluator inputs and frozen memory for safe result reuse.

    Supervision enters only this evaluator-side digest, never the agent prompt.
    Hash all content, not just filenames; an unchanged sample ID is not evidence
    that input, labels, memory or effective model configuration stayed unchanged.
    """
    def digest(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, default=_json_default).encode()
        return hashlib.sha256(encoded).hexdigest()
    receipt = Path(cfg.workspace_dir) / "training_lineage.json"
    return {"effective_config_sha256": digest(cfg.model_dump(mode="json", exclude={"llm": {"api_key", "azure_api_key"}})),
            "datasets_sha256": {name: digest([asdict(sample) for sample in samples]) for name, samples in datasets.items()},
            "memory_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.exists() else None,
            "task_keys": [spec.key for spec in specs], "samples_per_task": samples_per_task}


def _sanitize_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text or ""))


def _canonical_sample_id_from_sample(spec: InferenceTaskSpec, sample_idx: int, sample: MTBenchSample) -> str:
    info = get_dataset_info(spec.dataset_name)
    if str(info.get("format") or "") == "parquet":
        return f"row:{int(sample.metadata.get('row_idx', sample_idx))}"
    return f"file:{sample.file}"


def _sample_manifest_entry(spec: InferenceTaskSpec, selected: SelectedInferenceSample) -> dict[str, Any]:
    sample_idx = int(selected.sample_idx)
    sample = selected.sample
    selection_entry = selected.selection_entry if isinstance(selected.selection_entry, dict) else None
    info = get_dataset_info(spec.dataset_name)
    dataset_root = (REPO_ROOT / str(info["path"])).resolve()
    base = {
        "position": int(selection_entry.get("position")) if selection_entry and selection_entry.get("position") is not None else sample_idx + 1,
        "sample_idx": sample_idx,
        "sample_id": sample.file,
        "canonical_sample_id": _canonical_sample_id_from_sample(spec, sample_idx, sample),
        "dataset_name": spec.dataset_name,
        "task_name": spec.task_name,
        "pipeline_task": pipeline_task_for_benchmark_task(spec.task_name),
    }
    if selection_entry:
        for key in (
            "official_prepared_filename",
            "official_effective_rank",
            "sample_id",
            "canonical_sample_id",
        ):
            value = selection_entry.get(key)
            if value is not None and str(value).strip():
                base[key] = value
    if str(info.get("format") or "") == "parquet":
        parquet_name = str(sample.metadata.get("parquet_file") or "").strip()
        base.update(
            {
                "source_type": "parquet",
                "source_dataset_path": str(dataset_root),
                "source_parquet_path": str((dataset_root / "data" / parquet_name).resolve()),
                "source_row_index": int(sample.metadata.get("row_idx", sample_idx)),
            }
        )
        return base
    base.update(
        {
            "source_type": "json",
            "source_dataset_path": str(dataset_root),
            "source_path": str((dataset_root / sample.file).resolve()),
        }
    )
    return base


def _sample_ref(spec: InferenceTaskSpec, sample_idx: int, sample: MTBenchSample) -> str:
    return f"{spec.dataset_name}:{sample.file}:{sample_idx}:{spec.task_name}"


def _resume_identity_from_sample(spec: InferenceTaskSpec, sample_idx: int, sample: MTBenchSample) -> str:
    task_key = _row_task_key(spec)
    canonical_sample_id = _canonical_sample_id_from_sample(spec, sample_idx, sample)
    source_row_index = sample.metadata.get("row_idx")
    source_row_text = "" if source_row_index is None else str(int(source_row_index))
    return f"{task_key}|{canonical_sample_id}|{source_row_text}"


def _resume_identity_from_row(row: dict[str, Any]) -> str:
    task_key = str(row.get("task_key") or "").strip()
    canonical_sample_id = str(row.get("canonical_sample_id") or "").strip()
    source_row_index = row.get("source_row_index")
    source_row_text = "" if source_row_index is None or str(source_row_index).strip() == "" else str(int(source_row_index))
    return f"{task_key}|{canonical_sample_id}|{source_row_text}"


def _sample_key(spec: InferenceTaskSpec, sample_idx: int, sample: MTBenchSample) -> str:
    del spec
    return f"{sample.dataset}:{sample.file}:{sample_idx}"


def _row_task_key(spec: InferenceTaskSpec) -> str:
    return f"{spec.dataset_name}::{spec.task_name}"


def _parse_task_concurrency_overrides(
    specs: list[InferenceTaskSpec],
    raw_values: list[str] | None,
) -> dict[str, int]:
    if not raw_values:
        return {}
    key_aliases: dict[str, str] = {}
    for spec in specs:
        key_aliases[spec.key] = spec.key
        key_aliases[_row_task_key(spec)] = spec.key
    overrides: dict[str, int] = {}
    for raw in raw_values:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(
                f"Invalid --task-concurrency-override {raw!r}. Expected KEY=INT."
            )
        name, value_text = text.split("=", 1)
        lookup = key_aliases.get(str(name).strip())
        if lookup is None:
            raise SystemExit(
                f"Unknown task in --task-concurrency-override: {name!r}. "
                "Use official task key like finance_pred_macd_long or dataset::task like finance_aligned_long::indicator_macd."
            )
        try:
            value = int(value_text)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid concurrency value in --task-concurrency-override {raw!r}: {value_text!r}"
            ) from exc
        if value <= 0:
            raise SystemExit(
                f"Task concurrency override must be > 0, got {value} for {lookup}."
            )
        overrides[lookup] = value
    return overrides


def _task_manifests_dir(run_dir: Path, run_id: str) -> Path:
    if run_dir.name == run_id:
        return run_dir / "task_manifests"
    return run_dir / f"{run_id}_task_manifests"


def _extract_tool_chain_simple(result: dict[str, Any]) -> list[str]:
    chain = result.get("tool_chain")
    if isinstance(chain, list):
        cleaned = [str(tool).strip() for tool in chain if str(tool).strip()]
        if cleaned:
            return cleaned
    execution_log = result.get("execution_log")
    if isinstance(execution_log, list):
        return [
            str(entry.get("tool") or "").strip()
            for entry in execution_log
            if isinstance(entry, dict) and entry.get("success") and str(entry.get("tool") or "").strip()
        ]
    return []


def _reasoning_preview(result: dict[str, Any]) -> str:
    preview = str(result.get("reasoning_preview") or "").strip()
    if preview:
        return preview
    execution_context = result.get("execution_context")
    if isinstance(execution_context, dict):
        public_reasoning = execution_context.get("public_reasoning")
        if isinstance(public_reasoning, list):
            parts = [str(item).strip() for item in public_reasoning if str(item).strip()]
            if parts:
                return " ".join(parts[:3])
    evidence = str(result.get("evidence_summary") or "").strip()
    return evidence


def _row_correct(metrics: dict[str, Any]) -> bool | None:
    value = metrics.get("correct")
    if isinstance(value, bool):
        return value
    accuracy = metrics.get("accuracy")
    if isinstance(accuracy, (int, float)):
        return bool(float(accuracy) >= 1.0)
    return None


def _row_status_from_result(result: dict[str, Any]) -> str:
    raw_status = str(result.get("status") or "").strip().lower()
    if raw_status in {"ok", "completed"} and not result.get("error"):
        return "completed"
    if (
        not raw_status
        and not result.get("error")
        and result.get("prediction") is not None
    ):
        return "completed"
    return raw_status or "failed"


def _row_is_completed(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip().lower() == "completed"


def _artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = apply_official_metric_policy_to_row(row)
    if "row_recorded" in normalized:
        normalized["official_recorded"] = normalized.pop("row_recorded")
    if "summary_included" in normalized:
        normalized["official_summary_included"] = normalized.pop("summary_included")
    if "summary_exclude_reason" in normalized:
        normalized["official_summary_skip_reason"] = normalized.pop("summary_exclude_reason")
    return normalized


def _scope_summary_from_rows(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    official_rows = summarize_official_inference_rows(rows)
    normalized_rows = [apply_official_metric_policy_to_row(row) for row in rows]
    rows_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        dataset = str(row.get("dataset") or "").strip()
        task = str(row.get("task") or "").strip()
        if dataset and task:
            rows_by_scope[f"{dataset}::{task}"].append(row)
    scopes: dict[str, Any] = {}
    for item in official_rows:
        dataset = str(item.get("dataset") or "").strip()
        task = str(item.get("task") or "").strip()
        if not dataset or not task:
            continue
        scope_key = f"{dataset}::{task}"
        scope_rows = rows_by_scope.get(scope_key, [])
        completed_n = sum(_row_is_completed(row) for row in scope_rows)
        failed_n = max(len(scope_rows) - completed_n, 0)
        correct_n = sum(
            row.get("correct") is True
            for row in scope_rows
            if bool(row.get("summary_included"))
        )
        incorrect_n = sum(
            row.get("correct") is False
            for row in scope_rows
            if bool(row.get("summary_included"))
        )
        payload: dict[str, Any] = {
            "n": int(item.get("total") or 0),
            "completed_n": int(completed_n),
            "failed_n": int(failed_n),
            "official_recorded_n": int(item.get("row_recorded_n") or 0),
            "official_included_n": int(item.get("summary_included_n") or 0),
            "official_filtered_n": int(item.get("threshold_filtered_n") or 0),
            "correct_n": int(correct_n),
            "incorrect_n": int(incorrect_n),
        }
        for key, value in item.items():
            if key in {
                "dataset",
                "task",
                "total",
                "completed",
                "errors",
                "row_recorded_n",
                "summary_included_n",
                "threshold_filtered_n",
                "invalid_skipped_n",
                "avg_total_ms",
            }:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                payload[key] = value
        scopes[f"{dataset}::{task}"] = payload

    return {"run_id": run_id, "total_rows": len(rows), "scopes": scopes,
            "all_valid": summarize_official_inference_rows(rows, apply_thresholds=False)}


class ArtifactWriter:
    def __init__(self, output_root: Path, run_id: str):
        self.output_root = output_root
        self.run_id = run_id
        self.rows_path = output_root / f"{run_id}_rows.jsonl"
        self.summary_path = output_root / f"{run_id}_summary.json"
        self.manifest_path = output_root / f"{run_id}_manifest.json"
        self.failed_rows_path = output_root / f"{run_id}_failed_rows.json"
        self._lock = asyncio.Lock()

    def dataset_path(self, dataset_name: str) -> Path:
        return self.output_root / f"{self.run_id}_{dataset_name}.json"

    def owned_paths(self) -> tuple[Path, ...]:
        """Exact output ownership; shared run-name prefixes confer no ownership."""
        datasets = sorted({spec.dataset_name for spec in BENCHMARK_TASKS})
        return (self.rows_path, self.summary_path, self.manifest_path,
                self.failed_rows_path, _task_manifests_dir(self.output_root, self.run_id),
                *(self.dataset_path(name) for name in datasets))

    async def append_row(self, row: dict[str, Any]) -> None:
        async with self._lock:
            with open(self.rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    async def write_snapshot(
        self,
        *,
        rows: list[dict[str, Any]],
        rows_by_dataset: dict[str, list[dict[str, Any]]],
        manifest: dict[str, Any],
        failed_rows: list[dict[str, Any]],
    ) -> None:
        async with self._lock:
            _write_json(self.summary_path, _scope_summary_from_rows(self.run_id, rows))
            _write_json(self.manifest_path, manifest)
            _write_json(self.failed_rows_path, failed_rows)
            for dataset_name, dataset_rows in rows_by_dataset.items():
                _write_json(
                    self.dataset_path(dataset_name),
                    dataset_rows,
                )


def _load_existing_rows(rows_path: Path) -> list[dict[str, Any]]:
    if not rows_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(rows_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"rows file contains invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Result row must be an object at line {line_no}: {rows_path}")
            task_key = str(payload.get("task_key") or "").strip()
            canonical = str(payload.get("canonical_sample_id") or "").strip()
            parts = task_key.split("::")
            if len(parts) != 2 or not canonical:
                raise ValueError(f"Result row has missing/invalid identity at line {line_no}: {rows_path}")
            if not any(s.dataset_name == parts[0] and s.task_name == parts[1]
                       for task_set in TASK_SETS for s in benchmark_tasks(task_set)):
                raise ValueError(f"Unsupported dataset/task: {parts}")
            if payload.get("dataset") != parts[0] or payload.get("task") != parts[1]:
                raise ValueError(f"Result row has inconsistent identity at line {line_no}: {rows_path}")
            identity = _resume_identity_from_row(payload)
            if identity in seen:
                raise ValueError(f"Duplicate result identity at line {line_no}: {identity}. Preserve this evidence; no score-based deduplication is allowed.")
            seen.add(identity)
            rows.append(payload)
    return rows


@contextmanager
def _run_output_lease(output_root: Path, run_id: str):
    """Hold one OS-backed writer lease for this logical run's whole lifecycle.

    The persistent empty lock file is metadata, not a successful result. Closing
    the descriptor releases ownership even after a process failure; do not unlink
    an active lock file or another process could lock a different inode.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be one filename component using letters, digits, _, - or .")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f".{run_id}.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Run {run_id!r} already has an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _admit_run_output(*, output_root: Path, run_id: str, resume: bool, clear_output: bool) -> None:
    """A fresh launch must not append into an existing logical run.

    Resume and explicit clearing are different user actions; never combine them
    or implicitly convert a fresh request into a continuation.
    """
    if resume and clear_output:
        raise ValueError("--resume and --clear-output cannot be combined")
    if not resume and not clear_output and output_root.exists():
        if any(path.exists() for path in ArtifactWriter(output_root, run_id).owned_paths()):
            raise FileExistsError(f"An existing run is present for {run_id!r}. Use --resume for unchanged continuation, a new run ID/directory, or explicitly --clear-output.")


def _clear_run_output(*, output_root: Path, run_id: str, clear_output: bool) -> None:
    if not clear_output or not output_root.exists():
        return
    for path in ArtifactWriter(output_root, run_id).owned_paths():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _rows_by_dataset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = str(row.get("dataset") or "").strip()
        if dataset:
            grouped[dataset].append(row)
    return dict(grouped)


def _rebuild_manifest(
    *,
    existing: dict[str, Any] | None,
    run_id: str,
    task_set: str,
    config_path: Path,
    cfg: TimeClawConfig,
    llm_max_inflight_requests: int,
    per_task_concurrency: int,
    active_work_item_limit: int | None,
    samples_per_task: int,
    specs: list[InferenceTaskSpec],
    total_work_items: int,
    completed_rows: int,
    completed_by_task: dict[str, int],
    logic_fingerprint: dict[str, Any],
    selection_manifests_dir: Path | None,
    task_manifests_dir: Path,
    task_concurrency_overrides: dict[str, int],
    disable_memory: bool,
) -> dict[str, Any]:
    manifest = dict(existing or {})
    manifest.update(
        {
            "run_id": run_id,
            "mode": "inference",
            "disable_memory": bool(disable_memory),
            "official_mtbench_is_full_test_set": True,
            "global_concurrency": int(llm_max_inflight_requests),
            "task_set": task_set,
            "config_path": str(config_path),
            "api_base": str(cfg.llm.api_base or ""),
            "llm_max_inflight_requests": int(llm_max_inflight_requests),
            "per_task_concurrency": int(per_task_concurrency),
            "task_concurrency_overrides": dict(task_concurrency_overrides),
            "active_work_item_limit": active_work_item_limit,
            "max_samples_per_task": int(samples_per_task),
            "samples_per_task": int(samples_per_task),
            "task_keys": [spec.key for spec in specs],
            "task_pairs": [
                {"dataset": spec.dataset_name, "task": spec.task_name}
                for spec in specs
            ],
            "selection_manifests_dir": (
                str(selection_manifests_dir.resolve()) if selection_manifests_dir is not None else ""
            ),
            "task_manifests_dir": str(task_manifests_dir.resolve()),
            "total_work_items": int(total_work_items),
            "completed_rows": int(completed_rows),
            "completed_by_task": dict(completed_by_task),
            "logic_fingerprint": logic_fingerprint,
        }
    )
    existing_total = int((existing or {}).get("total_work_items") or 0)
    existing_task_keys = list((existing or {}).get("task_keys") or [])
    if existing and (
        existing_total > int(total_work_items)
        or len(existing_task_keys) > len(specs)
    ):
        for key in (
            "official_mtbench_is_full_test_set",
            "task_set",
            "max_samples_per_task",
            "samples_per_task",
            "task_keys",
            "task_pairs",
            "total_work_items",
        ):
            if key in existing:
                manifest[key] = existing[key]
    manifest.setdefault("started_at", time.time())
    if int(manifest.get("completed_rows") or 0) < int(manifest.get("total_work_items") or 0):
        manifest.pop("finished_at", None)
    return manifest


def _select_from_explicit_manifests(
    *,
    specs: list[InferenceTaskSpec],
    datasets: dict[str, list[MTBenchSample]],
    manifests_dir: Path,
) -> dict[str, list[SelectedInferenceSample]]:
    selected_by_task: dict[str, list[SelectedInferenceSample]] = {}
    for spec in specs:
        manifest_path = manifests_dir / f"{spec.key}.samples_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing selection manifest for {spec.key}: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries") or []
        if not isinstance(entries, list):
            raise RuntimeError(f"Selection manifest entries must be a list: {manifest_path}")
        if str(manifest.get("dataset_name") or "").strip() != spec.dataset_name:
            raise RuntimeError(
                f"Selection manifest dataset mismatch for {spec.key}: "
                f"{manifest.get('dataset_name')} != {spec.dataset_name}"
            )
        if str(manifest.get("task_name") or "").strip() != spec.task_name:
            raise RuntimeError(
                f"Selection manifest task mismatch for {spec.key}: "
                f"{manifest.get('task_name')} != {spec.task_name}"
            )
        samples = datasets[spec.dataset_name]
        canonical_to_item = {
            _canonical_sample_id_from_sample(spec, sample_idx, sample): (sample_idx, sample)
            for sample_idx, sample in enumerate(samples)
        }
        chosen: list[SelectedInferenceSample] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            canonical_sample_id = str(entry.get("canonical_sample_id") or "").strip()
            if not canonical_sample_id:
                raise RuntimeError(
                    f"Selection manifest entry missing canonical_sample_id: {manifest_path}"
                )
            item = canonical_to_item.get(canonical_sample_id)
            if item is None:
                raise RuntimeError(
                    f"Selection manifest sample not found in dataset {spec.dataset_name}: "
                    f"{canonical_sample_id}"
                )
            sample_idx, sample = item
            actual_canonical = _canonical_sample_id_from_sample(spec, sample_idx, sample)
            if actual_canonical != canonical_sample_id:
                raise RuntimeError(
                    f"Selection manifest canonical mismatch for {spec.key}: "
                    f"{canonical_sample_id} != {actual_canonical}"
                )
            if _extract_ground_truth(sample, spec.task_name) is None:
                raise RuntimeError(
                    f"Selection manifest picked sample without ground truth for {spec.key}: "
                    f"{canonical_sample_id}"
                )
            scoreable, invalid_reason = validate_mtbench_sample_for_scoring(sample, spec.task_name)
            if not scoreable:
                raise RuntimeError(
                    f"Selection manifest picked sample not scoreable for {spec.key}: "
                    f"{canonical_sample_id} ({invalid_reason})"
                )
            if canonical_sample_id in seen:
                continue
            seen.add(canonical_sample_id)
            chosen.append(SelectedInferenceSample(sample_idx=sample_idx, sample=sample, selection_entry=dict(entry)))
        selected_by_task[spec.key] = chosen

    return selected_by_task


async def _run_one(
    *,
    runner: PipelineRunner,
    spec: InferenceTaskSpec,
    selected: SelectedInferenceSample,
    task_sem: asyncio.Semaphore,
    run_id: str,
) -> dict[str, Any]:
    async with task_sem:
        sample_idx = int(selected.sample_idx)
        sample = selected.sample
        selection_entry = selected.selection_entry if isinstance(selected.selection_entry, dict) else None
        canonical_sample_id = _canonical_sample_id_from_sample(spec, sample_idx, sample)
        sample_ref = _sample_ref(spec, sample_idx, sample)
        source_metadata = {
            "dataset_name": spec.dataset_name,
            "sample_ref": sample_ref,
            "source_file": sample.file,
            "sample_idx": sample_idx,
            "canonical_sample_id": canonical_sample_id,
            "task_name": spec.task_name,
            "pipeline_task": pipeline_task_for_benchmark_task(spec.task_name),
            "domain": sample.domain,
            "inference_run": True,
            "run_id": run_id,
        }
        if selection_entry is not None:
            official_prepared_filename = str(selection_entry.get("official_prepared_filename") or "").strip()
            if official_prepared_filename:
                source_metadata["official_prepared_filename"] = official_prepared_filename
        result: dict[str, Any] = {}
        ground_truth = _extract_ground_truth(sample, spec.task_name)
        try:
            result = await runner.run_inference(
                sample,
                spec.task_name,
                source_metadata=source_metadata,
                verbose=False,
                full_dump=False,
            )
            scoreable, invalid_reason = validate_mtbench_sample_for_scoring(sample, spec.task_name)
            metrics = (
                score_mtbench_prediction(result.get("prediction"), ground_truth, spec.task_name)
                if scoreable
                else {"skipped": invalid_reason or "invalid_ground_truth"}
            )
            metrics = backfill_scored_metrics(
                result.get("prediction"),
                ground_truth,
                spec.task_name,
                metrics,
            )
            scope_keys = scope_keys_for_dataset_task(spec.dataset_name, spec.task_name)
            tool_chain = _record_tool_chain(
                result,
                metrics,
                task_type=source_metadata["pipeline_task"],
                domain=sample.domain,
                scope_keys=scope_keys,
            )
            if not tool_chain:
                tool_chain = _extract_tool_chain_simple(result)
            row = {
                "run_id": run_id,
                "dataset": spec.dataset_name,
                "task": spec.task_name,
                "task_key": _row_task_key(spec),
                "idx": sample_idx,
                "sample_idx": sample_idx,
                "sample_key": _sample_key(spec, sample_idx, sample),
                "sample_ref": sample_ref,
                "file": sample.file,
                "source_file": sample.file,
                "canonical_sample_id": canonical_sample_id,
                "source_row_index": sample.metadata.get("row_idx"),
                "official_prepared_filename": (
                    str(selection_entry.get("official_prepared_filename") or "").strip()
                    if selection_entry is not None
                    else ""
                ),
                "trace_id": result.get("trace_id"),
                "status": _row_status_from_result(result),
                "answer_type": result.get("answer_type"),
                "prediction": result.get("prediction"),
                "raw_answer": result.get("raw_answer"),
                "metrics": metrics,
                "correct": _row_correct(metrics),
                "gate_accepted": result.get("gate_accepted"),
                "gate_rejections": list(result.get("gate_rejections") or []),
                "confidence": result.get("confidence"),
                "tool_chain": tool_chain,
                "total_ms": result.get("total_ms"),
                "error": result.get("error"),
                "error_type": result.get("error_type"),
                "reasoning_preview": _reasoning_preview(result),
                "evidence_summary": result.get("evidence_summary"),
                "execution_context": result.get("execution_context"),
                "control_context": result.get("control_context"),
                "execution_log": result.get("execution_log", []),
                "ground_truth": ground_truth,
            }
            return _artifact_row(row)
        except Exception as exc:
            print(
                f"[Inference] sample failed without row fallback: "
                f"{spec.dataset_name}::{spec.task_name}:{sample_idx} {type(exc).__name__}: {exc}",
                flush=True,
            )
            row = {
                "run_id": run_id,
                "dataset": spec.dataset_name,
                "task": spec.task_name,
                "task_key": _row_task_key(spec),
                "idx": sample_idx,
                "sample_idx": sample_idx,
                "sample_key": _sample_key(spec, sample_idx, sample),
                "sample_ref": sample_ref,
                "file": sample.file,
                "source_file": sample.file,
                "canonical_sample_id": canonical_sample_id,
                "source_row_index": sample.metadata.get("row_idx"),
                "official_prepared_filename": (
                    str(selection_entry.get("official_prepared_filename") or "").strip()
                    if selection_entry is not None
                    else ""
                ),
                "trace_id": result.get("trace_id"),
                "status": "failed",
                "answer_type": result.get("answer_type"),
                "prediction": result.get("prediction"),
                "raw_answer": result.get("raw_answer"),
                "metrics": {"skipped": "execution_error"},
                "correct": None,
                "gate_accepted": result.get("gate_accepted"),
                "gate_rejections": list(result.get("gate_rejections") or []),
                "confidence": result.get("confidence"),
                "tool_chain": _extract_tool_chain_simple(result),
                "total_ms": result.get("total_ms"),
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "reasoning_preview": _reasoning_preview(result),
                "evidence_summary": result.get("evidence_summary"),
                "execution_context": result.get("execution_context"),
                "control_context": result.get("control_context"),
                "execution_log": result.get("execution_log", []),
                "ground_truth": ground_truth,
            }
            return _artifact_row(row)


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Run MTBench inference through TimeClaw's real run_inference path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--task-set", choices=TASK_SETS, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--samples-per-task", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--llm-max-inflight-requests", type=int, default=5)
    parser.add_argument("--per-task-concurrency", type=int, default=1)
    parser.add_argument(
        "--task-concurrency-override",
        action="append",
        help=(
            "Optional single-task concurrency override. "
            "New definition: accepts official task key or dataset::task, for example "
            "finance_pred_macd_long=14 or finance_aligned_long::indicator_macd=14."
        ),
    )
    parser.add_argument("--task", action="append", help="Optional task key filter. Repeatable.")
    parser.add_argument(
        "--selection-manifests-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit per-task sample manifests. New definition: when set, "
            "these manifests become the authoritative sample set and samples-per-task "
            "is ignored for selection."
        ),
    )
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing rows only when the sample set and logic fingerprint are unchanged.",
    )
    args = parser.parse_args()

    run_id = str(args.run_id or "").strip()
    run_dir = args.output_root
    if not run_id:
        if args.resume and run_dir.name.startswith("infer"):
            run_id = run_dir.name
        elif run_dir.name.startswith("infer"):
            run_id = run_dir.name
        else:
            run_id = f"infer_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            run_dir = run_dir / run_id

    cfg = TimeClawConfig.load(args.config)
    from timeclaw.benchmark.train_loop import _configure_workspace_roots
    cfg.llm.max_inflight_requests = int(args.llm_max_inflight_requests)

    args.task_set = args.task_set or cfg.benchmark.task_set
    cfg.benchmark.task_set = args.task_set
    specs = _task_specs(args.task_set, args.task)
    task_concurrency_overrides = _parse_task_concurrency_overrides(
        specs,
        args.task_concurrency_override,
    )
    datasets: dict[str, list[MTBenchSample]] = {}
    for spec in specs:
        if spec.dataset_name not in datasets:
            datasets[spec.dataset_name] = load_dataset(spec.dataset_name, config=cfg)
    with workspace_access(cfg, write=False):
        assert_inference_lineage(cfg, datasets, disable_memory=False)
        logic_fingerprint = _compute_selection_aware_logic_fingerprint(
            args.config, args.selection_manifests_dir, disable_memory=False,
            request_bindings=_request_bindings(cfg, datasets, specs, int(args.samples_per_task)),
        )
        with _run_output_lease(run_dir, run_id):
            _admit_run_output(output_root=run_dir, run_id=run_id,
                              resume=bool(args.resume), clear_output=bool(args.clear_output))
            _configure_workspace_roots(cfg.workspace_dir)
            _clear_run_output(output_root=run_dir, run_id=run_id, clear_output=bool(args.clear_output))
            run_dir.mkdir(parents=True, exist_ok=True)
            selected_by_task: dict[str, list[SelectedInferenceSample]] = {}
            if args.selection_manifests_dir is not None:
                selected_by_task = _select_from_explicit_manifests(
                    specs=specs,
                    datasets=datasets,
                    manifests_dir=args.selection_manifests_dir,
                )
            else:
                for spec in specs:
                    samples = datasets[spec.dataset_name]
                    selected = samples if int(args.samples_per_task) <= 0 else samples[: int(args.samples_per_task)]
                    chosen: list[SelectedInferenceSample] = []
                    for sample_idx, sample in enumerate(selected):
                        if _extract_ground_truth(sample, spec.task_name) is None:
                            continue
                        scoreable, _ = validate_mtbench_sample_for_scoring(sample, spec.task_name)
                        if not scoreable:
                            continue
                        chosen.append(SelectedInferenceSample(sample_idx=sample_idx, sample=sample))
                    selected_by_task[spec.key] = chosen
            total_work_items = sum(len(selected_by_task[spec.key]) for spec in specs)

            stream_limits = [
                int(task_concurrency_overrides.get(spec.key, int(args.per_task_concurrency)))
                for spec in specs
            ]
            active_limit = resolve_active_work_item_limit(
                stream_limits=stream_limits,
                llm_max_inflight_requests=int(args.llm_max_inflight_requests),
            )

            writer = ArtifactWriter(run_dir, run_id)
            existing_manifest: dict[str, Any] | None = None
            existing_rows: list[dict[str, Any]] = []
            if args.resume and writer.manifest_path.exists():
                existing_manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
                existing_fp = (existing_manifest or {}).get("logic_fingerprint")
                existing_digest = str(existing_fp.get("digest") or "").strip() if isinstance(existing_fp, dict) else ""
                if existing_digest != str(logic_fingerprint["digest"]):
                    raise SystemExit(
                        "Resume refused: the existing inference artifacts were produced by a different logic/config fingerprint. "
                        "Use a new output directory or clear the old artifacts before rerunning."
                    )
            elif args.resume and writer.rows_path.exists():
                raise SystemExit(
                    "Resume refused: rows file exists but manifest with logic fingerprint is missing. "
                    "Use a new output directory or clear the old artifacts before rerunning."
                )
            if args.resume and writer.rows_path.exists():
                existing_rows = _load_existing_rows(writer.rows_path)
                expected_identities = {
                    _resume_identity_from_sample(spec, selected.sample_idx, selected.sample)
                    for spec in specs for selected in selected_by_task[spec.key]
                }
                unexpected = {_resume_identity_from_row(row) for row in existing_rows} - expected_identities
                if unexpected:
                    raise ValueError(f"Result identity is outside the frozen sample selection: {sorted(unexpected)[:3]}")

            preserve_existing_layout = bool(
                args.resume
                and existing_manifest
                and (
                    int(existing_manifest.get("total_work_items") or 0) > int(total_work_items)
                    or len(list(existing_manifest.get("task_keys") or [])) > len(specs)
                )
            )

            manifests_dir = _task_manifests_dir(run_dir, run_id)
            manifests_dir.mkdir(parents=True, exist_ok=True)
            if not preserve_existing_layout:
                for spec in specs:
                    entries = [_sample_manifest_entry(spec, selected) for selected in selected_by_task[spec.key]]
                    _write_json(
                        manifests_dir / f"{spec.key}.samples_manifest.json",
                        {
                            "task_key": spec.key,
                            "dataset_name": spec.dataset_name,
                            "task_name": spec.task_name,
                            "selected_sample_count": len(entries),
                            "logic_fingerprint_digest": logic_fingerprint["digest"],
                            "entries": entries,
                        },
                    )

            completed_resume_ids = {
                _resume_identity_from_row(row)
                for row in existing_rows
                if isinstance(row, dict)
            }
            rows: list[dict[str, Any]] = list(existing_rows)
            rows_by_dataset = _rows_by_dataset(rows)
            completed_by_task: dict[str, int] = defaultdict(int)
            for row in rows:
                task_key = str(row.get("task_key") or "").strip()
                if task_key:
                    completed_by_task[task_key] += 1

            manifest = _rebuild_manifest(
                existing=existing_manifest,
                run_id=run_id,
                task_set=args.task_set,
                config_path=args.config,
                cfg=cfg,
                llm_max_inflight_requests=int(args.llm_max_inflight_requests),
                per_task_concurrency=int(args.per_task_concurrency),
                active_work_item_limit=active_limit,
                samples_per_task=int(args.samples_per_task),
                specs=specs,
                total_work_items=total_work_items,
                completed_rows=len(rows),
                completed_by_task=dict(completed_by_task),
                logic_fingerprint=logic_fingerprint,
                selection_manifests_dir=args.selection_manifests_dir,
                task_manifests_dir=manifests_dir,
                task_concurrency_overrides=task_concurrency_overrides,
                disable_memory=False,
            )
            # The append-only rows are authoritative. Sidecars can lag after a
            # crash, so reconstruct them instead of adopting stale failure totals.
            failed_rows = [row for row in rows if str(row.get("status")) != "completed"]

            pending_items: list[tuple[InferenceTaskSpec, SelectedInferenceSample]] = []
            for spec in specs:
                for selected in selected_by_task[spec.key]:
                    if _resume_identity_from_sample(spec, selected.sample_idx, selected.sample) in completed_resume_ids:
                        continue
                    pending_items.append((spec, selected))

            pending_items = interleave_items_round_robin(pending_items, key_fn=lambda item: item[0].key)
            task_sems = {
                spec.key: asyncio.Semaphore(
                    int(task_concurrency_overrides.get(spec.key, int(args.per_task_concurrency)))
                )
                for spec in specs
            }
            # Persist admission before any paid prediction can write its first row.
            # Otherwise a first-batch interruption leaves rows without a fingerprint
            # and a legitimate resume cannot establish their provenance.
            await writer.write_snapshot(
                rows=rows, rows_by_dataset=rows_by_dataset,
                manifest=manifest, failed_rows=failed_rows,
            )
            runner = PipelineRunner(
                cfg,
                disable_memory=False,
                drain_memory_on_shutdown=False,
            )

            async def _factory(spec: InferenceTaskSpec, selected: SelectedInferenceSample):
                async def _run() -> dict[str, Any]:
                    return await _run_one(
                        runner=runner,
                        spec=spec,
                        selected=selected,
                        task_sem=task_sems[spec.key],
                        run_id=run_id,
                    )
                return _run

            try:
                factories = [await _factory(spec, selected) for spec, selected in pending_items]
                async for done in iter_completed_tasks_with_retry_queue(factories, max_active=active_limit):
                    row = done.result()
                    rows.append(row)
                    rows_by_dataset.setdefault(str(row["dataset"]), []).append(row)
                    completed_by_task[str(row["task_key"])] += 1
                    manifest["completed_rows"] = int(manifest.get("completed_rows") or 0) + 1
                    manifest["completed_by_task"] = dict(completed_by_task)
                    await writer.append_row(row)
                    if str(row.get("status") or "").strip().lower() != "completed":
                        failed_rows.append(row)
                    if int(manifest["completed_rows"]) % 20 == 0:
                        await writer.write_snapshot(
                            rows=rows,
                            rows_by_dataset=rows_by_dataset,
                            manifest=manifest,
                            failed_rows=failed_rows,
                        )
                        print(
                            f"completed_rows={manifest['completed_rows']} last={row['task_key']}:{row['idx']}",
                            flush=True,
                        )

                if int(manifest.get("completed_rows") or 0) >= int(manifest.get("total_work_items") or 0):
                    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
                else:
                    manifest.pop("finished_at", None)
                await writer.write_snapshot(
                    rows=rows,
                    rows_by_dataset=rows_by_dataset,
                    manifest=manifest,
                    failed_rows=failed_rows,
                )
                return 0 if rows and all(str(row.get("status")) == "completed" for row in rows) else 1
            finally:
                try:
                    await runner.shutdown()
                finally:
                    assert_inference_lineage(cfg, datasets, disable_memory=False)


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
