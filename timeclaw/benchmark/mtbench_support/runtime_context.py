from __future__ import annotations

from typing import Any

from timeclaw.data_integrity import DataIntegrityError

from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark.mtbench_prompts import resolve_forecast_horizon
from timeclaw.benchmark.mtbench_support.scope_keys import scope_keys_for_dataset_task
from timeclaw.benchmark.mtbench_support.task_descriptions import build_task_description
from timeclaw.text.task_pruning import prepare_task_text, task_text_data, task_text_view


def pipeline_task_for_benchmark_task(task: str) -> str:
    """Map benchmark-facing task ids to PipelineRunner core task ids.

    New definition:
        ``pipeline_task_for_benchmark_task`` is the canonical shared mapping
        used by both training and MTBench-style real inference.  This avoids
        train/test/runtime each maintaining slightly different task routing.
    """
    return {
        "mcqa": "qa",
        "trend": "classification",
        "trend_past": "classification",
        "indicator_macd": "indicator",
        "indicator_upper_bb": "indicator",
    }.get(str(task or "").strip(), str(task or "").strip())


def _optional_text_heavy_backend(pipeline: Any) -> Any | None:
    """Best-effort access to the text pruning backend.

    New definition:
        Lightweight train/eval stubs used in tests may not expose a full
        ``router`` object. In that case task preparation should keep using the
        sample's raw text instead of crashing before ``pipeline.run(...)``.
    """
    router = getattr(pipeline, "router", None)
    if router is None:
        return None
    router_cfg = getattr(router, "config", None)
    model_cfg = getattr(router_cfg, "models", None)
    if model_cfg is not None and getattr(model_cfg, "text_heavy", None) is None:
        return None
    get_backend = getattr(router, "get_backend", None)
    if not callable(get_backend):
        return None
    return get_backend("text_heavy")


async def prepare_task_run_kwargs(
    sample: MTBenchSample,
    pipeline: Any,
    task: str,
    *,
    source_metadata: dict[str, Any] | None = None,
    mode: str = "test",
    ground_truth: Any = None,
    verbose: bool | None = None,
    full_dump: bool | None = None,
) -> dict[str, Any]:
    """Build the runner kwargs shared by task-aware train/test/runtime paths.

    New definition:
        ``prepare_task_run_kwargs`` materializes the exact training-aligned
        task-aware runner
        call contract for a known MTBench sample:
        1. task text pruning,
        2. official task description construction,
        3. canonical scope-key routing into memory.

        Generic callers that do not have an MTBench dataset/task identity still
        use ``PipelineRunner.run(...)`` directly; this helper is only for
        aligned MTBench-style inference/training paths.
    """
    if mode not in {"train", "test"}:
        raise DataIntegrityError(f"Unsupported execution mode: {mode!r}")
    if mode != "train" and ground_truth is not None:
        raise DataIntegrityError("ground_truth is training-only; test scoring must happen outside the agent")

    task_name = str(task or "").strip()
    if not task_name:
        raise ValueError("task is required")

    dataset_name = str(getattr(sample, "dataset", "") or "").strip()
    pipeline_task = pipeline_task_for_benchmark_task(task_name)
    horizon = resolve_forecast_horizon(sample) if pipeline_task == "forecast" else None

    text_heavy = _optional_text_heavy_backend(pipeline)
    if text_heavy is not None:
        await prepare_task_text(sample, task_name, llm=text_heavy)
    description = build_task_description(sample, task_name)
    text_data = task_text_data(sample, task_name)
    task_view = task_text_view(sample, task_name)

    merged_source_metadata = dict(source_metadata or {})
    if mode == "train" and "training_source_admission" in sample.metadata:
        merged_source_metadata["training_source_admission"] = dict(sample.metadata["training_source_admission"])
    merged_source_metadata.setdefault("dataset_name", dataset_name)
    merged_source_metadata.setdefault("task_name", task_name)
    merged_source_metadata.setdefault("pipeline_task", pipeline_task)
    merged_source_metadata.setdefault("domain", sample.domain)
    raw_text_ref = str(task_view.get("raw_text_ref") or "").strip()
    cleaned_text_ref = str(task_view.get("cleaned_text_ref") or "").strip()
    if raw_text_ref:
        merged_source_metadata["raw_text_ref"] = raw_text_ref
    if cleaned_text_ref:
        merged_source_metadata["cleaned_text_ref"] = cleaned_text_ref

    kwargs: dict[str, Any] = {
        "data": sample.input_window,
        "timestamps": sample.timestamps,
        "task_type": pipeline_task,
        "description": description,
        "horizon": horizon,
        "text_data": text_data,
        "domain": sample.domain,
        "scope_keys": scope_keys_for_dataset_task(dataset_name, task_name),
        "mode": mode,
        "source_metadata": merged_source_metadata,
    }
    if ground_truth is not None:
        kwargs["ground_truth"] = ground_truth
    if verbose is not None:
        kwargs["verbose"] = verbose
    if full_dump is not None:
        kwargs["full_dump"] = full_dump
    return kwargs


async def prepare_mtbench_run_kwargs(
    sample: MTBenchSample,
    pipeline: Any,
    task: str,
    *,
    source_metadata: dict[str, Any] | None = None,
    mode: str = "test",
    ground_truth: Any = None,
    verbose: bool | None = None,
    full_dump: bool | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for ``prepare_task_run_kwargs``.

    New definition:
        The old name is kept so existing training/eval code continues to work
        unchanged. New real-inference code should prefer the training-aligned
        neutral name ``prepare_task_run_kwargs``.
    """
    return await prepare_task_run_kwargs(
        sample,
        pipeline,
        task,
        source_metadata=source_metadata,
        mode=mode,
        ground_truth=ground_truth,
        verbose=verbose,
        full_dump=full_dump,
    )
