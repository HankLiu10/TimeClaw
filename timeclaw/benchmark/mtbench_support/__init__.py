"""MTBench ↔ pipeline glue: scope keys, task text, sample subsets, learning analysis.

Entry points (train/eval/baseline) stay in ``timeclaw.benchmark`` root; this package
holds shared helpers so ``train_loop`` / ``runner`` stay thin.
"""

from timeclaw.benchmark.mtbench_support.learning_analysis import (
    generate_learning_analysis,
    render_learning_analysis_text,
    render_learning_summary_from_analysis,
)
from timeclaw.benchmark.mtbench_support.official_metric_policy import (
    apply_official_metric_policy_to_row,
    summarize_official_inference_rows,
)
from timeclaw.benchmark.mtbench_support.runtime_context import (
    pipeline_task_for_benchmark_task,
    prepare_task_run_kwargs,
    prepare_mtbench_run_kwargs,
)
from timeclaw.benchmark.mtbench_support.task_scoring import (
    align_numeric_prediction_to_ground_truth,
    backfill_scored_metrics,
    canonical_task_type,
    score_mtbench_prediction,
    validate_mtbench_sample_for_scoring,
)
from timeclaw.benchmark.mtbench_support.sample_selection import (
    load_sample_indices,
    select_samples_by_index,
    select_samples_with_text_char_cap,
    select_samples_with_original_indices,
)
from timeclaw.benchmark.mtbench_support.scope_keys import (
    expand_dataset_tasks,
    scope_keys_for_dataset_task,
)
from timeclaw.benchmark.mtbench_support.task_descriptions import build_task_description

__all__ = [
    "build_task_description",
    "expand_dataset_tasks",
    "generate_learning_analysis",
    "load_sample_indices",
    "apply_official_metric_policy_to_row",
    "align_numeric_prediction_to_ground_truth",
    "backfill_scored_metrics",
    "pipeline_task_for_benchmark_task",
    "prepare_task_run_kwargs",
    "prepare_mtbench_run_kwargs",
    "canonical_task_type",
    "render_learning_analysis_text",
    "render_learning_summary_from_analysis",
    "score_mtbench_prediction",
    "scope_keys_for_dataset_task",
    "select_samples_by_index",
    "select_samples_with_text_char_cap",
    "select_samples_with_original_indices",
    "summarize_official_inference_rows",
    "validate_mtbench_sample_for_scoring",
]
