"""MTBench ↔ pipeline glue: scope keys, task text, sample subsets, learning analysis.

Entry points (train/eval/baseline) stay in ``timeclaw.benchmark`` root; this package
holds shared helpers so ``pipeline_eval`` / ``train_loop`` / ``runner`` stay thin.
"""

from timeclaw.benchmark.mtbench_support.learning_analysis import (
    generate_learning_analysis,
    render_learning_analysis_text,
)
from timeclaw.benchmark.mtbench_support.sample_selection import (
    load_sample_indices,
    select_samples_by_index,
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
    "render_learning_analysis_text",
    "scope_keys_for_dataset_task",
    "select_samples_by_index",
]
