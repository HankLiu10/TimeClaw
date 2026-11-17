from __future__ import annotations

from typing import Any

from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.text.task_pruning import task_text_view


def _normalize_official_prompt_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Upstream official finance BB prompt contains the typo "values values".
    return text.replace("values values", "values")


def _join_prompt(system_prompt: str, user_prompt: str) -> str:
    system_prompt = _normalize_official_prompt_text(system_prompt)
    user_prompt = _normalize_official_prompt_text(user_prompt)
    if system_prompt.strip():
        return f"System prompt:\n{system_prompt}\n\nUser prompt:\n{user_prompt}"
    return f"User prompt:\n{user_prompt}"


def build_task_description(sample: MTBenchSample, task: str) -> str:
    try:
        sys_p, user_p = prompts.build_prompt(sample, task)
    except KeyError:
        return sample.metadata.get("description") or f"{task} task on {sample.domain} data."
    description = _join_prompt(sys_p, user_p)
    task_view = task_text_view(sample, task)
    raw_text_ref = str(task_view.get("raw_text_ref") or "").strip()
    removed_summary = task_view.get("removed_summary")
    removed_lines = []
    if isinstance(removed_summary, list):
        removed_lines = [str(item).strip() for item in removed_summary if str(item).strip()]
    if raw_text_ref:
        removed_section = ""
        if removed_lines:
            removed_section = "\n".join(
                f"- Pruning removed: {item}" for item in removed_lines
            )
            removed_section = f"{removed_section}\n"
        description = (
            f"{description}\n\n"
            "Additional TimeClaw text access note:\n"
            f"- The original long text is not embedded here.\n"
            f"{removed_section}"
            f"- If you need to inspect the raw source, call `read_raw_text` with raw_text_ref=\"{raw_text_ref}\"."
        )
    return description
