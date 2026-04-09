from __future__ import annotations

from typing import Any

from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.benchmark.mtbench_adapter import MTBenchSample


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
    return _join_prompt(sys_p, user_p)
