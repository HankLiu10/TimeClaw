"""Skill Distillation: LLM analyses ExperiencePool to extract tool selection skills.

Supports multi-domain (weather/finance) and multi-task (forecast, trend,
indicator, mcqa, correlation) skill extraction.  Skills are partitioned
by domain and task type so the agent gets domain-specific guidance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.core.config import TimeClawConfig
from timeclaw.llm.router import ModelRouter
from timeclaw.memory.experience_pool import ExperiencePool

SKILLS_DIR = Path(".timeclaw/skills")
SKILLS_FILE = SKILLS_DIR / "tool_selection_skills.json"
SKILLS_REPORT = SKILLS_DIR / "skill_report.md"

DISTILL_PROMPT = """\
You are analysing tool/model performance data from a training phase across
multiple domains and task types.

## Training Data Summary
Total samples: {total_samples}
Domains: {domains}
Task types: {task_types}

### Per-Domain Per-Task Performance
{domain_task_summary}

### Per-Tool Forecast Performance (if applicable)
{tool_summary}

### Representative Sample Records
{sample_records}

## Your Task
Produce STRUCTURED SKILLS — rules an agent can follow to select the best
tool or approach for a given domain, task type, and data fingerprint.

Return JSON:
{{
  "skills": [
    {{
      "id": "skill_NNN",
      "domain": "weather|finance|all",
      "task": "forecast|trend|indicator|mcqa|correlation|all",
      "condition": "when this applies",
      "recommended_tool": "tool_name or approach",
      "confidence": 0.0-1.0,
      "evidence": "brief evidence"
    }}
  ],
  "tool_rankings": {{
    "overall_best": "tool_name",
    "by_domain": {{
      "weather": "tool_name",
      "finance": "tool_name"
    }},
    "by_scenario": {{
      "short_series": "tool_name",
      "strong_seasonality": "tool_name",
      "high_volatility": "tool_name",
      "trending": "tool_name"
    }}
  }},
  "insights": "2-3 paragraph summary"
}}
"""


def _build_domain_task_summary(pool: ExperiencePool) -> tuple[str, set, set]:
    """Aggregate metrics by domain x task from training entries."""
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    domains: set[str] = set()
    tasks: set[str] = set()

    for entry in pool._entries:
        if not entry.trace_id.startswith("train_"):
            continue
        domain = entry.metadata.get("domain", "unknown")
        task = entry.metadata.get("task", entry.task_type)
        domains.add(domain)
        tasks.add(task)

        for metric_name, val in entry.ground_truth_metrics.items():
            if isinstance(val, (int, float)):
                key = f"{domain}/{task}"
                stats[key][metric_name].append(val)

    lines = []
    for key in sorted(stats.keys()):
        metric_strs = []
        for m, vals in stats[key].items():
            metric_strs.append(f"{m}={np.mean(vals):.4f} (n={len(vals)})")
        lines.append(f"- **{key}**: {', '.join(metric_strs)}")

    return "\n".join(lines) if lines else "No data.", domains, tasks


def _build_tool_summary(pool: ExperiencePool) -> str:
    """Aggregate per-tool forecast metrics."""
    tool_data: dict[str, dict[str, Any]] = {}

    for entry in pool._entries:
        if not entry.trace_id.startswith("train_"):
            continue
        if entry.metadata.get("task") != "forecast":
            continue
        for tool_name, metrics in entry.ground_truth_metrics.items():
            if tool_name not in tool_data:
                tool_data[tool_name] = {
                    "runs": 0, "successes": 0, "mae_vals": [], "best": 0,
                }
            td = tool_data[tool_name]
            td["runs"] += 1
            if "mae" in metrics:
                v = metrics["mae"]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    td["successes"] += 1
                    td["mae_vals"].append(v)
            if entry.metadata.get("best_tool") == tool_name:
                td["best"] += 1

    lines = []
    for t, td in sorted(tool_data.items()):
        s = td["successes"]
        avg_mae = np.mean(td["mae_vals"]) if s else float("inf")
        lines.append(
            f"- **{t}**: {td['runs']} runs, {s} success, "
            f"avg_MAE={avg_mae:.4f}, best_in={td['best']} "
            f"(win_rate={td['best']/td['runs']*100:.1f}%)"
        )
    return "\n".join(lines) if lines else "No forecast data."


def _build_sample_records(pool: ExperiencePool, max_records: int = 60) -> str:
    train_entries = [e for e in pool._entries if e.trace_id.startswith("train_")]
    if not train_entries:
        return "No training records."

    import random
    selected = random.sample(train_entries, min(max_records, len(train_entries)))

    lines = []
    for entry in selected:
        domain = entry.metadata.get("domain", "?")
        task = entry.metadata.get("task", entry.task_type)
        best = entry.metadata.get("best_tool", "")
        metrics_str = "; ".join(
            f"{k}={v}" for k, v in entry.ground_truth_metrics.items()
            if isinstance(v, (int, float))
        )
        lines.append(
            f"- [{domain}/{task}] fp=({entry.fingerprint_summary[:500]}), "
            f"best={best}, {metrics_str}"
        )
    return "\n".join(lines)


async def distill_skills(
    config: TimeClawConfig | None = None,
) -> dict[str, Any]:
    """Run LLM-driven skill distillation from ExperiencePool."""
    cfg = config or TimeClawConfig.load()
    router = ModelRouter(cfg.llm)
    llm = router.get_backend("reasoner")
    pool = ExperiencePool(storage_dir=f"{cfg.workspace_dir}/experiences")

    train_count = sum(1 for e in pool._entries if e.trace_id.startswith("train_"))
    if train_count < 5:
        logger.warning(f"[SkillDistill] Only {train_count} entries, need >= 5")

    domain_task_summary, domains, tasks = _build_domain_task_summary(pool)
    tool_summary = _build_tool_summary(pool)
    sample_records = _build_sample_records(pool)

    prompt = DISTILL_PROMPT.format(
        total_samples=train_count,
        domains=", ".join(sorted(domains)) or "none",
        task_types=", ".join(sorted(tasks)) or "none",
        domain_task_summary=domain_task_summary,
        tool_summary=tool_summary,
        sample_records=sample_records,
    )

    messages = [
        {"role": "system", "content": "You are a machine learning analyst specialising in tool selection optimisation."},
        {"role": "user", "content": prompt},
    ]

    logger.info(f"[SkillDistill] Analysing {train_count} experiences ({domains}, {tasks})...")

    skills_data = None
    for attempt in range(3):
        response = await llm.complete(messages, response_format="json")
        if response.parsed:
            skills_data = response.parsed
            break
        raw_preview = (response.text or "")[:2000]
        logger.warning(f"[SkillDistill] Attempt {attempt+1}/3 unparseable, raw preview: {raw_preview}")
        if response.text:
            import re
            m = re.search(r'\{[\s\S]*"skills"[\s\S]*\}', response.text)
            if m:
                try:
                    skills_data = json.loads(m.group(0))
                    logger.info("[SkillDistill] Recovered JSON via regex fallback")
                    break
                except json.JSONDecodeError:
                    pass

    if not skills_data:
        logger.error("[SkillDistill] All 3 attempts failed to parse skills")
        return {}

    skills_data = response.parsed

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_FILE.write_text(json.dumps(skills_data, indent=2, ensure_ascii=False))
    logger.info(f"[SkillDistill] Saved skills to {SKILLS_FILE}")

    _generate_report(skills_data, domain_task_summary, tool_summary, train_count, domains, tasks)

    return skills_data


def _generate_report(
    skills_data: dict[str, Any],
    domain_task_summary: str,
    tool_summary: str,
    train_count: int,
    domains: set,
    tasks: set,
) -> None:
    lines = [
        "# TimeClaw Skill Distillation Report",
        "",
        f"Generated from {train_count} training samples.",
        f"Domains: {', '.join(sorted(domains))}",
        f"Tasks: {', '.join(sorted(tasks))}",
        "",
        "## Domain x Task Performance",
        domain_task_summary,
        "",
        "## Forecast Tool Performance",
        tool_summary,
        "",
    ]

    if "insights" in skills_data:
        lines.extend(["## Key Insights", skills_data["insights"], ""])

    if "skills" in skills_data:
        lines.append("## Learned Skills")
        for skill in skills_data["skills"]:
            lines.append(
                f"- **{skill.get('id', '?')}** [{skill.get('domain', 'all')}/{skill.get('task', 'all')}] "
                f"(conf={skill.get('confidence', '?')}): "
                f"When {skill.get('condition', '?')} -> **{skill.get('recommended_tool', '?')}**"
            )
        lines.append("")

    rankings = skills_data.get("tool_rankings", {})
    if rankings:
        lines.append("## Tool Rankings")
        lines.append(f"- Overall: **{rankings.get('overall_best', '?')}**")
        for domain, tool in rankings.get("by_domain", {}).items():
            lines.append(f"- {domain}: **{tool}**")
        for scenario, tool in rankings.get("by_scenario", {}).items():
            lines.append(f"- {scenario}: **{tool}**")

    SKILLS_REPORT.write_text("\n".join(lines))
    logger.info(f"[SkillDistill] Report saved to {SKILLS_REPORT}")


def load_skills() -> dict[str, Any]:
    """Load previously distilled skills."""
    if SKILLS_FILE.exists():
        try:
            return json.loads(SKILLS_FILE.read_text())
        except Exception:
            pass
    return {}


async def main() -> None:
    result = await distill_skills()
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
