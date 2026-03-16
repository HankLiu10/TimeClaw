"""Prompt templates for the Executor Agent — mini-ReAct architecture.

The Executor uses a step-by-step Think→Act→Observe loop instead of
planning the entire tool chain upfront. Each step, the LLM sees the
results of previous tools and decides what to do next.

Tool discovery is fully dynamic: all registered tools are exposed to the
agent via their self-descriptions (from ClawTool.description or MCP).
No static candidate-tool mappings or modality/task-type filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeclaw.clawhub.tool_base import ClawTool


def get_all_tool_names() -> list[str]:
    """Return the names of ALL registered tools (no filtering)."""
    from timeclaw.clawhub.tool_registry import list_tool_instances
    return [t.name for t in list_tool_instances()]


_CATEGORY_ORDER = [
    "generic_llm", "prediction", "profiling", "detection",
    "decomposition", "processing", "relation", "text", "image", "validation",
]

_CATEGORY_LABELS = {
    "generic_llm": "Generic Multimodal LLM",
    "prediction": "Prediction / Forecasting",
    "profiling": "Profiling / Statistics",
    "detection": "Detection",
    "decomposition": "Decomposition",
    "processing": "Processing",
    "relation": "Relation Analysis",
    "text": "Text Analysis",
    "image": "Image Analysis",
    "validation": "Validation",
}


def build_tool_guide_text(
    tool_names: list[str],
    experience_insights: dict[str, str] | None = None,
    mcp_tools: dict | None = None,
) -> str:
    """Build a categorized tool guide from dynamic sources.

    Tools are grouped by category and ordered so that generic_llm
    appears first, giving the agent equal visibility over LLM-based
    reasoning alongside specialized tools.

    Description priority:
        1. MCP ToolDescriptor.description (from running MCP servers)
        2. ClawTool.description (from the tool registry)
        3. Tool name only (if neither is available — should not happen)

    Experience notes (from TOOLS.md) are appended when available.
    """
    from timeclaw.clawhub.tool_registry import _TOOL_INSTANCES

    def _get_desc(name: str) -> str | None:
        if mcp_tools and name in mcp_tools:
            entry = mcp_tools[name]
            d = entry.description if hasattr(entry, "description") else str(entry)
            if d:
                return d
        obj = _TOOL_INSTANCES.get(name)
        return obj.description if obj else None

    def _get_category(name: str) -> str:
        if mcp_tools and name in mcp_tools:
            entry = mcp_tools[name]
            srv = entry.server_name if hasattr(entry, "server_name") else ""
            if "generic-llm" in srv:
                return "generic_llm"
            for cat in _CATEGORY_ORDER:
                if cat in srv:
                    return cat
        obj = _TOOL_INSTANCES.get(name)
        if obj:
            return obj.category.value if hasattr(obj.category, "value") else str(obj.category)
        return "other"

    grouped: dict[str, list[str]] = {c: [] for c in _CATEGORY_ORDER}
    grouped["other"] = []
    for name in tool_names:
        cat = _get_category(name)
        grouped.setdefault(cat, []).append(name)

    lines: list[str] = []
    for cat in _CATEGORY_ORDER + ["other"]:
        tools_in_cat = grouped.get(cat, [])
        if not tools_in_cat:
            continue
        label = _CATEGORY_LABELS.get(cat, cat.title())
        lines.append(f"\n### {label}")
        for name in tools_in_cat:
            desc = _get_desc(name)
            if desc:
                lines.append(f"- **{name}**: {desc}")
            else:
                lines.append(f"- **{name}**")
            if experience_insights and name in experience_insights:
                lines.append(f"  NOTE: {experience_insights[name]}")

    return "\n".join(lines)


# ── Mini-ReAct Prompt Templates ──────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are the Executor Agent of TimeClaw, a multimodal time-series analysis system.
You solve tasks step-by-step by selecting and orchestrating tools.

## How you work
Each turn, you see the results of all previous steps. You then decide:
- Run ONE more tool (provide tool name and args), OR
- Declare "done" if you have enough information to produce a final result.

## Rules
1. ONLY use tools from the available list — do not invent tool names.
2. Read each tool's description carefully to understand what it does, what inputs
   it needs, and what outputs it produces. Choose tools based on the task at hand.
3. Use results from previous steps to inform your next choice.
   Example: if detect_seasonality found period=7, pass period=7 to stl_decompose.
4. Keep total steps ≤ {max_steps}. Be efficient — don't run tools that won't help.
5. When you have produced the result the task requires, set done=true.
6. You have access to tools across ALL categories (time-series, text, image,
   decomposition, detection, prediction, etc.). Choose whichever tools are
   appropriate for the data and task — there are no modality restrictions.

## Output Format (strict JSON)

To call a tool:
{{
  "thought": "what I learned from previous results and what I should do next",
  "done": false,
  "tool": "tool_name",
  "args": {{"key": "value"}}
}}

When done — you MUST include `answer` with the final result:
{{
  "thought": "summary of analysis and reasoning for the final answer",
  "done": true,
  "answer": "<the final result — array for forecast, dict for indicator, label for classification, text for QA>",
  "answer_type": "forecast | indicator | classification | qa | correlation",
  "confidence": 0.85,
  "reasoning": "brief explanation of how the answer was derived from tool outputs"
}}

To spawn parallel sub-agents for independent sub-tasks (max {max_subagents}):
{{
  "thought": "why this task benefits from parallel sub-agents",
  "done": false,
  "tool": "spawn_subagent",
  "args": {{
    "tasks": [
      {{"task": "sub-task description 1", "tools_hint": "optional tool names"}},
      {{"task": "sub-task description 2", "tools_hint": "optional tool names"}}
    ]
  }}
}}
Each sub-agent runs an independent ReAct loop with the same data and tools.
Their results are returned to you for synthesis. Use this when a task
decomposes into genuinely independent sub-problems that each require
multi-step reasoning.

IMPORTANT: `done: true` without `answer` is INVALID. You must always provide a
concrete answer. If tools failed, still give your best-effort result based on
available information.
"""

REACT_FIRST_TURN = """\
Task: {task_type}
Description: {task_description}
Data fingerprint: {fingerprint}

{reanalysis_section}\
{strategy_section}\
{skills_section}\
{l1_section}\
## Available Tools
{tool_guide}

This is step 1. Analyze the task description, data fingerprint, and available tools.
Decide which tool to run first.
"""

REACT_NEXT_TURN = """\
Step {step}/{max_steps}. Decide: run another tool, or declare done?
"""

# ── Sub-agent prompt (lightweight, shares memory) ────────────────

SUBAGENT_SYSTEM_PROMPT = """\
You are a sub-agent of TimeClaw, working on a specific sub-task assigned by
the main agent. You have the same tools and data as the main agent.

## Rules
1. Focus ONLY on the sub-task described below. Do not try to solve the
   overall problem — the main agent will synthesize your result.
2. ONLY use tools from the available list.
3. Keep steps ≤ {max_steps}. Be concise and efficient.
4. When done, set done=true with your answer for this sub-task.

## Output Format (strict JSON)

To call a tool:
{{
  "thought": "reasoning",
  "done": false,
  "tool": "tool_name",
  "args": {{"key": "value"}}
}}

When done:
{{
  "thought": "summary",
  "done": true,
  "answer": "<result for this sub-task>",
  "answer_type": "forecast | indicator | classification | qa | correlation",
  "confidence": 0.85,
  "reasoning": "how the answer was derived"
}}
"""

SUBAGENT_FIRST_TURN = """\
Sub-task: {task_description}
Data fingerprint: {fingerprint}

{memory_section}\
## Available Tools
{tool_guide}

Begin working on this sub-task.
"""
