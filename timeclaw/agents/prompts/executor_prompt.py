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


def build_tool_guide_text(
    tool_names: list[str],
    experience_insights: dict[str, str] | None = None,
    mcp_tools: dict | None = None,
) -> str:
    """Build a tool guide from dynamic sources only.

    Description priority:
        1. MCP ToolDescriptor.description (from running MCP servers)
        2. ClawTool.description (from the tool registry)
        3. Tool name only (if neither is available — should not happen)

    Experience notes (from TOOLS.md) are appended when available.
    """
    from timeclaw.clawhub.tool_registry import _TOOL_INSTANCES

    lines: list[str] = []
    for name in tool_names:
        desc = None
        if mcp_tools and name in mcp_tools:
            mcp_entry = mcp_tools[name]
            desc = mcp_entry.description if hasattr(mcp_entry, "description") else str(mcp_entry)
        if not desc:
            tool_obj = _TOOL_INSTANCES.get(name)
            if tool_obj:
                desc = tool_obj.description
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
{{
  "thought": "what I learned from previous results and what I should do next",
  "done": false,
  "tool": "tool_name",
  "args": {{"key": "value"}}
}}

When done:
{{
  "thought": "summary of what was accomplished",
  "done": true
}}
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
Step {step}/{max_steps}. Previous results:

{history}

Decide: run another tool, or declare done?
"""
