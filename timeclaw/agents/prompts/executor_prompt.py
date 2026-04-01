"""Prompt templates for the Executor Agent — mini-ReAct architecture.

The Executor uses a step-by-step Think→Act→Observe loop instead of
planning the entire tool chain upfront. Each step, the LLM sees the
results of previous tools and decides what to do next.

Tool discovery is fully dynamic, but tool exposure is filtered by the
currently available modalities inferred from the sample.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeclaw.clawhub.tool_base import ClawTool


def get_all_tool_names(modalities: list[str] | None = None) -> list[str]:
    """Return tool names compatible with the currently available modalities."""
    from timeclaw.clawhub.tool_registry import list_tool_instances

    available = set(modalities or ["timeseries"])
    return [
        t.name for t in list_tool_instances()
        if t.required_modalities().issubset(available)
    ]


def build_tool_name_text(tool_names: list[str]) -> str:
    """Return the minimal tool list required for the executor prompt."""
    if not tool_names:
        return "(none)"
    return ", ".join(tool_names)


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
6. You only see tools compatible with the currently available modalities.
   If text or image data is absent, tools that require them will not be shown.
7. No tool is mandatory as a first or final step, including any `llm_reasoning_*`
   wrapper — pick tools from evidence and the task; do not assume you must
   end with LLM-only reasoning if another chain already satisfies the task.

## Output Format (strict JSON)

To call a tool:
{{
  "thought": "Reason: <why this next step is justified>. Action: <what tool to run next and why>",
  "done": false,
  "tool": "tool_name",
  "args": {{"key": "value"}}
}}

When done — you MUST include `answer` with the final result:
{{
  "thought": "Reason: <why the evidence supports the final answer>. Action: <finish and return the answer>",
  "done": true,
  "answer": "<the final result — array for forecast, dict for indicator, label for classification, text for QA>",
  "answer_type": "forecast | indicator | classification | qa | correlation",
  "confidence": 0.85,
  "reasoning": "brief explanation of how the answer was derived from tool outputs"
}}

To spawn parallel sub-agents for independent sub-tasks (max {max_subagents}):
{{
  "thought": "Reason: <why parallel branches are justified>. Action: <spawn sub-agents for the listed sub-tasks>",
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
Every `thought` MUST explicitly contain these two labeled parts:
- `Reason:`
- `Action:`
"""

REACT_FIRST_TURN = """\
Task: {task_type}
Description: {task_description}
Data fingerprint: {fingerprint}
{profiling_section}

{reanalysis_section}\
{skills_section}\
## Available Tools
{tool_names}

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
5. No tool is mandatory as a final step, including any `llm_reasoning_*` tool.

## Output Format (strict JSON)

To call a tool:
{{
  "thought": "Reason: <why this is justified>. Action: <what tool to run next and why>",
  "done": false,
  "tool": "tool_name",
  "args": {{"key": "value"}}
}}

When done:
{{
  "thought": "Reason: <why the evidence supports the answer>. Action: <finish and return the answer>",
  "done": true,
  "answer": "<result for this sub-task>",
  "answer_type": "forecast | indicator | classification | qa | correlation",
  "confidence": 0.85,
  "reasoning": "how the answer was derived"
}}
"""

SUBAGENT_FIRST_TURN = """\
Parent task type: {parent_task_type}
Parent task description: {parent_task_description}
Sub-task: {task_description}
Data fingerprint: {fingerprint}
{profiling_section}

{tool_notes_section}\
{memory_section}\
## Available Tools
{tool_names}

Begin working on this sub-task.
"""

# ── Training / Learning Mode ──────────────────────────────────────

TRAIN_MODE_HINT = (
    "\n\n## Learning Mode\n"
    "You are currently in **learning mode**. Your goal is to **explore "
    "multiple approaches** and learn which works best.\n\n"
    "Completion contract:\n"
    "- You MUST NOT finish learning mode before calling a ground-truth evaluation tool.\n"
    "- You MUST explore at least 2 valid candidate paths before finishing, unless tool availability makes that impossible.\n"
    "- A training run is only complete after at least one valid candidate has been evaluated with "
    "`evaluate_batch_against_gt`, `batch_evaluate_against_gt`, or `evaluate_against_gt`.\n"
    "- If you have candidate predictions but no evaluation yet, continue working instead of returning done=true.\n\n"
    "**Workflow:**\n"
    "1. Plan different solution approaches (e.g. distinct statistical pipelines, "
    "different time-series forecast tools, different preprocessing + model "
    "combinations, or independent multi-step chains).\n"
    "2. Use `spawn_subagent` to run them in parallel. Each sub-agent "
    "tries one approach and returns its result to you.\n"
    "3. Collect only the sub-agent results that are valid predictions for the task. "
    "Use `evaluate_batch_against_gt` to evaluate them together.\n"
    "4. Compare. Optionally try more approaches.\n"
    "5. Finish with answer_type=\"learning_summary\" and "
    "answer={\"insight\": \"...\", \"recommendation\": \"...\"}.\n\n"
    "**Rules:**\n"
    "- You are NOT required to include any `llm_reasoning_*` branch among your "
    "candidates; two valid paths may both be non-LLM-reasoning tools only.\n"
    "- Do NOT evaluate invalid results such as explanations, error messages, or wrong-typed outputs.\n"
    "- Prefer `evaluate_batch_against_gt` after collecting multiple valid approaches.\n"
    "- A forecast answer must be a numeric array; an indicator answer must be a numeric JSON object; "
    "a classification/correlation answer must be a label or a JSON object containing the label.\n"
    "- After evaluation, if you finish with a task result instead of a learning summary, the final answer must "
    "reuse one already-evaluated candidate prediction exactly; do not rewrite, summarize, or transform it.\n"
    "- answer_type MUST be \"learning_summary\", not the task type.\n"
    "- If a tool's description says \"Insufficient data\", prioritize "
    "including it in your exploration — it needs more test runs."
)

DEFERRED_TRAIN_MODE_HINT = (
    "\n\n## Learning Mode\n"
    "You are currently in **learning mode** with external learning-note recording.\n\n"
    "Completion contract:\n"
    "- You MUST NOT finish before calling a ground-truth evaluation tool.\n"
    "- You MUST explore at least 2 valid candidate paths before finishing, unless tool availability makes that impossible.\n"
    "- A training run is only complete after at least one valid candidate has been evaluated with "
    "`evaluate_batch_against_gt`, `batch_evaluate_against_gt`, or `evaluate_against_gt`.\n"
    "- The framework will record the learning note externally, so your FINAL answer must use the original task type, not `learning_summary`.\n\n"
    "**Workflow:**\n"
    "1. Explore multiple valid approaches.\n"
    "2. Use `spawn_subagent` to run independent approaches in parallel when helpful.\n"
    "3. Collect only task-valid candidate predictions.\n"
    "4. Prefer `evaluate_batch_against_gt` after collecting multiple valid candidates.\n"
    "5. Finish with a task-typed final answer after at least one successful GT evaluation.\n\n"
    "**Rules:**\n"
    "- You are NOT required to include any `llm_reasoning_*` branch among your "
    "candidates; two valid paths may both be non-LLM-reasoning tools only.\n"
    "- Do NOT evaluate invalid results such as explanations, error messages, or wrong-typed outputs.\n"
    "- A forecast answer must be a numeric array; an indicator answer must be a numeric JSON object; "
    "a classification/correlation answer must be a label or a JSON object containing the label.\n"
    "- New definition: an `evaluated winner` is a candidate that has already been scored by a GT evaluation tool "
    "and is strongest among the currently evaluated candidates.\n"
    "- When finishing, choose an evaluated winner and copy its task-valid prediction as the final answer.\n"
    "- Do NOT rewrite, summarize, reformat, or verbally describe the winner prediction.\n"
    "- For forecast specifically, the final `answer` must be the raw numeric array from the evaluated winner.\n"
    "- If your current final answer is prose, explanation, or a learning summary, it is invalid for deferred learning mode.\n"
)
