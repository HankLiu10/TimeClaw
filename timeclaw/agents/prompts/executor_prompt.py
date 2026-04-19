"""Prompt templates for the Executor Agent — mini-ReAct architecture.

The Executor uses a step-by-step Think→Act→Observe loop instead of
planning the entire tool chain upfront. Each step, the LLM sees the
results of previous tools and decides what to do next.

Tool discovery is fully dynamic, but tool exposure is filtered by the
currently available modalities inferred from the sample.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from timeclaw.clawhub.tool_base import ClawTool


_AGENT_HIDDEN_TOOLS = {
    "ensemble_forecast",
}


def filter_agent_available_tool_names(tool_names: list[str]) -> list[str]:
    """Hide tools whose argument contracts do not fit the current agent loop.

    New definition:
        - These tools still exist in ClawHub and can be used by code paths that
          prepare their full structured inputs explicitly.
        - They are removed only from the ReAct agent's visible tool list when
          the current loop cannot reliably construct the required arguments from
          its artifact state.

    Example:
        `ensemble_forecast` requires a `forecasts` list assembled from multiple
        earlier forecast outputs, but the current executor only carries forward
        the latest working artifact rather than a structured forecast bundle.
    """
    return [
        name for name in tool_names
        if str(name).strip() and str(name).strip() not in _AGENT_HIDDEN_TOOLS
    ]


def get_all_tool_names(modalities: list[str] | None = None) -> list[str]:
    """Return tool names compatible with the currently available modalities."""
    from timeclaw.clawhub.tool_registry import list_tool_instances

    available = set(modalities or ["timeseries"])
    tool_names = [
        t.name for t in list_tool_instances()
        if t.required_modalities().issubset(available)
    ]
    return filter_agent_available_tool_names(tool_names)


def build_tool_name_text(tool_names: list[str]) -> str:
    """Return a concise tool catalog with descriptions and argument contracts."""
    if not tool_names:
        return "(none)"

    from timeclaw.clawhub.tool_registry import get_tool_instance

    runtime_only_tools: dict[str, dict[str, Any]] = {
        "spawn_subagent": {
            "description": (
                "Spawn parallel sub-agents for genuinely independent candidate paths or sub-problems. "
                "Use this only when the branches can be explored separately and later synthesized."
            ),
            "required": [
                (
                    "tasks",
                    "array",
                    "List of sub-task specs. Each item should be an object with `task`, `slot_id`, and optional `tools_hint` / `branch_metadata`.",
                ),
            ],
            "optional": [],
        },
        "evaluate_against_gt": {
            "description": (
                "Evaluate one concrete prediction against ground truth during learning mode."
            ),
            "required": [
                ("prediction", "any", "One task-valid prediction payload"),
                ("task_type", "string", "Current task type"),
            ],
            "optional": [],
        },
        "evaluate_batch_against_gt": {
            "description": (
                "Evaluate multiple candidate predictions against ground truth during learning mode. "
                "If the latest spawned sub-agents already produced valid candidates, you may omit "
                "`predictions` and the framework will auto-fill them."
            ),
            "required": [
                ("task_type", "string", "Current task type"),
            ],
            "optional": [
                (
                    "predictions",
                    "array",
                    "Optional explicit candidate list. Use only when auto-fill from recent sub-agent results is unavailable.",
                ),
            ],
        },
        "batch_evaluate_against_gt": {
            "description": (
                "Alias of `evaluate_batch_against_gt` with the same compact-call behavior."
            ),
            "required": [
                ("task_type", "string", "Current task type"),
            ],
            "optional": [
                (
                    "predictions",
                    "array",
                    "Optional explicit candidate list. Use only when auto-fill from recent sub-agent results is unavailable.",
                ),
            ],
        },
    }

    def _fmt_param(
        name: str,
        spec: dict[str, Any] | tuple[str, str, str],
        *,
        required: bool,
    ) -> str:
        if isinstance(spec, tuple):
            param_name, param_type, param_desc = spec
        else:
            param_name = name
            param_type = str(spec.get("type", "any"))
            param_desc = str(spec.get("description", "")).strip()
        label = f"{param_name}<{param_type}>"
        if required:
            label += " required"
        if param_desc:
            return f"{label}: {param_desc.rstrip('.')}"
        return label

    def _fmt_input_contract(tool: ClawTool) -> str:
        contract = tool.input_contract()
        if not isinstance(contract, dict) or not contract:
            return ""
        parts: list[str] = []
        artifact_types = [
            str(item).strip()
            for item in (contract.get("artifact_types") or [])
            if str(item).strip()
        ]
        if artifact_types:
            parts.append("artifact=" + "/".join(artifact_types))
        min_series_channels = contract.get("min_series_channels")
        if min_series_channels is not None:
            parts.append(f"min_series_channels={int(min_series_channels)}")
        if not parts:
            return ""
        return "Current-state preconditions: " + "; ".join(parts) + "."

    def _fmt_framework_series_source_arg(tool: ClawTool) -> str:
        required_modalities = tool.required_modalities()
        if "timeseries" not in required_modalities:
            return ""
        return (
            "Framework-supported optional arg: "
            "source_ref<string>: select which accessible series source to read from. "
            "Use `current` for the current working series, `original_input` for the raw input, "
            "or a `step_<n>_<tool>` ref listed in the latest state update."
        )

    lines: list[str] = []
    for tool_name in tool_names:
        framework_source_text = ""
        if tool_name in runtime_only_tools:
            meta = runtime_only_tools[tool_name]
            description = str(meta["description"]).strip()
            required_args = [
                _fmt_param(name, spec, required=True)
                for name, spec in ((item[0], item) for item in meta.get("required", []))
            ]
            optional_args = [
                _fmt_param(name, spec, required=False)
                for name, spec in ((item[0], item) for item in meta.get("optional", []))
            ]
        else:
            try:
                tool = get_tool_instance(tool_name)
            except KeyError:
                lines.append(f"- {tool_name}: metadata unavailable.")
                continue

            description = str(tool.description or "").strip().replace("\n", " ")
            schema = tool.parameters_schema or {}
            properties = schema.get("properties", {}) or {}
            required = set(schema.get("required", []) or [])
            ordered_names = [name for name in properties if name in required] + [
                name for name in properties if name not in required
            ]
            required_args = []
            optional_args = []
            for name in ordered_names:
                spec = properties.get(name, {})
                if name in required:
                    required_args.append(_fmt_param(name, spec, required=True))
                else:
                    optional_args.append(_fmt_param(name, spec, required=False))
            contract_text = _fmt_input_contract(tool)
            framework_source_text = _fmt_framework_series_source_arg(tool)

        arg_parts: list[str] = []
        if required_args:
            arg_parts.append("Required args: " + "; ".join(required_args))
        if optional_args:
            arg_parts.append("Optional args: " + "; ".join(optional_args))
        if tool_name not in runtime_only_tools and contract_text:
            arg_parts.append(contract_text)
        if tool_name not in runtime_only_tools and framework_source_text:
            arg_parts.append(framework_source_text)
        if not arg_parts:
            arg_parts.append("Args: none.")
        lines.append(f"- {tool_name}: {description} {' '.join(arg_parts)}")

    return "\n".join(lines)


# ── Mini-ReAct Prompt Templates ──────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are TimeClaw, a multimodal time-series analysis system.
You solve tasks step-by-step by selecting and orchestrating tools.

## How you work
Each turn, you see the results of all previous steps. You then decide:
- Run ONE more tool (provide tool name and args), OR
- Declare "done" if you have enough information to produce a final result.
- Per turn, do EXACTLY ONE thing. Return exactly ONE top-level JSON object:
  either one tool call or one final answer, never both in the same turn.
- This reply is ONLY for the current turn. Do not include a multi-step plan, a
  sequence like "first ... then ...", or any second action reserved for a later
  turn. The framework, not you, will ask for the next turn separately.

## Rules
1. ONLY use tools from the available list — do not invent tool names.
2. Read each tool's description carefully and choose tools based on the task.
3. Use results from previous steps to inform the next choice.
4. Keep total steps ≤ {max_steps}. Be efficient.
5. When you have produced the required result, set done=true.
6. If you choose a tool call in this turn, stop after that tool-call JSON. Do NOT
   also assume the tool's output or append a final answer in the same turn.
7. For task-typed final answers, do NOT invent new task content in the final step.
   You may only finish by returning the exact task answer that was already
   produced by a real tool in this branch. Do not describe packaging, JSON
   repair, or payload wrapping. If no tool has produced such an answer yet,
   call another tool.

## Output Format (strict JSON)

Return exactly ONE top-level JSON object per turn. Do not append a second JSON
object, and do not combine a tool call with a final answer in the same reply.
If you choose a tool call, output ONLY the tool-call object.
If you choose a final answer, output ONLY the final-answer object.
Exactly one action type is valid per turn:
- tool-call JSON must NOT include `answer`, `answer_type`, `confidence`, or `reasoning`
- final-answer JSON must NOT include `tool` or `args`
Never append a backup `done=true` answer after a tool call.
Never append a backup tool call after a final answer.
Do not describe what you would do on the next turn. Output only the action you
are taking right now.
Do not simulate tool execution transcripts such as `assistant to=tool ...`,
`tool result=...`, markdown code fences, or any pseudo-log after the JSON.

To call a tool:
{{
  "thought": "Reason: <why this next step is justified>. Action: <what tool to run next and why>",
  "done": false,
  "tool": "tool_name",
  "args": {{"key": "value"}}
}}
Tool-call JSON must not include `answer`, `answer_type`, `confidence`, `reasoning`,
`execution_context`, or `control_context`.

When done — you MUST include `answer` with the final result:
{{
  "thought": "Reason: <why the evidence supports the final answer>. Action: <finish and return the answer>",
  "done": true,
  "answer": "<the final result — array for forecast, dict for indicator, label for classification, text for QA; or, in training main-agent mode, a JSON object like {{\"insight\":\"...\",\"recommendation\":\"...\"}}>",
  "answer_type": "forecast | indicator | classification | qa | correlation | learning_summary (training main-agent only)",
  "confidence": 0.85,
  "reasoning": "numbered step-by-step explanation focused on substantive analysis. Write it as a numbered expansion of the same execution-side evidence you put in `execution_context.public_reasoning`, and reuse the same literal tool/path names there. If you directly used analysis tools yourself, include one numbered item per substantive analysis tool in order. If your top-level work was only orchestration (for example spawning candidates and comparing them), do NOT describe orchestration-only tools such as spawn_subagent or evaluate_* as steps; instead explain the candidate approaches considered, the decisive evidence, and why the winner was selected.",
  "execution_context": {{
    "tool_chain": ["the substantive tool chain that produced this answer"],
    "public_reasoning": ["1-3 short reusable evidence sentences grounded in actual tool outputs or data properties"],
    "evidence_summary": "one concise execution-side evidence summary"
  }},
  "control_context": {{
    "completion_notes": "orchestration / completion notes reserved for controller or trace use; use an empty string if none"
  }}
}}
Final-answer JSON must not include `tool` or `args`.
`execution_context` is for reusable task evidence only. Do NOT put branch control,
completion mechanics, packaging, or prompt-contract language inside it.
`control_context` is the only place where orchestration/completion mechanics may appear.
In training main-agent mode, the only valid final `answer_type` is `learning_summary`.

IMPORTANT: `done: true` without `answer` is INVALID. You must always provide a
concrete answer when you intentionally finish.
If a tool call is still needed, do NOT append a backup final answer in the same
reply.
A placeholder final answer is INVALID. Do not return `done=true` merely to say
that you are waiting for tool results, pending sub-agent outputs, or future
evaluation/comparison.
Every `thought` MUST explicitly contain these two labeled parts:
- `Reason:`
- `Action:`
"""

REACT_FIRST_TURN = """\
## Objective
### Task Prompt
{task_prompt}

{task_boundary_section}\
{execution_context_section}\
{control_context_section}\
{output_contract_section}\
## Observation
### Sample Fingerprint
{fingerprint}
{profiling_section}\
{support_section}\
## Decision
{decision_section}

### Available Tools
{tool_names}
"""

REACT_NEXT_TURN = """\
## Decision
Turn {step}/{max_steps}. Re-read the objective and the accumulated observations.
This message asks for the current turn only. Do not plan multiple future turns
inside the same reply.
{decision_section}
"""

REACT_BUDGET_ALERT = """\
Budget alert: only {remaining_turns} turn(s) remain including this one. {budget_guidance}
"""

TRAIN_COMPLETION_BUDGET_ALERT = """\
Budget alert: only {remaining_turns} turn(s) remain including this one, and the training completion contract is already satisfied (at least 2 valid candidate paths plus GT evaluation). Do NOT start another exploration branch or extra evaluation. Finish with `done=true` and answer_type="learning_summary".
"""

FINAL_STEP_MUST_FINISH = """\
Final decision step: this is your last turn. You MUST return `done=true` with the required final answer now. Do not call another tool. Wrap the task payload inside the full final JSON object with `thought`, `done`, `answer`, and `answer_type`; do not return a bare array, bare object, or bare label.
"""

# ── Sub-agent prompt (shares the same core prompt) ────────────────

SUBAGENT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT

SUBAGENT_FIRST_TURN = REACT_FIRST_TURN

def build_visible_tool_guidance(
    tool_names: list[str],
    *,
    defer_learning_record: bool = False,
) -> str:
    visible = {str(name).strip() for name in tool_names if str(name).strip()}
    sections: list[str] = []

    if "spawn_subagent" in visible:
        sections.append(
            "## Spawn Guidance\n"
            "- If you think this step would benefit from exploring multiple follow-up branches for comparison, prefer using `spawn_subagent`.\n"
            "- In training mode, the first spawn round should create at least 2 distinct candidate branches.\n"
            "- When you spawn branches, build them from the provided candidate slots / post-dropout visible tool subsets.\n"
            "- For each spawned branch, choose a `slot_id` whose `visible_tools` fully cover every tool named in that branch's `tools_hint`.\n"
            "- Do not assign a branch to a slot if even one hinted tool is missing from that slot's `visible_tools`.\n"
            "- If a preferred family anchor is not visible in the chosen slot, keep `tools_hint` slot-local and put that explicit anchor in `branch_metadata.anchor`.\n"
            "- Different branches should use different slot tool subsets or path suggestions, not the same path repeated.\n"
            "- Each branch should be a short candidate path, not a full plan.\n"
        )

    eval_visible = visible.intersection({
        "evaluate_against_gt",
        "evaluate_batch_against_gt",
        "batch_evaluate_against_gt",
    })
    if eval_visible:
        lines = [
            "## Evaluate Guidance",
            "- `evaluate_*` is for post-hoc validation and retrospective review, not for solving the task itself.",
            "- After calling an evaluate tool, do not adjust, rewrite, or refine the candidate output in this run.",
            "- After evaluation, you may only finish or summarize using already-produced candidates.",
            "- Do not use evaluate outputs or any revealed ground truth as task evidence inside the solution path.",
            "- Before finishing this learning run, evaluate at least one task-valid candidate.",
            "- If multiple reasonable candidates exist, compare at least 2 candidate paths before finishing unless tool availability makes that impossible.",
        ]
        if defer_learning_record:
            lines.append(
                "- If you finish with a task-typed final answer, copy one already-evaluated candidate exactly."
            )
        else:
            lines.append(
                '- If you finish the learning run itself, `answer_type` must be `"learning_summary"`.'
            )
        sections.append("\n".join(lines) + "\n")

    if not sections:
        return ""
    return "\n\n" + "\n".join(section.rstrip() for section in sections)
