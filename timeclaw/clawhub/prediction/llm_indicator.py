"""Backward-compatible alias — the tool has moved to generic_llm/llm_reasoning.py.

Kept here so existing code importing ``llm_indicator`` does not break.
The tool is NOT registered from this location (is_stub=True); the
canonical registration is in ``generic_llm.llm_reasoning``.
"""

from timeclaw.clawhub.generic_llm.llm_reasoning import LLMReasoningTool

# Alias with the old name so any ``from ...llm_indicator import tool`` still works
tool = LLMReasoningTool()
tool.is_stub = True  # prevent double-registration by server_factory
