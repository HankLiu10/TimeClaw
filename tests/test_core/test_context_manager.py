"""Tests for ContextManager."""

from timeclaw.core.context_manager import ContextManager


def test_estimate_tokens():
    cm = ContextManager()
    assert cm.estimate_tokens("hello world") == 2
    assert cm.estimate_tokens("a" * 400) == 100


def test_fit_budget_no_pruning():
    cm = ContextManager(max_tokens=5000, reserve_tokens=1000)
    sections = {"system": "short", "user": "also short"}
    result = cm.fit_budget(sections)
    assert result["system"] == "short"
    assert result["user"] == "also short"


def test_fit_budget_with_pruning():
    cm = ContextManager(max_tokens=200, reserve_tokens=50)
    sections = {
        "system": "important " * 10,
        "user": "long user content " * 500,
    }
    result = cm.fit_budget(sections)
    assert len(result["user"]) < len(sections["user"])
    assert "[truncated]" in result["user"]


def test_prune_evidence_summary():
    cm = ContextManager(prune_observation_chars=50)
    lines = []
    for i in range(10):
        lines.append(f"Step {i}")
        lines.append(f"  thought: step {i} reasoning")
        lines.append(f"  observation: {'x' * 200}")
    summary = "\n".join(lines)

    pruned = cm.prune_evidence_summary(summary, max_recent_steps=3)
    assert "[observation truncated" in pruned
    assert len(pruned) < len(summary)


def test_prune_tool_descriptions():
    cm = ContextManager()
    tools = [
        {"name": f"tool_{i}", "description": f"description for tool {i}" * 10, "category": "test"}
        for i in range(50)
    ]
    pruned = cm.prune_tool_descriptions(tools, budget=500)
    assert len(pruned) < len(tools)
    for t in pruned:
        assert len(t["description"]) <= 100
