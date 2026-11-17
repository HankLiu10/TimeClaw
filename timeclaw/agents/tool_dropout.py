"""Training-time tool dropout for sub-agent exploration."""

from __future__ import annotations

import random
from typing import Any

DEFAULT_ALPHA = 1.0


def keep_probability(
    tool_name: str,
    scope_counts: dict[str, Any],
    competing_tools: list[str],
    *,
    alpha: float = DEFAULT_ALPHA,
    never_drop: set[str] | None = None,
) -> float:
    """Return the keep probability for one tool under the final design formula.

    New definition:
        ``competing_tools`` is the exact tool subset that participates in the
        dropout race for this sub-agent. ``never_drop`` tools are excluded from
        the shared ``n_min`` scale and always return ``1.0``.
    """
    protected = never_drop or set()
    if tool_name in protected:
        return 1.0

    pool = [name for name in competing_tools if name not in protected]
    if not pool:
        return 1.0

    counts = {
        name: max(0, int(scope_counts.get(name, 0) or 0))
        for name in pool
    }
    n_i = counts.get(tool_name, 0)
    n_min = min(counts.values()) if counts else 0
    return float(((1 + n_min) / (1 + n_i)) ** alpha)


def apply_dropout(
    candidates: list[str],
    scope_counts: dict[str, Any] | None,
    *,
    alpha: float = DEFAULT_ALPHA,
    never_drop: set[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Apply one Bernoulli dropout mask to the current candidate tools."""
    if not candidates:
        return [], {}

    protected = set(never_drop or set())
    counts = scope_counts or {}
    rng = rng or random.Random()

    keep_probs = {
        name: keep_probability(
            name,
            counts,
            candidates,
            alpha=alpha,
            never_drop=protected,
        )
        for name in candidates
    }

    visible: list[str] = []
    for name in candidates:
        p_keep = keep_probs[name]
        if name in protected or rng.random() < p_keep:
            visible.append(name)

    return visible, keep_probs


def generate_dropout_candidates(
    candidates: list[str],
    scope_counts: dict[str, Any] | None,
    *,
    alpha: float = DEFAULT_ALPHA,
    never_drop: set[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Wrapper kept as the main API used by the executor."""
    return apply_dropout(
        candidates,
        scope_counts,
        alpha=alpha,
        never_drop=never_drop,
        rng=rng,
    )
