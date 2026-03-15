"""Evidence log, quality gate, and related data structures for the reasoning engine."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class EvidenceEntry(BaseModel):
    """A single step in the evidence-gathering trace."""

    step: int
    thought: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    observation: Any = None
    critic_feedback: str = ""
    timestamp: float = Field(default_factory=time.time)

    @property
    def is_successful(self) -> bool:
        return self.observation is not None


class EvidenceLog(BaseModel):
    """Accumulates evidence entries and tracks predicate coverage."""

    entries: list[EvidenceEntry] = Field(default_factory=list)
    covered_predicates: set[str] = Field(default_factory=set)
    gap_predicates: set[str] = Field(default_factory=set)
    required_predicates: set[str] = Field(default_factory=set)

    def append(self, entry: EvidenceEntry) -> None:
        self.entries.append(entry)

    def mark_covered(self, predicate: str) -> None:
        self.covered_predicates.add(predicate)
        self.gap_predicates.discard(predicate)

    def set_required(self, predicates: set[str]) -> None:
        self.required_predicates = predicates
        self.gap_predicates = predicates - self.covered_predicates

    @property
    def is_sufficient(self) -> bool:
        return len(self.gap_predicates) == 0

    @property
    def num_steps(self) -> int:
        return len(self.entries)

    def summary(self) -> str:
        lines = [f"Evidence Log ({self.num_steps} steps):"]
        for e in self.entries:
            status = "OK" if e.is_successful else "FAIL"
            lines.append(f"  [{status}] Step {e.step}: {e.tool_name} -> {e.thought[:500]}")
        lines.append(f"  Covered: {self.covered_predicates}")
        lines.append(f"  Gaps: {self.gap_predicates}")
        return "\n".join(lines)


class QualityGateResult(BaseModel):
    """Result of the SafeAuto quality gate check."""

    accepted: bool
    rejections: list[str] = Field(default_factory=list)
    checked_at: float = Field(default_factory=time.time)

    @property
    def rejection_summary(self) -> str:
        if self.accepted:
            return "PASSED"
        return "REJECTED: " + "; ".join(self.rejections)


class QualityGate:
    """SafeAuto quality gate: validates outputs against FOL, physical, and cross-modal constraints.

    Supported constraint types:
    - bound: value range [lower, upper]
    - monotonic_increasing / monotonic_decreasing: trend enforcement
    - range: explicit min/max range check
    - sum_equals: sum of predictions must equal target
    - non_negative: all values >= 0
    - physical: domain-specific physical rules (temperature, energy, etc.)
    - temporal: continuity and trend consistency with historical data
    - cross_modal: cross-modality consistency (text sentiment vs numeric trend)
    """

    def check(
        self,
        prediction: Any,
        evidence_log: EvidenceLog,
        constraints: list[dict[str, Any]] | None = None,
    ) -> QualityGateResult:
        import numpy as np

        rejections: list[str] = []

        if prediction is None:
            rejections.append("prediction is None")

        if not evidence_log.is_sufficient:
            gaps = evidence_log.gap_predicates
            rejections.append(f"insufficient evidence, missing predicates: {gaps}")

        if constraints:
            for c in constraints:
                violation = self._check_constraint(prediction, c)
                if violation:
                    rejections.append(violation)

        return QualityGateResult(accepted=len(rejections) == 0, rejections=rejections)

    def _check_constraint(self, prediction: Any, constraint: dict[str, Any]) -> str | None:
        """Dispatch to the appropriate constraint checker."""
        import numpy as np

        ctype = constraint.get("type", "")
        checkers = {
            "bound": self._check_bound,
            "range": self._check_range,
            "monotonic_increasing": self._check_monotonic_inc,
            "monotonic_decreasing": self._check_monotonic_dec,
            "sum_equals": self._check_sum_equals,
            "non_negative": self._check_non_negative,
            "physical": self._check_physical,
            "temporal": self._check_temporal,
            "cross_modal": self._check_cross_modal,
        }

        checker = checkers.get(ctype)
        if checker is None:
            return None
        try:
            return checker(prediction, constraint)
        except Exception as e:
            return f"constraint check error ({ctype}): {e}"

    @staticmethod
    def _check_bound(prediction: Any, constraint: dict[str, Any]) -> str | None:
        import numpy as np

        arr = np.asarray(prediction, dtype=float)
        lower = constraint.get("lower")
        upper = constraint.get("upper")
        if lower is not None and np.any(arr < lower):
            return f"bound violation: values below {lower}"
        if upper is not None and np.any(arr > upper):
            return f"bound violation: values above {upper}"
        return None

    @staticmethod
    def _check_range(prediction: Any, constraint: dict[str, Any]) -> str | None:
        import numpy as np

        arr = np.asarray(prediction, dtype=float)
        vmin = constraint.get("min")
        vmax = constraint.get("max")
        if vmin is not None and np.any(arr < vmin):
            return f"range violation: values below min={vmin}"
        if vmax is not None and np.any(arr > vmax):
            return f"range violation: values above max={vmax}"
        return None

    @staticmethod
    def _check_monotonic_inc(prediction: Any, constraint: dict[str, Any]) -> str | None:
        import numpy as np

        arr = np.asarray(prediction, dtype=float).ravel()
        tolerance = constraint.get("tolerance", 0.0)
        diffs = np.diff(arr)
        if np.any(diffs < -tolerance):
            n_violations = int(np.sum(diffs < -tolerance))
            return f"monotonic_increasing violation: {n_violations} decreasing steps"
        return None

    @staticmethod
    def _check_monotonic_dec(prediction: Any, constraint: dict[str, Any]) -> str | None:
        import numpy as np

        arr = np.asarray(prediction, dtype=float).ravel()
        tolerance = constraint.get("tolerance", 0.0)
        diffs = np.diff(arr)
        if np.any(diffs > tolerance):
            n_violations = int(np.sum(diffs > tolerance))
            return f"monotonic_decreasing violation: {n_violations} increasing steps"
        return None

    @staticmethod
    def _check_sum_equals(prediction: Any, constraint: dict[str, Any]) -> str | None:
        import numpy as np

        arr = np.asarray(prediction, dtype=float)
        target = constraint.get("target")
        if target is None:
            return None
        tolerance = constraint.get("tolerance", 1e-6)
        actual = float(np.sum(arr))
        if abs(actual - target) > tolerance:
            return f"sum_equals violation: sum={actual:.6f}, expected={target}"
        return None

    @staticmethod
    def _check_non_negative(prediction: Any, constraint: dict[str, Any]) -> str | None:
        import numpy as np

        arr = np.asarray(prediction, dtype=float)
        if np.any(arr < 0):
            n_neg = int(np.sum(arr < 0))
            return f"non_negative violation: {n_neg} negative values"
        return None

    @staticmethod
    def _check_physical(prediction: Any, constraint: dict[str, Any]) -> str | None:
        """Domain-specific physical constraints.

        Examples:
        - {"type": "physical", "domain": "temperature", "unit": "celsius", "min": -273.15, "max": 1000}
        - {"type": "physical", "domain": "energy", "conservation": true}
        - {"type": "physical", "domain": "pressure", "min": 0}
        """
        import numpy as np

        arr = np.asarray(prediction, dtype=float)
        domain = constraint.get("domain", "")

        if "min" in constraint:
            if np.any(arr < constraint["min"]):
                return f"physical violation ({domain}): values below physical minimum {constraint['min']}"
        if "max" in constraint:
            if np.any(arr > constraint["max"]):
                return f"physical violation ({domain}): values above physical maximum {constraint['max']}"

        if constraint.get("conservation") and "reference_sum" in constraint:
            ref_sum = constraint["reference_sum"]
            tolerance = constraint.get("tolerance", 0.01 * abs(ref_sum)) if ref_sum != 0 else 1e-6
            actual_sum = float(np.sum(arr))
            if abs(actual_sum - ref_sum) > tolerance:
                return (
                    f"physical violation ({domain}): conservation law broken, "
                    f"sum={actual_sum:.4f} vs reference={ref_sum:.4f}"
                )

        return None

    @staticmethod
    def _check_temporal(prediction: Any, constraint: dict[str, Any]) -> str | None:
        """Temporal constraints: continuity with history, trend consistency.

        Examples:
        - {"type": "temporal", "history_last": 100.0, "max_jump": 50.0}
        - {"type": "temporal", "history_trend": "up", "min_trend_agreement": 0.5}
        """
        import numpy as np

        arr = np.asarray(prediction, dtype=float).ravel()
        if len(arr) == 0:
            return None

        history_last = constraint.get("history_last")
        if history_last is not None:
            max_jump = constraint.get("max_jump", float("inf"))
            jump = abs(float(arr[0]) - history_last)
            if jump > max_jump:
                return (
                    f"temporal violation: discontinuity at boundary, "
                    f"jump={jump:.4f} > max_jump={max_jump}"
                )

        history_trend = constraint.get("history_trend")
        if history_trend and len(arr) > 1:
            pred_trend = "up" if arr[-1] > arr[0] else "down" if arr[-1] < arr[0] else "flat"
            min_agreement = constraint.get("min_trend_agreement", 0.0)
            if min_agreement > 0 and pred_trend != history_trend:
                return (
                    f"temporal violation: predicted trend '{pred_trend}' "
                    f"contradicts historical trend '{history_trend}'"
                )

        return None

    @staticmethod
    def _check_cross_modal(prediction: Any, constraint: dict[str, Any]) -> str | None:
        """Cross-modal consistency: check if numeric prediction aligns with text/image evidence.

        Examples:
        - {"type": "cross_modal", "text_sentiment": "negative", "prediction_direction": "up"}
          -> warns about contradiction
        - {"type": "cross_modal", "image_label": "fault", "prediction_anomaly": false}
          -> warns about contradiction
        """
        text_sentiment = constraint.get("text_sentiment")
        pred_direction = constraint.get("prediction_direction")

        if text_sentiment and pred_direction:
            contradictions = {
                ("negative", "up"): "text sentiment is negative but prediction trends up",
                ("positive", "down"): "text sentiment is positive but prediction trends down",
            }
            key = (text_sentiment, pred_direction)
            if key in contradictions:
                severity = constraint.get("severity", "warning")
                if severity == "error":
                    return f"cross_modal violation: {contradictions[key]}"

        image_label = constraint.get("image_label")
        pred_anomaly = constraint.get("prediction_anomaly")
        if image_label and pred_anomaly is not None:
            if image_label in ("fault", "anomaly", "degraded") and not pred_anomaly:
                severity = constraint.get("severity", "warning")
                if severity == "error":
                    return (
                        f"cross_modal violation: image shows '{image_label}' "
                        f"but prediction indicates no anomaly"
                    )

        return None
