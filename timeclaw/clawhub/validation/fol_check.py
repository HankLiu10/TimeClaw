"""First-order logic check."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class FOLCheckTool(ClawTool):
    """First-order logic check for time series / predictions."""

    name: str = "fol_check"
    category: ToolCategory = ToolCategory.VALIDATION
    description: str = (
        "Validates predictions against formal first-order logic constraints such as monotonicity, bounds, and custom rules. "
        "Input: prediction array and a set of constraint definitions. "
        "Output: dict with 'passed' boolean and list of 'violations' describing which constraints were broken. "
        "Use when predictions must satisfy domain-specific logical invariants beyond simple range checks. "
        "Params: prediction (required, the values to validate), constraints (required, list of FOL constraint definitions)."
    )
    io_type: ToolIOType = ToolIOType.XP_TO_ACCEPT
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "prediction": {
                "type": "array",
                "description": "Prediction values to validate.",
            },
            "constraints": {
                "type": "array",
                "description": "List of constraint dicts, each with a 'type' field.",
            },
        },
        "required": ["prediction", "constraints"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        prediction = np.asarray(params.get("prediction", ts_data), dtype=float)
        constraints = params.get("constraints", [])

        violations: list[dict[str, Any]] = []

        for constraint in constraints:
            ctype = constraint.get("type", "")

            if ctype == "monotonic_increasing":
                diffs = np.diff(prediction)
                if not np.all(diffs >= 0):
                    bad_idx = int(np.argmax(diffs < 0))
                    violations.append({
                        "constraint_type": ctype,
                        "details": f"Decrease at index {bad_idx}: "
                                   f"{prediction[bad_idx]} -> {prediction[bad_idx + 1]}",
                    })

            elif ctype == "monotonic_decreasing":
                diffs = np.diff(prediction)
                if not np.all(diffs <= 0):
                    bad_idx = int(np.argmax(diffs > 0))
                    violations.append({
                        "constraint_type": ctype,
                        "details": f"Increase at index {bad_idx}: "
                                   f"{prediction[bad_idx]} -> {prediction[bad_idx + 1]}",
                    })

            elif ctype == "range":
                lower = constraint.get("lower", -np.inf)
                upper = constraint.get("upper", np.inf)
                out_of_range = (prediction < lower) | (prediction > upper)
                if np.any(out_of_range):
                    n_out = int(np.sum(out_of_range))
                    violations.append({
                        "constraint_type": ctype,
                        "details": f"{n_out} values outside [{lower}, {upper}]; "
                                   f"min={float(prediction.min())}, max={float(prediction.max())}",
                    })

            elif ctype == "sum_equals":
                target = constraint.get("target", 0.0)
                tolerance = constraint.get("tolerance", 1e-6)
                actual_sum = float(np.sum(prediction))
                if abs(actual_sum - target) > tolerance:
                    violations.append({
                        "constraint_type": ctype,
                        "details": f"Sum {actual_sum} != target {target} "
                                   f"(diff={abs(actual_sum - target)}, tol={tolerance})",
                    })

            elif ctype == "non_negative":
                neg_mask = prediction < 0
                if np.any(neg_mask):
                    n_neg = int(np.sum(neg_mask))
                    violations.append({
                        "constraint_type": ctype,
                        "details": f"{n_neg} negative values; min={float(prediction.min())}",
                    })

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "n_constraints": len(constraints),
            "n_violations": len(violations),
        }


tool = FOLCheckTool()
