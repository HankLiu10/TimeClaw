"""Tests for the evidence log and quality gate."""

from __future__ import annotations

import numpy as np

from timeclaw.core.evidence import EvidenceEntry, EvidenceLog, QualityGate, QualityGateResult


class TestEvidenceLog:
    def test_empty_log(self):
        log = EvidenceLog()
        assert log.num_steps == 0
        assert log.is_sufficient

    def test_append_entry(self):
        log = EvidenceLog()
        entry = EvidenceEntry(step=0, thought="test", tool_name="basic_stats")
        log.append(entry)
        assert log.num_steps == 1

    def test_predicate_tracking(self):
        log = EvidenceLog()
        log.set_required({"trend", "stationarity", "anomaly"})
        assert not log.is_sufficient
        assert log.gap_predicates == {"trend", "stationarity", "anomaly"}

        log.mark_covered("trend")
        assert "trend" in log.covered_predicates
        assert "trend" not in log.gap_predicates
        assert not log.is_sufficient

        log.mark_covered("stationarity")
        log.mark_covered("anomaly")
        assert log.is_sufficient

    def test_summary(self):
        log = EvidenceLog()
        log.append(EvidenceEntry(step=0, thought="check trend", tool_name="detect_trend"))
        summary = log.summary()
        assert "detect_trend" in summary
        assert "Step 0" in summary

    def test_summary_without_predicates_falls_back_to_tool_coverage(self):
        log = EvidenceLog()
        log.append(
            EvidenceEntry(
                step=0,
                thought="forecast next window",
                tool_name="timesfm2_forecast",
                observation={"forecast": [1.0]},
            )
        )
        log.append(EvidenceEntry(step=1, thought="classify the label", tool_name="llm_reasoning_classification"))

        summary = log.summary()

        assert "Tool coverage:" in summary
        assert "timesfm2_forecast x1" in summary
        assert "Successful steps: 1/2" in summary


class TestQualityGate:
    def test_accept_valid_prediction(self):
        gate = QualityGate()
        log = EvidenceLog()
        result = gate.check(np.array([1.0, 2.0, 3.0]), log)
        assert result.accepted

    def test_reject_none_prediction(self):
        gate = QualityGate()
        log = EvidenceLog()
        result = gate.check(None, log)
        assert not result.accepted
        assert any("None" in r for r in result.rejections)

    def test_reject_insufficient_evidence(self):
        gate = QualityGate()
        log = EvidenceLog()
        log.set_required({"trend"})
        result = gate.check(np.array([1.0]), log)
        assert not result.accepted

    def test_bound_constraint(self):
        gate = QualityGate()
        log = EvidenceLog()
        constraints = [{"type": "bound", "lower": 0, "upper": 10}]
        result = gate.check(np.array([5.0, 15.0]), log, constraints)
        assert not result.accepted
        assert any("above" in r for r in result.rejections)

    def test_bound_constraint_pass(self):
        gate = QualityGate()
        log = EvidenceLog()
        constraints = [{"type": "bound", "lower": 0, "upper": 100}]
        result = gate.check(np.array([5.0, 50.0]), log, constraints)
        assert result.accepted
