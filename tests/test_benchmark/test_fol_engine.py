"""Tests for the extended QualityGate FOL engine."""

import numpy as np
import pytest

from timeclaw.core.evidence import QualityGate, EvidenceLog, QualityGateResult


@pytest.fixture
def gate():
    return QualityGate()


@pytest.fixture
def sufficient_log():
    log = EvidenceLog()
    log.set_required({"pred_a"})
    log.mark_covered("pred_a")
    return log


class TestBoundConstraint:
    def test_pass(self, gate, sufficient_log):
        result = gate.check(np.array([5, 10, 15]), sufficient_log, [{"type": "bound", "lower": 0, "upper": 20}])
        assert result.accepted

    def test_fail_lower(self, gate, sufficient_log):
        result = gate.check(np.array([-1, 5, 10]), sufficient_log, [{"type": "bound", "lower": 0}])
        assert not result.accepted
        assert "below" in result.rejections[0]


class TestMonotonicConstraint:
    def test_increasing_pass(self, gate, sufficient_log):
        result = gate.check(np.array([1, 2, 3, 4]), sufficient_log, [{"type": "monotonic_increasing"}])
        assert result.accepted

    def test_increasing_fail(self, gate, sufficient_log):
        result = gate.check(np.array([1, 3, 2, 4]), sufficient_log, [{"type": "monotonic_increasing"}])
        assert not result.accepted

    def test_decreasing_pass(self, gate, sufficient_log):
        result = gate.check(np.array([4, 3, 2, 1]), sufficient_log, [{"type": "monotonic_decreasing"}])
        assert result.accepted


class TestSumEquals:
    def test_pass(self, gate, sufficient_log):
        result = gate.check(np.array([0.25, 0.25, 0.25, 0.25]), sufficient_log, [{"type": "sum_equals", "target": 1.0}])
        assert result.accepted

    def test_fail(self, gate, sufficient_log):
        result = gate.check(np.array([0.3, 0.3, 0.3, 0.3]), sufficient_log, [{"type": "sum_equals", "target": 1.0}])
        assert not result.accepted


class TestNonNegative:
    def test_pass(self, gate, sufficient_log):
        result = gate.check(np.array([0, 1, 2]), sufficient_log, [{"type": "non_negative"}])
        assert result.accepted

    def test_fail(self, gate, sufficient_log):
        result = gate.check(np.array([1, -1, 2]), sufficient_log, [{"type": "non_negative"}])
        assert not result.accepted


class TestPhysicalConstraint:
    def test_temperature(self, gate, sufficient_log):
        result = gate.check(
            np.array([20, 25, 30]),
            sufficient_log,
            [{"type": "physical", "domain": "temperature", "min": -273.15, "max": 1000}],
        )
        assert result.accepted

    def test_temperature_violation(self, gate, sufficient_log):
        result = gate.check(
            np.array([-300, 25]),
            sufficient_log,
            [{"type": "physical", "domain": "temperature", "min": -273.15}],
        )
        assert not result.accepted


class TestTemporalConstraint:
    def test_continuity_pass(self, gate, sufficient_log):
        result = gate.check(
            np.array([101, 102, 103]),
            sufficient_log,
            [{"type": "temporal", "history_last": 100.0, "max_jump": 5.0}],
        )
        assert result.accepted

    def test_continuity_fail(self, gate, sufficient_log):
        result = gate.check(
            np.array([200, 201, 202]),
            sufficient_log,
            [{"type": "temporal", "history_last": 100.0, "max_jump": 5.0}],
        )
        assert not result.accepted


class TestCrossModalConstraint:
    def test_no_error_by_default(self, gate, sufficient_log):
        result = gate.check(
            np.array([1, 2, 3]),
            sufficient_log,
            [{"type": "cross_modal", "text_sentiment": "negative", "prediction_direction": "up"}],
        )
        assert result.accepted  # default severity is warning, not error

    def test_error_severity(self, gate, sufficient_log):
        result = gate.check(
            np.array([1, 2, 3]),
            sufficient_log,
            [{"type": "cross_modal", "text_sentiment": "negative", "prediction_direction": "up", "severity": "error"}],
        )
        assert not result.accepted
