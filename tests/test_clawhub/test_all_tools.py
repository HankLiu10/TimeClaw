"""Comprehensive async tests for all non-stub ClawHub tools.

Uses discover_and_register to load all tools, then exercises each one
via tool.execute(data, **params) and validates ToolResult.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from timeclaw.clawhub.tool_registry import (
    clear_registry,
    discover_and_register,
    get_tool_instance,
)


@pytest.fixture(scope="module", autouse=True)
def _register_tools():
    """Discover and register all tools once per module."""
    clear_registry()
    count = discover_and_register()
    assert count > 0, "No tools were discovered"
    yield
    clear_registry()


@pytest.fixture
def ts_data():
    """200-point univariate time series with trend and seasonality."""
    np.random.seed(42)
    t = np.arange(200)
    return t * 0.5 + np.sin(t * 0.1) * 10 + np.random.randn(200) * 2


@pytest.fixture
def ts_multivariate():
    """200x3 multivariate time series."""
    np.random.seed(42)
    t = np.arange(200)
    ch1 = t * 0.5 + np.random.randn(200) * 2
    ch2 = np.sin(t * 0.1) * 10 + np.random.randn(200)
    ch3 = np.random.randn(200) * 5
    return np.column_stack([ch1, ch2, ch3])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_tool(name: str, data: Any, **params: Any):
    tool = get_tool_instance(name)
    result = await tool.execute(data, **params)
    return result


# ---------------------------------------------------------------------------
# Processing tools
# ---------------------------------------------------------------------------

class TestProcessingTools:
    async def test_handle_missing(self, ts_data):
        data_with_nan = ts_data.copy()
        data_with_nan[10] = np.nan
        data_with_nan[50] = np.nan
        result = await _run_tool("handle_missing", data_with_nan, strategy="mean")
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["missing_count"] == 2

    async def test_select_channels(self, ts_multivariate):
        result = await _run_tool("select_channels", ts_multivariate, channels=[0, 1])
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["selected_channels"] == [0, 1]

    async def test_segment(self, ts_data):
        result = await _run_tool("segment", ts_data, window_size=50, step_size=25)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["num_segments"] > 0
        assert result.data["window_size"] == 50

    async def test_resample(self, ts_data):
        result = await _run_tool("resample", ts_data, target_length=100)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["target_length"] == 100
        assert len(result.data["data"]) == 100

    async def test_normalize(self, ts_data):
        result = await _run_tool("normalize", ts_data, method="zscore")
        assert result.success is True
        normalized = np.asarray(result.data)
        assert abs(np.mean(normalized)) < 0.01
        assert abs(np.std(normalized) - 1.0) < 0.01

    async def test_slice(self, ts_data):
        result = await _run_tool("slice", ts_data, start=10, end=50)
        assert result.success is True
        sliced = np.asarray(result.data)
        assert len(sliced) == 40

    async def test_value_at(self, ts_data):
        result = await _run_tool("value_at", ts_data, index=-1)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["resolved_index"] == len(ts_data) - 1
        assert result.data["value"] == pytest.approx(float(ts_data[-1]))

    async def test_window_stats(self, ts_data):
        result = await _run_tool("window_stats", ts_data, start=-10, end=len(ts_data))
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["length"] == 10
        assert result.data["resolved_start"] == len(ts_data) - 10
        assert result.data["resolved_end"] == len(ts_data)

    async def test_window_stats_reports_current_sequence_bounds_in_error(self, ts_data):
        result = await _run_tool("window_stats", ts_data[:24], start=144, end=168)
        assert result.success is False
        assert "Current sequence length is 24" in result.error
        assert "first 24 points are start=0, end=24" in result.error


# ---------------------------------------------------------------------------
# Profiling tools
# ---------------------------------------------------------------------------

class TestProfilingTools:
    async def test_basic_stats(self, ts_data):
        result = await _run_tool("basic_stats", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)
        for key in ("mean", "std", "min", "max", "skewness", "kurtosis"):
            assert key in result.data

    async def test_autocorrelation(self, ts_data):
        result = await _run_tool("autocorrelation", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "acf" in result.data

    async def test_stationarity_test(self, ts_data):
        result = await _run_tool("stationarity_test", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "adf" in result.data

    async def test_stationarity_test_handles_constant_series(self):
        data = np.ones(32)
        result = await _run_tool("stationarity_test", data)
        assert result.success is True
        assert result.data["is_constant"] is True
        assert result.data["conclusion"] == "constant_stationary"
        assert result.data["adf"]["is_stationary"] is True
        assert result.data["kpss"]["is_stationary"] is True

    async def test_hurst_exponent(self, ts_data):
        result = await _run_tool("hurst_exponent", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_catch22_features(self, ts_data):
        try:
            result = await _run_tool("catch22_features", ts_data)
            assert result.success is True
            assert isinstance(result.data, dict)
        except Exception:
            pytest.skip("pycatch22 not installed")

    async def test_distribution_fit(self, ts_data):
        result = await _run_tool("distribution_fit", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_entropy(self, ts_data):
        result = await _run_tool("entropy", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_spectral_density(self, ts_data):
        result = await _run_tool("spectral_density", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)


# ---------------------------------------------------------------------------
# Detection tools
# ---------------------------------------------------------------------------

class TestDetectionTools:
    async def test_detect_trend(self, ts_data):
        result = await _run_tool("detect_trend", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_detect_anomaly(self, ts_data):
        result = await _run_tool("detect_anomaly", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_detect_changepoint(self, ts_data):
        try:
            result = await _run_tool("detect_changepoint", ts_data)
            assert result.success is True
            assert isinstance(result.data, dict)
        except Exception:
            pytest.skip("ruptures not installed")

    async def test_detect_drift(self, ts_data):
        result = await _run_tool("detect_drift", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_detect_seasonality(self, ts_data):
        result = await _run_tool("detect_seasonality", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_detect_regime(self, ts_data):
        result = await _run_tool("detect_regime", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_classify_noise_type(self, ts_data):
        result = await _run_tool("classify_noise_type", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)


# ---------------------------------------------------------------------------
# Decomposition tools
# ---------------------------------------------------------------------------

class TestDecompositionTools:
    async def test_stl_decompose(self, ts_data):
        result = await _run_tool("stl_decompose", ts_data, period=20)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "trend" in result.data

    async def test_fft_spectrum(self, ts_data):
        result = await _run_tool("fft_spectrum", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_wavelet_transform(self, ts_data):
        try:
            result = await _run_tool("wavelet_transform", ts_data)
            assert result.success is True
            assert isinstance(result.data, dict)
        except Exception:
            pytest.skip("PyWavelets not installed")

    async def test_emd_decompose(self, ts_data):
        try:
            result = await _run_tool("emd_decompose", ts_data)
            assert result.success is True
            assert isinstance(result.data, dict)
        except Exception:
            pytest.skip("EMD-signal not installed")

    async def test_ssa_decompose(self, ts_data):
        result = await _run_tool("ssa_decompose", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_variance_decompose(self, ts_data):
        result = await _run_tool("variance_decompose", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)


# ---------------------------------------------------------------------------
# Relation tools (all need 2D data)
# ---------------------------------------------------------------------------

class TestRelationTools:
    async def test_correlation_matrix(self, ts_multivariate):
        result = await _run_tool("correlation_matrix", ts_multivariate, method="pearson")
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "matrix" in result.data

    async def test_lagged_correlation(self, ts_multivariate):
        result = await _run_tool("lagged_correlation", ts_multivariate)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_dtw_distance(self, ts_multivariate):
        result = await _run_tool("dtw_distance", ts_multivariate)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_mutual_information(self, ts_multivariate):
        result = await _run_tool("mutual_information", ts_multivariate)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_heterogeneity_score(self, ts_multivariate):
        result = await _run_tool("heterogeneity_score", ts_multivariate)
        assert result.success is True
        assert isinstance(result.data, dict)

    async def test_transfer_entropy(self, ts_multivariate):
        result = await _run_tool("transfer_entropy", ts_multivariate)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "transfer_entropy_xy" in result.data

    async def test_granger_causality(self, ts_multivariate):
        result = await _run_tool("granger_causality", ts_multivariate)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "results" in result.data

    async def test_cointegration_test(self, ts_multivariate):
        result = await _run_tool("cointegration_test", ts_multivariate)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "p_value" in result.data


# ---------------------------------------------------------------------------
# Prediction tools
# ---------------------------------------------------------------------------

class TestPredictionTools:
    async def test_arima_forecast(self, ts_data):
        result = await _run_tool("arima_forecast", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "forecast" in result.data
        assert len(result.data["forecast"]) == 10

    async def test_ets_forecast(self, ts_data):
        result = await _run_tool("ets_forecast", ts_data)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "forecast" in result.data


# ---------------------------------------------------------------------------
# Validation tools
# ---------------------------------------------------------------------------

class TestValidationTools:
    async def test_bound_check(self, ts_data):
        result = await _run_tool("bound_check", ts_data, lower=-100, upper=200)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["within_bounds"] is True

    async def test_bound_check_violation(self, ts_data):
        result = await _run_tool("bound_check", ts_data, lower=0, upper=10)
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["violations"] > 0

    async def test_fol_check(self, ts_data):
        result = await _run_tool(
            "fol_check",
            ts_data,
            prediction=[1, 2, 3],
            constraints=[{"type": "non_negative"}],
        )
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["passed"] is True

    async def test_fol_check_violation(self, ts_data):
        result = await _run_tool(
            "fol_check",
            ts_data,
            prediction=[-1, 2, 3],
            constraints=[{"type": "non_negative"}],
        )
        assert result.success is True
        assert result.data["passed"] is False
        assert result.data["n_violations"] == 1

    async def test_consistency_check(self, ts_data):
        historical = ts_data.tolist()
        prediction = [ts_data[-1] + 0.5, ts_data[-1] + 1.0, ts_data[-1] + 1.5]
        result = await _run_tool(
            "consistency_check",
            ts_data,
            prediction=prediction,
            historical=historical,
        )
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "consistent" in result.data
        assert "checks" in result.data

    async def test_uncertainty_quantify(self, ts_data):
        historical = ts_data.tolist()
        prediction = [ts_data[-1] + 0.5, ts_data[-1] + 1.0, ts_data[-1] + 1.5]
        result = await _run_tool(
            "uncertainty_quantify",
            ts_data,
            prediction=prediction,
            historical=historical,
        )
        assert result.success is True
        assert isinstance(result.data, dict)
        assert "lower_bound" in result.data
        assert "upper_bound" in result.data
        assert result.data["confidence"] == 0.95
