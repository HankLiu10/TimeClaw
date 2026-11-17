"""Exercise the original task wrapper with evaluator-only data unavailable."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark.mtbench_support.runtime_context import prepare_task_run_kwargs
from timeclaw.data_integrity import DataIntegrityError


class _ForbiddenFuture:
    def __len__(self):
        raise AssertionError("Future targets must not be read")

    def __array__(self, *_args, **_kwargs):
        raise AssertionError("Future targets must not be read")

    def get(self, *_args, **_kwargs):
        raise AssertionError("Evaluation labels must not be read")


def _sample(**changes):
    sample = MTBenchSample(
        file="visible-request.json",
        dataset="weather_aligned_short",
        domain="weather",
        task_types=["forecast"],
        input_window=np.arange(336, dtype=float),
        output_window=None,
        text="",
        timestamps=[1704067200.0 + 600 * index for index in range(336)],
        start_year=2024,
        forecast_horizon=96,
    )
    return replace(sample, **changes)


@pytest.mark.asyncio
@pytest.mark.parametrize("future", [None, _ForbiddenFuture()])
async def test_visible_forecast_uses_public_horizon_without_future_access(future):
    sample = _sample(output_window=future, labels=_ForbiddenFuture())

    kwargs = await prepare_task_run_kwargs(sample, SimpleNamespace(), "forecast")

    assert kwargs["horizon"] == 96
    assert "exactly equal to 96" in kwargs["description"]
    assert kwargs["timestamps"] == sample.timestamps
    assert kwargs["data"] is sample.input_window
    assert "ground_truth" not in kwargs


@pytest.mark.asyncio
async def test_generic_dataset_keeps_existing_metadata_description():
    sample = _sample(
        dataset="weather",
        output_window=_ForbiddenFuture(),
        labels=_ForbiddenFuture(),
        metadata={"description": "Forecast 96 ten-minute temperature readings."},
    )

    kwargs = await prepare_task_run_kwargs(sample, SimpleNamespace(), "forecast")

    assert kwargs["description"] == sample.metadata["description"]
    assert kwargs["horizon"] == 96
    assert kwargs["timestamps"] == sample.timestamps


@pytest.mark.asyncio
@pytest.mark.parametrize("horizon", [1, 24, 72])
async def test_legacy_benchmark_forecast_length_is_preserved(horizon):
    sample = _sample(forecast_horizon=None, output_window=np.full(horizon, 987654.321))

    kwargs = await prepare_task_run_kwargs(sample, SimpleNamespace(), "forecast")

    assert kwargs["horizon"] == horizon
    assert f"exactly equal to {horizon}" in kwargs["description"]
    assert "987654.321" not in kwargs["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("horizon", [0, -1, 1.5, True, "96"])
async def test_invalid_explicit_horizon_is_rejected_before_any_model_access(horizon):
    sample = _sample(forecast_horizon=horizon, output_window=_ForbiddenFuture())

    class UnreachablePipeline:
        @property
        def router(self):
            raise AssertionError("An invalid request must fail before backend access")

    with pytest.raises(DataIntegrityError, match="positive integer"):
        await prepare_task_run_kwargs(sample, UnreachablePipeline(), "forecast")


@pytest.mark.asyncio
async def test_empty_legacy_target_cannot_silently_set_zero_horizon():
    sample = _sample(forecast_horizon=None, output_window=np.array([]))
    with pytest.raises(DataIntegrityError, match="positive integer"):
        await prepare_task_run_kwargs(sample, SimpleNamespace(), "forecast")


@pytest.mark.asyncio
async def test_finance_missing_prediction_timestamps_rejects_future_price_sentinel():
    sample = _sample(
        dataset="finance_aligned_short",
        domain="finance",
        output_window=np.full(96, 987654.321),
    )

    with pytest.raises(DataIntegrityError, match="metadata.output_timestamps") as caught:
        await prepare_task_run_kwargs(sample, SimpleNamespace(), "forecast")

    assert "987654.321" not in str(caught.value)


@pytest.mark.asyncio
async def test_finance_visible_forecast_needs_timestamps_but_no_prices_or_labels():
    sample = _sample(
        dataset="finance_aligned_short",
        domain="finance",
        output_window=_ForbiddenFuture(),
        labels=_ForbiddenFuture(),
        metadata={"output_timestamps": [1704268800.0 + 600 * index for index in range(96)]},
    )

    kwargs = await prepare_task_run_kwargs(sample, SimpleNamespace(), "forecast")

    assert kwargs["horizon"] == 96
    assert "exactly 96" in kwargs["description"]
    assert kwargs["timestamps"] == sample.timestamps
