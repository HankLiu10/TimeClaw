from __future__ import annotations

import numpy as np

from timeclaw.sbm.base import SBMConfig
from timeclaw.sbm.timesfm_adapter import TimesFMAdapter


class _DummyHParams:
    def __init__(self, horizon_len: int):
        self.horizon_len = horizon_len


class _DummyTimesFMModel:
    def __init__(self, horizon_len: int):
        self.horizon_len = horizon_len
        self.hparams = _DummyHParams(horizon_len)
        self.quantiles: list[float] = []

    def forecast(self, inputs, freq, forecast_context_len):
        assert len(inputs) == 1
        horizon = int(self.horizon_len)
        point = np.arange(horizon, dtype=np.float32)[None, :]
        full = point[..., None]
        return point, full


def test_timesfm_adapter_grows_runtime_horizon_monotonically():
    cfg = SBMConfig(
        model_name="timesfm2",
        prediction_length=128,
        context_length=32,
    )
    adapter = TimesFMAdapter(cfg)
    adapter._model = _DummyTimesFMModel(horizon_len=128)

    output = adapter.forecast(np.arange(16, dtype=np.float32), prediction_length=200)

    assert adapter.config.prediction_length == 200
    assert adapter._model.horizon_len == 200
    assert adapter._model.hparams.horizon_len == 200
    assert output.point_forecast.shape == (200,)


def test_timesfm_adapter_does_not_shrink_runtime_horizon():
    cfg = SBMConfig(
        model_name="timesfm2",
        prediction_length=200,
        context_length=32,
    )
    adapter = TimesFMAdapter(cfg)
    adapter._model = _DummyTimesFMModel(horizon_len=200)

    output = adapter.forecast(np.arange(16, dtype=np.float32), prediction_length=78)

    assert adapter.config.prediction_length == 200
    assert adapter._model.horizon_len == 200
    assert output.point_forecast.shape == (78,)
