"""Tests for SBM base classes and registry."""

import numpy as np
import pytest
from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask
from timeclaw.sbm.registry import SBMRegistry


class MockSBM(SBMBase):
    name = "mock_sbm"
    supported_tasks = [SBMTask.FORECAST]

    def load(self):
        self._model = "loaded"

    def forecast(self, context, prediction_length=None, **kwargs):
        pred_len = prediction_length or self.config.prediction_length
        return SBMOutput(
            point_forecast=np.full(pred_len, np.mean(context)),
            metadata={"model": "mock"},
        )


def test_sbm_config():
    config = SBMConfig(model_name="test", device="cpu")
    assert config.model_name == "test"
    assert config.context_length == 512


def test_sbm_output():
    output = SBMOutput(point_forecast=np.array([1.0, 2.0, 3.0]))
    assert output.to_numpy() is not None
    assert len(output.to_numpy()) == 3


def test_mock_sbm():
    config = SBMConfig(model_name="mock")
    sbm = MockSBM(config)
    assert not sbm.is_loaded
    sbm.load()
    assert sbm.is_loaded
    result = sbm.forecast(np.array([1.0, 2.0, 3.0]), prediction_length=10)
    assert len(result.point_forecast) == 10
    sbm.unload()
    assert not sbm.is_loaded


def test_sbm_profile():
    sbm = MockSBM(SBMConfig(model_name="mock"))
    profile = sbm.get_profile()
    assert profile["name"] == "mock_sbm"
    assert "forecast" in profile["tasks"]


def test_registry():
    reg = SBMRegistry()
    reg.register("mock", MockSBM, SBMConfig(model_name="mock"))
    assert "mock" in reg.list_available()
    inst = reg.get("mock")
    assert isinstance(inst, MockSBM)
    reg.unload_all()


def test_registry_missing():
    reg = SBMRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")
