"""Regression checks for explicit model environments and failed forecasts.

No model is loaded and no subprocess or network connection is created here.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from timeclaw.core.config import TimeClawConfig
from timeclaw.sbm import moirai_adapter, service_manager


@pytest.fixture
def model_launcher(monkeypatch, tmp_path):
    """Exercise the real launcher without creating a model service."""
    config = TimeClawConfig(workspace_dir=str(tmp_path))
    monkeypatch.setattr(service_manager, "_server_processes", {})
    monkeypatch.setattr(service_manager, "_server_urls", {})
    monkeypatch.setattr(service_manager, "_server_log_paths", {})
    monkeypatch.setattr(service_manager, "_reuse_existing_healthy_listener", lambda **_: None)
    monkeypatch.setattr(service_manager, "_ensure_target_port_available", lambda *_, **__: None)
    monkeypatch.setattr(service_manager, "_server_is_healthy", lambda _: True)
    process = SimpleNamespace(poll=lambda: None)
    launch = Mock(return_value=process)
    monkeypatch.setattr(service_manager.subprocess, "Popen", launch)
    yield config, launch
    for call in launch.call_args_list:
        call.kwargs["stdout"].close()


def test_model_launch_honors_each_explicit_interpreter(model_launcher):
    config, launch = model_launcher
    # Parse the same nested config field used by YAML configuration, including
    # two models whose dependencies require different Python environments.
    config = TimeClawConfig.model_validate({
        "workspace_dir": config.workspace_dir,
        "sbm": {"managed_server_python": {
            "moirai1_1_r_base": "/isolated/moirai/bin/python",
            "aurora": "/isolated/aurora/bin/python",
        }},
    })
    for ordinal, name in enumerate(("moirai1_1_r_base", "aurora")):
        service_manager._start_one_model_server(config, model_name=name, ordinal=ordinal)

    commands = [call.args[0] for call in launch.call_args_list]
    assert [command[0] for command in commands] == [
        "/isolated/moirai/bin/python", "/isolated/aurora/bin/python",
    ]
    assert [command[command.index("--models") + 1] for command in commands] == [
        "moirai1_1_r_base", "aurora",
    ]
    assert [command[command.index("--port") + 1] for command in commands] == ["8100", "8101"]


def test_unspecified_model_keeps_calling_interpreter(model_launcher):
    config, launch = model_launcher
    config.sbm.managed_server_python = {"aurora": "/isolated/aurora/bin/python"}
    service_manager._start_one_model_server(config, model_name="chronos2", ordinal=0)
    assert launch.call_args.args[0][0] == service_manager.sys.executable


def test_missing_explicit_interpreter_is_not_replaced(model_launcher):
    config, launch = model_launcher
    config.sbm.managed_server_python = {"aurora": "/missing/env/bin/python"}
    launch.side_effect = FileNotFoundError("requested interpreter is absent")
    with pytest.raises(FileNotFoundError, match="requested interpreter is absent"):
        service_manager._start_one_model_server(config, model_name="aurora", ordinal=0)
    assert launch.call_count == 1
    assert launch.call_args.args[0][0] == "/missing/env/bin/python"
    assert not service_manager._server_processes
    assert launch.call_args.kwargs["stdout"].closed


@pytest.mark.parametrize("forecasts_empty", [True, False])
def test_moirai_empty_predictor_cannot_become_a_successful_zero_forecast(
    monkeypatch, forecasts_empty,
):
    # Keep the real adapter dataset/forecast conversion path; replace only the
    # external neural predictor. Zero is itself a valid prediction, so the
    # success case verifies real zero output is retained, not forbidden.
    forecasts = [] if forecasts_empty else [SimpleNamespace(samples=np.zeros((2, 3)))]
    predictor = SimpleNamespace(predict=Mock(return_value=iter(forecasts)))
    model = SimpleNamespace(create_predictor=Mock(return_value=predictor))
    model_factory = Mock(return_value=model)
    monkeypatch.setattr(moirai_adapter, "_ensure_moirai_imports", lambda: (model_factory, object))
    adapter = moirai_adapter.MoiraiAdapter(moirai_adapter.MOIRAI_DEFAULT_CONFIG.model_copy())
    adapter._model = True
    adapter._module = object()

    if forecasts_empty:
        with pytest.raises(RuntimeError, match="Moirai produced no forecast"):
            adapter.forecast(np.arange(8.0), prediction_length=3)
    else:
        output = adapter.forecast(np.arange(8.0), prediction_length=3)
        np.testing.assert_array_equal(output.point_forecast, np.zeros(3))
        assert output.metadata["prediction_length"] == 3
    predictor.predict.assert_called_once()
    assert model_factory.call_args.kwargs["prediction_length"] == 3
