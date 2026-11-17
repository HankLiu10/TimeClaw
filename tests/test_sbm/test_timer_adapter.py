from __future__ import annotations

import numpy as np
import pytest

from timeclaw.sbm.timer_adapter import TIMER_DEFAULT_CONFIG, TimerAdapter


def test_timer_impute_is_explicitly_not_implemented():
    adapter = TimerAdapter(TIMER_DEFAULT_CONFIG)
    adapter._model = object()

    with pytest.raises(NotImplementedError, match="not implemented"):
        adapter.impute(np.array([1.0, 2.0]), np.array([False, True]))


def test_sundial_profile_advertises_only_implemented_capabilities():
    adapter=TimerAdapter(TIMER_DEFAULT_CONFIG)
    assert adapter.get_profile()['tasks']==['forecast']


def test_unimplemented_sundial_imputation_never_loads_weights(monkeypatch):
    adapter=TimerAdapter(TIMER_DEFAULT_CONFIG)
    def reject_load():raise AssertionError('Unsupported imputation must not load model weights')
    monkeypatch.setattr(adapter,'load',reject_load)
    with pytest.raises(NotImplementedError,match='not implemented'):
        adapter.impute(np.array([1.0,2.0]),np.array([False,True]))
