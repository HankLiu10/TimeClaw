from __future__ import annotations

import numpy as np
import pytest

from timeclaw.sbm.timer_adapter import TIMER_DEFAULT_CONFIG, TimerAdapter


def test_timer_impute_is_explicitly_not_implemented():
    adapter = TimerAdapter(TIMER_DEFAULT_CONFIG)
    adapter._model = object()

    with pytest.raises(NotImplementedError, match="not implemented"):
        adapter.impute(np.array([1.0, 2.0]), np.array([False, True]))
