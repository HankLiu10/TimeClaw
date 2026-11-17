from __future__ import annotations

from timeclaw.sbm.moirai_adapter import MOIRAI_DEFAULT_CONFIG


def test_moirai_default_checkpoint_is_r_large():
    assert MOIRAI_DEFAULT_CONFIG.model_name == "moirai1_1_r_base"
    assert MOIRAI_DEFAULT_CONFIG.model_path == "Salesforce/moirai-1.1-R-large"
