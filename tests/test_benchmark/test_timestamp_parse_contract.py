from datetime import datetime, timezone

import pytest

from timeclaw.benchmark.mtbench_adapter import _parse_timestamp_to_epoch
from timeclaw.data_integrity import DataIntegrityError


@pytest.mark.parametrize("stamp", [
    "2014-12-04T04:00:00+00:00", "2014-12-04T04:00:00Z",
    "2014-12-03T20:00:00-08:00", "2014-12-04T13:00:00+09:00",
])
def test_iso_timezone_preserves_the_real_instant(stamp):
    expected = datetime(2014, 12, 4, 4, tzinfo=timezone.utc).timestamp()
    assert _parse_timestamp_to_epoch(stamp) == expected


@pytest.mark.parametrize("stamp", ["not a date", "", None, float("nan"), True])
def test_invalid_timestamp_must_not_turn_into_1970(stamp):
    with pytest.raises(DataIntegrityError):
        _parse_timestamp_to_epoch(stamp)
