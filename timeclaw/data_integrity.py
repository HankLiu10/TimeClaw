"""Reject known contaminated inputs and explicitly quarantined artifacts.

Quarantine means an artifact is preserved for audit but cannot be used by a
new experiment. Rebuilding from admissible evidence, not deleting a marker,
is required to produce a replacement. These checks are not a general proof
that arbitrary natural-language input is free of future information.
"""
from pathlib import Path
from typing import Any

QUARANTINE_FILE = "DATA_QUARANTINE.json"


class DataIntegrityError(ValueError):
    """A known-invalid dataset or learned artifact reached an execution path."""


def assert_not_quarantined(path: str | Path) -> None:
    location = Path(path).resolve()
    for directory in (location, *location.parents):
        marker = directory / QUARANTINE_FILE
        if marker.is_file():
            raise DataIntegrityError(
                f"Artifact is quarantined: {location}; read {marker}. "
                "Rebuild from clean, time-valid source evidence before reuse."
            )


def assert_no_target_summary(value: Any, *, source: str) -> None:
    """Recognize the observed weather generator disclosure, including copied QA."""
    if isinstance(value, str):
        if "24-hour output window beginning at the event anchor" in value.lower():
            raise DataIntegrityError(f"Weather target-derived statistics in agent input: {source}")
    elif isinstance(value, dict):
        for item in value.values():
            assert_no_target_summary(item, source=source)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_target_summary(item, source=source)


def assert_weather_record_admissible(record: dict, *, source: str) -> None:
    assert_no_target_summary(record.get("text", ""), source=source)
    assert_no_target_summary(record.get("question", ""), source=source)
    status = record.get("data_integrity", {}).get("status")
    if status in {"retrospective_only", "quarantined"}:
        raise DataIntegrityError(f"Weather record is {status}: {source}")
    if "official storm data summary (noaa storm events)" in str(record.get("text", "")).lower():
        raise DataIntegrityError(f"Weather record is retrospective_only (post-event Storm Events narrative): {source}")
