"""Strict checks for values that come from storage or a form."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ..errors import MeterDataError
from ..roi import Roi


def number(value: object, name: str) -> float:
    """Return ``value`` as a finite float; ints count, bools and strings do not."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MeterDataError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise MeterDataError(f"{name} must be finite, got {value!r}")
    return float(value)


def integer(value: object, name: str) -> int:
    """Return ``value`` as an int; bools and floats do not count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeterDataError(f"{name} must be an integer, got {value!r}")
    return value


def flag(value: object, name: str) -> bool:
    """Return ``value`` if it is a bool; ``"false"`` or 0 are not."""
    if not isinstance(value, bool):
        raise MeterDataError(f"{name} must be true or false, got {value!r}")
    return value


def aware(value: object, name: str) -> datetime:
    """Return ``value`` if it is a timezone-aware datetime."""
    if not isinstance(value, datetime):
        raise MeterDataError(f"{name} must be a datetime, got {value!r}")
    if value.utcoffset() is None:
        raise MeterDataError(f"{name} must be timezone-aware, got {value!r}")
    return value


def mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return ``value`` if it is a mapping."""
    if not isinstance(value, Mapping):
        raise MeterDataError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def roi(value: object, name: str) -> Roi:
    """Return a region from four finite numbers; raises RoiError if out of frame."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise MeterDataError(f"{name} must be four numbers, got {value!r}")
    if len(value) != 4:
        raise MeterDataError(f"{name} must be four numbers, got {len(value)}")
    left, top, right, bottom = (number(v, name) for v in value)
    return Roi(left, top, right, bottom)


def within(value: float, low: float, high: float, name: str) -> float:
    """Return ``value`` if ``low <= value <= high``."""
    if not low <= value <= high:
        raise MeterDataError(f"{name} must be {low} to {high}, got {value!r}")
    return value
