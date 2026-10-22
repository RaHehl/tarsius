from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tarsius.meter import (
    MeterReading,
    Plausibility,
    PlausibilityState,
    Rejection,
)

T0 = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


def _reading(value: float) -> MeterReading:
    return MeterReading(value, value, None, 0.9, 0.0, 0.001)


def test_first_value_is_plausible() -> None:
    assert Plausibility().check(627.0, T0) is None


def test_decrease_is_rejected() -> None:
    guard = Plausibility(state=PlausibilityState(627.5, T0))
    assert guard.check(627.4, T0 + timedelta(minutes=5)) is Rejection.DECREASED


def test_tolerance_allows_jitter() -> None:
    guard = Plausibility(tolerance=0.001, state=PlausibilityState(627.5, T0))
    assert guard.check(627.4995, T0 + timedelta(minutes=5)) is None


def test_rate_limit() -> None:
    guard = Plausibility(max_rate=6.0, state=PlausibilityState(100.0, T0))
    later = T0 + timedelta(minutes=10)
    assert guard.check(100.9, later) is None
    assert guard.check(101.1, later) is Rejection.TOO_FAST


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(days=2)])
def test_no_history_to_compare(delta: timedelta) -> None:
    # Same moment, or so long ago that anything may have happened since.
    guard = Plausibility(state=PlausibilityState(500.0, T0))
    assert guard.check(1.0, T0 + delta) is None


def test_review_remembers_only_plausible_values() -> None:
    guard = Plausibility(max_rate=6.0)
    assert guard.review(_reading(100.0), T0).accepted
    wrong = guard.review(_reading(180.0), T0 + timedelta(minutes=5))
    assert wrong.rejection is Rejection.TOO_FAST
    assert wrong.value is None
    assert wrong.candidate == 180.0
    assert guard.state == PlausibilityState(100.0, T0)
    assert guard.review(_reading(100.2), T0 + timedelta(minutes=10)).accepted
    assert guard.state == PlausibilityState(100.2, T0 + timedelta(minutes=10))


def test_review_passes_rejected_readings_through() -> None:
    guard = Plausibility()
    rejected = _reading(5.0).rejected(Rejection.FAULT)
    assert guard.review(rejected, T0) is rejected
    assert guard.state is None


def test_state_round_trip() -> None:
    state = PlausibilityState(627.8846, T0)
    assert PlausibilityState.from_dict(state.as_dict()) == state


@pytest.mark.parametrize("data", [{}, {"value": 1.0}, {"value": 1.0, "at": None}])
def test_invalid_state(data: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid plausibility state"):
        PlausibilityState.from_dict(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"max_rate": 0}, "max_rate"), ({"tolerance": -1}, "tolerance")],
)
def test_invalid_settings(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Plausibility(**kwargs)
