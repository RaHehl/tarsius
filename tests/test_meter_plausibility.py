from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tarsius import MeterDataError
from tarsius.meter import (
    MeterReading,
    Plausibility,
    PlausibilityState,
    Rejection,
)

T0 = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def _reading(value: float) -> MeterReading:
    return MeterReading(value, value, None, 0.9, 0.0, 0.001)


def _guard(value: float = 627.5, **kwargs: Any) -> Plausibility:
    return Plausibility(state=PlausibilityState(value, T0), **kwargs)


def test_nothing_is_accepted_before_a_confirmed_value() -> None:
    guard = Plausibility()
    assert guard.check(627.0, T0) is Rejection.UNCONFIRMED
    assert guard.review(_reading(627.0), T0).rejection is Rejection.UNCONFIRMED
    assert guard.state is None
    guard.accept(627.0, T0)
    assert guard.review(_reading(627.01), T0 + MINUTE).accepted


def test_decrease_is_rejected() -> None:
    assert _guard().check(627.4, T0 + 5 * MINUTE) is Rejection.DECREASED


def test_tolerance_allows_jitter() -> None:
    guard = _guard(tolerance=0.001)
    assert guard.check(627.4995, T0 + 5 * MINUTE) is None
    assert guard.check(627.4989, T0 + 5 * MINUTE) is Rejection.DECREASED


def test_rate_limit() -> None:
    guard = _guard(100.0, max_rate=6.0)
    later = T0 + 10 * MINUTE
    assert guard.check(100.9, later) is None
    assert guard.check(101.1, later) is Rejection.TOO_FAST


def test_rate_limit_includes_the_tolerance() -> None:
    guard = _guard(100.0, max_rate=6.0, tolerance=0.25)
    later = T0 + 10 * MINUTE
    assert guard.check(101.2, later) is None
    assert guard.check(101.3, later) is Rejection.TOO_FAST


def test_default_rate_suits_household_water() -> None:
    guard = _guard(100.0)
    assert guard.max_rate == 4.0
    assert guard.check(100.3, T0 + 5 * MINUTE) is None
    assert guard.check(101.0, T0 + 5 * MINUTE) is Rejection.TOO_FAST


@pytest.mark.parametrize("delta", [timedelta(0), -timedelta(hours=1)])
def test_no_time_passed_allows_no_rise(delta: timedelta) -> None:
    guard = _guard(500.0, tolerance=0.001)
    assert guard.check(1e6, T0 + delta) is Rejection.TOO_FAST
    assert guard.check(1.0, T0 + delta) is Rejection.DECREASED
    assert guard.check(500.001, T0 + delta) is None


def test_clock_stepping_back_keeps_the_state_time() -> None:
    guard = _guard(500.0, tolerance=0.001)
    assert guard.review(_reading(500.0005), T0 - MINUTE).accepted
    assert guard.state == PlausibilityState(500.0005, T0)


def test_jitter_within_the_tolerance_holds_the_value() -> None:
    guard = _guard(500.0, tolerance=0.001)
    reading = guard.review(_reading(499.9995), T0 + MINUTE)
    assert reading.accepted
    assert reading.value == 500.0
    assert reading.candidate == 499.9995
    assert guard.state == PlausibilityState(500.0, T0)


def test_long_gap_is_still_checked() -> None:
    guard = _guard(500.0)
    later = T0 + timedelta(days=2)
    assert guard.check(1.0, later) is Rejection.DECREASED
    assert guard.check(500.0 + 4.0 * 48 + 0.01, later) is Rejection.TOO_FAST
    assert guard.check(500.0 + 4.0 * 48, later) is None


def test_review_remembers_only_plausible_values() -> None:
    guard = _guard(100.0, max_rate=6.0)
    wrong = guard.review(_reading(180.0), T0 + 5 * MINUTE)
    assert wrong.rejection is Rejection.TOO_FAST
    assert wrong.value is None
    assert wrong.candidate == 180.0
    assert guard.state == PlausibilityState(100.0, T0)
    assert guard.review(_reading(100.2), T0 + 10 * MINUTE).accepted
    assert guard.state == PlausibilityState(100.2, T0 + 10 * MINUTE)


def test_review_passes_rejected_readings_through() -> None:
    guard = _guard()
    rejected = _reading(5.0).rejected(Rejection.FAULT)
    assert guard.review(rejected, T0) is rejected
    assert guard.state == PlausibilityState(627.5, T0)


def test_agreeing_lower_readings_replace_the_value() -> None:
    guard = _guard(628.5)
    results = [
        guard.review(_reading(value), T0 + (i + 1) * MINUTE)
        for i, value in enumerate((627.50, 627.51, 627.52))
    ]
    assert [r.rejection for r in results] == [Rejection.DECREASED] * 2 + [None]
    assert results[-1].value == 627.52
    assert guard.state == PlausibilityState(627.52, T0 + 3 * MINUTE)


def test_disagreeing_lower_readings_start_over() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(627.5), T0 + MINUTE).accepted
    # Lower than the pending one: a new candidate, not the second vote.
    assert not guard.review(_reading(600.0), T0 + 2 * MINUTE).accepted
    assert guard.review(_reading(600.0), T0 + 3 * MINUTE).accepted


def test_a_plausible_reading_drops_pending_lower_ones() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(627.5), T0 + MINUTE).accepted
    assert guard.review(_reading(628.5), T0 + 2 * MINUTE).accepted
    assert not guard.review(_reading(627.5), T0 + 3 * MINUTE).accepted


def test_a_too_fast_reading_drops_pending_lower_ones() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(627.5), T0 + MINUTE).accepted
    assert guard.review(_reading(900.0), T0 + 2 * MINUTE).rejection is (
        Rejection.TOO_FAST
    )
    assert not guard.review(_reading(627.5), T0 + 3 * MINUTE).accepted


def test_accept_drops_pending_lower_ones() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(627.5), T0 + MINUTE).accepted
    guard.accept(628.5, T0 + 2 * MINUTE)
    assert not guard.review(_reading(627.5), T0 + 3 * MINUTE).accepted


def test_wrap_counts_on() -> None:
    guard = _guard(99999.8, period=100000.0)
    reading = guard.review(_reading(0.1), T0 + 5 * MINUTE)
    assert reading.value == pytest.approx(100000.1)
    assert reading.candidate == 0.1
    assert guard.state is not None
    assert guard.state.value == pytest.approx(100000.1)
    assert guard.review(_reading(0.2), T0 + 10 * MINUTE).value == pytest.approx(
        100000.2
    )


def test_wrap_does_not_hide_a_decrease() -> None:
    guard = _guard(5.0, period=100000.0)
    assert guard.check(4.0, T0 + 5 * MINUTE) is Rejection.DECREASED
    assert guard.check(99999.0, T0 + 5 * MINUTE) is Rejection.DECREASED


def test_without_period_a_wrap_is_a_decrease() -> None:
    assert _guard(99999.8).check(0.1, T0 + 5 * MINUTE) is Rejection.DECREASED


def test_state_round_trip() -> None:
    state = PlausibilityState(627.8846, T0)
    assert PlausibilityState.from_dict(state.as_dict()) == state


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"value": 1.0},
        {"value": 1.0, "at": None},
        {"value": 1.0, "at": "garbage"},
        {"value": 1.0, "at": "2026-10-22T12:00:00"},
        {"value": float("nan"), "at": T0.isoformat()},
        {"value": "1.0", "at": T0.isoformat()},
        {"value": True, "at": T0.isoformat()},
        [1, 2],
    ],
)
def test_invalid_state(data: Any) -> None:
    with pytest.raises(MeterDataError, match="invalid plausibility state"):
        PlausibilityState.from_dict(data)


def test_naive_times_are_rejected() -> None:
    naive = datetime(2026, 10, 22, 12, 0)
    with pytest.raises(MeterDataError, match="timezone-aware"):
        PlausibilityState(1.0, naive)
    with pytest.raises(MeterDataError, match="timezone-aware"):
        _guard().check(627.6, naive)
    with pytest.raises(MeterDataError, match="timezone-aware"):
        Plausibility().accept(627.6, naive)


def test_times_must_be_datetimes() -> None:
    with pytest.raises(MeterDataError, match="must be a datetime"):
        PlausibilityState(1.0, "2026-10-22")  # type: ignore[arg-type]


def test_values_must_be_finite() -> None:
    with pytest.raises(MeterDataError, match="finite"):
        _guard().check(float("nan"), T0 + MINUTE)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_rate": 0}, "max_rate"),
        ({"max_rate": -1.0}, "max_rate"),
        ({"max_rate": float("nan")}, "max_rate"),
        ({"max_rate": float("inf")}, "max_rate"),
        ({"tolerance": -1}, "tolerance"),
        ({"tolerance": float("nan")}, "tolerance"),
        ({"period": 0.0}, "period"),
        ({"period": float("inf")}, "period"),
        ({"confirmations": 1}, "confirmations"),
        ({"confirmations": 2.5}, "confirmations"),
        ({"state": {"value": 1.0}}, "state"),
    ],
)
def test_invalid_settings(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(MeterDataError, match=match):
        Plausibility(**kwargs)
