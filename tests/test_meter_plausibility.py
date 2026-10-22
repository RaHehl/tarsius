from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tarsius import MeterDataError
from tarsius.meter import (
    Dispute,
    MeterReading,
    Plausibility,
    PlausibilityState,
    Reading,
    Rejection,
)

T0 = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def _vt(guard: Plausibility) -> tuple[float, datetime]:
    assert guard.state is not None
    return guard.state.value, guard.state.at


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
    assert _vt(guard) == (500.0005, T0)


def test_jitter_within_the_tolerance_holds_the_value() -> None:
    guard = _guard(500.0, tolerance=0.001)
    reading = guard.review(_reading(499.9995), T0 + MINUTE)
    assert reading.accepted
    assert reading.value == 500.0
    assert reading.candidate == 499.9995
    assert _vt(guard) == (500.0, T0 + MINUTE)


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
    assert _vt(guard) == (100.0, T0)
    assert guard.review(_reading(100.2), T0 + 10 * MINUTE).accepted
    assert _vt(guard) == (100.2, T0 + 10 * MINUTE)


def test_review_passes_rejected_readings_through() -> None:
    guard = _guard()
    rejected = _reading(5.0).rejected(Rejection.FAULT)
    assert guard.review(rejected, T0) is rejected
    assert _vt(guard) == (627.5, T0)


def test_agreeing_lower_readings_replace_the_value() -> None:
    guard = _guard(628.5)
    results = [
        guard.review(_reading(value), T0 + (i + 1) * MINUTE)
        for i, value in enumerate((512.345, 512.346, 512.347))
    ]
    assert [r.rejection for r in results] == [Rejection.DECREASED] * 2 + [None]
    assert results[-1].value == 512.347
    assert _vt(guard) == (512.347, T0 + 3 * MINUTE)


def test_disagreeing_lower_readings_start_over() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(512.345), T0 + MINUTE).accepted
    # Lower than the pending one: a new candidate, not the second vote.
    assert not guard.review(_reading(498.765), T0 + 2 * MINUTE).accepted
    assert guard.review(_reading(498.765), T0 + 3 * MINUTE).accepted


def test_a_plausible_reading_drops_pending_lower_ones() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(512.345), T0 + MINUTE).accepted
    assert guard.review(_reading(628.5), T0 + 2 * MINUTE).accepted
    assert not guard.review(_reading(512.345), T0 + 3 * MINUTE).accepted


def test_a_too_fast_reading_drops_pending_lower_ones() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(512.345), T0 + MINUTE).accepted
    assert guard.review(_reading(900.0), T0 + 2 * MINUTE).rejection is (
        Rejection.TOO_FAST
    )
    assert not guard.review(_reading(512.345), T0 + 3 * MINUTE).accepted


def test_accept_drops_pending_lower_ones() -> None:
    guard = _guard(628.5, confirmations=2)
    assert not guard.review(_reading(512.345), T0 + MINUTE).accepted
    guard.accept(628.5, T0 + 2 * MINUTE)
    assert not guard.review(_reading(512.345), T0 + 3 * MINUTE).accepted


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


def _meter(
    value: float,
    *alternatives: float,
    rejection: Rejection | None = None,
) -> MeterReading:
    """A reading of five wheels and four dials, like the maintainer's meter."""
    dials = tuple(Reading(0.0, 0.9) for _ in range(4))
    reading = MeterReading(
        value, value, None, 0.9, 0.1, 0.0001, (), dials, tuple(alternatives)
    )
    return reading if rejection is None else reading.rejected(rejection)


def test_alternative_that_fits_is_taken() -> None:
    # A biased tens wheel reads 658.45; jumped instead it is 668.45.
    guard = _guard(668.4)
    reading = guard.review(_meter(658.45, 668.45), T0 + 5 * MINUTE)
    assert reading.accepted
    assert reading.value == 668.45
    assert reading.candidate == 658.45
    assert _vt(guard) == (668.45, T0 + 5 * MINUTE)


def test_ambiguous_reading_is_resolved_by_the_last_value() -> None:
    guard = _guard(619.95)
    reading = _meter(620.97, 619.97, rejection=Rejection.AMBIGUOUS)
    assert guard.review(reading, T0 + 5 * MINUTE).value == 619.97


def test_smallest_rise_wins_when_several_fit() -> None:
    # After a long gap a reading 10 higher fits too; the smaller rise wins.
    guard = _guard(600.0)
    reading = guard.review(_meter(615.5, 605.5), T0 + timedelta(hours=5))
    assert reading.value == 605.5


@pytest.mark.parametrize(
    ("values", "rejection"),
    [
        ((658.45, 678.45), Rejection.AMBIGUOUS),
        ((658.45, 648.45), Rejection.DECREASED),
        ((688.45, 678.45), Rejection.TOO_FAST),
    ],
)
def test_no_fit(values: tuple[float, ...], rejection: Rejection) -> None:
    guard = _guard(668.4)
    assert guard.review(_meter(*values), T0 + 5 * MINUTE).rejection is rejection
    assert _vt(guard) == (668.4, T0)


def test_other_rejections_pass_through() -> None:
    guard = _guard(668.4)
    reading = _meter(668.45, rejection=Rejection.INCONSISTENT)
    assert guard.review(reading, T0 + 5 * MINUTE) is reading


def test_a_misread_seen_again_never_replaces_the_value() -> None:
    # Ten below, read the same way again and again: a biased wheel.
    guard = _guard(668.4)
    for i in range(1, 10):
        reading = guard.review(_meter(658.41), T0 + i * 5 * MINUTE)
        assert reading.rejection is Rejection.DECREASED
    assert _vt(guard) == (668.4, T0)


@pytest.mark.parametrize(("last", "new"), [(62700.0, 3.0), (6200.0, 3.0)])
def test_new_meter_is_not_a_wrap(last: float, new: float) -> None:
    guard = _guard(last, period=100000.0 if last > 10000 else 10000.0)
    results = [
        guard.review(_meter(new + 0.001 * i), T0 + i * 5 * MINUTE) for i in (1, 2, 3)
    ]
    assert [r.rejection for r in results] == [Rejection.DECREASED] * 2 + [None]
    assert results[-1].value == pytest.approx(new + 0.003)


def test_clock_that_was_ahead_is_repaired() -> None:
    guard = _guard(500.0)
    # The state was saved an hour ahead; a reading now is checked from now.
    now = T0 - timedelta(hours=1)
    assert guard.check(500.0, now) is None
    guard.review(_reading(500.0), now)
    assert _vt(guard) == (500.0, now)
    reading = guard.review(_reading(500.3), now + 5 * MINUTE)
    assert reading.accepted
    assert _vt(guard) == (500.3, now + 5 * MINUTE)


def test_small_clock_step_back_allows_no_rise() -> None:
    guard = _guard(500.0)
    assert guard.check(500.1, T0 - 5 * MINUTE) is Rejection.TOO_FAST


def test_after_a_long_gap_a_new_meter_is_still_taken() -> None:
    guard = _guard(628.5)
    later = T0 + timedelta(days=30)
    results = [guard.review(_meter(512.345), later + i * 5 * MINUTE) for i in (1, 2, 3)]
    assert [r.rejection for r in results] == [Rejection.DECREASED] * 2 + [None]


def test_small_wheels_do_not_block_a_new_meter() -> None:
    # Three decimal wheels: a drop of 3.456 is no misread of the 0.01 wheel.
    guard = _guard(123.456)
    results = [guard.review(_reading(120.0), T0 + i * 5 * MINUTE) for i in (1, 2, 3)]
    assert [r.rejection for r in results] == [Rejection.DECREASED] * 2 + [None]
    # A drop of a whole unit is a misread of the units wheel.
    guard = _guard(123.456)
    for i in (1, 2, 3, 4):
        assert guard.review(_reading(122.456), T0 + i * 5 * MINUTE).rejection is (
            Rejection.DECREASED
        )


def test_a_misread_at_rest_stays_out_for_hours() -> None:
    guard = _guard(668.4)
    for i in range(1, 30):
        later = T0 + i * timedelta(hours=1)
        assert guard.review(_meter(658.4), later).rejection is Rejection.DECREASED


def test_a_misread_with_jitter_below_the_last_value() -> None:
    guard = _guard(668.4, tolerance=0.001)
    for i in range(1, 6):
        reading = guard.review(_meter(658.3999), T0 + i * 5 * MINUTE)
        assert reading.rejection is Rejection.DECREASED


def test_a_misread_stays_out_while_the_true_value_moves_on() -> None:
    # Seen once right after the last value, the offset is remembered.
    guard = _guard(668.4)
    assert guard.review(_meter(658.4), T0 + 5 * MINUTE).rejection is (
        Rejection.DECREASED
    )
    for i in range(1, 20):
        later = T0 + i * timedelta(hours=6)
        reading = guard.review(_meter(658.4 + 0.1 * i), later)
        assert reading.rejection is Rejection.DECREASED
    guard.accept(500.0, T0 + timedelta(days=6))
    assert guard.state is not None
    assert not guard.state.misreads


def test_jitter_after_a_gap_holds_instead_of_a_higher_reading() -> None:
    # A coupled counter after a dark night: the composed value is 0.0003
    # low, the jumped reading of its tens wheel 10 higher would fit now.
    guard = _guard(627.7, tolerance=0.0001)
    later = T0 + timedelta(hours=3)
    reading = guard.review(_meter(627.6997, 637.6997, 617.6997), later)
    assert reading.value == 627.7
    assert guard.state is not None
    assert (guard.state.value, guard.state.at, guard.state.alternative) == (
        627.7,
        later,
        False,
    )


def test_lower_alternative_wins_over_a_fitting_higher_value() -> None:
    # A coupled wheel read a tenth high composes 10 too much after a gap.
    guard = _guard(600.0)
    reading = guard.review(_meter(610.2, 600.2), T0 + timedelta(hours=3))
    assert reading.value == 600.2


def test_a_value_from_an_alternative_can_be_replaced() -> None:
    guard = _guard(668.4)
    guard.review(_meter(658.45, 668.45), T0 + 5 * MINUTE)
    assert guard.state is not None
    assert guard.state.alternative
    results = [
        guard.review(_meter(658.46 + 0.001 * i), T0 + (i + 2) * 5 * MINUTE)
        for i in range(3)
    ]
    assert [r.rejection for r in results] == [Rejection.DECREASED] * 2 + [None]
    assert results[-1].value == pytest.approx(658.462)


def test_lasting_misread_drops_become_a_dispute() -> None:
    guard = _guard(668.4)
    for i in range(1, 80):
        reading = guard.review(_meter(658.4), T0 + i * 5 * MINUTE)
        assert reading.rejection is Rejection.DECREASED
        if i * 5 < 6 * 60:
            assert guard.dispute is None
    dispute = guard.dispute
    assert dispute is not None
    assert (dispute.since, dispute.count, dispute.value) == (T0 + 5 * MINUTE, 79, 658.4)
    # Readings that agree again settle it.
    guard.review(_meter(668.41), T0 + 80 * 5 * MINUTE)
    assert guard.dispute is None


def test_few_misread_drops_are_no_dispute() -> None:
    guard = _guard(668.4)
    for i in (1, 2):
        guard.review(_meter(658.4), T0 + i * timedelta(hours=4))
    assert guard.dispute is None


def test_accept_settles_a_dispute() -> None:
    guard = _guard(668.4)
    for i in range(1, 80):
        guard.review(_meter(658.4), T0 + i * 5 * MINUTE)
    guard.accept(658.4, T0 + timedelta(days=1))
    assert guard.dispute is None


def test_pending_lower_readings_survive_a_restart() -> None:
    guard = _guard(628.5)
    for i in (1, 2):
        guard.review(_reading(512.345), T0 + i * MINUTE)
    restarted = Plausibility(state=guard.state)
    assert restarted.review(_reading(512.345), T0 + 3 * MINUTE).accepted


def test_full_state_round_trip() -> None:
    state = PlausibilityState(
        627.8846,
        T0,
        alternative=True,
        pending=((620.0, T0 + MINUTE),),
        misreads=(10.0,),
        dispute=Dispute(T0, T0 + MINUTE, 2, 617.8),
    )
    restored = PlausibilityState.from_dict(json.loads(json.dumps(state.as_dict())))
    assert restored == state


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"alternative": "yes"}, "alternative"),
        ({"pending": "x"}, "pending must be a list"),
        ({"pending": [[1.0]]}, "pending must be"),
        ({"pending": [[1.0, "soon"]]}, "pending time"),
        ({"misreads": [-10.0]}, "positive"),
        ({"misreads": ["10"]}, "misread"),
        ({"dispute": {"since": T0.isoformat()}}, "dispute is missing"),
        ({"dispute": [1]}, "dispute must be a mapping"),
        (
            {
                "dispute": {
                    "since": T0.isoformat(),
                    "last": T0.isoformat(),
                    "count": 0,
                    "value": 1.0,
                }
            },
            "count",
        ),
    ],
)
def test_invalid_state_fields(change: dict[str, Any], match: str) -> None:
    data = PlausibilityState(1.0, T0).as_dict() | change
    with pytest.raises(MeterDataError, match=f"invalid plausibility state: .*{match}"):
        PlausibilityState.from_dict(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pending": ((1.0,),)}, "pending must be"),
        ({"dispute": {"count": 1}}, "dispute must be a Dispute"),
    ],
)
def test_invalid_state_objects(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(MeterDataError, match=match):
        PlausibilityState(1.0, T0, **kwargs)


def test_new_meter_taken_with_a_clock_stepped_back_keeps_the_time() -> None:
    guard = _guard(628.5)
    for i in (3, 2, 1):
        reading = guard.review(_reading(512.345), T0 - i * MINUTE)
    assert reading.accepted
    assert _vt(guard) == (512.345, T0)
