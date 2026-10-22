from __future__ import annotations

import pytest

from tarsius.meter import compose


@pytest.mark.parametrize(
    ("wheels", "dials", "value"),
    [
        # Without dials the last wheel keeps its fraction: 601.6, not 602.
        ([0.0, 6.0, 0.0, 1.6], [], 601.6),
        # A rolling last wheel carries the one above it.
        ([0.0, 0.0, 5.95, 9.5], [], 59.5),
        # Upper wheels that jump: a leading 0 must stay 0.
        ([0.0, 6.0, 2.0, 7.6], [], 627.6),
        # The maintainer's meter, dials coupled to the last wheel.
        ([9.98, 9.98, 5.9, 1.91, 7.8], [8.8, 8.4, 4.6, 6.3], 627.8846),
        # Wheels just below a mark round up through the carry.
        ([9.9, 9.9, 9.9], [], 999.9),
        ([], [3.4, 2.0], 0.32),
    ],
)
def test_values(wheels: list[float], dials: list[float], value: float) -> None:
    assert compose(wheels, dials).value == pytest.approx(value, abs=1e-3)


def test_coupled_wheel_resolves_against_the_one_below() -> None:
    # 1.95 alone rounds to 2, but the wheel below at 9.5 says 1 (still rolling).
    assert compose([1.95, 9.5]).value == pytest.approx(19.5)


def test_dial_digit_resolves_against_the_dial_below() -> None:
    # x0.1 at 4.97 is 4, not 5, while x0.01 shows 9.7.
    assert compose([3.0], [4.97, 9.7]).value == pytest.approx(3.497)


def test_residual_is_small_when_positions_agree() -> None:
    result = compose([0.0, 6.0, 2.0, 7.88], [8.85, 8.46, 4.63, 6.3])
    assert result.residual < 0.05


def test_residual_flags_contradicting_dials() -> None:
    # x0.01 at 1.2 needs x0.1 at x.12, it shows 6.6.
    assert compose([5.0], [6.6, 1.2]).residual == pytest.approx(0.48, abs=0.01)


def test_residual_flags_a_wheel_between_marks() -> None:
    assert compose([3.5, 0.0]).residual == pytest.approx(0.5)


def test_empty() -> None:
    assert compose([]).value == 0.0


def test_dial_digit_counts_back_over_zero() -> None:
    # x0.1 just past 0 while x0.01 still shows 9.7: the tenth is 9, not 0.
    assert compose([], [0.02, 9.7]).value == pytest.approx(0.997)


@pytest.mark.parametrize("last", [8.95, 8.99, 9.0, 9.03])
def test_wheel_above_the_dials_is_coupled(last: float) -> None:
    # Dials at 0.99: the last wheel is just before its carry to 9.
    result = compose([6.0, 2.0, 7.0, last], [9.9])
    assert result.value == pytest.approx(6278.99)
    assert not result.ambiguous


def test_upper_wheel_near_a_carry_is_ambiguous() -> None:
    # Jumped 2 at 2.0 or coupled 1 at 1.997, the wheel at 1.99 fits both.
    assert compose([1.99, 9.97]).ambiguous
    assert compose([2.0, 9.97], [9.7]).ambiguous


def test_upper_wheel_away_from_a_carry_is_not_ambiguous() -> None:
    assert not compose([1.99, 5.0]).ambiguous
    # Both readings fit equally well but give the same digit.
    assert not compose([3.05, 1.0]).ambiguous
    assert not compose([0.0, 6.0, 2.0, 7.88], [8.85, 8.46, 4.63, 6.3]).ambiguous


def test_jumped_wheel_wins_by_more_than_the_margin() -> None:
    # Coupled 1 at 1.9 is 0.1 off, jumped 2 fits: no ambiguity, 2 wins.
    result = compose([2.0, 9.0])
    assert result.value == pytest.approx(29.0)
    assert not result.ambiguous


def test_alternatives_fit_about_as_well() -> None:
    # A 6 read 5.897 above a units wheel at 8.45: jumped it is 0.1 off,
    # close enough; a 6 at 6.0 above 7.0 read coupled is 0.3 off, dropped.
    close = compose([0.0, 0.0, 6.0, 5.897, 8.45], [4.5, 5.0])
    assert [round(v, 2) for v, _ in close.alternatives] == [668.45]
    far = compose([0.0, 0.0, 6.0, 6.0, 7.0], [0.0, 0.0])
    assert far.alternatives == ()
