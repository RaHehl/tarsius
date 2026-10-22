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
