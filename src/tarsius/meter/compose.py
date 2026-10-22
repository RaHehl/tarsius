"""
Turn wheel positions and dial angles into one number.

Composition is not concatenation: a wheel at 1.7 shows 1 or 2 depending on
the wheels below it. On a coupled counter wheel k sits at its digit plus the
value of everything below it, divided by ten::

    p_k = D_k + R_(k-1) / 10

so the digit is recovered by subtracting what the lower wheels contribute.
Many counters couple only the last wheel and let the upper ones jump, so each
upper wheel is resolved both ways and the reading that explains its observed
position better wins. Dials are geared and always coupled, and so is the
wheel right above them.

Near a carry the two ways can disagree by one digit while fitting about
equally well: a coupled wheel just before its carry sits where a jumped one
sits just after it. Such a reading is ambiguous rather than a coin toss.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .engine import circular_distance

# Two readings of an upper wheel that differ by a digit and fit closer than
# this, in digits, are ambiguous.
AMBIGUITY = 0.05


@dataclass(frozen=True, slots=True)
class Composition:
    """A composed value and how well the positions fit each other."""

    value: float
    # Largest distance between an observed position and the position the
    # composed value implies for it, in digits; 0 for a perfect fit, at
    # most 0.5. Large values mean the wheels contradict each other.
    residual: float
    # An upper wheel fits two digits about equally well, near a carry.
    ambiguous: bool = False


def compose(wheels: Sequence[float], dials: Sequence[float] = ()) -> Composition:
    """
    Compose positions, most significant first, into one value.

    ``wheels`` are the wheel positions (0 to below 10), ``dials`` the pointer
    positions below them, dial k counting 10^-(k+1) units. The fraction of
    the least significant element is kept: it is real information on a
    rolling wheel or dial.
    """
    residual = 0.0
    ambiguous = False
    # Value of everything below the current element, in its own digit units.
    below = 0.0
    fraction = 0.0
    if dials:
        below = dials[-1] % 10.0
        for position in reversed(dials[:-1]):
            digit = round(position - below / 10.0) % 10
            residual = max(residual, circular_distance(position, digit + below / 10))
            below = digit + below / 10.0
        fraction = below / 10.0
    rest = list(reversed(wheels))
    digits: list[int] = []
    if not dials and rest:
        # Without dials the last wheel is the resolution: 601.6, not 602.
        position = rest.pop(0) % 10.0
        digits.append(math.floor(position))
        fraction = position - math.floor(position)
        below = position
    else:
        below = fraction * 10.0
    for index, position in enumerate(rest):
        coupled = round(position - below / 10.0) % 10
        coupled_error = circular_distance(position, coupled + below / 10.0)
        if dials and index == 0:
            # The wheel above the dials is geared to them, it cannot jump.
            digit, below, error = coupled, coupled + below / 10.0, coupled_error
        else:
            jumped = round(position) % 10
            jumped_error = circular_distance(position, jumped)
            if coupled_error <= jumped_error:
                digit, below, error = coupled, coupled + below / 10.0, coupled_error
            else:
                digit, below, error = jumped, float(jumped), jumped_error
            if coupled != jumped and abs(coupled_error - jumped_error) < AMBIGUITY:
                ambiguous = True
        residual = max(residual, error)
        digits.append(digit)
    whole = 0
    for digit in reversed(digits):
        whole = whole * 10 + digit
    return Composition(whole + fraction, residual, ambiguous)
