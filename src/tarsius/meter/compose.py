"""
Turn wheel positions and dial angles into one number.

Composition is not concatenation: a wheel at 1.7 shows 1 or 2 depending on
the wheels below it. On a coupled counter wheel k sits at its digit plus the
value of everything below it, divided by ten::

    p_k = D_k + R_(k-1) / 10

so the digit is recovered by subtracting what the lower wheels contribute.
Many counters couple only the last wheel and let the upper ones jump, so each
upper wheel is resolved both ways and the reading that explains its observed
position better wins. Dials are geared and always coupled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .engine import circular_distance


@dataclass(frozen=True, slots=True)
class Composition:
    """A composed value and how well the positions fit each other."""

    value: float
    # Largest distance between an observed position and the position the
    # composed value implies for it, in digits; 0 for a perfect fit, at
    # most 0.5. Large values mean the wheels contradict each other.
    residual: float


def compose(wheels: Sequence[float], dials: Sequence[float] = ()) -> Composition:
    """
    Compose positions, most significant first, into one value.

    ``wheels`` are the wheel positions (0 to below 10), ``dials`` the pointer
    positions below them, dial k counting 10^-(k+1) units. The fraction of
    the least significant element is kept: it is real information on a
    rolling wheel or dial.
    """
    residual = 0.0
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
    for position in rest:
        coupled = round(position - below / 10.0) % 10
        jumped = round(position) % 10
        coupled_error = circular_distance(position, coupled + below / 10.0)
        jumped_error = circular_distance(position, jumped)
        if coupled_error <= jumped_error:
            digit, below = coupled, coupled + below / 10.0
        else:
            digit, below = jumped, float(jumped)
        residual = max(residual, min(coupled_error, jumped_error))
        digits.append(digit)
    whole = 0
    for digit in reversed(digits):
        whole = whole * 10 + digit
    return Composition(whole + fraction, residual)
