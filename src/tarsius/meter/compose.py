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
    # Other values the positions allow when upper wheels are read the other
    # way, coupled instead of jumped or back, as (value, residual), best
    # first, at most ALTERNATIVE_MARGIN worse. Near a carry, or with a reader
    # biased by a tenth of a digit,
    # one of them can be the right one; Plausibility picks it by the last
    # value.
    alternatives: tuple[tuple[float, float], ...] = ()


# Alternatives fit at most this much worse than the composed value.
ALTERNATIVE_MARGIN = 0.15
# At most this many alternatives are searched; each upper wheel near a carry
# doubles them.
MAX_ALTERNATIVES = 64


def compose(
    wheels: Sequence[float], dials: Sequence[float] = (), *, max_residual: float = 0.5
) -> Composition:
    """
    Compose positions, most significant first, into one value.

    ``wheels`` are the wheel positions (0 to below 10), ``dials`` the pointer
    positions below them, dial k counting 10^-(k+1) units. The fraction of
    the least significant element is kept: it is real information on a
    rolling wheel or dial. Alternatives above ``max_residual`` are left out.
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
    # The wheel above the dials is geared to them, it cannot jump.
    forced = 1 if dials else 0
    ambiguous = False
    # Depth-first over both readings of each upper wheel; the first choice
    # each time gives the composed value itself.
    found: list[tuple[float, float]] = []
    stack = [(0, below, tuple(digits), residual)]
    while stack and len(found) <= MAX_ALTERNATIVES:
        index, below, path, worst = stack.pop()
        if index == len(rest):
            found.append((_number(path) + fraction, worst))
            continue
        options = _options(rest[index], below, forced=index < forced)
        if not found and len(options) == 2:
            (first, _, error), (second, _, other) = options
            if first != second and abs(error - other) < AMBIGUITY:
                ambiguous = True
        # Pushed in reverse, so the first choice is explored first.
        for digit, next_below, error in reversed(options):
            # The composed value itself is followed whatever its residual.
            primary = not found and (digit, next_below, error) == options[0]
            if primary or max(worst, error) <= max_residual:
                stack.append((index + 1, next_below, (*path, digit), max(worst, error)))
    value, residual = found[0]
    # Only readings that fit about as well: on jumping wheels the coupled
    # reading of a wheel above a 6 or more fits poorly and would stand as a
    # value one step lower all the time.
    others = {
        (v, r)
        for v, r in found[1:]
        if v != value and r <= residual + ALTERNATIVE_MARGIN
    }
    alternatives = sorted(others, key=lambda item: (item[1], item[0]))
    return Composition(value, residual, ambiguous, tuple(alternatives))


def _options(
    position: float, below: float, *, forced: bool
) -> list[tuple[int, float, float]]:
    """Return (digit, below for the next wheel, error), the better fit first."""
    coupled = round(position - below / 10.0) % 10
    coupled_error = circular_distance(position, coupled + below / 10.0)
    coupled_option = (coupled, coupled + below / 10.0, coupled_error)
    if forced:
        return [coupled_option]
    jumped = round(position) % 10
    jumped_option = (jumped, float(jumped), circular_distance(position, jumped))
    if coupled == jumped:
        # Same digit either way; the better fit sets what the next wheel sees.
        return [min(coupled_option, jumped_option, key=lambda o: o[2])]
    if coupled_error <= jumped_option[2]:
        return [coupled_option, jumped_option]
    return [jumped_option, coupled_option]


def _number(digits: Sequence[int]) -> int:
    whole = 0
    for digit in reversed(digits):
        whole = whole * 10 + digit
    return whole
