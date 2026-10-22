"""A meter counts up, and no faster than its pipe allows."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..errors import MeterDataError
from . import checks
from .reader import MeterReading, Rejection

# Units per hour. Suits a household water meter in m³: a Q3 2.5 meter is
# built for 3.125 m³/h at most. Pass your own for other meters and units.
MAX_RATE = 4.0
# Consecutive readings that must agree on a lower value before it is taken.
CONFIRMATIONS = 3
# The counter wraps only from its top tenth to its bottom tenth.
WRAP_ZONE = 0.1
# A state this far ahead of the reading's time came from a clock that was
# ahead; its time is taken as the reading's.
CLOCK_SLACK = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class PlausibilityState:
    """The last accepted value; store it to keep checking across restarts."""

    value: float
    # Timezone-aware; naive datetimes are rejected.
    at: datetime

    def __post_init__(self) -> None:
        """Reject a value that is not finite or a naive time."""
        checks.number(self.value, "value")
        checks.aware(self.at, "at")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible form."""
        return {"value": self.value, "at": self.at.isoformat()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlausibilityState:
        """Rebuild the state from :meth:`as_dict`; raises MeterDataError."""
        try:
            data = checks.mapping(data, "state")
            at = data["at"]
            if not isinstance(at, str):
                raise MeterDataError(f"at must be an ISO time, got {at!r}")
            try:
                parsed = datetime.fromisoformat(at)
            except ValueError as err:
                raise MeterDataError(f"at must be an ISO time, got {at!r}") from err
            return cls(data["value"], parsed)
        except KeyError as err:
            raise MeterDataError(f"invalid plausibility state: missing {err}") from err
        except MeterDataError as err:
            raise MeterDataError(f"invalid plausibility state: {err}") from err


class Plausibility:
    """
    Rejects readings that go backwards or rise faster than ``max_rate``.

    Needs no model and catches the gross errors confidence does not: a
    confidently wrong upper wheel moves the value by whole units at once.

    ``max_rate`` is in units per hour: about 4 for household water in m³
    (4000 in litres), the meter's Qmax for gas in m³ (6 for a G4), the
    supply's peak power for electricity in kWh (about 45 on 3 x 63 A). The
    allowed rise grows with the time since the last accepted value, also
    after long gaps. ``tolerance`` lets a value drop or overshoot by that
    much, e.g. one unit of the finest dial that jitters around a mark.

    Nothing is accepted unchecked: seed the state with a value the user
    confirmed, :meth:`accept` or ``state``; until then every reading is
    UNCONFIRMED. A lower value is taken only once ``confirmations``
    consecutive readings agree on it, so a meter exchange or a corrected
    misread settles by itself while a single misread does not. The readings
    agree when each passes the usual checks against the one before.

    A lower value is taken only when all readings the wheels allow are
    lower, and not when the drop is a whole number of steps of the lowest
    upper wheel (10 m³ on a meter with four dials): that is a misread seen
    again, not a new meter. Such a meter needs :meth:`accept`.

    With ``period``, e.g. :attr:`MeterLayout.period`, the counter may wrap
    from its top tenth to its bottom tenth: 99999.9 then 00000.2 is a rise
    of 0.3. Values then count on past the wrap (100000.2), in the state and
    in reviewed readings. 62700 then 3 is not a wrap but a drop.

    A state more than ten minutes ahead of a reading's time, from a clock
    that was ahead, is moved back to it.

    Not thread-safe; use one per meter and serialize its calls. Pending
    lower readings live in memory only.
    """

    def __init__(
        self,
        *,
        max_rate: float = MAX_RATE,
        tolerance: float = 0.0,
        period: float | None = None,
        confirmations: int = CONFIRMATIONS,
        state: PlausibilityState | None = None,
    ) -> None:
        """Start from ``state``; raises MeterDataError for invalid settings."""
        if checks.number(max_rate, "max_rate") <= 0:
            raise MeterDataError("max_rate must be positive")
        if checks.number(tolerance, "tolerance") < 0:
            raise MeterDataError("tolerance must not be negative")
        if period is not None and checks.number(period, "period") <= 0:
            raise MeterDataError("period must be positive")
        if checks.integer(confirmations, "confirmations") < 2:
            raise MeterDataError("confirmations must be at least 2")
        if state is not None and not isinstance(state, PlausibilityState):
            raise MeterDataError(f"state must be a PlausibilityState, got {state!r}")
        self.max_rate = float(max_rate)
        self.tolerance = float(tolerance)
        self.period = None if period is None else float(period)
        self.confirmations = confirmations
        self.state = state
        self._pending: list[PlausibilityState] = []

    def check(self, value: float, at: datetime) -> Rejection | None:
        """Return why ``value`` at ``at`` is implausible, or None; changes nothing."""
        value = checks.number(value, "value")
        at = checks.aware(at, "at")
        if (last := self._last(at)) is None:
            return Rejection.UNCONFIRMED
        return self._against(last, value, at)[0]

    def accept(self, value: float, at: datetime) -> None:
        """Take ``value`` as the last accepted one, e.g. a confirmed reading."""
        self.state = PlausibilityState(value, at)
        self._pending.clear()

    def review(self, reading: MeterReading, at: datetime) -> MeterReading:
        """
        Check a reading against the last accepted value and return it.

        The reading's candidate and its alternatives are checked; the one
        that fits is taken and remembered, also for a reading the reader
        found ambiguous. When several fit, the smallest rise wins. Other
        rejected readings are returned unchanged. An implausible reading
        comes back rejected with UNCONFIRMED, DECREASED (all lower),
        TOO_FAST (all higher) or AMBIGUOUS (some lower, some higher). A drop
        within ``tolerance`` comes back with the last value, which it keeps.
        With ``period`` the value counts on past a wrap.
        """
        if reading.rejection not in (None, Rejection.AMBIGUOUS):
            return reading
        at = checks.aware(at, "at")
        values = (reading.candidate, *reading.alternatives)
        hypotheses = [checks.number(v, "value") for v in values]
        if (last := self._last(at)) is None:
            return reading.rejected(Rejection.UNCONFIRMED)
        # A clock that was ahead must not hold back every later reading.
        self.state = last
        judged = [self._against(last, v, at) for v in hypotheses]
        fitting = [value for rejection, value in judged if rejection is None]
        if not fitting:
            return self._unfitting(reading, judged, at)
        self._pending.clear()
        value = min(fitting)
        if value < last.value:
            # Jitter within the tolerance holds the value instead of
            # lowering it, so a total does not wobble.
            value = last.value
        else:
            # The clock may step back; the state's time never does.
            self.state = PlausibilityState(value, max(at, last.at))
        return dataclasses.replace(reading, value=value, rejection=None)

    def _unfitting(
        self,
        reading: MeterReading,
        judged: list[tuple[Rejection | None, float]],
        at: datetime,
    ) -> MeterReading:
        """Return a reading none of whose values fit, or a lower one agreed on."""
        assert self.state is not None
        rejections = {rejection for rejection, _ in judged}
        if rejections == {Rejection.DECREASED}:
            value = judged[0][1]
            if not self._agreed_lower(reading, value, at):
                return reading.rejected(Rejection.DECREASED)
            self._pending.clear()
            self.state = PlausibilityState(value, max(at, self.state.at))
            return dataclasses.replace(reading, value=value, rejection=None)
        self._pending.clear()
        if rejections == {Rejection.TOO_FAST}:
            return reading.rejected(Rejection.TOO_FAST)
        return reading.rejected(Rejection.AMBIGUOUS)

    def _last(self, at: datetime) -> PlausibilityState | None:
        """Return the state, its time pulled back if it lies well ahead of ``at``."""
        if self.state is None:
            return None
        if self.state.at - at > CLOCK_SLACK:
            return PlausibilityState(self.state.value, at)
        return self.state

    def _against(
        self, last: PlausibilityState, value: float, at: datetime
    ) -> tuple[Rejection | None, float]:
        value = self._unwrapped(last.value, value)
        # No time passed, or the clock stepped back: no rise allowed.
        hours = max(0.0, (at - last.at).total_seconds() / 3600)
        rise = value - last.value
        if rise < -self.tolerance:
            return Rejection.DECREASED, value
        if rise > self._allowed(hours):
            return Rejection.TOO_FAST, value
        return None, value

    def _allowed(self, hours: float) -> float:
        return self.max_rate * hours + self.tolerance

    def _unwrapped(self, last: float, value: float) -> float:
        """Return ``value`` counted on from ``last``, across a wrap of the counter."""
        if self.period is None:
            return value
        cycle = math.floor(last / self.period) * self.period
        position = last - cycle
        value += cycle
        # Only from the top of the counter to its bottom, or back by jitter:
        # 62700 then 3 is a new meter or a misread, not a wrap.
        if position > self.period * (1 - WRAP_ZONE) and value - cycle < (
            self.period * WRAP_ZONE
        ):
            value += self.period
        elif position < self.period * WRAP_ZONE and value - cycle > self.period * (
            1 - WRAP_ZONE
        ):
            value -= self.period
        return value

    def _agreed_lower(self, reading: MeterReading, value: float, at: datetime) -> bool:
        """
        Collect a lower reading; return True once enough agree in a row.

        A drop that misread upper wheels explain never counts: three
        readings that agree on it are more likely one bias seen three times
        than a new meter.
        """
        assert self.state is not None
        hours = max(0.0, (at - self.state.at).total_seconds() / 3600)
        if _misread(
            self.state.value - value,
            self._allowed(hours),
            self.tolerance,
            reading.resolution * 10 ** (len(reading.dials) + 1),
        ):
            self._pending.clear()
            return False
        if self._pending:
            rejection, _ = self._against(self._pending[-1], value, at)
            if rejection is not None:
                self._pending.clear()
        self._pending.append(PlausibilityState(value, at))
        return len(self._pending) >= self.confirmations


# Upper wheels a misread may touch at once.
MISREAD_WHEELS = 2


def _misread(drop: float, allowed: float, tolerance: float, step: float) -> bool:
    """
    Return True when misread upper wheels explain a drop.

    ``step`` is the unit of the lowest wheel that can jump. A reading that
    is one or two such wheels a digit or more off, while the true value rose
    by up to ``allowed``, is lower by a number of steps with at most two
    digits other than 0.
    """
    low = max(1, math.ceil((drop - tolerance) / step))
    high = math.floor((drop + allowed) / step)
    if high - low > 1000:
        # A long gap: any drop could be a misread.
        return True
    return any(
        sum(c != "0" for c in str(n)) <= MISREAD_WHEELS for n in range(low, high + 1)
    )
