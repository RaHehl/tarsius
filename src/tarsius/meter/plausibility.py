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
# Upper wheels a misread may touch at once.
MISREAD_WHEELS = 2
# A misread keeps the lower wheels right, so the true value lies just above
# the last one in their digits: within this share of the wheel's unit. A
# drop this small is jitter that holds the value, never a misread.
MISREAD_SHARE = 0.02
# A drop of this many units of the last wheel or dial also holds the value.
HOLD_UNITS = 5
# Misreads followed at once.
MAX_MISREADS = 4
# Lower readings that only a misread explains, for this long and this often,
# are a dispute the user should settle: the meter or the last value is wrong.
DISPUTE_AFTER = timedelta(hours=6)
DISPUTE_READS = 12


def _time(value: object, name: str) -> datetime:
    """Return an aware datetime from an ISO string."""
    if not isinstance(value, str):
        raise MeterDataError(f"{name} must be an ISO time, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as err:
        raise MeterDataError(f"{name} must be an ISO time, got {value!r}") from err
    return checks.aware(parsed, name)


def _list(value: object, name: str) -> list[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple):
        raise MeterDataError(f"{name} must be a list, got {value!r}")
    return list(value)


@dataclass(frozen=True, slots=True)
class Dispute:
    """Lower readings that only a misread of upper wheels explains."""

    # First and latest of them.
    since: datetime
    last: datetime
    count: int
    # The latest of them, e.g. to offer as the value to confirm.
    value: float

    def __post_init__(self) -> None:
        """Reject a dispute no review produces; raises MeterDataError."""
        checks.aware(self.since, "since")
        checks.aware(self.last, "last")
        if checks.integer(self.count, "count") < 1:
            raise MeterDataError("count must be at least 1")
        checks.number(self.value, "value")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible form."""
        return {
            "since": self.since.isoformat(),
            "last": self.last.isoformat(),
            "count": self.count,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: object) -> Dispute:
        """Rebuild a dispute from :meth:`as_dict`; raises MeterDataError."""
        data = checks.mapping(data, "dispute")
        try:
            return cls(
                _time(data["since"], "since"),
                _time(data["last"], "last"),
                data["count"],
                data["value"],
            )
        except KeyError as err:
            raise MeterDataError(f"dispute is missing {err}") from err


@dataclass(frozen=True, slots=True)
class PlausibilityState:
    """
    The last accepted value and what the checks remember about it.

    Store :meth:`as_dict` to keep checking across restarts.
    """

    value: float
    # Timezone-aware; naive datetimes are rejected.
    at: datetime
    # How far the value lies above the composed value of its reading; not
    # 0 when an alternative was taken.
    offset: float = 0.0
    # Lower readings that agree so far, (value, time), oldest first.
    pending: tuple[tuple[float, datetime], ...] = ()
    # Misreads followed, (offset, latest such reading, its time).
    misreads: tuple[tuple[float, float, datetime], ...] = ()
    dispute: Dispute | None = None

    def __post_init__(self) -> None:
        """Reject values no review produces; raises MeterDataError."""
        checks.number(self.value, "value")
        checks.aware(self.at, "at")
        checks.number(self.offset, "offset")
        object.__setattr__(self, "pending", tuple(self.pending))
        object.__setattr__(self, "misreads", tuple(self.misreads))
        for item in self.pending:
            if not isinstance(item, tuple) or len(item) != 2:
                raise MeterDataError(f"pending must be (value, time), got {item!r}")
            checks.number(item[0], "pending value")
            checks.aware(item[1], "pending time")
        for misread in self.misreads:
            if not isinstance(misread, tuple) or len(misread) != 3:
                raise MeterDataError(
                    f"misreads must be (offset, value, time), got {misread!r}"
                )
            if checks.number(misread[0], "misread offset") <= 0:
                raise MeterDataError("misread offsets must be positive")
            checks.number(misread[1], "misread value")
            checks.aware(misread[2], "misread time")
        if self.dispute is not None and not isinstance(self.dispute, Dispute):
            raise MeterDataError(f"dispute must be a Dispute, got {self.dispute!r}")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible form."""
        return {
            "value": self.value,
            "at": self.at.isoformat(),
            "offset": self.offset,
            "pending": [[value, at.isoformat()] for value, at in self.pending],
            "misreads": [
                [offset, value, at.isoformat()] for offset, value, at in self.misreads
            ],
            "dispute": None if self.dispute is None else self.dispute.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlausibilityState:
        """Rebuild the state from :meth:`as_dict`; raises MeterDataError."""
        try:
            data = checks.mapping(data, "state")
            pending = []
            for item in _list(data.get("pending", []), "pending"):
                pair = _list(item, "pending item")
                if len(pair) != 2:
                    raise MeterDataError(f"pending must be (value, time), got {item!r}")
                pending.append((pair[0], _time(pair[1], "pending time")))
            misreads = []
            for item in _list(data.get("misreads", []), "misreads"):
                triple = _list(item, "misread")
                if len(triple) != 3:
                    raise MeterDataError(
                        f"misreads must be (offset, value, time), got {item!r}"
                    )
                misreads.append(
                    (triple[0], triple[1], _time(triple[2], "misread time"))
                )
            dispute = data.get("dispute")
            return cls(
                data["value"],
                _time(data["at"], "at"),
                data.get("offset", 0.0),
                tuple(pending),
                tuple(misreads),
                None if dispute is None else Dispute.from_dict(dispute),
            )
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
    UNCONFIRMED.

    The composed value of a reading comes first. A small drop below the last
    value is jitter and holds it. Its alternatives, the values the upper
    wheels allow when read the other way, are taken only when the composed
    value cannot be right: a lower one when it rose too far or both fit, a
    higher one only when the composed value dropped by more than jitter.
    So a gap that allows a rise of a whole step does not let a higher
    reading of a coupled counter in.

    A lower value is taken once ``confirmations`` consecutive readings agree
    on it and all alternatives are lower too, so a meter exchange settles by
    itself. A drop of whole digits of one or two upper wheels is a misread
    seen again and never settles by itself, unless the last value came from
    an alternative: the data cannot tell a persistent misread from a wrong
    last value. When such drops last, :attr:`dispute` reports them so the
    user can confirm the value.

    With ``period``, e.g. :attr:`MeterLayout.period`, the counter may wrap
    from its top tenth to its bottom tenth: 99999.9 then 00000.2 is a rise
    of 0.3. Values then count on past the wrap (100000.2), in the state and
    in reviewed readings. 62700 then 3 is not a wrap but a drop.

    A state more than ten minutes ahead of a reading's time, from a clock
    that was ahead, is moved back to it. Everything the checks remember is
    part of :attr:`state`. Not thread-safe; use one per meter and serialize
    its calls.
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

    @property
    def dispute(self) -> Dispute | None:
        """
        Return lasting lower readings that only a misread explains, or None.

        Set after DISPUTE_AFTER and DISPUTE_READS of them; cleared by an
        accepted reading or :meth:`accept`. Either the meter is misread or
        the last value is too high; ask the user for the value.
        """
        if self.state is None or (dispute := self.state.dispute) is None:
            return None
        if dispute.last - dispute.since < DISPUTE_AFTER:
            return None
        return dispute if dispute.count >= DISPUTE_READS else None

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

    def review(self, reading: MeterReading, at: datetime) -> MeterReading:
        """
        Check a reading against the last accepted value and return it.

        Readings the reader rejected for anything but AMBIGUOUS are returned
        unchanged; they break a run of agreeing lower readings. Otherwise
        the composed value or an alternative is taken as the class
        describes, or the reading comes back rejected with UNCONFIRMED,
        DECREASED (all lower), TOO_FAST (all higher) or AMBIGUOUS (some
        lower, some higher). A held value comes back as the last value.
        With ``period`` the value counts on past a wrap.
        """
        if reading.rejection not in (None, Rejection.AMBIGUOUS):
            if self.state is not None and self.state.pending:
                self.state = dataclasses.replace(self.state, pending=())
            return reading
        at = checks.aware(at, "at")
        candidate = checks.number(reading.candidate, "value")
        alternatives = [checks.number(v, "value") for v in reading.alternatives]
        if (last := self._last(at)) is None:
            return reading.rejected(Rejection.UNCONFIRMED)
        # A clock that was ahead must not hold back every later reading.
        self.state = last
        step = _step(reading)
        hold = max(MISREAD_SHARE * step, HOLD_UNITS * reading.resolution)
        rejection, value = self._against(last, candidate, at)
        # (value, how far above the composed value)
        options: list[tuple[float, float]] = []
        if rejection is None:
            options.append((value, 0.0))
        elif rejection is Rejection.DECREASED and last.value - value <= hold:
            options.append((last.value, last.offset))
        judged = [self._against(last, v, at) for v in alternatives]
        for other_rejection, other in judged:
            # A higher reading only as a misread: just above the last value.
            if other_rejection is None and (
                other < value or other - last.value <= MISREAD_SHARE * step
            ):
                options.append((other, other - value))
        if options:
            return self._take(reading, *min(options), at)
        rejections = {rejection, *(r for r, _ in judged)}
        if rejections == {Rejection.DECREASED}:
            return self._lower(reading, value, at, step, hold)
        if rejections == {Rejection.TOO_FAST}:
            self.state = dataclasses.replace(
                last, pending=(), dispute=_disputed(last, value, at)
            )
            return reading.rejected(Rejection.TOO_FAST)
        self.state = dataclasses.replace(last, pending=())
        return reading.rejected(Rejection.AMBIGUOUS)

    def _take(
        self, reading: MeterReading, value: float, offset: float, at: datetime
    ) -> MeterReading:
        """Remember an accepted value; a lower one holds the last."""
        assert self.state is not None
        last = self.state
        if value <= last.value:
            # Jitter holds the value instead of lowering it, so a total
            # does not wobble; the meter stands there now.
            value, offset = last.value, last.offset
        # The clock may step back; the state's time never does.
        self.state = dataclasses.replace(
            last,
            value=value,
            at=max(at, last.at),
            offset=offset,
            pending=(),
            dispute=None,
        )
        return dataclasses.replace(reading, value=value, rejection=None)

    def _lower(
        self,
        reading: MeterReading,
        value: float,
        at: datetime,
        step: float,
        hold: float,
    ) -> MeterReading:
        """Handle a reading whose values are all lower than the last one."""
        assert self.state is not None
        last = self.state
        hours = max(0.0, (at - last.at).total_seconds() / 3600)
        offset = _misread(value, last.value, self._allowed(hours), hold, step)
        if offset is None:
            offset = self._followed(value, at, hold)
        # The last value came from an alternative this much higher: readings
        # that fall back by it may be right after all.
        waived = offset is not None and math.isclose(
            offset, last.offset, rel_tol=1e-6, abs_tol=self.tolerance
        )
        if offset is not None and not waived:
            followed = [m for m in last.misreads if m[0] != offset]
            misreads = (*followed, (offset, value, at))[-MAX_MISREADS:]
            self.state = dataclasses.replace(
                last,
                pending=(),
                misreads=misreads,
                dispute=_disputed(last, value, at),
            )
            return reading.rejected(Rejection.DECREASED)
        pending = last.pending
        if pending:
            previous = PlausibilityState(*pending[-1])
            if self._against(previous, value, at)[0] is not None:
                pending = ()
        pending = (*pending, (value, at))
        if len(pending) < self.confirmations:
            self.state = dataclasses.replace(last, pending=pending)
            return reading.rejected(Rejection.DECREASED)
        # Enough agree: a new meter, or a last value that was wrong.
        self.state = PlausibilityState(value, max(at, last.at))
        return dataclasses.replace(reading, value=value, rejection=None)

    def _followed(self, value: float, at: datetime, hold: float) -> float | None:
        """
        Return the offset of a misread followed so far that ``value`` continues.

        A misread at rest keeps its lower wheels right while the true value
        moves on, so it is followed from reading to reading: the next one
        lies a hold below to a rise above the latest. A misread not seen for
        so long that the allowed rise reaches its offset is forgotten.
        """
        assert self.state is not None
        for offset, seen, seen_at in self.state.misreads:
            hours = max(0.0, (at - seen_at).total_seconds() / 3600)
            allowed = self._allowed(hours)
            if allowed < offset and -hold <= value - seen <= allowed:
                return offset
        return None

    def _last(self, at: datetime) -> PlausibilityState | None:
        """Return the state, its time pulled back if it lies well ahead of ``at``."""
        if self.state is None:
            return None
        if self.state.at - at > CLOCK_SLACK:
            return dataclasses.replace(self.state, at=at)
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


def _disputed(last: PlausibilityState, value: float, at: datetime) -> Dispute:
    """Return the dispute of ``last`` continued by a reading, or a new one."""
    dispute = last.dispute
    if dispute is None or (dispute.value < last.value) != (value < last.value):
        return Dispute(at, at, 1, value)
    return Dispute(dispute.since, at, dispute.count + 1, value)


def _step(reading: MeterReading) -> float:
    """Return the unit of the lowest wheel of a reading that can jump."""
    return float(reading.resolution * 10 ** (len(reading.dials) + 1))


def _misread(
    value: float, last: float, allowed: float, hold: float, step: float
) -> float | None:
    """
    Return the offset of a misread of upper wheels that explains a drop.

    ``step`` is the unit of the lowest wheel that can jump. A reading with
    one or two wheels a digit or more off shows the lower wheels right: the
    true value is ``value`` plus a multiple of a wheel's unit with at most
    two digits other than 0, and lies a small rise above ``last``. A new
    meter matches that only by chance. The lower wheels may read up to
    ``hold`` low. None when no misread explains it.
    """
    unit = step
    while unit <= last - value + allowed + hold:
        rise = (value - last) % unit
        if rise > unit - hold:
            rise -= unit
        steps = round((last + rise - value) / unit)
        if (
            steps >= 1
            and rise <= min(allowed, MISREAD_SHARE * unit)
            and sum(c != "0" for c in str(steps)) <= MISREAD_WHEELS
        ):
            return steps * unit
        unit *= 10
    return None
