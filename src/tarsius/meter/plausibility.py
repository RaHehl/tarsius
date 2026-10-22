"""A meter counts up, and no faster than its pipe allows."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..errors import MeterDataError
from . import checks
from .reader import MeterReading, Rejection

# Units per hour. Suits a household water meter in m³: a Q3 2.5 meter is
# built for 3.125 m³/h at most. Pass your own for other meters and units.
MAX_RATE = 4.0
# Consecutive readings that must agree on a lower value before it is taken.
CONFIRMATIONS = 3


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

    With ``period``, e.g. :attr:`MeterLayout.period`, the counter may wrap:
    99999.9 then 00000.2 is a rise of 0.3. Values then count on past the
    wrap (100000.2), in the state and in reviewed readings.

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
        return self._judge(value, at)[0]

    def accept(self, value: float, at: datetime) -> None:
        """Take ``value`` as the last accepted one, e.g. a confirmed reading."""
        self.state = PlausibilityState(value, at)
        self._pending.clear()

    def review(self, reading: MeterReading, at: datetime) -> MeterReading:
        """
        Check an accepted reading, remember it if plausible, and return it.

        A rejected reading is returned unchanged and not remembered. An
        implausible one comes back rejected with UNCONFIRMED, DECREASED or
        TOO_FAST. A drop within ``tolerance`` comes back with the last value,
        which it keeps. With ``period`` the value counts on past a wrap.
        """
        if reading.value is None:
            return reading
        rejection, value = self._judge(reading.value, at)
        if rejection is Rejection.DECREASED:
            if not self._agreed_lower(value, at):
                return reading.rejected(rejection)
        elif rejection is not None:
            self._pending.clear()
            return reading.rejected(rejection)
        self._pending.clear()
        assert self.state is not None
        if rejection is None and value < self.state.value:
            # Jitter within the tolerance holds the value instead of
            # lowering it, so a total does not wobble.
            value = self.state.value
        else:
            # The clock may step back; the state's time never does.
            self.state = PlausibilityState(value, max(at, self.state.at))
        if value == reading.value:
            return reading
        return dataclasses.replace(reading, value=value)

    def _judge(self, value: float, at: datetime) -> tuple[Rejection | None, float]:
        value = checks.number(value, "value")
        at = checks.aware(at, "at")
        if self.state is None:
            return Rejection.UNCONFIRMED, value
        return self._against(self.state, value, at)

    def _against(
        self, last: PlausibilityState, value: float, at: datetime
    ) -> tuple[Rejection | None, float]:
        if self.period is not None:
            # The representation nearest the last value: a wrap is a rise.
            value += self.period * round((last.value - value) / self.period)
        # No time passed, or the clock stepped back: no rise allowed.
        hours = max(0.0, (at - last.at).total_seconds() / 3600)
        rise = value - last.value
        if rise < -self.tolerance:
            return Rejection.DECREASED, value
        if rise > self.max_rate * hours + self.tolerance:
            return Rejection.TOO_FAST, value
        return None, value

    def _agreed_lower(self, value: float, at: datetime) -> bool:
        """Collect a lower reading; return True once enough agree in a row."""
        if self._pending:
            rejection, _ = self._against(self._pending[-1], value, at)
            if rejection is not None:
                self._pending.clear()
        self._pending.append(PlausibilityState(value, at))
        return len(self._pending) >= self.confirmations
