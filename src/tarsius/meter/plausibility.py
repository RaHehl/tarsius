"""A meter counts up, and no faster than its pipe allows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .reader import MeterReading, Rejection

# 0.3 units per minute, what the maintainer's previous reader used.
MAX_RATE = 18.0
MAX_GAP = timedelta(days=1)


@dataclass(frozen=True, slots=True)
class PlausibilityState:
    """The last accepted value; store it to keep checking across restarts."""

    value: float
    at: datetime

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible form."""
        return {"value": self.value, "at": self.at.isoformat()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlausibilityState:
        """Rebuild the state from :meth:`as_dict`; raises ValueError if invalid."""
        try:
            return cls(float(data["value"]), datetime.fromisoformat(data["at"]))
        except (KeyError, TypeError) as err:
            raise ValueError(f"invalid plausibility state: {err!r}") from err


class Plausibility:
    """
    Rejects readings that go backwards or rise faster than ``max_rate``.

    Needs no model and catches the gross errors confidence does not: a
    confidently wrong upper wheel moves the value by whole units at once.
    ``max_rate`` is in units per hour, e.g. the meter's Q4 flow in m³/h.
    ``tolerance`` lets a value drop by that much without rejection, e.g. one
    unit of the finest dial that jitters around a mark. After more than
    ``max_gap`` without an accepted value, the next one is accepted as is.
    Not thread-safe; use one per meter and serialize its calls.
    """

    def __init__(
        self,
        *,
        max_rate: float = MAX_RATE,
        tolerance: float = 0.0,
        max_gap: timedelta = MAX_GAP,
        state: PlausibilityState | None = None,
    ) -> None:
        """Start from ``state``, or with nothing to compare against."""
        if max_rate <= 0:
            raise ValueError("max_rate must be positive")
        if tolerance < 0:
            raise ValueError("tolerance must not be negative")
        self.max_rate = max_rate
        self.tolerance = tolerance
        self.max_gap = max_gap
        self.state = state

    def check(self, value: float, at: datetime) -> Rejection | None:
        """Return why ``value`` at ``at`` is implausible, or None."""
        if self.state is None:
            return None
        elapsed = at - self.state.at
        if elapsed <= timedelta(0) or elapsed > self.max_gap:
            return None
        if value < self.state.value - self.tolerance:
            return Rejection.DECREASED
        hours = elapsed.total_seconds() / 3600
        if value - self.state.value > self.max_rate * hours + self.tolerance:
            return Rejection.TOO_FAST
        return None

    def accept(self, value: float, at: datetime) -> None:
        """Remember ``value`` as the last accepted one."""
        self.state = PlausibilityState(value, at)

    def review(self, reading: MeterReading, at: datetime) -> MeterReading:
        """
        Check an accepted reading, remember it if plausible, and return it.

        A rejected reading is returned unchanged and not remembered. An
        implausible one comes back rejected with DECREASED or TOO_FAST.
        """
        if reading.value is None:
            return reading
        if rejection := self.check(reading.value, at):
            return reading.rejected(rejection)
        self.accept(reading.value, at)
        return reading
