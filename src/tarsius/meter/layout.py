"""Where a meter's wheels and dials are in the camera frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import MeterDataError
from ..roi import Roi
from . import checks

# More wheels than any utility meter has; guards against a typo in a form.
MAX_WHEELS = 12
MAX_DIALS = 8


@dataclass(frozen=True, slots=True)
class Dial:
    """One pointer dial: a square-ish region centred on its axis."""

    roi: Roi
    # The pointer turns counter-clockwise, read as its mirror image.
    counterclockwise: bool = False

    def __post_init__(self) -> None:
        """Reject a region that is not a Roi or a flag that is not a bool."""
        if not isinstance(self.roi, Roi):
            raise MeterDataError(f"dial region must be a Roi, got {self.roi!r}")
        checks.flag(self.counterclockwise, "counterclockwise")


@dataclass(frozen=True, slots=True)
class MeterLayout:
    """
    The regions of one meter, fixed per installation.

    Regions are fractions of the frame after it has been turned by
    ``rotation`` degrees counter-clockwise about its centre, keeping its size
    (as ``PIL.Image.rotate`` does). With ``rotation`` 0 they are fractions of
    the frame as the camera delivers it.
    """

    # The counter window holding all wheels, tight around the digits.
    digits: Roi
    # Number of wheels in the window.
    wheels: int
    # Pointer dials below the wheels, most significant (x0.1) first.
    dials: tuple[Dial, ...] = ()
    # Rightmost wheels that count fractions, e.g. red ones after the point.
    decimal_wheels: int = 0
    rotation: float = 0.0

    def __post_init__(self) -> None:
        """Reject layouts no meter has; raises MeterDataError."""
        if not isinstance(self.digits, Roi):
            raise MeterDataError(f"digits must be a Roi, got {self.digits!r}")
        if not 1 <= checks.integer(self.wheels, "wheels") <= MAX_WHEELS:
            raise MeterDataError(f"wheels must be 1 to {MAX_WHEELS}")
        decimal_wheels = checks.integer(self.decimal_wheels, "decimal_wheels")
        if not 0 <= decimal_wheels <= self.wheels:
            raise MeterDataError("decimal_wheels must be 0 to the number of wheels")
        # A tuple keeps the layout hashable, a list from JSON would not.
        object.__setattr__(self, "dials", tuple(self.dials))
        if len(self.dials) > MAX_DIALS:
            raise MeterDataError(f"at most {MAX_DIALS} dials")
        for dial in self.dials:
            if not isinstance(dial, Dial):
                raise MeterDataError(f"dials must be Dial objects, got {dial!r}")
        rotation = checks.number(self.rotation, "rotation")
        if not -360.0 <= rotation <= 360.0:
            raise MeterDataError("rotation must be within -360 to 360 degrees")
        object.__setattr__(self, "rotation", rotation)

    @property
    def decimals(self) -> int:
        """Return the number of places after the point the wheels and dials show."""
        return self.decimal_wheels + len(self.dials)

    @property
    def period(self) -> float:
        """
        Return the value at which the counter wraps to zero.

        Five whole wheels count up to 99999.x and then show 00000 again, so
        the period is 100000; pass it to :class:`Plausibility`.
        """
        return float(10 ** (self.wheels - self.decimal_wheels))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible form, e.g. for a config entry."""
        return {
            "digits": list(self.digits.as_tuple()),
            "wheels": self.wheels,
            "dials": [
                {"roi": list(d.roi.as_tuple()), "counterclockwise": d.counterclockwise}
                for d in self.dials
            ],
            "decimal_wheels": self.decimal_wheels,
            "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MeterLayout:
        """
        Rebuild a layout from :meth:`as_dict`.

        Raises MeterDataError, or RoiError for a region outside the frame;
        both are ValueErrors. Nothing is coerced: ``"false"`` is no bool and
        5.9 no wheel count.
        """
        try:
            data = checks.mapping(data, "meter layout")
            dials = data.get("dials", [])
            if isinstance(dials, str | bytes | Mapping) or not isinstance(
                dials, Sequence
            ):
                raise MeterDataError(f"dials must be a list, got {dials!r}")
            return cls(
                digits=checks.roi(data["digits"], "digits"),
                wheels=checks.integer(data["wheels"], "wheels"),
                dials=tuple(_dial(checks.mapping(d, "dial")) for d in dials),
                decimal_wheels=checks.integer(
                    data.get("decimal_wheels", 0), "decimal_wheels"
                ),
                rotation=checks.number(data.get("rotation", 0.0), "rotation"),
            )
        except KeyError as err:
            raise MeterDataError(f"invalid meter layout: missing {err}") from err
        except MeterDataError as err:
            raise MeterDataError(f"invalid meter layout: {err}") from err


def _dial(data: Mapping[str, Any]) -> Dial:
    return Dial(
        checks.roi(data["roi"], "dial region"),
        checks.flag(data.get("counterclockwise", False), "counterclockwise"),
    )
