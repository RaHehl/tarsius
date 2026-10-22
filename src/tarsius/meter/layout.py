"""Where a meter's wheels and dials are in the camera frame."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..roi import Roi

# More wheels than any utility meter has; guards against a typo in a form.
MAX_WHEELS = 12
MAX_DIALS = 8


@dataclass(frozen=True, slots=True)
class Dial:
    """One pointer dial: a square-ish region centred on its axis."""

    roi: Roi
    # The pointer turns counter-clockwise, read as its mirror image.
    counterclockwise: bool = False


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
        """Reject layouts no meter has."""
        if not 1 <= self.wheels <= MAX_WHEELS:
            raise ValueError(f"wheels must be 1 to {MAX_WHEELS}")
        if not 0 <= self.decimal_wheels <= self.wheels:
            raise ValueError("decimal_wheels must be 0 to the number of wheels")
        if len(self.dials) > MAX_DIALS:
            raise ValueError(f"at most {MAX_DIALS} dials")
        if not -360.0 <= self.rotation <= 360.0:
            raise ValueError("rotation must be within -360 to 360 degrees")
        # A tuple keeps the layout hashable, a list from JSON would not.
        object.__setattr__(self, "dials", tuple(self.dials))

    @property
    def decimals(self) -> int:
        """Return the number of places after the point the wheels and dials show."""
        return self.decimal_wheels + len(self.dials)

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
        """Rebuild a layout from :meth:`as_dict`; raises ValueError if invalid."""
        try:
            return cls(
                digits=Roi(*(float(v) for v in data["digits"])),
                wheels=int(data["wheels"]),
                dials=tuple(
                    Dial(
                        Roi(*(float(v) for v in dial["roi"])),
                        bool(dial.get("counterclockwise", False)),
                    )
                    for dial in data.get("dials", ())
                ),
                decimal_wheels=int(data.get("decimal_wheels", 0)),
                rotation=float(data.get("rotation", 0.0)),
            )
        except (KeyError, TypeError) as err:
            raise ValueError(f"invalid meter layout: {err!r}") from err
