"""
Cut a counter window into wheel cells, and settle how once per installation.

The raster and which way up the window is are properties of the
installation, not of the frame, so they are searched once and stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from ..errors import MeterDataError
from . import checks
from .compose import compose
from .engine import Exposure, Model, Reading

# Search grid in units of the cell pitch (offset) and of the nominal cell
# (width, height). Small on purpose: it corrects framing, not placement.
OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)
WIDTHS = (0.72, 0.86, 1.0)
# 0.72 was worth 2.4 points of whole readings on loosely found windows.
HEIGHTS = (0.72, 0.86, 1.0)
# Orientation log-odds margin at or below which the window is upside down.
# Calibrated on synthetic strips: 70 % of turned strips caught, 7 % false.
UPSIDE_DOWN_MARGIN = -4.0


@dataclass(frozen=True, slots=True)
class Raster:
    """How cells are cut from the window."""

    # Shift of the whole raster, in cell pitches.
    offset: float = 0.0
    # Cell width as a fraction of the pitch.
    width: float = 1.0
    # Cell height as a fraction of the window height.
    height: float = 1.0

    def __post_init__(self) -> None:
        """Reject rasters that cut no sensible cells; raises MeterDataError."""
        checks.within(checks.number(self.offset, "offset"), -0.5, 0.5, "offset")
        checks.within(checks.number(self.width, "width"), 0.1, 1.0, "width")
        checks.within(checks.number(self.height, "height"), 0.1, 1.0, "height")


@dataclass(frozen=True, slots=True)
class Calibration:
    """
    Raster and orientation of one installation, from :meth:`MeterReader.calibrate`.

    Store it next to the layout and pass it to every read. It belongs to the
    layout it was made for; calibrate again after changing the layout.
    """

    raster: Raster = Raster()
    # The meter is mounted upside down: wheels and dials are read turned.
    upside_down: bool = False
    # Orientation evidence: log-odds upright minus turned, summed over frames.
    orientation_margin: float = 0.0
    # Mean wheel confidence on the chosen raster, 0 to 1.
    score: float = 0.0
    # Share of the search grid reading the same whole digits as the chosen
    # raster, 0 to 1. Around 0.8 when the window is found well, around 0.3
    # when the readings are wrong; low values mean the window needs fixing.
    agreement: float = 0.0
    # Exposure of each wheel, then each dial, on the calibration frames; a
    # read faults on departures from it. Empty: fixed limits only.
    exposure: tuple[Exposure, ...] = ()

    def __post_init__(self) -> None:
        """Reject values a calibration cannot have; raises MeterDataError."""
        if not isinstance(self.raster, Raster):
            raise MeterDataError(f"raster must be a Raster, got {self.raster!r}")
        object.__setattr__(self, "exposure", tuple(self.exposure))
        for item in self.exposure:
            if not isinstance(item, Exposure):
                raise MeterDataError(f"exposure must be Exposure, got {item!r}")
        checks.flag(self.upside_down, "upside_down")
        checks.number(self.orientation_margin, "orientation_margin")
        checks.within(checks.number(self.score, "score"), 0.0, 1.0, "score")
        checks.within(checks.number(self.agreement, "agreement"), 0.0, 1.0, "agreement")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible form."""
        return {
            "offset": self.raster.offset,
            "width": self.raster.width,
            "height": self.raster.height,
            "upside_down": self.upside_down,
            "orientation_margin": self.orientation_margin,
            "score": self.score,
            "agreement": self.agreement,
            "exposure": [item.as_list() for item in self.exposure],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Calibration:
        """Rebuild a calibration from :meth:`as_dict`; raises MeterDataError."""
        try:
            data = checks.mapping(data, "calibration")
            return cls(
                Raster(data["offset"], data["width"], data["height"]),
                data["upside_down"],
                data.get("orientation_margin", 0.0),
                data.get("score", 0.0),
                data.get("agreement", 0.0),
                _exposure(data.get("exposure", [])),
            )
        except KeyError as err:
            raise MeterDataError(f"invalid calibration: missing {err}") from err
        except MeterDataError as err:
            raise MeterDataError(f"invalid calibration: {err}") from err


def _exposure(data: object) -> tuple[Exposure, ...]:
    if isinstance(data, str | bytes) or not isinstance(data, Sequence):
        raise MeterDataError(f"exposure must be a list, got {data!r}")
    items = []
    for item in data:
        if isinstance(item, str | bytes) or not isinstance(item, Sequence):
            raise MeterDataError(f"exposure must be lists of 5 numbers, got {item!r}")
        if len(item) != 5:
            raise MeterDataError(f"exposure must be lists of 5 numbers, got {item!r}")
        items.append(Exposure(*item))
    return tuple(items)


def pixels(image: Image.Image, hw: tuple[int, int]) -> npt.NDArray[np.float32]:
    """Return a crop resized to a reader's ``(height, width)`` as float RGB."""
    h, w = hw
    return np.asarray(
        image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), np.float32
    )


def median_exposure(samples: Sequence[Sequence[Exposure]]) -> tuple[Exposure, ...]:
    """Return the per-element median of several frames' exposures."""
    return tuple(
        Exposure(*(float(v) for v in np.median([e.as_list() for e in column], axis=0)))
        for column in zip(*samples, strict=True)
    )


GRID = tuple(Raster(o, w, h) for o in OFFSETS for w in WIDTHS for h in HEIGHTS)


def cells(window: Image.Image, wheels: int, raster: Raster) -> list[Image.Image]:
    """Cut the window into one cell per wheel, left to right."""
    width, height = window.size
    pitch = width / wheels
    cell_w, cell_h = pitch * raster.width, height * raster.height
    y0 = (height - cell_h) / 2
    out = []
    for i in range(wheels):
        cx = pitch * (i + 0.5 + raster.offset)
        left, top = int(cx - cell_w / 2), int(y0)
        # At least one pixel, so a tiny window is judged blank, not an error.
        right = max(int(cx + cell_w / 2), left + 1)
        bottom = max(int(y0 + cell_h), top + 1)
        out.append(window.crop((left, top, right, bottom)))
    return out


def read_cells(
    model: Model, crops: Sequence[Image.Image], *, views: int = 1
) -> list[Reading]:
    """Read each crop with ``model``."""
    return [model.inspect(pixels(c, model.input_hw), views=views) for c in crops]


def orientation_margin(model: Model, window: Image.Image, wheels: int) -> float:
    """
    Return log-odds that the window is upright rather than turned.

    Compared against the window's own rotation rather than a fixed cutoff,
    since the orientation head reads real crops as more upright across the
    board. The cells are independent evidence about one question, so their
    log-odds add.
    """

    def evidence(image: Image.Image) -> float:
        readings = read_cells(model, cells(image, wheels, Raster()))
        p = np.clip([r.upside_down for r in readings], 1e-4, 1 - 1e-4)
        return float(np.sum(np.log((1 - p) / p)))

    return evidence(window) - evidence(window.rotate(180))


def whole_digits(readings: Sequence[Reading]) -> int:
    """Return the whole number the wheels show, the key for agreement."""
    return int(compose([r.value for r in readings]).value)


def calibrate(model: Model, windows: Sequence[Image.Image], wheels: int) -> Calibration:
    """
    Settle orientation and raster for an installation from upright-cut windows.

    Alignment is scored by the value head, orientation by the orientation
    head: scoring both by orientation let a raster that merely looked upright
    win, and did worse than not calibrating.
    """
    margin = sum(orientation_margin(model, w, wheels) for w in windows)
    upside_down = margin <= UPSIDE_DOWN_MARGIN
    views = [w.rotate(180) if upside_down else w for w in windows]
    scores = np.zeros(len(GRID))
    keys: list[list[int]] = [[] for _ in windows]
    for g, raster in enumerate(GRID):
        for i, view in enumerate(views):
            readings = read_cells(model, cells(view, wheels, raster))
            scores[g] += float(np.mean([r.confidence for r in readings]))
            keys[i].append(whole_digits(readings))
    best = int(np.argmax(scores))
    agreement = float(
        np.mean([np.mean([k == frame[best] for k in frame]) for frame in keys])
    )
    return Calibration(
        GRID[best], upside_down, margin, float(scores[best] / len(views)), agreement
    )
