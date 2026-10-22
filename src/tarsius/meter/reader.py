"""Read a meter from a frame: wheels, dials, carry rule and rejection."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageStat

from ..errors import BackendError, MeterDataError
from . import checks
from .compose import compose
from .engine import (
    DIAL_GLARE_SHARE,
    GLARE_SHARE,
    Exposure,
    Fault,
    Model,
    Reading,
    exposure_fault,
)
from .frame import crop, decode_frame
from .layout import MeterLayout
from .strip import Calibration, calibrate, cells, median_exposure, pixels

DIGITS_MODEL = "digits.npz"
DIALS_MODEL = "dials.npz"

# Accepted readings this sure of every digit move the exposure baseline by
# ADAPT_RATE each: half way in about 14 readings.
ADAPT_CONFIDENCE = 0.7
ADAPT_RATE = 0.05
# Defaults chosen on the maintainer's meter, see the README.
MIN_CONFIDENCE = 0.5
# A window without digits leaves every wheel unsure: its mean confidence
# stayed below 0.67 on a foreign meter's housing, readable windows above
# 0.82 on both meters.
MIN_MEAN_CONFIDENCE = 0.75
MAX_RESIDUAL = 0.4


class Rejection(StrEnum):
    """Why a reading has no value."""

    # A wheel or dial shows a capture fault: focus, glare, dark or offset.
    FAULT = "fault"
    # A wheel or dial is not sure enough of its digit.
    LOW_CONFIDENCE = "low_confidence"
    # The wheels and dials contradict each other under the carry rule.
    INCONSISTENT = "inconsistent"
    # An upper wheel near a carry fits two digits about equally well.
    AMBIGUOUS = "ambiguous"
    # Lower than the last accepted value, from Plausibility.
    DECREASED = "decreased"
    # Rose faster than the meter can count, from Plausibility.
    TOO_FAST = "too_fast"
    # No confirmed value to compare against yet, from Plausibility.
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True, slots=True)
class MeterReading:
    """Outcome of one read."""

    # Set only when no rejection applies.
    value: float | None
    # The composed value, also when rejected, e.g. to ask "is this right?".
    candidate: float
    rejection: Rejection | None
    # Lowest confidence of all wheels and dials, 0 to 1.
    confidence: float
    # Largest carry-rule misfit in digits, 0 to 0.5; see compose().
    residual: float
    # Unit of the last wheel or dial; the value carries one more place.
    resolution: float
    wheels: tuple[Reading, ...] = ()
    dials: tuple[Reading, ...] = ()
    # Other values the wheels allow near a carry, best fit first, within
    # max_residual; Plausibility picks the one that fits the last value.
    alternatives: tuple[float, ...] = ()
    # How badly each alternative fits, like residual; empty: as the value.
    alternative_residuals: tuple[float, ...] = ()
    # Mean grey of the frame, 0 when unknown; see MeterReader.adapt().
    light: float = 0.0

    @property
    def accepted(self) -> bool:
        """Return True when the reading has a value."""
        return self.rejection is None

    @property
    def faults(self) -> tuple[Fault, ...]:
        """Return the distinct capture faults of wheels and dials, worst first."""
        found = {
            r.fault for r in (*self.wheels, *self.dials) if r.fault is not Fault.OK
        }
        return tuple(f for f in Fault if f in found)

    def rejected(self, rejection: Rejection) -> MeterReading:
        """Return a copy without value, rejected for ``rejection``."""
        return dataclasses.replace(self, value=None, rejection=rejection)


class _Models:
    """Readers loaded once per process; loading holds the lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, Model] = {}

    def get(self, name: str) -> Model:
        with self._lock:
            if (model := self._models.get(name)) is None:
                resource = files("tarsius.meter") / "models" / name
                if not resource.is_file():
                    raise BackendError(f"bundled reader model {name} is missing")
                with resource.open("rb") as handle:
                    model = self._models[name] = Model.load(handle)
            return model


_SHARED = _Models()


class MeterReader:
    """
    Reads meters with the bundled wheel and dial readers.

    The readers load on first use, about 2.6 MB, shared by every reader of
    the process. All methods block, call them in an executor. They are safe
    to call from several threads.
    """

    def __init__(
        self,
        *,
        min_confidence: float = MIN_CONFIDENCE,
        max_residual: float = MAX_RESIDUAL,
        views: int = 1,
        min_mean_confidence: float = MIN_MEAN_CONFIDENCE,
    ) -> None:
        """
        Create a reader; loads nothing yet.

        A reading is rejected when a wheel or dial is less sure than
        ``min_confidence``, the wheels and dials on average less sure than
        ``min_mean_confidence`` (a window without digits), or the carry
        rule leaves a misfit above ``max_residual`` digits. ``views`` of 2
        to 5 also reads every crop shifted by a pixel and averages, which
        costs as many times the time.
        """
        if not 0.0 <= checks.number(min_confidence, "min_confidence") <= 1.0:
            raise MeterDataError("min_confidence must be 0 to 1")
        if not 0.0 < checks.number(max_residual, "max_residual") <= 0.5:
            raise MeterDataError("max_residual must be above 0 and at most 0.5")
        if not 1 <= checks.integer(views, "views") <= 5:
            raise MeterDataError("views must be 1 to 5")
        if not 0.0 <= checks.number(min_mean_confidence, "min_mean_confidence") <= 1.0:
            raise MeterDataError("min_mean_confidence must be 0 to 1")
        self._min_mean_confidence = min_mean_confidence
        self._min_confidence = min_confidence
        self._max_residual = max_residual
        self._views = views

    def load(self) -> None:
        """Load the readers now instead of on first use; raises BackendError."""
        _SHARED.get(DIGITS_MODEL)
        _SHARED.get(DIALS_MODEL)

    def calibrate(
        self,
        images: bytes | Image.Image | Sequence[bytes | Image.Image],
        layout: MeterLayout,
    ) -> Calibration:
        """
        Find orientation and cell raster of an installation.

        Pass one frame or several from different moments; more frames give
        steadier evidence. Costs about 50 reads of the wheels per frame, so
        run it at setup or after the camera moved, not per reading.
        """
        frames = [images] if isinstance(images, bytes | Image.Image) else images
        if not frames:
            raise ValueError("calibration needs at least one frame")
        decoded = [decode_frame(frame) for frame in frames]
        windows = [crop(f, layout.digits, layout.rotation) for f in decoded]
        model = _SHARED.get(DIGITS_MODEL)
        calibration = calibrate(model, windows, layout.wheels)
        # What each wheel and dial looks like now, to tell glare from a drum,
        # each frame scaled to the frames' middle light level.
        lights = [_light(frame) for frame in decoded]
        light = float(np.median(lights))
        samples = [
            [
                _exposure(p, light / level)
                for p in (
                    *self._wheel_pixels(window, layout, calibration),
                    *self._dial_pixels(frame, layout, calibration),
                )
            ]
            for frame, window, level in zip(decoded, windows, lights, strict=True)
        ]
        return dataclasses.replace(
            calibration, exposure=median_exposure(samples), light=light
        )

    def read(
        self,
        image: bytes | Image.Image,
        layout: MeterLayout,
        calibration: Calibration | None = None,
    ) -> MeterReading:
        """
        Read the meter in a frame.

        Raises ImageError for undecodable data, BlankFrameError for a frame
        that shows nothing and BackendError if a reader cannot load. Without
        ``calibration`` the window is cut into equal cells, upright, and
        exposure faults use fixed limits instead of the calibrated look.
        Each crop's exposure is taken at the calibration's light level.
        """
        calibration = calibration or Calibration()
        frame = decode_frame(image)
        light = _light(frame)
        scale = calibration.light / light if calibration.light else 1.0
        window = crop(frame, layout.digits, layout.rotation)
        digits = _SHARED.get(DIGITS_MODEL)
        wheel_pixels = self._wheel_pixels(window, layout, calibration)
        wheels = [digits.inspect(p, views=self._views) for p in wheel_pixels]
        dial_pixels = self._dial_pixels(frame, layout, calibration)
        dials: list[Reading] = []
        if layout.dials:
            model = _SHARED.get(DIALS_MODEL)
            dials = [model.inspect(p, views=self._views) for p in dial_pixels]
        readings = (*wheels, *dials)
        scaled = [_exposure(p, scale) for p in (*wheel_pixels, *dial_pixels)]
        baseline: tuple[Exposure | None, ...] = calibration.exposure
        if len(baseline) != len(readings):
            baseline = (None,) * len(readings)
        shares = glare_shares(layout)
        checked = [
            _exposed(r, exposure, base, share)
            for r, exposure, base, share in zip(
                readings, scaled, baseline, shares, strict=True
            )
        ]
        reading = self._judge(
            layout, tuple(checked[: layout.wheels]), tuple(checked[layout.wheels :])
        )
        return dataclasses.replace(reading, light=light)

    def adapt(self, calibration: Calibration, reading: MeterReading) -> Calibration:
        """
        Return the calibration with its exposure moved towards a reading's.

        Pass readings that were accepted, by Plausibility as well; others,
        and unsure, faulted or poorly fitting ones, leave it unchanged. Each
        such reading moves the baseline by ADAPT_RATE, so slow changes of
        the scene are learnt while a sudden glare spot still stands out. A
        calibration without exposure takes the reading's as it is. Store
        the result in place of the calibration.
        """
        if (
            reading.value is None
            or reading.faults
            or reading.confidence < ADAPT_CONFIDENCE
            or reading.residual > self._max_residual / 2
        ):
            return calibration
        current = [r.exposure for r in (*reading.wheels, *reading.dials)]
        if any(e is None for e in current) or not reading.light:
            return calibration
        observed = [e for e in current if e is not None]
        if len(calibration.exposure) != len(observed) or not calibration.light:
            return dataclasses.replace(
                calibration, exposure=tuple(observed), light=reading.light
            )
        moved = tuple(
            Exposure(
                *(
                    old + ADAPT_RATE * (new - old)
                    for old, new in zip(base.as_list(), now.as_list(), strict=True)
                )
            )
            for base, now in zip(calibration.exposure, observed, strict=True)
        )
        return dataclasses.replace(calibration, exposure=moved)

    @staticmethod
    def _wheel_pixels(
        window: Image.Image, layout: MeterLayout, calibration: Calibration
    ) -> list[npt.NDArray[np.float32]]:
        if calibration.upside_down:
            window = window.rotate(180)
        hw = _SHARED.get(DIGITS_MODEL).input_hw
        return [pixels(c, hw) for c in cells(window, layout.wheels, calibration.raster)]

    @staticmethod
    def _dial_pixels(
        frame: Image.Image, layout: MeterLayout, calibration: Calibration
    ) -> list[npt.NDArray[np.float32]]:
        if not layout.dials:
            return []
        hw = _SHARED.get(DIALS_MODEL).input_hw
        out = []
        for dial in layout.dials:
            face = crop(frame, dial.roi, layout.rotation)
            if calibration.upside_down:
                face = face.rotate(180)
            if dial.counterclockwise:
                # A counter-clockwise dial is the mirror of a clockwise one.
                face = face.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            out.append(pixels(face, hw))
        return out

    def _judge(
        self,
        layout: MeterLayout,
        wheels: tuple[Reading, ...],
        dials: tuple[Reading, ...],
    ) -> MeterReading:
        composition = compose(
            [r.value for r in wheels],
            [r.value for r in dials],
            max_residual=self._max_residual,
        )
        scale = 10**layout.decimal_wheels
        candidate = round(composition.value / scale, layout.decimals + 1)
        alternatives = tuple(
            round(value / scale, layout.decimals + 1)
            for value, _ in composition.alternatives
        )
        resolution = 10.0**-layout.decimals
        confidence = min(r.confidence for r in (*wheels, *dials))
        reading = MeterReading(
            candidate,
            candidate,
            None,
            confidence,
            composition.residual,
            resolution,
            wheels,
            dials,
            alternatives,
            tuple(residual for _, residual in composition.alternatives),
        )
        if reading.faults:
            return reading.rejected(Rejection.FAULT)
        mean = sum(r.confidence for r in (*wheels, *dials)) / len((*wheels, *dials))
        if confidence < self._min_confidence or mean < self._min_mean_confidence:
            return reading.rejected(Rejection.LOW_CONFIDENCE)
        if composition.residual > self._max_residual:
            return reading.rejected(Rejection.INCONSISTENT)
        if composition.ambiguous:
            return reading.rejected(Rejection.AMBIGUOUS)
        return reading


def _exposed(
    reading: Reading,
    exposure: Exposure,
    baseline: Exposure | None,
    glare_share: float,
) -> Reading:
    """
    Return the reading with its scaled exposure and the fault it shows.

    ``reading.exposure`` is the crop as the camera saw it, ``exposure`` the
    same scaled to the calibration's light level.
    """
    fault = exposure_fault(
        exposure,
        baseline,
        reading.confidence,
        glare_share=glare_share,
        raw=reading.exposure,
    )
    if fault is Fault.OK:
        return dataclasses.replace(reading, exposure=exposure)
    return dataclasses.replace(reading, exposure=exposure, fault=fault)


def glare_shares(layout: MeterLayout) -> tuple[float, ...]:
    """Return the clipped share allowed without a calibration, wheels then dials."""
    return (GLARE_SHARE,) * layout.wheels + (DIAL_GLARE_SHARE,) * len(layout.dials)


def _light(frame: Image.Image) -> float:
    """Return the mean grey level of a frame, at least 1."""
    return max(1.0, float(ImageStat.Stat(frame.convert("L")).mean[0]))


def _exposure(pixels: npt.NDArray[np.float32], scale: float) -> Exposure:
    """Return a crop's exposure as if lit ``scale`` times as bright."""
    if scale == 1.0:
        return Exposure.of(pixels)
    return Exposure.of(np.clip(pixels * np.float32(scale), 0.0, 255.0))
