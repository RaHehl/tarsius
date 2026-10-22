"""Read a meter from a frame: wheels, dials, carry rule and rejection."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files

import numpy as np
from PIL import Image

from ..errors import BackendError, MeterDataError
from . import checks
from .compose import compose
from .engine import DIAL_GLARE_SHARE, Fault, Model, Reading
from .frame import crop, decode_frame
from .layout import MeterLayout
from .strip import Calibration, calibrate, cells, read_cells

DIGITS_MODEL = "digits.npz"
DIALS_MODEL = "dials.npz"

# Defaults chosen on the maintainer's meter, see the README.
MIN_CONFIDENCE = 0.5
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
    ) -> None:
        """
        Create a reader; loads nothing yet.

        A reading is rejected when a wheel or dial is less sure than
        ``min_confidence`` or the carry rule leaves a misfit above
        ``max_residual`` digits. ``views`` of 2 to 5 also reads every crop
        shifted by a pixel and averages, which costs as many times the time.
        """
        if not 0.0 <= checks.number(min_confidence, "min_confidence") <= 1.0:
            raise MeterDataError("min_confidence must be 0 to 1")
        if not 0.0 < checks.number(max_residual, "max_residual") <= 0.5:
            raise MeterDataError("max_residual must be above 0 and at most 0.5")
        if not 1 <= checks.integer(views, "views") <= 5:
            raise MeterDataError("views must be 1 to 5")
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
        windows = [
            crop(decode_frame(frame), layout.digits, layout.rotation)
            for frame in frames
        ]
        return calibrate(_SHARED.get(DIGITS_MODEL), windows, layout.wheels)

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
        ``calibration`` the window is cut into equal cells, upright.
        """
        calibration = calibration or Calibration()
        frame = decode_frame(image)
        window = crop(frame, layout.digits, layout.rotation)
        if calibration.upside_down:
            window = window.rotate(180)
        wheels = read_cells(
            _SHARED.get(DIGITS_MODEL),
            cells(window, layout.wheels, calibration.raster),
            views=self._views,
        )
        dials: list[Reading] = []
        if layout.dials:
            model = _SHARED.get(DIALS_MODEL)
            h, w = model.input_hw
            for dial in layout.dials:
                face = crop(frame, dial.roi, layout.rotation)
                if calibration.upside_down:
                    face = face.rotate(180)
                if dial.counterclockwise:
                    # A counter-clockwise dial is the mirror of a clockwise one.
                    face = face.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                pixels = np.asarray(
                    face.resize((w, h), Image.Resampling.BILINEAR), np.float32
                )
                dials.append(
                    model.inspect(
                        pixels, views=self._views, glare_share=DIAL_GLARE_SHARE
                    )
                )
        return self._judge(layout, tuple(wheels), tuple(dials))

    def _judge(
        self,
        layout: MeterLayout,
        wheels: tuple[Reading, ...],
        dials: tuple[Reading, ...],
    ) -> MeterReading:
        composition = compose([r.value for r in wheels], [r.value for r in dials])
        scale = 10**layout.decimal_wheels
        candidate = round(composition.value / scale, layout.decimals + 1)
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
        )
        if reading.faults:
            return reading.rejected(Rejection.FAULT)
        if confidence < self._min_confidence:
            return reading.rejected(Rejection.LOW_CONFIDENCE)
        if composition.residual > self._max_residual:
            return reading.rejected(Rejection.INCONSISTENT)
        if composition.ambiguous:
            return reading.rejected(Rejection.AMBIGUOUS)
        return reading
