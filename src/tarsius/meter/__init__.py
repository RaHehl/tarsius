"""
Read utility meters: counter wheels and pointer dials in a camera frame.

The caller gives the regions (:class:`MeterLayout`), :class:`MeterReader`
reads them with small bundled numpy networks and composes the value with the
carry rule, :class:`Plausibility` rejects values that go backwards or rise
too fast. ``tarsius.meter.finder`` can suggest the regions; it is alpha.
"""

from .compose import Composition, compose
from .engine import Exposure, Fault, Reading
from .frame import METER_FRAME_SIDE
from .layout import Dial, MeterLayout
from .plausibility import Dispute, Plausibility, PlausibilityState
from .reader import MeterReader, MeterReading, Rejection
from .strip import Calibration, Raster

__all__ = [
    "METER_FRAME_SIDE",
    "Calibration",
    "Composition",
    "Dial",
    "Dispute",
    "Exposure",
    "Fault",
    "MeterLayout",
    "MeterReader",
    "MeterReading",
    "Plausibility",
    "PlausibilityState",
    "Raster",
    "Reading",
    "Rejection",
    "compose",
]
