"""Frame decoding and crops of the rotated frame."""

from __future__ import annotations

import math

from PIL import Image, ImageStat

from ..errors import BlankFrameError
from ..image import load_image
from ..roi import Roi

# Longer side frames are normalized to. Larger than for state detection:
# a wheel must keep enough pixels when the meter is small in the frame.
METER_FRAME_SIDE = 2048
# Grey level spread below which the camera shows nothing, as in StateDetector.
BLANK_STDDEV = 4.0


def decode_frame(image: bytes | Image.Image) -> Image.Image:
    """
    Return an RGB frame, decoded and bounded to ``METER_FRAME_SIDE``.

    Raises ImageError for undecodable data and BlankFrameError for a frame
    that shows nothing.
    """
    if isinstance(image, Image.Image):
        frame = image.convert("RGB")
        if max(frame.size) > METER_FRAME_SIDE:
            frame.thumbnail((METER_FRAME_SIDE, METER_FRAME_SIDE))
    else:
        frame = load_image(image, max_side=METER_FRAME_SIDE)
    if ImageStat.Stat(frame.convert("L")).stddev[0] < BLANK_STDDEV:
        raise BlankFrameError("the frame shows nothing, the camera may be offline")
    return frame


def crop(frame: Image.Image, roi: Roi, rotation: float = 0.0) -> Image.Image:
    """
    Return ``roi`` of the frame turned by ``rotation`` degrees.

    Same pixels as ``frame.rotate(rotation, BILINEAR).crop(...)`` without
    turning the whole frame.
    """
    left, top, right, bottom = roi.box(frame.size)
    if not rotation % 360:
        return frame.crop((left, top, right, bottom))
    # Output to input mapping of Image.rotate, shifted to the box.
    angle = -math.radians(rotation)
    a, b = math.cos(angle), math.sin(angle)
    cx, cy = frame.width / 2, frame.height / 2
    c = cx - a * cx - b * cy
    f = cy + b * cx - a * cy
    return frame.transform(
        (right - left, bottom - top),
        Image.Transform.AFFINE,
        (a, b, a * left + b * top + c, -b, a, -b * left + a * top + f),
        resample=Image.Resampling.BILINEAR,
    )
