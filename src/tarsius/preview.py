"""Frames with regions drawn on them, for setup and debugging views."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import ImageDraw

from .image import FRAME_SIDE, encode_jpeg, load_image
from .roi import Roi

OUTLINE = (255, 0, 255)


def outline(
    data: bytes, rois: Roi | Sequence[Roi], *, max_side: int = FRAME_SIDE
) -> bytes:
    """
    Return the frame as JPEG with each region outlined.

    The outline uses the same box rule as the crop; at the detector's frame
    side it marks exactly the compared pixels.
    """
    image = load_image(data, max_side=max_side)
    draw = ImageDraw.Draw(image)
    width = max(2, round(max(image.size) / 240))
    for roi in [rois] if isinstance(rois, Roi) else rois:
        left, top, right, bottom = roi.box(image.size)
        draw.rectangle((left, top, right - 1, bottom - 1), outline=OUTLINE, width=width)
    return encode_jpeg(image, quality=85)
