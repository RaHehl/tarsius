"""Image decoding."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .errors import ImageError


def load_image(data: bytes, *, draft_side: int = 1024) -> Image.Image:
    """
    Decode an encoded image to RGB.

    JPEGs are decoded at the smallest scale whose longer side is still at
    least ``draft_side``, since full 4K decoding costs more than the feature
    extraction itself. Other formats are decoded at full size.
    """
    if draft_side < 1:
        raise ValueError("draft_side must be positive")
    try:
        image = Image.open(BytesIO(data))
        width, height = image.size
        scale = draft_side / max(width, height)
        if scale < 1:
            # draft keeps both sides at or above the request.
            image.draft(
                "RGB", (max(1, int(width * scale)), max(1, int(height * scale)))
            )
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as err:
        raise ImageError(f"cannot decode image: {err}") from err
