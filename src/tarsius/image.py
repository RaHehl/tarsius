"""Image decoding and encoding."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .errors import ImageError

# Longer side frames are normalized to. References and new frames must share
# it, or the same scene compares as different.
FRAME_SIDE = 1024


def load_image(data: bytes, *, max_side: int = FRAME_SIDE) -> Image.Image:
    """
    Decode an encoded image to RGB with the longer side at most ``max_side``.

    JPEGs are decoded at a reduced scale first, since full 4K decoding costs
    more than the feature extraction itself.
    """
    if max_side < 1:
        raise ValueError("max_side must be positive")
    try:
        encoded = Image.open(BytesIO(data))
        width, height = encoded.size
        scale = max_side / max(width, height)
        if scale < 1:
            # draft keeps both sides at or above the request.
            encoded.draft(
                "RGB", (max(1, int(width * scale)), max(1, int(height * scale)))
            )
        image = encoded.convert("RGB")
    except (UnidentifiedImageError, OSError) as err:
        raise ImageError(f"cannot decode image: {err}") from err
    image.thumbnail((max_side, max_side))
    return image


def encode_jpeg(image: Image.Image, *, quality: int = 95) -> bytes:
    """Encode an image as JPEG."""
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()
