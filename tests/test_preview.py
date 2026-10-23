from __future__ import annotations

from PIL import Image

from tarsius import Roi, load_image, outline
from tarsius.image import encode_jpeg
from tarsius.preview import HIGHLIGHT

MAGENTA = (255, 0, 255)


def _close(pixel: tuple[int, int, int], color: tuple[int, int, int]) -> bool:
    return all(abs(a - b) < 60 for a, b in zip(pixel, color, strict=True))


def test_outline_draws_the_crop_box() -> None:
    frame = encode_jpeg(Image.new("RGB", (1920, 1080), (40, 40, 40)))
    roi = Roi(0.25, 0.25, 0.75, 0.75)

    image = load_image(outline(frame, roi, max_side=960))

    assert image.size == (960, 540)
    left, top, right, bottom = roi.box(image.size)
    assert _close(image.getpixel((left + 1, (top + bottom) // 2)), MAGENTA)
    assert _close(image.getpixel((right - 2, (top + bottom) // 2)), MAGENTA)
    assert _close(image.getpixel(((left + right) // 2, (top + bottom) // 2)), (40,) * 3)


def test_outline_several_regions() -> None:
    frame = encode_jpeg(Image.new("RGB", (400, 200), (0, 0, 0)))
    rois = [Roi(0.0, 0.0, 0.4, 0.5), Roi(0.6, 0.5, 1.0, 1.0)]

    image = load_image(outline(frame, rois))

    assert _close(image.getpixel((1, 50)), MAGENTA)
    assert _close(image.getpixel((398, 150)), MAGENTA)


def test_outline_highlights_one_region() -> None:
    frame = encode_jpeg(Image.new("RGB", (400, 200), (0, 0, 0)))
    roi = Roi(0.0, 0.0, 0.4, 0.5)
    current = Roi(0.6, 0.5, 1.0, 1.0)

    image = load_image(outline(frame, [roi], highlight=current))

    assert _close(image.getpixel((1, 50)), MAGENTA)
    assert _close(image.getpixel((398, 150)), HIGHLIGHT)
    assert not _close(image.getpixel((398, 150)), MAGENTA)


def test_outline_highlight_on_a_light_frame() -> None:
    frame = encode_jpeg(Image.new("RGB", (400, 200), (240, 240, 240)))

    image = load_image(outline(frame, [], highlight=Roi(0.25, 0.25, 0.75, 0.75)))

    assert _close(image.getpixel((101, 100)), HIGHLIGHT)


def test_outline_without_highlight_is_unchanged() -> None:
    frame = encode_jpeg(Image.new("RGB", (400, 200), (30, 60, 90)))
    rois = [Roi(0.1, 0.1, 0.4, 0.5), Roi(0.6, 0.5, 0.9, 0.9)]

    assert outline(frame, rois, highlight=None) == outline(frame, rois)
