from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from tarsius import FULL_FRAME, ImageError, Roi, load_image


def _jpeg(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, "JPEG")
    return buffer.getvalue()


def test_crop_rounds_fractions() -> None:
    image = Image.new("RGB", (100, 100))
    assert Roi(0.29, 0.5, 0.71, 1.0).crop(image).size == (42, 50)


def test_crop_never_empty() -> None:
    image = Image.new("RGB", (10, 10))
    assert Roi(0.999, 0.999, 1.0, 1.0).crop(image).size == (1, 1)
    assert Roi(0.0, 0.0, 0.001, 0.001).crop(image).size == (1, 1)


def test_full_frame() -> None:
    image = Image.new("RGB", (64, 48))
    assert FULL_FRAME.crop(image).size == (64, 48)


@pytest.mark.parametrize(
    "bounds",
    [(0.5, 0.0, 0.5, 1.0), (0.0, 0.6, 1.0, 0.4), (-0.1, 0.0, 1.0, 1.0), (0, 0, 1, 1.2)],
)
def test_invalid_roi(bounds: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="invalid"):
        Roi(*bounds)


def test_from_cells_bounding_box() -> None:
    roi = Roi.from_cells([(1, 0), (2, 1), (1, 1)], columns=4, rows=3)
    assert roi.as_tuple() == pytest.approx((0.25, 0.0, 0.75, 2 / 3))
    corner = Roi.from_cells([(3, 2)], columns=4, rows=3)
    assert corner.as_tuple() == pytest.approx((0.75, 2 / 3, 1.0, 1.0))


@pytest.mark.parametrize("cells", [[], [(4, 0)], [(0, 3)], [(-1, 0)]])
def test_from_cells_invalid(cells: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError):
        Roi.from_cells(cells, columns=4, rows=3)


def test_load_image_drafts_large_jpeg() -> None:
    image = load_image(_jpeg((2688, 1512)), draft_side=1024)
    assert image.mode == "RGB"
    assert image.size == (1344, 756)


def test_load_image_keeps_small_jpeg() -> None:
    assert load_image(_jpeg((640, 360))).size == (640, 360)


def test_load_image_extreme_aspect() -> None:
    assert load_image(_jpeg((4000, 8)), draft_side=64).size[1] >= 1


def test_load_image_png_full_size() -> None:
    buffer = BytesIO()
    Image.new("L", (2000, 30)).save(buffer, "PNG")
    image = load_image(buffer.getvalue())
    assert image.size == (2000, 30)
    assert image.mode == "RGB"


@pytest.mark.parametrize(
    "data", [b"", b"<html>502 Bad Gateway</html>", _jpeg((64, 64))[:60]]
)
def test_load_image_bad_data(data: bytes) -> None:
    with pytest.raises(ImageError):
        load_image(data)


def test_load_image_rejects_bad_side() -> None:
    with pytest.raises(ValueError, match="positive"):
        load_image(_jpeg((10, 10)), draft_side=0)
