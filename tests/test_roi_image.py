from __future__ import annotations

import copy
import pickle
from io import BytesIO

import pytest
from PIL import Image

from tarsius import FULL_FRAME, ImageError, Roi, RoiError, load_image
from tarsius.image import encode_jpeg


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
    with pytest.raises(RoiError, match="invalid"):
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
    image = load_image(_jpeg((2688, 1512)), max_side=1024)
    assert image.mode == "RGB"
    assert image.size == (1024, 576)


def test_load_image_keeps_small_jpeg() -> None:
    assert load_image(_jpeg((640, 360))).size == (640, 360)


def test_load_image_extreme_aspect() -> None:
    assert load_image(_jpeg((4000, 8)), max_side=64).size[1] >= 1


def test_load_image_png_bounded() -> None:
    buffer = BytesIO()
    Image.new("L", (2000, 30)).save(buffer, "PNG")
    image = load_image(buffer.getvalue())
    assert image.size == (1024, 15)
    assert image.mode == "RGB"


@pytest.mark.parametrize(
    "data", [b"", b"<html>502 Bad Gateway</html>", _jpeg((64, 64))[:60]]
)
def test_load_image_bad_data(data: bytes) -> None:
    with pytest.raises(ImageError):
        load_image(data)


def test_load_image_rejects_bad_side() -> None:
    with pytest.raises(ValueError, match="positive"):
        load_image(_jpeg((10, 10)), max_side=0)


@pytest.mark.parametrize(
    ("bounds", "axis"),
    [((50, 0, 50, 100), "horizontal"), ((0, 60, 100, 40), "vertical")],
)
def test_roi_error_axis(bounds: tuple[float, float, float, float], axis: str) -> None:
    with pytest.raises(RoiError) as err:
        Roi.from_percent(*bounds)
    assert err.value.axis == axis


def test_percent_roundtrip() -> None:
    roi = Roi.from_percent(12.5, 10, 56, 45.5)
    assert roi.as_tuple() == pytest.approx((0.125, 0.1, 0.56, 0.455))
    assert Roi.from_percent(7, 14, 29, 57).as_percent() == (7, 14, 29, 57)
    assert Roi(1 / 3, 0, 1, 1).as_percent() == (33.333333, 0.0, 100.0, 100.0)
    assert all(type(v) is float for v in Roi(0.3, 0, 1, 1).as_percent())


def test_box_matches_crop() -> None:
    roi = Roi(0.29, 0.5, 0.71, 1.0)
    assert roi.box((100, 100)) == (29, 50, 71, 100)
    assert Roi(0.0, 0.29, 1.0, 0.71).box((100, 100)) == (0, 29, 100, 71)
    assert Roi(0.999, 0.0, 1.0, 0.001).box((10, 10)) == (9, 0, 10, 1)


def test_encode_jpeg_roundtrip() -> None:
    image = Image.new("RGB", (40, 30), (200, 10, 10))
    assert load_image(encode_jpeg(image)).size == (40, 30)


def test_roi_error_survives_copy() -> None:
    for bounds, axis in (
        ((0.5, 0, 0.4, 1), "horizontal"),
        ((0, 0.6, 1, 0.5), "vertical"),
    ):
        with pytest.raises(RoiError) as err:
            Roi(*bounds)
        err.value.add_note("seen in a test")
        for clone in (copy.copy(err.value), pickle.loads(pickle.dumps(err.value))):
            assert clone.axis == axis
            assert str(clone) == str(err.value)
            assert clone.__notes__ == ["seen in a test"]


def test_encode_jpeg_accepts_alpha() -> None:
    image = Image.new("RGBA", (10, 10), (1, 2, 3, 128))
    assert load_image(encode_jpeg(image)).mode == "RGB"


def test_load_image_rejects_huge_non_jpeg() -> None:
    buffer = BytesIO()
    Image.new("1", (5000, 5000)).save(buffer, "PNG")
    with pytest.raises(ImageError, match="too large"):
        load_image(buffer.getvalue())
