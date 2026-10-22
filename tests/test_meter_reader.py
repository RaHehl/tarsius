from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from meter_scenes import dial_face, layout, meter_frame
from PIL import Image, ImageFilter

from tarsius import BackendError, BlankFrameError, ImageError, Roi
from tarsius.image import encode_jpeg
from tarsius.meter import (
    Calibration,
    Dial,
    MeterLayout,
    MeterReader,
    Rejection,
)
from tarsius.meter import reader as reader_module


@pytest.fixture(scope="module")
def reader() -> MeterReader:
    return MeterReader()


def _flipped(roi: Roi) -> Roi:
    left, top, right, bottom = roi.as_tuple()
    return Roi(1 - right, 1 - bottom, 1 - left, 1 - top)


def test_reads_wheels_and_dials(reader: MeterReader) -> None:
    # The last wheel stands at 7.884, coupled to dials at 8.84 and 8.4.
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    reading = reader.read(frame, layout())
    assert reading.accepted
    assert reading.value == pytest.approx(627.884, abs=0.01)
    assert reading.candidate == reading.value
    assert reading.resolution == pytest.approx(0.01)
    assert len(reading.wheels) == 5
    assert len(reading.dials) == 2
    assert reading.faults == ()
    assert reading.residual < 0.2
    assert 0.5 < reading.confidence <= 1.0


def test_reads_jpeg_bytes(reader: MeterReader) -> None:
    data = encode_jpeg(meter_frame("01234", ()))
    assert reader.read(data, layout(dials=0)).value == pytest.approx(1234, abs=0.2)


def test_rolling_last_wheel_without_dials(reader: MeterReader) -> None:
    reading = reader.read(meter_frame("00629", (), roll=0.5), layout(dials=0))
    assert reading.value == pytest.approx(629.5, abs=0.15)


def test_decimal_wheels(reader: MeterReader) -> None:
    base = layout(dials=0)
    decimal = MeterLayout(base.digits, base.wheels, decimal_wheels=2)
    reading = reader.read(meter_frame("12345", ()), decimal)
    assert reading.value == pytest.approx(123.45, abs=0.005)
    assert reading.resolution == pytest.approx(0.01)


def test_rotated_frame(reader: MeterReader) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884, rotation=20)
    reading = reader.read(frame, layout(rotation=20))
    assert reading.value == pytest.approx(627.884, abs=0.01)


def test_counterclockwise_dial(reader: MeterReader) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    base = layout()
    # Mirror the second dial in the frame: it now turns counter-clockwise.
    left, top, right, _bottom = base.dials[1].roi.box(frame.size)
    face = dial_face(8.4, (right - left) // 2).transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )
    frame.paste(face, (left, top))
    ccw = MeterLayout(
        base.digits, base.wheels, (base.dials[0], Dial(base.dials[1].roi, True))
    )
    assert reader.read(frame, ccw).value == pytest.approx(627.884, abs=0.01)


def test_calibration_of_an_upright_meter(reader: MeterReader) -> None:
    frames = [meter_frame("00627", (8.84, 8.4), roll=0.884, seed=s) for s in (1, 2)]
    calibration = reader.calibrate(frames, layout())
    assert not calibration.upside_down
    assert calibration.orientation_margin > 0
    assert calibration.score > 0.8
    assert calibration.agreement > 0.5
    reading = reader.read(frames[0], layout(), calibration)
    assert reading.value == pytest.approx(627.884, abs=0.01)


def test_calibration_finds_an_upside_down_meter(reader: MeterReader) -> None:
    frame = meter_frame("13579", (2.46, 4.6), roll=0.246, rotation=180)
    base = layout()
    turned = MeterLayout(
        _flipped(base.digits), 5, tuple(Dial(_flipped(d.roi)) for d in base.dials)
    )
    calibration = reader.calibrate(encode_jpeg(frame), turned)
    assert calibration.upside_down
    reading = reader.read(frame, turned, calibration)
    assert reading.value == pytest.approx(13579.246, abs=0.01)


def test_calibration_needs_a_frame(reader: MeterReader) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        reader.calibrate([], layout())


def test_region_without_digits_is_rejected(reader: MeterReader) -> None:
    # The window points at the plain face below the counter.
    plain = MeterLayout(Roi(0.4, 0.4, 0.6, 0.5), 4)
    reading = reader.read(meter_frame(), plain)
    assert not reading.accepted
    assert reading.value is None
    assert reading.rejection in (Rejection.FAULT, Rejection.LOW_CONFIDENCE)


def test_blurred_wheels_report_a_fault(reader: MeterReader) -> None:
    frame = meter_frame()
    box = layout().digits.box(frame.size)
    frame.paste(frame.crop(box).filter(ImageFilter.GaussianBlur(6)), box[:2])
    reading = reader.read(frame, layout())
    assert reading.rejection is Rejection.FAULT
    assert reading.faults
    assert reading.candidate is not None


def test_load_ahead(reader: MeterReader) -> None:
    reader.load()


def test_contradicting_dials_are_rejected(reader: MeterReader) -> None:
    # x0.01 at 1.2 needs x0.1 just past a mark, it stands at 6.6.
    reading = reader.read(meter_frame("70315", (6.6, 1.2)), layout())
    assert reading.rejection is Rejection.INCONSISTENT
    assert reading.residual > 0.4


def test_low_confidence_threshold() -> None:
    strict = MeterReader(min_confidence=1.0)
    reading = strict.read(meter_frame(), layout())
    assert reading.rejection is Rejection.LOW_CONFIDENCE


def test_views_average_shifted_crops() -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    reading = MeterReader(views=3).read(frame, layout())
    assert reading.value == pytest.approx(627.884, abs=0.01)


def test_blank_frame(reader: MeterReader) -> None:
    with pytest.raises(BlankFrameError):
        reader.read(Image.new("RGB", (640, 480), (128, 128, 128)), layout())


def test_undecodable_frame(reader: MeterReader) -> None:
    with pytest.raises(ImageError):
        reader.read(b"<html>not a frame</html>", layout())


def test_large_frame_is_bounded(reader: MeterReader) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884).resize((3200, 2400))
    assert reader.read(frame, layout()).value == pytest.approx(627.884, abs=0.01)


def test_threads_share_the_readers(reader: MeterReader) -> None:
    frames = [meter_frame("00627", (8.84, 8.4), roll=0.884, seed=s) for s in range(4)]
    with ThreadPoolExecutor(4) as pool:
        readings = list(pool.map(lambda f: reader.read(f, layout()), frames * 2))
    assert readings[:4] == readings[4:]
    assert np.allclose([r.candidate for r in readings], 627.884, atol=0.01)


def test_default_calibration_is_upright_and_nominal(reader: MeterReader) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    assert reader.read(frame, layout()) == reader.read(frame, layout(), Calibration())


def test_missing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reader_module, "_SHARED", reader_module._Models())
    monkeypatch.setattr(reader_module, "files", lambda _package: tmp_path)
    with pytest.raises(BackendError, match="missing"):
        MeterReader().load()


def test_unloadable_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "digits.npz").write_bytes(b"garbage")
    monkeypatch.setattr(reader_module, "_SHARED", reader_module._Models())
    monkeypatch.setattr(reader_module, "files", lambda _package: tmp_path)
    with pytest.raises(BackendError, match="cannot load"):
        MeterReader().read(meter_frame(), layout())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_confidence": 1.5}, "min_confidence"),
        ({"max_residual": 0.0}, "max_residual"),
        ({"views": 6}, "views"),
    ],
)
def test_invalid_settings(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MeterReader(**kwargs)  # type: ignore[arg-type]


def test_bundled_models_match_their_checksums() -> None:
    models = files("tarsius.meter") / "models"
    sums = dict(
        reversed(line.split())
        for line in (models / "SHA256SUMS").read_text().splitlines()
    )
    bundled = {
        f.name for f in models.iterdir() if f.name not in ("MODELS.md", "SHA256SUMS")
    }
    assert set(sums) == bundled
    for name, digest in sums.items():
        assert hashlib.sha256((models / name).read_bytes()).hexdigest() == digest
