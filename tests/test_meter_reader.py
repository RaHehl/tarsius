from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from meter_scenes import dial_face, layout, meter_frame
from PIL import Image, ImageDraw, ImageFilter

from tarsius import (
    BackendError,
    BlankFrameError,
    ImageError,
    MeterDataError,
    Roi,
)
from tarsius.image import encode_jpeg
from tarsius.meter import (
    Calibration,
    Dial,
    Fault,
    MeterLayout,
    MeterReader,
    Reading,
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
    reading = reader.read(meter_frame("00624", (), roll=0.5), layout(dials=0))
    assert reading.value == pytest.approx(624.5, abs=0.15)


def test_upper_wheel_near_a_carry_is_ambiguous(reader: MeterReader) -> None:
    # The jumped 2 at 2.0 and a coupled 1 at 1.95 fit about equally well.
    reading = reader.read(meter_frame("00629", (), roll=0.5), layout(dials=0))
    assert reading.rejection is Rejection.AMBIGUOUS
    assert reading.candidate == pytest.approx(629.5, abs=0.15)


@pytest.mark.parametrize("tenths", [9.0, 9.3, 9.6, 9.9, 9.96])
def test_wheel_above_the_dials_stays_coupled_near_a_carry(
    reader: MeterReader, tenths: float
) -> None:
    # The last wheel rolls towards 8 with the dials; it must not read 8 early.
    hundredths = tenths * 10 % 10
    frame = meter_frame("00627", (tenths, hundredths), roll=tenths / 10)
    reading = reader.read(frame, layout())
    assert reading.accepted
    assert reading.value == pytest.approx(627 + tenths / 10, abs=0.01)


@pytest.mark.parametrize("fill", [(255, 255, 255), (0, 0, 0), (128, 128, 128)])
def test_covered_wheel_is_a_fault(reader: MeterReader, fill: tuple[int, ...]) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    left, top, right, bottom = layout().digits.box(frame.size)
    pitch = (right - left) / 5
    covered = (int(left + 2 * pitch), top, int(left + 3 * pitch), bottom)
    ImageDraw.Draw(frame).rectangle(covered, fill=fill)
    reading = reader.read(frame, layout())
    assert reading.rejection is Rejection.FAULT
    assert reading.wheels[2].fault in (Fault.GLARE, Fault.DARK)


@pytest.mark.parametrize(
    ("fill", "fault"), [((255, 255, 255), Fault.GLARE), ((5, 5, 5), Fault.DARK)]
)
def test_blank_window_is_a_fault(
    reader: MeterReader, fill: tuple[int, ...], fault: Fault
) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    ImageDraw.Draw(frame).rectangle(layout().digits.box(frame.size), fill=fill)
    reading = reader.read(frame, layout())
    assert reading.rejection is Rejection.FAULT
    assert reading.faults == (fault,)


def test_glare_on_a_wheel_is_a_fault(reader: MeterReader) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    pixels = np.asarray(frame, np.float32)
    left, top, right, bottom = layout().digits.box(frame.size)
    y, x = np.ogrid[: pixels.shape[0], : pixels.shape[1]]
    centre = (left + 2.5 * (right - left) / 5, (top + bottom) / 2)
    blob = np.clip(3 - np.hypot(x - centre[0], y - centre[1]) / 12, 0, 1)
    pixels = np.clip(pixels + blob[..., None] * 400, 0, 255).astype(np.uint8)
    # Only a calibration tells a glare blob from a light drum.
    calibration = reader.calibrate(frame, layout())
    reading = reader.read(Image.fromarray(pixels), layout(), calibration)
    assert reading.rejection is Rejection.FAULT
    assert reading.wheels[2].fault is Fault.GLARE


def test_covered_dial_is_a_fault(reader: MeterReader) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    box = layout().dials[1].roi.box(frame.size)
    ImageDraw.Draw(frame).rectangle(box, fill=(255, 255, 255))
    reading = reader.read(frame, layout())
    assert reading.rejection is Rejection.FAULT
    assert reading.dials[1].fault is Fault.GLARE
    assert reading.faults == (Fault.GLARE,)


def test_tiny_window_is_a_fault(reader: MeterReader) -> None:
    tiny = MeterLayout(Roi(0.5, 0.5, 0.501, 0.501), 8)
    assert reader.read(meter_frame(), tiny).rejection is Rejection.FAULT


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


def _readings(*values: float, confidence: float = 0.9) -> tuple[Reading, ...]:
    return tuple(Reading(v, confidence) for v in values)


@pytest.mark.parametrize(("max_residual", "accepted"), [(0.5, True), (0.4, False)])
def test_residual_at_the_limit_is_accepted(
    max_residual: float, *, accepted: bool
) -> None:
    # 3.5 above 0.0 is half a digit off either way: residual exactly 0.5.
    reading = MeterReader(max_residual=max_residual)._judge(
        layout(wheels=2, dials=0), _readings(3.5, 0.0), ()
    )
    assert reading.residual == 0.5
    assert reading.accepted is accepted


def test_confidence_at_the_limit_is_accepted() -> None:
    judge = MeterReader(min_confidence=0.75)._judge
    shape = layout(wheels=2, dials=0)
    assert judge(shape, _readings(1.0, 2.0, confidence=0.75), ()).accepted
    low = judge(shape, _readings(1.0, 2.0, confidence=0.74), ())
    assert low.rejection is Rejection.LOW_CONFIDENCE


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
        ({"views": 2.0}, "views"),
        ({"min_confidence": float("nan")}, "min_confidence"),
        ({"max_residual": "0.4"}, "max_residual"),
    ],
)
def test_invalid_settings(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(MeterDataError, match=match):
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


def test_calibration_records_the_exposure(reader: MeterReader) -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884)
    calibration = reader.calibrate(frame, layout())
    assert len(calibration.exposure) == 7
    # Dark wheels with white digits, white dial faces.
    assert calibration.exposure[0].level < 128 < calibration.exposure[-1].level
    no_dials = reader.calibrate(meter_frame("00627", ()), layout(dials=0))
    assert len(no_dials.exposure) == 5


def test_calibration_of_another_layout_has_no_baseline(reader: MeterReader) -> None:
    # Made for two dials, read with none: the fixed limits apply.
    calibration = reader.calibrate(meter_frame(), layout())
    frame = meter_frame("00627", ())
    box = layout().digits.box(frame.size)
    ImageDraw.Draw(frame).rectangle(box, fill=(255, 255, 255))
    assert reader.read(frame, layout(dials=0), calibration).faults == (Fault.GLARE,)


def test_reading_without_exposure_keeps_its_fault() -> None:
    reading = Reading(1.0, 0.9)
    assert reader_module._exposed(reading, None, 0.4) is reading


def test_readings_offer_alternatives_near_a_carry(reader: MeterReader) -> None:
    reading = reader.read(meter_frame("00629", (), roll=0.5), layout(dials=0))
    assert reading.rejection is Rejection.AMBIGUOUS
    assert reading.alternatives
    assert {round(reading.candidate), *map(round, reading.alternatives)} >= {
        620,
        630,
    }
