from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from tarsius import BackendError, MeterDataError
from tarsius.meter import Fault
from tarsius.meter.engine import Exposure, Model, decode, exposure_fault


def _model(arch: list[dict[str, object]], arrays: dict[str, np.ndarray]) -> io.BytesIO:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        __arch__=np.frombuffer(json.dumps(arch).encode(), np.uint8),
        __input_hw__=np.array([4, 4], np.int32),
        **arrays,
    )
    buffer.seek(0)
    return buffer


def _skip_model(outputs: int) -> io.BytesIO:
    """A tiny net with every op the engine knows."""
    rng = np.random.default_rng(0)
    arch: list[dict[str, object]] = [
        {"op": "rescale", "scale": 1 / 255},
        {"op": "conv", "w": "w0", "b": "b0", "relu": True, "save": "skip"},
        {"op": "maxpool"},
        {"op": "upsample", "factor": 2},
        {"op": "add", "other": "skip"},
        {"op": "concat", "other": "skip"},
        {"op": "relu", "save": "joined"},
        {"op": "maxpool"},
        {"op": "relu", "from": "joined"},
        {"op": "flatten"},
        {"op": "dense", "w": "w1", "b": "b1"},
    ]
    return _model(
        arch,
        {
            "w0": rng.normal(size=(3, 3, 3, 2)).astype(np.float32),
            "b0": np.zeros(2, np.float32),
            "w1": rng.normal(size=(4 * 4 * 4, outputs)).astype(np.float32),
            "b1": np.zeros(outputs, np.float32),
        },
    )


@pytest.mark.parametrize("outputs", [100, 105, 107])
def test_heads_by_output_width(outputs: int) -> None:
    model = Model.load(_skip_model(outputs))
    crop = np.random.default_rng(1).uniform(0, 255, (4, 4, 3)).astype(np.float32)
    reading = model.inspect(crop, views=5)
    assert 0 <= reading.value < 10
    assert 0 <= reading.confidence <= 1
    if outputs == 100:
        assert reading.fault is Fault.OK
    assert (reading.upside_down > 0) == (outputs == 107)


def test_unknown_op() -> None:
    with pytest.raises(BackendError, match="unknown op"):
        Model.load(_model([{"op": "softplus"}], {}))


def test_garbage(tmp_path: Path) -> None:
    path = tmp_path / "model.npz"
    path.write_bytes(b"garbage")
    with pytest.raises(BackendError, match="cannot load"):
        Model.load(path)


def test_decode_refines_between_classes() -> None:
    probs = np.zeros(100, np.float32)
    probs[34] = probs[35] = 0.5
    value, confidence = decode(probs)
    assert value == pytest.approx(3.45)
    assert confidence == pytest.approx(1.0)


def test_decode_wraps_around_zero() -> None:
    probs = np.zeros(100, np.float32)
    probs[99] = probs[0] = 0.5
    value, _ = decode(probs)
    assert min(value, 10 - value) == pytest.approx(0.05)


def test_decode_competing_digit_lowers_confidence() -> None:
    probs = np.zeros(100, np.float32)
    probs[30] = 0.6
    probs[70] = 0.4
    _, confidence = decode(probs)
    assert confidence < 0.4


def _crop(rng_seed: int = 2) -> np.ndarray:
    return (
        np.random.default_rng(rng_seed).uniform(40, 200, (32, 20, 3)).astype(np.float32)
    )


def _exposure(crop: np.ndarray) -> Exposure:
    return Exposure.of(crop)


def test_exposure_ok_on_a_textured_crop() -> None:
    assert exposure_fault(_exposure(_crop()), None, 0.5) is Fault.OK


@pytest.mark.parametrize(
    ("level", "fault"), [(255.0, Fault.GLARE), (128.0, Fault.GLARE), (3.0, Fault.DARK)]
)
def test_exposure_blank_crop(level: float, fault: Fault) -> None:
    crop = np.full((32, 20, 3), level, np.float32)
    crop += np.random.default_rng(3).normal(0, 1, crop.shape).astype(np.float32)
    assert exposure_fault(_exposure(crop), None, 1.0) is fault


def test_exposure_blank_against_the_calibration() -> None:
    baseline = Exposure(spread=80.0, light=0.3, level=90.0)
    faint = Exposure(spread=3.5, light=0.3, level=90.0)
    assert exposure_fault(faint, None, 1.0) is Fault.OK
    assert exposure_fault(faint, baseline, 1.0) is Fault.DARK


def test_exposure_clipped_share_without_calibration() -> None:
    crop = _crop()
    crop[:14] = 255.0
    exposure = _exposure(crop)
    # A sure reading of a clipped crop stands, an unsure one is glare.
    assert exposure_fault(exposure, None, 0.9) is Fault.OK
    assert exposure_fault(exposure, None, 0.6) is Fault.GLARE


def test_exposure_dark_share() -> None:
    crop = _crop()
    crop[:26] = 0.0
    exposure = _exposure(crop)
    assert exposure_fault(exposure, None, 0.9) is Fault.OK
    assert exposure_fault(exposure, None, 0.6) is Fault.DARK
    assert exposure_fault(exposure, _exposure(_crop()), 0.6) is Fault.DARK


def test_glare_against_the_calibration() -> None:
    dark_drum = Exposure(spread=60.0, white=0.02, light=0.25, level=70.0)
    blob = Exposure(spread=90.0, white=0.45, light=0.6, level=150.0)
    assert exposure_fault(blob, dark_drum, 0.95) is Fault.GLARE
    # An overexposed light drum clips its background, its digits stay.
    light_drum = Exposure(spread=60.0, white=0.1, light=0.8, level=200.0)
    clipped = Exposure(spread=70.0, white=0.75, light=0.82, level=220.0)
    assert exposure_fault(clipped, light_drum, 0.95) is Fault.OK


def test_exposure_values_are_checked() -> None:
    with pytest.raises(MeterDataError, match="white"):
        Exposure(white=1.5)
    with pytest.raises(MeterDataError, match="spread"):
        Exposure(spread=float("nan"))


def test_inspect_reports_the_exposure() -> None:
    model = Model.load(_skip_model(100))
    blank = np.full((4, 4, 3), 250.0, np.float32)
    reading = model.inspect(blank)
    assert reading.exposure == Exposure(0.0, 1.0, 0.0, 1.0, 250.0)
    assert reading.fault is Fault.OK
