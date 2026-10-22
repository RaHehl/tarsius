from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from tarsius import BackendError
from tarsius.meter import Fault
from tarsius.meter.engine import DIAL_GLARE_SHARE, Model, decode, exposure_fault


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


def test_exposure_ok_on_a_textured_crop() -> None:
    assert exposure_fault(_crop()) is Fault.OK


@pytest.mark.parametrize(
    ("level", "fault"), [(255.0, Fault.GLARE), (128.0, Fault.GLARE), (3.0, Fault.DARK)]
)
def test_exposure_blank_crop(level: float, fault: Fault) -> None:
    crop = np.full((32, 20, 3), level, np.float32)
    crop += np.random.default_rng(3).normal(0, 2, crop.shape).astype(np.float32)
    assert exposure_fault(crop) is fault


def test_exposure_clipped_share() -> None:
    crop = _crop()
    crop[:14] = 255.0
    assert exposure_fault(crop) is Fault.GLARE
    # Dial faces are white: the same share is normal there.
    assert exposure_fault(crop, glare_share=DIAL_GLARE_SHARE) is Fault.OK


def test_exposure_dark_share() -> None:
    crop = _crop()
    crop[:26] = 0.0
    assert exposure_fault(crop) is Fault.DARK


def test_exposure_fault_overrides_the_fault_head() -> None:
    model = Model.load(_skip_model(100))
    blank = np.full((4, 4, 3), 250.0, np.float32)
    reading = model.inspect(blank)
    assert reading.fault is Fault.GLARE
    assert 0 <= reading.value < 10
