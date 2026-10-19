from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tarsius import (
    BackendError,
    Classifier,
    NcnnBackend,
    Roi,
    Status,
    extract_features,
)
from tarsius.backends import ncnn as ncnn_module

if os.environ.get("TARSIUS_REQUIRE_NCNN") != "1":
    # CI sets this where ncnn is installed: a silent skip would hide a missing model.
    pytest.importorskip("ncnn")

CAR_SPOT = Roi(0.28, 0.22, 0.72, 0.88)
GOLDEN = Path(__file__).parent / "data" / "golden_squeezenet11.npy"


@pytest.fixture(scope="module")
def backend() -> NcnnBackend:
    try:
        return NcnnBackend()
    except BackendError:
        if os.environ.get("TARSIUS_REQUIRE_NCNN") == "1":
            raise
        pytest.skip("SqueezeNet model not built, run scripts/build_model.sh")


def test_missing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncnn_module, "files", lambda _package: tmp_path)
    with pytest.raises(BackendError, match="missing"):
        NcnnBackend()


def test_unloadable_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    for name in ("squeezenet_v1.1.param", "squeezenet_v1.1.bin"):
        (tmp_path / "models" / name).write_text("garbage")
    monkeypatch.setattr(ncnn_module, "files", lambda _package: tmp_path)
    with pytest.raises(BackendError, match="could not load"):
        NcnnBackend()


def test_embedding_shape(make_scene, backend: NcnnBackend) -> None:
    vector, query = backend.extract(make_scene(car=True, seed=0))
    assert vector.shape == (512,)
    assert vector.dtype == np.float32
    assert query.shape == (1, 512)
    assert np.linalg.norm(vector) == pytest.approx(1.0, rel=1e-5)


def test_golden_embedding(make_scene, backend: NcnnBackend) -> None:
    """Model, ncnn options and preprocessing must keep the id's features stable.

    If this fails after a deliberate change, bump NcnnBackend.id and
    regenerate the golden file.
    """
    vector, _ = backend.extract(make_scene(car=True, seed=0))
    assert float(vector @ np.load(GOLDEN)) > 0.9999


@pytest.mark.parametrize("car", [True, False])
def test_classify_with_rough_region(
    make_scene, backend: NcnnBackend, car: bool
) -> None:
    classifier = Classifier.for_backend(backend)
    for seed in range(2):
        for has_car, state in ((True, "occupied"), (False, "free")):
            image = make_scene(car=has_car, seed=seed)
            classifier.add(
                f"{state}-{seed}",
                extract_features(backend, image, CAR_SPOT),
                state=state,
            )
    for seed in range(100, 105):
        features = extract_features(backend, make_scene(car=car, seed=seed), CAR_SPOT)
        result = classifier.classify(features)
        assert result.status is Status.CONFIDENT
        assert result.state == ("occupied" if car else "free")
