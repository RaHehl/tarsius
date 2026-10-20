from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tarsius import (
    BackendError,
    Classifier,
    NcnnBackend,
    Roi,
    StateDetector,
    Status,
    extract_features,
)
from tarsius.backends import ncnn as ncnn_module

CAR_SPOT = Roi(0.28, 0.22, 0.72, 0.88)
GOLDEN = Path(__file__).parent / "data" / "golden_squeezenet11.npy"


@pytest.fixture(scope="module")
def backend() -> NcnnBackend:
    try:
        return NcnnBackend()
    except BackendError:
        if os.environ.get("TARSIUS_REQUIRE_MODEL") == "1":
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


@pytest.mark.usefixtures("backend")
def test_state_detector_shares_its_default_backend() -> None:
    """Without a backend, state detectors use ncnn, loaded once."""
    first = StateDetector(CAR_SPOT, ("on", "off"))
    second = StateDetector(Roi(0, 0, 1, 1), ("a", "b", "c"))

    assert first.backend is second.backend
    assert first.backend_id == NcnnBackend.id


def test_fp16_is_off(backend: NcnnBackend) -> None:
    """fp16 differs between CPUs, features must not depend on the hardware."""
    opt = backend._net.opt
    assert not (opt.use_fp16_arithmetic or opt.use_fp16_storage or opt.use_fp16_packed)


class _CountingNet:
    """Stands in for the ncnn net and records how many inferences overlap."""

    def __init__(self) -> None:
        self.running = 0
        self.most = 0
        self._count = threading.Lock()

    def create_extractor(self) -> _CountingExtractor:
        return _CountingExtractor(self)


class _CountingExtractor:
    def __init__(self, net: _CountingNet) -> None:
        self._net = net

    def input(self, name: str, mat: object) -> None:
        pass

    def extract(self, name: str) -> tuple[int, np.ndarray]:
        with self._net._count:
            self._net.running += 1
            self._net.most = max(self._net.most, self._net.running)
        time.sleep(0.02)
        with self._net._count:
            self._net.running -= 1
        return 0, np.ones((512, 13, 13), dtype=np.float32)


def test_one_inference_at_a_time(
    backend: NcnnBackend, monkeypatch: pytest.MonkeyPatch, make_scene
) -> None:
    """ncnn keeps buffers per overlapping inference, so they take turns."""
    net = _CountingNet()
    monkeypatch.setattr(backend, "_net", net)
    image = make_scene(car=True, seed=0)
    threads = [
        threading.Thread(target=backend.extract, args=(image,)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert net.most == 1
