"""Benchmarks: `pytest tests/benchmarks --codspeed`; plain pytest runs each once."""

from __future__ import annotations

import os
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from tarsius import (
    BackendError,
    Classifier,
    EdgeBackend,
    Features,
    NcnnBackend,
    Roi,
    extract_features,
    load_image,
)

SPOT = Roi(0.28, 0.22, 0.72, 0.88)


@pytest.fixture(scope="module")
def frame_jpeg() -> bytes:
    """A 4K camera frame, so decoding cost matches a real snapshot."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (2160, 3840, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def test_load_4k_jpeg(benchmark, frame_jpeg: bytes) -> None:
    benchmark(load_image, frame_jpeg)


def test_edges_extract(benchmark, frame_jpeg: bytes) -> None:
    backend = EdgeBackend()
    image = load_image(frame_jpeg)
    benchmark(extract_features, backend, image, SPOT)


def test_ncnn_extract(benchmark, frame_jpeg: bytes) -> None:
    try:
        backend = NcnnBackend()
    except ImportError, BackendError:
        if os.environ.get("TARSIUS_REQUIRE_NCNN"):
            raise
        pytest.skip("ncnn or its model is not available")
    image = load_image(frame_jpeg)
    benchmark(extract_features, backend, image, SPOT)


def test_classify_200_references(benchmark) -> None:
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(201, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    classifier = Classifier("bench/1", margin=0.02)
    for index, vector in enumerate(vectors[:200]):
        features = Features("bench/1", vector, vector[None])
        classifier.add(str(index), features, state=("a", "b")[index % 2])
    query = Features("bench/1", vectors[200], vectors[200][None])
    benchmark(classifier.classify, query)
