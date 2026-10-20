from __future__ import annotations

import json
import struct
import threading

import numpy as np
import pytest
from PIL import Image

from tarsius import (
    FULL_FRAME,
    BackendError,
    BlankFrameError,
    BlankRegionError,
    EdgeBackend,
    Features,
    ImageError,
    LoadReport,
    Result,
    Roi,
    SkipReason,
    StateDetector,
    Status,
    load_image,
)
from tarsius import cache as cache_module
from tarsius import detector as detector_module
from tarsius.image import encode_jpeg

SPOT = Roi(0.28, 0.22, 0.72, 0.88)


def _detector(**kwargs) -> StateDetector:
    return StateDetector(SPOT, ("on", "off"), backend=EdgeBackend(), **kwargs)


def _frame(make_scene, *, car: bool, seed: int, **kwargs) -> bytes:
    return encode_jpeg(make_scene(car=car, seed=seed, **kwargs))


def _teach(detector: StateDetector, make_scene, key: str, *, car: bool, seed: int):
    prepared = detector.prepare(_frame(make_scene, car=car, seed=seed))
    return detector.add(key, prepared, state="on" if car else "off")


def _grey(noise: float = 0.0) -> bytes:
    pixels = np.full((180, 320, 3), 128.0)
    pixels += np.random.default_rng(0).normal(0, noise, pixels.shape)
    return encode_jpeg(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)))


@pytest.mark.parametrize(
    ("states", "match"),
    [
        (("on",), "at least two"),
        (("on", "off", "on"), "distinct"),
        ("ab", "distinct"),
    ],
    ids=["one", "duplicate", "string"],
)
def test_rejects_bad_states(states, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StateDetector(SPOT, states, backend=EdgeBackend())


@pytest.mark.parametrize(
    ("option", "match"),
    [({"max_per_state": 0}, "max_per_state"), ({"frame_side": 0}, "frame_side")],
)
def test_rejects_bad_options(option: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _detector(**option)


def test_settings_are_read_only() -> None:
    detector = _detector(max_per_state=7, frame_side=640)
    assert (detector.roi, detector.states) == (SPOT, ("on", "off"))
    assert (detector.max_per_state, detector.frame_side) == (7, 640)
    assert detector.backend_id == "edges/1"
    with pytest.raises(AttributeError):
        detector.roi = Roi()  # type: ignore[misc]


def test_not_ready_until_every_state(make_scene) -> None:
    detector = StateDetector(SPOT, ("on", "off", "half"), backend=EdgeBackend())
    _teach(detector, make_scene, "a", car=True, seed=0)
    _teach(detector, make_scene, "b", car=False, seed=0)

    assert detector.missing_states() == ["half"]
    assert not detector.ready
    frame = _frame(make_scene, car=True, seed=9)
    assert detector.classify(frame).status is Status.NOT_READY


@pytest.mark.parametrize("car", [True, False])
def test_classifies_after_teaching(make_scene, car: bool) -> None:
    detector = _detector()
    for seed in range(2):
        _teach(detector, make_scene, f"car{seed}", car=True, seed=seed)
        _teach(detector, make_scene, f"empty{seed}", car=False, seed=seed)

    result = detector.classify(_frame(make_scene, car=car, seed=50))

    assert detector.ready
    assert result.state == ("on" if car else "off")


def test_prepare_normalizes_the_original(make_scene) -> None:
    big = encode_jpeg(make_scene(car=True, seed=0).resize((2688, 1512)))

    prepared = _detector().prepare(big)

    assert load_image(prepared.original).size == (1024, 576)
    assert (prepared.roi, prepared.frame_side) == (SPOT, 1024)


def test_frame_side_used_everywhere(make_scene) -> None:
    """References, reloads and live frames all use the configured size."""
    detector = _detector(frame_side=2048)
    big = encode_jpeg(make_scene(car=True, seed=0).resize((3840, 2160)))
    empty = encode_jpeg(make_scene(car=False, seed=0).resize((3840, 2160)))
    car = detector.prepare(big)
    assert load_image(car.original, max_side=4096).size == (2048, 1152)

    detector.add("car", car, state="on")
    detector.add("empty", detector.prepare(empty), state="off")
    live = detector.classify(big)
    detector.load([("car", car.original, "on"), ("empty", empty, "off")])
    reloaded = detector.classify(big)

    # The reference comes from the stored JPEG, the live frame from the camera.
    assert live.scores["on"] == pytest.approx(1.0, abs=5e-3)
    assert reloaded.scores["on"] == pytest.approx(live.scores["on"], abs=1e-6)


def test_reload_gives_the_same_features(make_scene) -> None:
    detector = _detector()
    frame = encode_jpeg(make_scene(car=True, seed=0).resize((3840, 2160)))
    stored = detector.prepare(frame)
    reloaded = detector.prepare(stored.original)

    assert float(stored.features.vector @ reloaded.features.vector) > 0.999


@pytest.mark.parametrize("noise", [0.0, 1.5], ids=["flat", "noisy"])
def test_blank_frames_are_rejected(noise: float) -> None:
    detector = _detector()
    with pytest.raises(BlankFrameError):
        detector.prepare(_grey(noise))


def test_classify_rejects_blank_and_garbage(make_scene) -> None:
    detector = _detector()
    _teach(detector, make_scene, "a", car=True, seed=0)
    _teach(detector, make_scene, "b", car=False, seed=0)
    with pytest.raises(BlankFrameError):
        detector.classify(_grey(1.5))
    with pytest.raises(ImageError):
        detector.classify(b"<html>502</html>")


def test_prepare_rejects_garbage() -> None:
    with pytest.raises(ImageError):
        _detector().prepare(b"<html>502</html>")


def test_add_rejects_unknown_state(make_scene) -> None:
    detector = _detector()
    prepared = detector.prepare(_frame(make_scene, car=True, seed=0))
    with pytest.raises(ValueError, match="unknown state"):
        detector.add("a", prepared, state="half")


@pytest.mark.parametrize(
    "other",
    [{"roi": Roi(0, 0, 0.5, 0.5)}, {"frame_side": 640}],
    ids=["roi", "frame_side"],
)
def test_add_rejects_prepared_from_other_detector(make_scene, other) -> None:
    options = {"roi": SPOT, **other}
    roi = options.pop("roi")
    source = StateDetector(roi, ("on", "off"), backend=EdgeBackend(), **options)
    prepared = source.prepare(_frame(make_scene, car=True, seed=0))
    with pytest.raises(ValueError, match="another region"):
        _detector().add("a", prepared, state="on")


def test_add_replaces_a_key(make_scene) -> None:
    detector = _detector()
    _teach(detector, make_scene, "a", car=True, seed=0)
    prepared = detector.prepare(_frame(make_scene, car=False, seed=0))
    detector.add("a", prepared, state="off")
    assert detector.references() == {"a": "off"}


def test_load_skips_unusable_references(make_scene) -> None:
    detector = _detector()
    car = _frame(make_scene, car=True, seed=0)
    empty = _frame(make_scene, car=False, seed=0)

    report = detector.load(
        [
            ("car", car, "on"),
            ("empty", empty, "off"),
            ("grey", _grey(1.5), "off"),
            ("broken", car[:100], "on"),
            ("gone", car, "half"),
        ]
    )

    assert report == LoadReport(
        2,
        {
            "grey": SkipReason.BLANK,
            "broken": SkipReason.UNDECODABLE,
            "gone": SkipReason.UNKNOWN_STATE,
        },
        (),
        report.caches,
    )
    assert set(report.caches) == {"car", "empty"}
    assert detector.references() == {"car": "on", "empty": "off"}


def test_load_survives_decompression_bomb(
    make_scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    car = _frame(make_scene, car=True, seed=0)
    small = encode_jpeg(Image.new("RGB", (8, 8), (10, 200, 10)))
    # Make every normal frame count as a bomb, except the tiny one.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 20)
    report = _detector().load([("bomb", car, "on"), ("small", small, "on")])
    assert report.skipped["bomb"] is SkipReason.UNDECODABLE


def test_load_last_duplicate_wins(make_scene) -> None:
    detector = _detector()
    car = _frame(make_scene, car=True, seed=0)
    report = detector.load(
        [
            ("a", b"junk", "on"),
            ("a", car, "on"),
            ("b", _frame(make_scene, car=False, seed=0), "off"),
        ]
    )
    assert report == LoadReport(2, {}, (), report.caches)
    assert detector.references() == {"a": "on", "b": "off"}


def test_load_replaces_references_and_keeps_settings(make_scene) -> None:
    detector = _detector(k=1)
    _teach(detector, make_scene, "old", car=True, seed=0)
    detector.load(
        [
            ("new", _frame(make_scene, car=False, seed=0), "off"),
            ("car", _frame(make_scene, car=True, seed=1), "on"),
        ]
    )
    assert detector.references() == {"new": "off", "car": "on"}
    result = detector.classify(_frame(make_scene, car=True, seed=1))
    assert result.scores["on"] == pytest.approx(1.0, abs=1e-5)
    assert result.margin >= 0.02


def test_load_enforces_cap_for_every_state(make_scene) -> None:
    detector = _detector(max_per_state=1)
    report = detector.load(
        [
            ("car0", _frame(make_scene, car=True, seed=0), "on"),
            ("car1", _frame(make_scene, car=True, seed=1), "on"),
            ("empty0", _frame(make_scene, car=False, seed=0), "off"),
            ("empty1", _frame(make_scene, car=False, seed=1), "off"),
        ]
    )
    assert report.loaded == 2
    assert len(report.evicted) == 2
    assert detector.counts() == {"on": 1, "off": 1}


def test_cap_keeps_distinct_and_newest(make_scene) -> None:
    """A duplicate replaces the older copy, a distinct frame stays."""
    detector = _detector(max_per_state=2)
    distinct = detector.prepare(
        _frame(make_scene, car=True, seed=0, shift=(2, 2), clutter=False)
    )
    twin = _frame(make_scene, car=True, seed=1, shift=(-2, -2), clutter=False)
    assert detector.add("distinct", distinct, state="on") == []
    assert detector.add("twin1", detector.prepare(twin), state="on") == []

    assert detector.add("twin2", detector.prepare(twin), state="on") == ["twin1"]
    assert set(detector.references()) == {"distinct", "twin2"}


@pytest.mark.parametrize("keys", ["ref-a", ["ref-a", "ref-b"]], ids=["one", "several"])
def test_remove(make_scene, keys) -> None:
    detector = _detector()
    _teach(detector, make_scene, "ref-a", car=True, seed=0)
    _teach(detector, make_scene, "ref-b", car=True, seed=1)
    _teach(detector, make_scene, "ref-c", car=False, seed=0)

    detector.remove(keys)
    detector.remove(["missing"])

    expected = {"ref-b": "on", "ref-c": "off"} if keys == "ref-a" else {"ref-c": "off"}
    assert detector.references() == expected


class _BlockingBackend(EdgeBackend):
    """Pauses the first extraction until the test lets it go on."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def extract(self, image: Image.Image):
        if not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(5)
        return super().extract(image)


def test_add_during_load_is_kept(make_scene) -> None:
    """An add that arrives while load() runs is applied after it, not lost."""
    backend = _BlockingBackend()
    detector = StateDetector(SPOT, ("on", "off"), backend=backend)
    stored = [("s0", _frame(make_scene, car=False, seed=0), "off")]
    extra = StateDetector(SPOT, ("on", "off"), backend=EdgeBackend()).prepare(
        _frame(make_scene, car=True, seed=99)
    )
    loader = threading.Thread(target=detector.load, args=(stored,))
    loader.start()
    assert backend.entered.wait(5)
    adder = threading.Thread(
        target=detector.add, args=("extra", extra), kwargs={"state": "on"}
    )
    adder.start()
    adder.join(0.2)
    # The add waits for the load instead of racing it.
    assert adder.is_alive()
    backend.release.set()
    loader.join()
    adder.join()

    assert detector.references() == {"s0": "off", "extra": "on"}


def test_classify_during_changes_never_skips_a_state(make_scene) -> None:
    detector = StateDetector(SPOT, ("on", "off", "half"), backend=EdgeBackend())
    _teach(detector, make_scene, "a", car=True, seed=0)
    _teach(detector, make_scene, "b", car=False, seed=0)
    half = detector.prepare(_frame(make_scene, car=True, seed=1))
    frame = _frame(make_scene, car=True, seed=2)
    stop = threading.Event()

    def toggle() -> None:
        while not stop.is_set():
            detector.add("h", half, state="half")
            detector.remove("h")

    thread = threading.Thread(target=toggle)
    thread.start()
    try:
        for _ in range(200):
            result = detector.classify(frame)
            assert result.status is Status.NOT_READY or len(result.scores) == 3
    finally:
        stop.set()
        thread.join()


class _StubBackend:
    """Edge features, or a chosen failure for frames of a given width."""

    id = "edges/1"
    default_margin = 0.02

    def __init__(self, fail_width: int, error: Exception | None) -> None:
        self._edges = EdgeBackend()
        self._fail_width = fail_width
        self._error = error

    def extract(self, image: Image.Image):
        if image.width == self._fail_width:
            if self._error is not None:
                raise self._error
            vector, query = self._edges.extract(image)
            return np.zeros_like(vector), np.zeros_like(query)
        return self._edges.extract(image)


def test_load_skips_region_without_structure(make_scene) -> None:
    detector = StateDetector(FULL_FRAME, ("on", "off"), backend=_StubBackend(110, None))
    odd = encode_jpeg(make_scene(car=True, seed=0).resize((110, 60)))
    report = detector.load(
        [("odd", odd, "on"), ("car", _frame(make_scene, car=True, seed=0), "on")]
    )
    assert report.skipped == {"odd": SkipReason.BLANK}
    assert detector.backend is not None


def test_load_backend_failure_keeps_references(make_scene) -> None:
    """A broken backend must not look like every reference being unusable."""
    detector = StateDetector(
        FULL_FRAME,
        ("on", "off"),
        backend=_StubBackend(110, BackendError("ncnn failed")),
    )
    detector.load([("car", _frame(make_scene, car=True, seed=0), "on")])
    odd = encode_jpeg(make_scene(car=True, seed=0).resize((110, 60)))
    with pytest.raises(BackendError):
        detector.load([("odd", odd, "on")])
    assert detector.references() == {"car": "on"}


def test_classify_snapshot_missing_a_state(
    make_scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a state vanished between the check and the lookup, it is not ready."""
    detector = _detector()
    _teach(detector, make_scene, "a", car=True, seed=0)
    _teach(detector, make_scene, "b", car=False, seed=0)
    monkeypatch.setattr(
        detector._classifier,
        "classify",
        lambda features: Result(Status.CONFIDENT, "on", "on", 1.0, {"on": 1.0}),
    )
    frame = _frame(make_scene, car=True, seed=1)
    assert detector.classify(frame).status is Status.NOT_READY


def test_uniform_region_in_a_real_frame_is_usable(make_scene) -> None:
    """A region may be plain in one state, only a blank frame is refused."""
    corner = Roi(0.0, 0.85, 0.15, 1.0)
    detector = StateDetector(corner, ("on", "off"), backend=EdgeBackend())
    frame = make_scene(car=True, seed=0, clutter=False)
    region = np.asarray(corner.crop(frame).convert("L"), dtype=float)
    assert region.std() < 4.0
    prepared = detector.prepare(encode_jpeg(frame))
    assert detector.add("plain", prepared, state="off") == []


def _greedy_reference(vectors: dict[str, np.ndarray], cap: int) -> list[str]:
    """Drop the most redundant one at a time, recomputing from scratch."""
    keys = list(vectors)
    dropped = []
    while len(keys) > cap:
        matrix = np.stack([vectors[key] for key in keys])
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, -np.inf)
        drop = keys[int(similarity.max(axis=1).argmax())]
        keys.remove(drop)
        dropped.append(drop)
    return dropped


def test_load_eviction_matches_one_by_one(make_scene) -> None:
    """Several evictions at once choose like repeated single evictions."""
    detector = _detector(max_per_state=3)
    frames = []
    # Distinct frames, so no near ties decide the order by float noise.
    for seed in range(8):
        image = make_scene(car=True, seed=seed, clutter=seed > 4)
        frames.append((f"k{seed}", encode_jpeg(image), "on"))
    vectors = {key: detector.prepare(data).features.vector for key, data, _ in frames}

    report = detector.load(frames)

    assert list(report.evicted) == _greedy_reference(vectors, 3)
    assert detector.counts() == {"on": 3}


def test_add_after_eviction_keeps_the_cap(make_scene) -> None:
    detector = _detector(max_per_state=1)
    _teach(detector, make_scene, "a", car=True, seed=0)
    _teach(detector, make_scene, "b", car=True, seed=1)
    _teach(detector, make_scene, "c", car=True, seed=2)
    assert detector.counts() == {"on": 1}
    assert len(detector.references()) == 1


def test_replaced_key_counts_as_newest(make_scene) -> None:
    detector = _detector(max_per_state=2)
    twin = detector.prepare(_frame(make_scene, car=True, seed=1, clutter=False))
    detector.add("old", twin, state="on")
    detector.add(
        "other", detector.prepare(_frame(make_scene, car=True, seed=4)), state="on"
    )
    # Re-teaching "old" makes it the newest copy of the twin.
    detector.add("old", twin, state="on")
    assert detector.add("new", twin, state="on") == ["old"]


def test_removed_keys_are_forgotten(make_scene) -> None:
    detector = _detector(max_per_state=1)
    _teach(detector, make_scene, "ref-1", car=True, seed=0)
    detector.remove("ref-1")
    assert detector.references() == {}
    # A forgotten key no longer counts against the cap.
    assert _teach(detector, make_scene, "ref-2", car=True, seed=1) == []


def test_load_keeps_k(make_scene) -> None:
    detector = _detector(k=1)
    car_far = _frame(make_scene, car=True, seed=0, shift=(4, 3))
    frames = [
        ("car", _frame(make_scene, car=True, seed=0), "on"),
        ("car_far", car_far, "on"),
        ("empty", _frame(make_scene, car=False, seed=0), "off"),
    ]
    detector.load(frames)
    result = detector.classify(_frame(make_scene, car=True, seed=0))
    # With k=1 only the best match counts.
    assert result.scores["on"] == pytest.approx(1.0, abs=1e-5)


def test_classify_region_without_structure_is_unsure(make_scene) -> None:
    """A dark or plain region in a working camera is unsure, not an error."""
    detector = StateDetector(FULL_FRAME, ("on", "off"), backend=_StubBackend(110, None))
    detector.add(
        "car", detector.prepare(_frame(make_scene, car=True, seed=0)), state="on"
    )
    detector.add(
        "empty", detector.prepare(_frame(make_scene, car=False, seed=0)), state="off"
    )
    plain = encode_jpeg(make_scene(car=True, seed=1).resize((110, 60)))

    result = detector.classify(plain)

    assert result.status is Status.UNSURE
    assert result.candidate is None
    assert result.scores == {"on": 0.0, "off": 0.0}


def test_dark_frame_with_structure_is_not_blank(make_scene) -> None:
    """A dim night frame still shows something, unlike an offline camera."""

    # Grey level spread 5 to 9, real IR night frames measured 24 and more.
    def dark(*, car: bool) -> bytes:
        pixels = np.asarray(make_scene(car=car, seed=0), dtype=np.float32) * 0.3
        return encode_jpeg(Image.fromarray(pixels.astype(np.uint8)))

    detector = _detector()
    detector.add("car", detector.prepare(dark(car=True)), state="on")
    detector.add("empty", detector.prepare(dark(car=False)), state="off")

    assert detector.classify(dark(car=True)).state == "on"


def test_blank_errors_tell_frame_from_region(make_scene) -> None:
    detector = StateDetector(FULL_FRAME, ("on", "off"), backend=_StubBackend(110, None))
    with pytest.raises(BlankFrameError):
        detector.prepare(_grey(1.5))
    plain = encode_jpeg(make_scene(car=True, seed=1).resize((110, 60)))
    with pytest.raises(BlankRegionError) as err:
        detector.prepare(plain)
    assert not isinstance(err.value, BlankFrameError)


def test_default_backend_is_shared_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed load is not kept, the next detector loads it and shares it."""
    loads = []

    def factory() -> EdgeBackend:
        loads.append(1)
        if len(loads) == 1:
            raise BackendError("model missing")
        return EdgeBackend()

    monkeypatch.setattr(detector_module, "_STATE_BACKEND", factory)
    with pytest.raises(BackendError):
        StateDetector(SPOT, ("on", "off"))

    first = StateDetector(SPOT, ("on", "off"))
    second = StateDetector(SPOT, ("a", "b"))

    assert first.backend is second.backend
    assert len(loads) == 2


class _CountingBackend(EdgeBackend):
    def __init__(self, backend_id: str = "edges/1") -> None:
        super().__init__()
        self.id = backend_id
        self.calls = 0

    def extract(self, image: Image.Image):
        self.calls += 1
        return super().extract(image)


def _counting_detector(
    backend: _CountingBackend, roi: Roi = SPOT, **kwargs
) -> StateDetector:
    return StateDetector(roi, ("on", "off"), backend=backend, **kwargs)


def _prepared_items(detector: StateDetector, make_scene):
    car = detector.prepare(_frame(make_scene, car=True, seed=0))
    empty = detector.prepare(_frame(make_scene, car=False, seed=0))
    items = [("car", car.original, "on"), ("empty", empty.original, "off")]
    return items, {"car": car.cache, "empty": empty.cache}


def test_load_uses_caches_instead_of_extracting(make_scene) -> None:
    items, caches = _prepared_items(_detector(), make_scene)
    backend = _CountingBackend()
    detector = _counting_detector(backend)

    report = detector.load(items, caches=caches)

    # One extraction checks that the cached vectors fit the backend.
    assert backend.calls == 1
    assert report.loaded == 2
    assert not report.caches
    frame = _frame(make_scene, car=True, seed=1)
    assert detector.classify(frame) == _detector_loaded(items).classify(frame)


def test_load_takes_caches_as_memoryview(make_scene) -> None:
    items, caches = _prepared_items(_detector(), make_scene)
    backend = _CountingBackend()

    report = _counting_detector(backend).load(
        items, caches={key: memoryview(blob) for key, blob in caches.items()}
    )

    assert backend.calls == 1
    assert not report.caches


def _detector_loaded(items) -> StateDetector:
    detector = _detector()
    detector.load(items)
    return detector


def test_load_reports_caches_that_spare_the_next_load(make_scene) -> None:
    items, _ = _prepared_items(_detector(), make_scene)
    report = _detector().load(items)
    backend = _CountingBackend()

    assert _counting_detector(backend).load(items, caches=report.caches).loaded == 2
    assert backend.calls == 1


@pytest.mark.parametrize(
    ("backend_id", "roi", "frame_side"),
    [
        ("edges/2", SPOT, 1024),
        ("edges/1", Roi(0.2, 0.2, 0.8, 0.9), 1024),
        ("edges/1", SPOT, 640),
        ("edges/1", Roi(SPOT.left, SPOT.top, SPOT.right, 0.9), 1024),
    ],
    ids=["backend", "roi", "frame_side", "roi_bottom"],
)
def test_load_ignores_caches_for_other_settings(
    make_scene, backend_id: str, roi: Roi, frame_side: int
) -> None:
    items, caches = _prepared_items(_detector(), make_scene)
    backend = _CountingBackend(backend_id)
    detector = _counting_detector(backend, roi, frame_side=frame_side)

    report = detector.load(items, caches=caches)

    assert backend.calls == 2
    assert set(report.caches) == {"car", "empty"}
    assert report.caches["car"] != caches["car"]


@pytest.mark.parametrize("package", ["tarsius", "pillow", "ncnn"])
def test_load_ignores_caches_from_another_release(
    make_scene, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    items, caches = _prepared_items(_detector(), make_scene)
    monkeypatch.setattr(
        cache_module, "_VERSIONS", cache_module._VERSIONS | {package: "99.0.0"}
    )
    backend = _CountingBackend()

    report = _counting_detector(backend).load(items, caches=caches)

    assert backend.calls == 2
    assert set(report.caches) == {"car", "empty"}


def _rewrite(blob: bytes, **changes) -> bytes:
    """Change header fields of a cache and keep it well-formed otherwise."""
    start = len(cache_module.MAGIC)
    (size,) = struct.unpack_from("<I", blob, start)
    header = json.loads(blob[start + 4 : start + 4 + size]) | changes
    encoded = json.dumps(header).encode()
    return b"".join(
        (
            cache_module.MAGIC,
            struct.pack("<I", len(encoded)),
            encoded,
            blob[start + 4 + size :],
        )
    )


@pytest.mark.parametrize(
    "damage",
    [
        lambda blob, other: other,
        lambda blob, other: b"",
        lambda blob, other: blob[:-4],
        lambda blob, other: blob + b"\0\0\0\0",
        lambda blob, other: b"x" + blob[1:],
        lambda blob, other: _rewrite(blob, vector=0),
        lambda blob, other: _rewrite(blob, vector="5"),
        lambda blob, other: _rewrite(blob, format=99),
        lambda blob, other: _scaled(blob, 100.0),
        lambda blob, other: _scaled(blob, np.inf),
        lambda blob, other: _scaled(blob, np.nan),
        lambda blob, other: (
            cache_module.MAGIC + struct.pack("<I", 200_000) + b"[" * 200_000
        ),
        lambda blob, other: cache_module.MAGIC + b"\x04\0\0\0null",
        lambda blob, other: cache_module.MAGIC + b"\xff\0\0\0{",
    ],
    ids=[
        "other_original",
        "empty",
        "truncated",
        "trailing",
        "magic",
        "length",
        "length_type",
        "format",
        "scaled",
        "infinite",
        "nan",
        "deep_header",
        "not_a_dict",
        "broken_header",
    ],
)
def test_load_extracts_again_for_unusable_caches(make_scene, damage) -> None:
    items, caches = _prepared_items(_detector(), make_scene)
    backend = _CountingBackend()

    report = _counting_detector(backend).load(
        items, caches={"car": damage(caches["car"], caches["empty"])}
    )

    assert backend.calls == 2
    assert report.caches["car"] == caches["car"]


def _scaled(blob: bytes, factor: float) -> bytes:
    """Multiply the cached vector, keeping the header and length."""
    start = len(cache_module.MAGIC)
    (size,) = struct.unpack_from("<I", blob, start)
    body = start + 4 + size
    vector = np.frombuffer(blob, "<f4", offset=body) * np.float32(factor)
    return blob[:body] + vector.astype("<f4").tobytes()


def _short(blob: bytes, original: bytes) -> bytes:
    """Return a well-formed cache of a unit vector with the wrong length."""
    features = Features("edges/1", np.full(5, 0.2**0.5, np.float32), np.ones((1, 5)))
    return cache_module.dump(features, original, SPOT, 1024)


@pytest.mark.parametrize(
    ("damaged", "calls"),
    [
        # The fresh extraction of the other item shows the length.
        pytest.param({"car"}, 1 + 2, id="one"),
        # One extraction to learn the length, then all again.
        pytest.param({"car", "empty"}, 1 + 2, id="all"),
    ],
)
def test_load_extracts_again_for_caches_of_another_length(
    make_scene, damaged: set[str], calls: int
) -> None:
    items, caches = _prepared_items(_detector(), make_scene)
    originals = {key: data for key, data, _ in items}
    caches |= {key: _short(caches[key], originals[key]) for key in damaged}
    if damaged == {"car"}:
        del caches["empty"]
    backend = _CountingBackend()
    detector = _counting_detector(backend)

    report = detector.load(items, caches=caches)

    assert backend.calls == calls
    assert report.loaded == 2
    assert set(report.caches) == {"car", "empty"}
    detector.add(
        "more", detector.prepare(_frame(make_scene, car=True, seed=2)), state="on"
    )
    assert detector.classify(_frame(make_scene, car=True, seed=3)).state == "on"


def test_load_ignores_a_cache_of_a_blank_region(make_scene) -> None:
    items, caches = _prepared_items(_detector(), make_scene)
    features = cache_module.restore(caches["car"], items[0][1], "edges/1", SPOT, 1024)
    assert features is not None
    blank = cache_module.dump(
        Features("edges/1", np.zeros_like(features.vector), features.query),
        items[0][1],
        SPOT,
        1024,
    )
    backend = _CountingBackend()

    _counting_detector(backend).load(items[:1], caches={"car": blank})

    assert backend.calls == 1


def test_load_reports_no_cache_for_evicted_keys(make_scene) -> None:
    detector = _detector(max_per_state=1)
    report = detector.load(
        [
            ("car0", _frame(make_scene, car=True, seed=0), "on"),
            ("car1", _frame(make_scene, car=True, seed=1), "on"),
            ("empty0", _frame(make_scene, car=False, seed=0), "off"),
        ]
    )
    assert report.evicted
    assert set(report.caches) == set(detector.references())
