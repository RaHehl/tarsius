from __future__ import annotations

import pytest
from PIL import Image

from tarsius import (
    BlankRegionError,
    EdgeBackend,
    ImageError,
    LoadReport,
    Roi,
    StateDetector,
    Status,
    encode_jpeg,
    load_image,
)

SPOT = Roi(0.28, 0.22, 0.72, 0.88)


def _jpeg(image: Image.Image) -> bytes:
    return encode_jpeg(image)


def _detector(**kwargs) -> StateDetector:
    return StateDetector(EdgeBackend(), SPOT, ("on", "off"), **kwargs)


def _teach(detector: StateDetector, make_scene, key: str, *, car: bool, seed: int):
    frame = _jpeg(make_scene(car=car, seed=seed))
    return detector.add(key, "on" if car else "off", detector.prepare(frame))


@pytest.mark.parametrize(
    "states", [("on",), ("on", "on"), ()], ids=["one", "duplicate", "none"]
)
def test_needs_two_distinct_states(states: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="distinct"):
        StateDetector(EdgeBackend(), SPOT, states)


def test_rejects_bad_cap() -> None:
    with pytest.raises(ValueError, match="max_per_state"):
        _detector(max_per_state=0)


def test_not_ready_until_every_state(make_scene) -> None:
    detector = StateDetector(EdgeBackend(), SPOT, ("on", "off", "half"))
    frame = _jpeg(make_scene(car=True, seed=9))
    _teach(detector, make_scene, "a", car=True, seed=0)
    _teach(detector, make_scene, "b", car=False, seed=0)

    assert detector.missing_states() == ["half"]
    assert not detector.ready
    assert detector.classify(frame).status is Status.NOT_READY


@pytest.mark.parametrize("car", [True, False])
def test_classifies_after_teaching(make_scene, car: bool) -> None:
    detector = _detector()
    for seed in range(2):
        _teach(detector, make_scene, f"car{seed}", car=True, seed=seed)
        _teach(detector, make_scene, f"empty{seed}", car=False, seed=seed)

    result = detector.classify(_jpeg(make_scene(car=car, seed=50)))

    assert detector.ready
    assert detector.backend_id == "edges/1"
    assert result.state == ("on" if car else "off")


def test_prepare_normalizes_the_original(make_scene) -> None:
    detector = _detector()
    big = make_scene(car=True, seed=0).resize((2688, 1512))

    prepared = detector.prepare(_jpeg(big))

    assert load_image(prepared.original).size == (1024, 576)


def test_frame_side_is_configurable(make_scene) -> None:
    detector = _detector(frame_side=2048)
    big = make_scene(car=True, seed=0).resize((3840, 2160))

    prepared = detector.prepare(_jpeg(big))

    assert detector.frame_side == 2048
    assert load_image(prepared.original, max_side=4096).size == (2048, 1152)


def test_stored_original_matches_live_frame(make_scene) -> None:
    """References and live frames go through the same normalization."""
    detector = _detector()
    frame = _jpeg(make_scene(car=True, seed=0).resize((3840, 2160)))
    stored = detector.prepare(frame)
    reloaded = detector.prepare(stored.original)
    live = detector.prepare(frame)

    assert float(reloaded.features.vector @ live.features.vector) > 0.995


def test_prepare_rejects_blank_and_garbage(make_scene) -> None:
    detector = _detector()
    grey = _jpeg(Image.new("RGB", (320, 180), (128, 128, 128)))
    with pytest.raises(BlankRegionError):
        detector.prepare(grey)
    with pytest.raises(ImageError):
        detector.prepare(b"<html>502</html>")


def test_add_rejects_unknown_state(make_scene) -> None:
    detector = _detector()
    prepared = detector.prepare(_jpeg(make_scene(car=True, seed=0)))
    with pytest.raises(ValueError, match="unknown state"):
        detector.add("a", "half", prepared)


def test_load_skips_unusable_references(make_scene) -> None:
    detector = _detector()
    car = _jpeg(make_scene(car=True, seed=0))
    empty = _jpeg(make_scene(car=False, seed=0))
    grey = _jpeg(Image.new("RGB", (320, 180), (128, 128, 128)))

    report = detector.load(
        [
            ("car", "on", car),
            ("empty", "off", empty),
            ("grey", "off", grey),
            ("broken", "on", car[:100]),
            ("gone", "half", car),
        ]
    )

    assert report == LoadReport(
        2, {"grey": "blank", "broken": "undecodable", "gone": "unknown_state"}, []
    )
    assert detector.references() == {"car": "on", "empty": "off"}
    assert detector.ready


def test_load_replaces_references(make_scene) -> None:
    detector = _detector()
    _teach(detector, make_scene, "old", car=True, seed=0)
    detector.load([("new", "off", _jpeg(make_scene(car=False, seed=0)))])
    assert detector.references() == {"new": "off"}


def test_load_enforces_cap(make_scene) -> None:
    detector = _detector(max_per_state=1)
    frames = [("a", "on", _jpeg(make_scene(car=True, seed=0)))] * 1 + [
        ("b", "on", _jpeg(make_scene(car=True, seed=1)))
    ]
    report = detector.load(frames)
    assert report.loaded == 1
    assert len(report.evicted) == 1
    assert detector.counts() == {"on": 1}


def test_cap_drops_the_most_redundant(make_scene) -> None:
    """A near duplicate goes, a distinct frame stays."""
    detector = _detector(max_per_state=2)
    distinct = make_scene(car=True, seed=0, shift=(2, 2), clutter=False)
    twin = make_scene(car=True, seed=1, shift=(-2, -2), clutter=False)
    assert detector.add("distinct", "on", detector.prepare(_jpeg(distinct))) == []
    assert detector.add("twin1", "on", detector.prepare(_jpeg(twin))) == []

    evicted = detector.add("twin2", "on", detector.prepare(_jpeg(twin)))

    assert evicted in (["twin1"], ["twin2"])
    assert "distinct" in detector.references()
    assert detector.counts() == {"on": 2}


def test_most_redundant_needs_two(make_scene) -> None:
    detector = _detector()
    assert detector.most_redundant("on") is None
    _teach(detector, make_scene, "a", car=True, seed=0)
    assert detector.most_redundant("on") is None
    _teach(detector, make_scene, "b", car=True, seed=1)
    assert detector.most_redundant("on") in ("a", "b")


def test_remove(make_scene) -> None:
    detector = _detector()
    _teach(detector, make_scene, "a", car=True, seed=0)
    _teach(detector, make_scene, "b", car=False, seed=0)
    detector.remove(["a", "missing"])
    assert detector.references() == {"b": "off"}
    assert detector.missing_states() == ["on"]
