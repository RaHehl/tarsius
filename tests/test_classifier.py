from __future__ import annotations

import numpy as np
import pytest

from tarsius import (
    FULL_FRAME,
    BackendMismatchError,
    BlankRegionError,
    Classifier,
    EdgeBackend,
    Features,
    Result,
    Roi,
    Status,
    extract_features,
    majority,
)
from tarsius.features import unit

CAR_SPOT = Roi(0.28, 0.22, 0.72, 0.88)
DIM = 4


def _features(*values: float, backend_id: str = "test/1") -> Features:
    vector = np.array(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return Features(backend_id, vector, vector[None])


def _toy(margin: float = 0.05, k: int = 3) -> Classifier:
    return Classifier("test/1", margin=margin, k=k)


def _trained(make_scene, backend: EdgeBackend, roi: Roi = CAR_SPOT) -> Classifier:
    classifier = Classifier.for_backend(backend)
    for seed in range(2):
        for car, state in ((True, "occupied"), (False, "free")):
            features = extract_features(backend, make_scene(car=car, seed=seed), roi)
            classifier.add(f"{state}-{seed}", features, state=state)
    return classifier


@pytest.mark.parametrize("night", [False, True])
@pytest.mark.parametrize("car", [True, False])
def test_edges_classify_new_frames(make_scene, car: bool, night: bool) -> None:
    backend = EdgeBackend()
    classifier = _trained(make_scene, backend)
    expected = "occupied" if car else "free"

    for seed in range(100, 110):
        image = make_scene(car=car, seed=seed, night=night)
        result = classifier.classify(extract_features(backend, image, CAR_SPOT))
        assert result.status is Status.CONFIDENT
        assert result.state == result.candidate == expected
        assert result.margin >= backend.default_margin
        assert result.nearest[expected].startswith(expected)


def test_roi_ignores_clutter_outside(make_scene) -> None:
    """Clutter that follows the car in the references fools the full frame only."""
    backend = EdgeBackend()
    query = make_scene(car=False, seed=50, clutter_seed=1)

    def answer(roi: Roi) -> Result:
        classifier = Classifier.for_backend(backend)
        for seed in range(2):
            for car, state, clutter in ((True, "occupied", 1), (False, "free", 2)):
                image = make_scene(car=car, seed=seed, clutter_seed=clutter)
                classifier.add(
                    f"{state}-{seed}",
                    extract_features(backend, image, roi),
                    state=state,
                )
        return classifier.classify(extract_features(backend, query, roi))

    assert answer(CAR_SPOT).state == "free"
    assert answer(FULL_FRAME).candidate == "occupied"


def test_edge_query_tolerates_shift(make_scene) -> None:
    backend = EdgeBackend()
    reference, _ = backend.extract(make_scene(car=True, seed=0, shift=(0, 0)))
    vector, query = backend.extract(make_scene(car=True, seed=0, shift=(5, 4)))

    assert query.shape == (25, vector.shape[0])
    np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1.0, rtol=1e-5)
    assert float((query @ reference).max()) > float(vector @ reference) + 0.05


def test_edges_use_gradients_not_brightness(make_scene) -> None:
    """A uniform brightness change must not change the features."""
    backend = EdgeBackend()
    image = make_scene(car=True, seed=0)
    darker = image.point(lambda value: value // 2)
    a, _ = backend.extract(image)
    b, _ = backend.extract(darker)
    assert float(a @ b) > 0.97


def test_not_ready_until_two_states() -> None:
    classifier = _toy()
    assert classifier.classify(_features(1, 0, 0, 0)) == Result(Status.NOT_READY)
    assert not classifier.ready

    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    assert classifier.classify(_features(1, 0, 0, 0)).status is Status.NOT_READY
    assert classifier.counts() == {"occupied": 1}

    classifier.add("b", _features(0, 1, 0, 0), state="free")
    assert classifier.ready


def test_unsure_keeps_candidate() -> None:
    classifier = _toy(margin=0.2)
    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    classifier.add("b", _features(0, 1, 0, 0), state="free")

    result = classifier.classify(_features(1, 0.9, 0, 0))

    assert result.status is Status.UNSURE
    assert result.state is None
    assert result.candidate == "occupied"
    assert 0 < result.margin < 0.2
    assert result.nearest == {"occupied": "a", "free": "b"}


def test_margin_is_inclusive() -> None:
    query = _features(1, 0.5, 0.2, 0)
    probe = _toy(margin=0.01)
    probe.add("a", _features(1, 0, 0, 0), state="occupied")
    probe.add("b", _features(0, 1, 0, 0), state="free")
    margin = probe.classify(query).margin

    exact = _toy(margin=margin)
    exact.add("a", _features(1, 0, 0, 0), state="occupied")
    exact.add("b", _features(0, 1, 0, 0), state="free")

    assert exact.classify(query).status is Status.CONFIDENT


def test_query_rows_use_best_match() -> None:
    classifier = _toy()
    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    classifier.add("b", _features(0, 1, 0, 0), state="free")
    rows = np.array([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.float32)

    result = classifier.classify(Features("test/1", rows[0], rows))

    assert result.state == "occupied"
    assert result.scores["occupied"] == pytest.approx(1.0)


def test_more_references_never_hurt() -> None:
    """k shrinks to the smallest state, so extra good references cannot lose."""
    query = _features(1, 0, 0, 0)
    occupied = [(0.97, 0.243), (0.85, 0.527), (0.80, 0.6)]
    classifier = _toy(margin=0.01)
    classifier.add("free", _features(0.90, 0.436, 0, 0), state="free")
    for index, (x, z) in enumerate(occupied):
        classifier.add(f"car-{index}", _features(x, 0, z, 0), state="occupied")

    result = classifier.classify(query)

    assert result.state == "occupied"
    assert result.scores["occupied"] == pytest.approx(0.97, abs=1e-3)


def test_top_k_averages_best_matches() -> None:
    classifier = _toy(margin=0.01, k=2)
    classifier.add("o1", _features(1, 0, 0, 0), state="occupied")
    classifier.add("o2", _features(0.6, 0.8, 0, 0), state="occupied")
    classifier.add("o3", _features(0, 0, 1, 0), state="occupied")
    classifier.add("f1", _features(0.8, 0, 0, 0.6), state="free")
    classifier.add("f2", _features(0.7, 0, 0, 0.714), state="free")

    result = classifier.classify(_features(1, 0, 0, 0))

    assert result.scores["occupied"] == pytest.approx((1.0 + 0.6) / 2, abs=1e-3)
    assert result.scores["free"] == pytest.approx((0.8 + 0.7) / 2, abs=1e-2)


def test_relabel_updates_lookup() -> None:
    classifier = _toy()
    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    classifier.add("b", _features(0, 1, 0, 0), state="free")
    classifier.add("c", _features(0, 0, 1, 0), state="other")
    assert classifier.classify(_features(1, 0, 0, 0)).state == "occupied"

    classifier.add("a", _features(1, 0, 0, 0), state="free")

    assert classifier.references() == {"a": "free", "b": "free", "c": "other"}
    assert classifier.classify(_features(1, 0, 0, 0)).state == "free"


def test_remove_updates_lookup() -> None:
    classifier = _toy()
    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    classifier.add("a2", _features(0, 0, 0, 1), state="occupied")
    classifier.add("b", _features(0.9, 0.436, 0, 0), state="free")
    assert classifier.classify(_features(1, 0, 0, 0)).state == "occupied"

    classifier.remove("a")
    classifier.remove("missing")

    assert len(classifier) == 2
    assert classifier.classify(_features(1, 0, 0, 0)).state == "free"


def test_clear() -> None:
    classifier = _toy()
    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    classifier.remove("a")
    assert classifier.classify(_features(1, 0, 0, 0)).status is Status.NOT_READY
    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    classifier.clear()
    assert len(classifier) == 0
    assert classifier.references() == {}
    assert classifier.counts() == {}


def test_rejects_other_backend(make_scene) -> None:
    classifier = _toy()
    features = extract_features(EdgeBackend(), make_scene(car=True, seed=0))
    with pytest.raises(BackendMismatchError, match="expected test/1"):
        classifier.add("a", features, state="occupied")
    with pytest.raises(ValueError, match="expected test/1"):
        classifier.classify(features)


def test_rejects_other_vector_length() -> None:
    classifier = _toy()
    classifier.add("a", _features(1, 0, 0, 0), state="occupied")
    with pytest.raises(BackendMismatchError, match="vector length 3"):
        classifier.add("b", _features(1, 0, 0), state="free")


def test_rejects_blank_reference(make_scene) -> None:
    """A grey frame from an offline camera would match nothing."""
    backend = EdgeBackend()
    blank = make_scene(car=False, seed=0, clutter=False).point(lambda _: 128)
    classifier = Classifier.for_backend(backend)
    with pytest.raises(BlankRegionError, match="no structure"):
        classifier.add("blank", extract_features(backend, blank), state="free")


@pytest.mark.parametrize(("margin", "k"), [(0.0, 3), (-0.1, 3), (0.02, 0)])
def test_rejects_bad_settings(margin: float, k: int) -> None:
    with pytest.raises(ValueError):
        Classifier("test/1", margin=margin, k=k)


def _vote(state: str | None, candidate: str | None = None) -> Result:
    scores = {"a": 0.1, "b": 0.1}
    if state is None:
        return Result(Status.UNSURE, None, candidate, 0.0, scores)
    return Result(Status.CONFIDENT, state, state, 0.5, scores)


@pytest.mark.parametrize(
    ("results", "status", "state", "candidate"),
    [
        pytest.param([], Status.NOT_READY, None, None, id="empty"),
        pytest.param(
            [Result(Status.NOT_READY)] * 2, Status.NOT_READY, None, None, id="not_ready"
        ),
        pytest.param(
            [_vote("a"), _vote("a"), _vote("b")],
            Status.CONFIDENT,
            "a",
            "a",
            id="two_of_three",
        ),
        pytest.param([_vote("a"), _vote("b")], Status.UNSURE, None, "a", id="tie"),
        pytest.param(
            [_vote("a"), _vote(None, "b"), _vote(None, "b")],
            Status.UNSURE,
            None,
            "b",
            id="mostly_unsure",
        ),
        pytest.param(
            [_vote(None, "b"), _vote(None, "b")],
            Status.UNSURE,
            None,
            "b",
            id="all_unsure",
        ),
        pytest.param(
            [_vote("a"), _vote("a"), _vote(None)],
            Status.CONFIDENT,
            "a",
            "a",
            id="one_unsure",
        ),
        pytest.param(
            [Result(Status.NOT_READY), _vote("a"), _vote("a")],
            Status.CONFIDENT,
            "a",
            "a",
            id="partly_ready",
        ),
        pytest.param(
            [Result(Status.NOT_READY)] * 2 + [_vote("a")],
            Status.NOT_READY,
            None,
            None,
            id="mostly_not_ready",
        ),
        pytest.param(
            [Result(Status.NOT_READY), _vote("a")],
            Status.NOT_READY,
            None,
            None,
            id="half_not_ready",
        ),
        pytest.param(
            [_vote("b"), _vote("a")], Status.UNSURE, None, "b", id="tie_goes_first"
        ),
    ],
)
def test_majority(
    results: list[Result], status: Status, state: str | None, candidate: str | None
) -> None:
    result = majority(results)
    assert (result.status, result.state, result.candidate) == (status, state, candidate)


def test_majority_reports_shares() -> None:
    result = majority([_vote("a"), _vote("a"), _vote("b"), _vote(None)])
    assert result.scores == {"a": 0.5, "b": 0.25}
    assert result.margin == 0.5


def test_majority_scores_every_state() -> None:
    """States without a confident vote still get a share, like classify."""
    result = majority([_vote(None, "b"), _vote(None, "b")])
    assert result.scores == {"a": 0.0, "b": 0.0}
    assert result.margin == 0.0


def test_majority_margin_follows_candidate() -> None:
    result = majority([_vote("a"), _vote(None, "b"), _vote(None, "b")])
    assert (result.candidate, result.margin) == ("b", 0.0)
    assert result.scores == {"a": pytest.approx(1 / 3), "b": 0.0}


@pytest.mark.parametrize("share", [0.49, 1.0])
def test_majority_rejects_share(share: float) -> None:
    with pytest.raises(ValueError, match="min_share"):
        majority([], min_share=share)


def test_majority_stricter_share() -> None:
    results = [_vote("a")] * 3 + [_vote(None)]
    assert majority(results, min_share=0.7).state == "a"
    assert majority(results, min_share=0.75).status is Status.UNSURE


def test_unit_ignores_offset_and_scale() -> None:
    """NCC: an edge map with more overall energy still matches its pattern."""
    pattern = np.array([[0.0, 1.0], [2.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(unit(pattern * 3 + 7), unit(pattern), atol=1e-6)
    assert float(np.linalg.norm(unit(pattern))) == pytest.approx(1.0)
    assert not unit(np.ones((2, 2), np.float32)).any()


def test_features_compare_by_identity() -> None:
    features = _features(1, 0, 0, 0)
    same = features
    assert features == same
    assert features != _features(1, 0, 0, 0)


def test_add_many_rebuilds_once() -> None:
    classifier = _toy()
    classifier.add_many(
        [
            ("a", _features(1, 0, 0, 0), "occupied"),
            ("b", _features(0, 1, 0, 0), "free"),
        ]
    )
    assert classifier.references() == {"a": "occupied", "b": "free"}
    classifier.add_many([])
    assert len(classifier) == 2


def test_add_many_rejects_mixed_lengths() -> None:
    classifier = _toy()
    with pytest.raises(BackendMismatchError, match="vector length 3"):
        classifier.add_many(
            [("a", _features(1, 0, 0, 0), "on"), ("b", _features(1, 0, 0), "off")]
        )
    assert len(classifier) == 0


def test_add_many_is_atomic(make_scene) -> None:
    """A blank item in a batch leaves the classifier unchanged."""
    backend = EdgeBackend()
    classifier = Classifier.for_backend(backend)
    good = extract_features(backend, make_scene(car=True, seed=0))
    blank = Features(backend.id, np.zeros_like(good.vector), np.zeros_like(good.query))
    with pytest.raises(BlankRegionError):
        classifier.add_many([("a", good, "on"), ("b", blank, "off")])
    assert len(classifier) == 0


def test_remove_many() -> None:
    classifier = _toy()
    classifier.add_many(
        [(key, _features(1, i, 0, 0), "on") for i, key in enumerate("abc")]
    )
    classifier.remove_many(["a", "c", "missing"])
    classifier.remove_many([])
    assert classifier.references() == {"b": "on"}
