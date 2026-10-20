"""Detect the state of one region from confirmed reference frames."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from PIL import Image, ImageStat

from . import cache
from .backends.ncnn import NcnnBackend
from .classifier import Classifier, Result, Status
from .errors import BlankFrameError, BlankRegionError, ImageError
from .features import FeatureBackend, Features, Vector, extract_features
from .image import FRAME_SIDE, encode_jpeg, load_image
from .roi import Roi

MAX_PER_STATE = 50
# Grey level spread of the whole frame below which the camera shows nothing.
# Real frames, IR nights included, stay above 20, a grey placeholder with
# sensor noise below 2. The region itself may be uniform in one state.
BLANK_STDDEV = 4.0
# Cosine above which a cached vector counts as the one extracted now.
_SAME_VECTOR = 1 - 1e-4


class SkipReason(StrEnum):
    """Why :meth:`StateDetector.load` did not use a stored item."""

    BLANK = "blank"
    UNDECODABLE = "undecodable"
    UNKNOWN_STATE = "unknown_state"


# Arrays do not compare with ==, so identity equality.
@dataclass(frozen=True, slots=True, eq=False)
class Prepared:
    """A frame checked and ready to become a reference of one detector."""

    # Normalized JPEG for the caller to store and pass to load() later.
    original: bytes
    features: Features
    roi: Roi
    frame_side: int

    @property
    def cache(self) -> bytes:
        """Return the features of the original, to store next to it for load()."""
        return cache.dump(self.features, self.original, self.roi, self.frame_side)


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Outcome of loading stored references."""

    loaded: int
    # Key -> reason for items that could not be used.
    skipped: Mapping[str, SkipReason] = field(default_factory=dict)
    # Keys dropped to stay within the per-state cap.
    evicted: tuple[str, ...] = ()
    # Key -> new cache for items extracted without a usable one. Only these
    # are listed: store them, keep the caches of the others and drop those
    # of skipped and evicted keys.
    caches: Mapping[str, bytes] = field(default_factory=dict)


class _SharedBackends:
    """
    Backends loaded once per process and shared by every task using them.

    Loading holds the lock, so loads of different backends wait for each
    other. Do not fork while another thread uses a shared backend.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backends: dict[Callable[[], FeatureBackend], FeatureBackend] = {}

    def get(self, factory: Callable[[], FeatureBackend]) -> FeatureBackend:
        """Return the shared backend, loading it first; a failure is not kept."""
        with self._lock:
            if (backend := self._backends.get(factory)) is None:
                backend = self._backends[factory] = factory()
            return backend


_SHARED = _SharedBackends()
# The backend StateDetector uses unless one is passed.
_STATE_BACKEND: Callable[[], FeatureBackend] = NcnnBackend


class StateDetector:
    """
    The state of one region, learned from confirmed frames.

    Owns the region and the references, and uses a backend shared per process.
    Callers store the ``original`` of each prepared frame and hand it back to
    :meth:`load`, so references follow the current region and backend. Storing
    the ``cache`` too spares the extraction on load while nothing changed.
    Originals are stored at ``frame_side``; raising it later means teaching
    again. The settings are fixed; build a new detector to change them.

    All methods block; call them in an executor. They are safe to call from
    several threads: changes are serialized, :meth:`classify` reads an
    immutable snapshot and never waits for them. Only inferences on the
    shared backend take turns.
    """

    def __init__(
        self,
        roi: Roi,
        states: Sequence[str],
        *,
        backend: FeatureBackend | None = None,
        max_per_state: int = MAX_PER_STATE,
        k: int = 3,
        frame_side: int = FRAME_SIDE,
    ) -> None:
        """
        Create a detector for the given states, without references.

        The detector picks its backend for this task, shared by all detectors of
        the process that don't pass one. The first one loads the model, so this
        can block and raise BackendError. ``backend_id`` tells which backend it
        is, and a stored original is simply re-extracted when it changes. Pass
        ``backend`` only to override it, e.g. in tests. ``frame_side`` bounds
        the longer side frames are normalized to; small details in a small
        region need more.
        """
        if isinstance(states, str):
            raise ValueError("states must be a sequence of distinct names")
        states = tuple(states)
        if len(set(states)) != len(states):
            raise ValueError("states must be a sequence of distinct names")
        if len(states) < 2:
            raise ValueError("states must be at least two distinct names")
        if max_per_state < 1:
            raise ValueError("max_per_state must be at least 1")
        if frame_side < 1:
            raise ValueError("frame_side must be positive")
        if backend is None:
            backend = _SHARED.get(_STATE_BACKEND)
        self._backend = backend
        self._roi = roi
        self._states = tuple(states)
        self._max_per_state = max_per_state
        self._frame_side = frame_side
        self._k = k
        self._classifier = Classifier.for_backend(backend, k=k)
        self._lock = threading.Lock()
        self._vectors: dict[str, Vector] = {}
        self._labels: dict[str, str] = {}

    @property
    def backend(self) -> FeatureBackend:
        """Return the feature backend."""
        return self._backend

    @property
    def backend_id(self) -> str:
        """Return the id of the backend in use."""
        return self._backend.id

    @property
    def roi(self) -> Roi:
        """Return the region."""
        return self._roi

    @property
    def states(self) -> tuple[str, ...]:
        """Return the declared states."""
        return self._states

    @property
    def max_per_state(self) -> int:
        """Return the reference cap per state."""
        return self._max_per_state

    @property
    def frame_side(self) -> int:
        """Return the longer side frames are normalized to."""
        return self._frame_side

    @property
    def ready(self) -> bool:
        """Return True once every state has a reference."""
        return not self.missing_states()

    def missing_states(self) -> list[str]:
        """Return the states without any reference, in declared order."""
        counts = self.counts()
        return [state for state in self._states if not counts.get(state)]

    def counts(self) -> dict[str, int]:
        """Return the number of references per state."""
        return self._classifier.counts()

    def references(self) -> dict[str, str]:
        """Return the state of every reference key."""
        return self._classifier.references()

    def prepare(self, data: bytes) -> Prepared:
        """
        Check a frame and extract its features.

        Raises ImageError for undecodable data, BlankFrameError when the whole
        frame shows nothing and BlankRegionError when the region has nothing
        to compare, so it cannot serve as a reference.
        """
        original = encode_jpeg(load_image(data, max_side=self._frame_side))
        # Extract from what is stored, so a reload gives the same features.
        stored = load_image(original, max_side=self._frame_side)
        features = self._checked_features(stored)
        return Prepared(original, features, self._roi, self._frame_side)

    def add(self, key: str, prepared: Prepared, *, state: str) -> list[str]:
        """
        Add or replace a reference; return the keys evicted to stay in the cap.

        When the cap is exceeded the most redundant reference goes; between
        two equal ones the older is dropped, so the newest stays.
        """
        self._check_state(state)
        if prepared.roi != self._roi or prepared.frame_side != self._frame_side:
            raise ValueError("prepared for another region or frame size")
        with self._lock:
            self._classifier.add(key, prepared.features, state=state)
            # Re-insert so a replaced key counts as the newest.
            self._forget([key])
            self._vectors[key] = prepared.features.vector
            self._labels[key] = state
            evicted = self._select_evictions(self._labels, self._vectors)
            if evicted:
                self._classifier.remove_many(evicted)
                self._forget(evicted)
            return evicted

    def load(
        self,
        items: Iterable[tuple[str, bytes, str]],
        *,
        caches: Mapping[str, bytes] | None = None,
    ) -> LoadReport:
        """
        Replace all references with stored ``(key, original, state)`` items.

        Everything not in ``items`` is dropped. Unusable items are skipped and
        reported instead of failing the load; for a repeated key the last item
        wins. A backend failure raises and leaves the references unchanged.

        ``caches`` maps keys to the ``cache`` of their prepared frame or of an
        earlier report. A cache made for other settings, another backend, a
        release of tarsius, Pillow or ncnn, or a damaged one is ignored, and
        the report holds a new one. One reference is extracted anyway to check
        the caches against the backend, if they disagree all are extracted.
        """
        latest = {key: (data, state) for key, data, state in items}
        with self._lock:
            features, labels, skipped, fresh = self._load_features(latest, caches or {})
            if features and not self._fits_backend(features, latest, fresh):
                # A cache of another length slipped through, extract them all.
                features, labels, skipped, fresh = self._load_features(latest, {})
            vectors = {key: f.vector for key, f in features.items()}
            evicted = self._select_evictions(labels, vectors)
            for key in evicted:
                del labels[key], vectors[key]
                fresh.pop(key, None)
            classifier = Classifier(
                self._backend.id, margin=self._classifier.margin, k=self._k
            )
            classifier.add_many(
                (key, features[key], state) for key, state in labels.items()
            )
            # Nothing changes until every step above succeeded.
            self._labels, self._vectors, self._classifier = labels, vectors, classifier
            return LoadReport(len(labels), skipped, tuple(evicted), fresh)

    def _load_features(
        self, latest: Mapping[str, tuple[bytes, str]], caches: Mapping[str, bytes]
    ) -> tuple[
        dict[str, Features], dict[str, str], dict[str, SkipReason], dict[str, bytes]
    ]:
        """Restore or extract the features of each item."""
        features_by_key: dict[str, Features] = {}
        labels: dict[str, str] = {}
        skipped: dict[str, SkipReason] = {}
        fresh: dict[str, bytes] = {}
        for key, (data, state) in latest.items():
            if state not in self._states:
                skipped[key] = SkipReason.UNKNOWN_STATE
                continue
            if (blob := caches.get(key)) is not None and (
                restored := cache.restore(
                    blob, data, self._backend.id, self._roi, self._frame_side
                )
            ):
                features_by_key[key] = restored
                labels[key] = state
                continue
            try:
                image = load_image(data, max_side=self._frame_side)
                extracted = self._checked_features(image)
            except ImageError:
                skipped[key] = SkipReason.UNDECODABLE
            except BlankRegionError:
                skipped[key] = SkipReason.BLANK
            else:
                features_by_key[key] = extracted
                labels[key] = state
                fresh[key] = cache.dump(extracted, data, self._roi, self._frame_side)
        return features_by_key, labels, skipped, fresh

    def _fits_backend(
        self,
        features: Mapping[str, Features],
        latest: Mapping[str, tuple[bytes, str]],
        fresh: Mapping[str, bytes],
    ) -> bool:
        """Check the cached vectors against what the backend gives now."""
        if fresh:
            length = features[next(iter(fresh))].vector.size
            return all(f.vector.size == length for f in features.values())
        # All came from caches: extract one and compare it with its cache.
        key = next(iter(features))
        try:
            image = load_image(latest[key][0], max_side=self._frame_side)
            probe = self._checked_features(image).vector
        except ImageError, BlankRegionError:
            # The full extraction skips it with its reason.
            return False
        cached = features[key].vector
        return probe.size == cached.size and float(probe @ cached) >= _SAME_VECTOR

    def remove(self, keys: str | Iterable[str]) -> None:
        """Remove one or more references; unknown keys are ignored."""
        keys = [keys] if isinstance(keys, str) else list(keys)
        with self._lock:
            self._classifier.remove_many(keys)
            self._forget(keys)

    def classify(self, data: bytes) -> Result:
        """
        Classify a frame; not ready until every state has a reference.

        Raises ImageError for undecodable data and BlankFrameError for a frame
        that shows nothing, e.g. from an offline camera. A region without
        structure is answered as unsure. While not ready the frame is not
        looked at.
        """
        classifier = self._classifier
        if len(classifier.counts()) < len(self._states):
            return Result(Status.NOT_READY)
        image = load_image(data, max_side=self._frame_side)
        self._check_frame(image)
        features = extract_features(self._backend, image, self._roi)
        if not features.has_structure:
            # Every reference would score the same, there is no candidate.
            return Result(Status.UNSURE, scores=dict.fromkeys(self._states, 0.0))
        result = classifier.classify(features)
        # A reference may have been removed since the check above.
        if len(result.scores) < len(self._states):
            return Result(Status.NOT_READY)
        return result

    def _check_frame(self, image: Image.Image) -> None:
        if ImageStat.Stat(image.convert("L")).stddev[0] < BLANK_STDDEV:
            raise BlankFrameError("the frame shows nothing, the camera may be offline")

    def _checked_features(self, image: Image.Image) -> Features:
        self._check_frame(image)
        features = extract_features(self._backend, image, self._roi)
        if not features.has_structure:
            raise BlankRegionError("the region has no structure to compare")
        return features

    def _check_state(self, state: str) -> None:
        if state not in self._states:
            raise ValueError(f"unknown state {state!r}, expected one of {self._states}")

    def _forget(self, keys: Iterable[str]) -> None:
        for key in keys:
            self._vectors.pop(key, None)
            self._labels.pop(key, None)

    def _select_evictions(
        self, labels: Mapping[str, str], vectors: Mapping[str, Vector]
    ) -> list[str]:
        """Pick the most redundant references until every state fits the cap."""
        evicted: list[str] = []
        for state in self._states:
            keys = [key for key, label in labels.items() if label == state]
            excess = len(keys) - self._max_per_state
            if excess <= 0:
                continue
            stacked = np.stack([vectors[key] for key in keys])
            similarity = stacked @ stacked.T
            np.fill_diagonal(similarity, -np.inf)
            alive = np.ones(len(keys), dtype=bool)
            for _ in range(excess):
                closest = np.where(alive, similarity.max(axis=1), -np.inf)
                # Keys are oldest first and argmax takes the first of equals.
                drop = int(np.argmax(closest))
                alive[drop] = False
                similarity[:, drop] = -np.inf
                similarity[drop, :] = -np.inf
                evicted.append(keys[drop])
        return evicted
