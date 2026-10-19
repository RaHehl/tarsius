"""Detect the state of one region from confirmed reference frames."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from PIL import Image, ImageStat

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


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Outcome of loading stored references."""

    loaded: int
    # Key -> reason for items that could not be used.
    skipped: Mapping[str, SkipReason] = field(default_factory=dict)
    # Keys dropped to stay within the per-state cap.
    evicted: tuple[str, ...] = ()


class StateDetector:
    """
    The state of one region, learned from confirmed frames.

    Owns the region, the backend and the references. Callers store the
    ``original`` of each prepared frame and hand it back to :meth:`load`, so
    references are always re-extracted with the current region and backend.
    Originals are stored at ``frame_side``; raising it later means teaching
    again. The settings are fixed; build a new detector to change them.

    All methods block; call them in an executor. They are safe to call from
    several threads: changes are serialized, :meth:`classify` reads an
    immutable snapshot and never waits for them.
    """

    def __init__(
        self,
        backend: FeatureBackend,
        roi: Roi,
        states: Sequence[str],
        *,
        max_per_state: int = MAX_PER_STATE,
        k: int = 3,
        frame_side: int = FRAME_SIDE,
    ) -> None:
        """
        Create a detector for the given states, without references.

        ``frame_side`` bounds the longer side frames are normalized to; small
        details in a small region need more.
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
        return Prepared(
            original, self._checked_features(stored), self._roi, self._frame_side
        )

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
            evicted = self._select_evictions()
            if evicted:
                self._classifier.remove_many(evicted)
                self._forget(evicted)
            return evicted

    def load(self, items: Iterable[tuple[str, bytes, str]]) -> LoadReport:
        """
        Replace all references with stored ``(key, original, state)`` items.

        Everything not in ``items`` is dropped. Unusable items are skipped and
        reported instead of failing the load; for a repeated key the last item
        wins. A backend failure raises and leaves the references unchanged.
        """
        latest = {key: (data, state) for key, data, state in items}
        with self._lock:
            features_by_key: dict[str, Features] = {}
            labels: dict[str, str] = {}
            skipped: dict[str, SkipReason] = {}
            for key, (data, state) in latest.items():
                if state not in self._states:
                    skipped[key] = SkipReason.UNKNOWN_STATE
                    continue
                try:
                    image = load_image(data, max_side=self._frame_side)
                    features = self._checked_features(image)
                except ImageError:
                    skipped[key] = SkipReason.UNDECODABLE
                except BlankRegionError:
                    skipped[key] = SkipReason.BLANK
                else:
                    features_by_key[key] = features
                    labels[key] = state
            self._labels = labels
            self._vectors = {key: f.vector for key, f in features_by_key.items()}
            evicted = self._select_evictions()
            self._forget(evicted)
            classifier = Classifier(
                self._backend.id, margin=self._classifier.margin, k=self._k
            )
            classifier.add_many(
                (key, features_by_key[key], state)
                for key, state in self._labels.items()
            )
            self._classifier = classifier
            return LoadReport(len(self._labels), skipped, tuple(evicted))

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
        result = classifier.classify(extract_features(self._backend, image, self._roi))
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

    def _select_evictions(self) -> list[str]:
        """Pick the most redundant references until every state fits the cap."""
        evicted: list[str] = []
        for state in self._states:
            keys = [key for key, label in self._labels.items() if label == state]
            excess = len(keys) - self._max_per_state
            if excess <= 0:
                continue
            vectors = np.stack([self._vectors[key] for key in keys])
            similarity = vectors @ vectors.T
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
