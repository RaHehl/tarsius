"""Detect the state of one region from confirmed reference frames."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .classifier import Classifier, Result, Status
from .errors import BlankRegionError, ImageError
from .features import FeatureBackend, Features, Vector, extract_features
from .image import FRAME_SIDE, encode_jpeg, load_image
from .roi import Roi

MAX_PER_STATE = 50


# Arrays do not compare with ==, so identity equality.
@dataclass(frozen=True, slots=True, eq=False)
class Prepared:
    """A frame checked and ready to become a reference."""

    # Normalized JPEG for the caller to store and pass to load() later.
    original: bytes
    features: Features


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Outcome of loading stored references."""

    loaded: int
    # Key -> reason for references that could not be used.
    skipped: Mapping[str, str] = field(default_factory=dict)
    # Keys dropped to stay within the per-state cap.
    evicted: Sequence[str] = ()


class StateDetector:
    """
    The state of one region, learned from confirmed frames.

    Owns the region, the backend and the references. Callers store the
    ``original`` of each prepared frame and hand it back to :meth:`load`,
    so references are always re-extracted with the current region and
    backend. All methods are blocking; run them in an executor. They may
    run concurrently: :meth:`classify` reads an immutable snapshot.
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
        if len(set(states)) < 2 or len(set(states)) != len(states):
            raise ValueError("states must be at least two distinct names")
        if max_per_state < 1:
            raise ValueError("max_per_state must be at least 1")
        self.backend = backend
        self.roi = roi
        self.states = tuple(states)
        self.max_per_state = max_per_state
        self.frame_side = frame_side
        self._k = k
        self._classifier = Classifier.for_backend(backend, k=k)
        self._lock = threading.Lock()
        self._vectors: dict[str, Vector] = {}
        self._labels: dict[str, str] = {}

    @property
    def backend_id(self) -> str:
        """Return the id of the backend in use."""
        return self.backend.id

    @property
    def ready(self) -> bool:
        """Return True once every state has a reference."""
        return not self.missing_states()

    def missing_states(self) -> list[str]:
        """Return the states without any reference, in declared order."""
        counts = self.counts()
        return [state for state in self.states if not counts.get(state)]

    def counts(self) -> dict[str, int]:
        """Return the number of references per state."""
        return self._classifier.counts()

    def references(self) -> dict[str, str]:
        """Return the state of every reference key."""
        return self._classifier.references()

    def prepare(self, data: bytes) -> Prepared:
        """
        Check a frame and extract its features.

        Raises ImageError for undecodable data and BlankRegionError when the
        region shows nothing to compare.
        """
        image = load_image(data, max_side=self.frame_side)
        return Prepared(encode_jpeg(image), self._checked_features(image))

    def add(self, key: str, state: str, prepared: Prepared) -> list[str]:
        """
        Add a reference; return the keys evicted to stay within the cap.

        The new key itself can be evicted when it adds nothing new.
        """
        self._check_state(state)
        with self._lock:
            self._classifier.add(key, prepared.features, state=state)
            self._vectors[key] = prepared.features.vector
            self._labels[key] = state
            return self._evict(state)

    def load(self, items: Iterable[tuple[str, str, bytes]]) -> LoadReport:
        """
        Replace all references with stored ``(key, state, original)`` items.

        Unusable items are skipped and reported instead of failing the load.
        """
        prepared: list[tuple[str, Features, str]] = []
        skipped: dict[str, str] = {}
        for key, state, data in items:
            if state not in self.states:
                skipped[key] = "unknown_state"
                continue
            try:
                features = self._checked_features(
                    load_image(data, max_side=self.frame_side)
                )
            except ImageError:
                skipped[key] = "undecodable"
            except BlankRegionError:
                skipped[key] = "blank"
            else:
                prepared.append((key, features, state))
        classifier = Classifier(
            self.backend.id, margin=self._classifier.margin, k=self._k
        )
        classifier.add_many(prepared)
        with self._lock:
            # Swap in one step, so a concurrent classify never sees it empty.
            self._classifier = classifier
            self._vectors = {key: features.vector for key, features, _ in prepared}
            self._labels = {key: state for key, _, state in prepared}
            evicted = [key for state in self.states for key in self._evict(state)]
        return LoadReport(len(prepared) - len(evicted), skipped, evicted)

    def remove(self, keys: Iterable[str]) -> None:
        """Remove references; unknown keys are ignored."""
        with self._lock:
            for key in keys:
                self._drop(key)

    def classify(self, data: bytes) -> Result:
        """Classify a frame; not ready until every state has a reference."""
        if not self.ready:
            return Result(Status.NOT_READY)
        image = load_image(data, max_side=self.frame_side)
        return self._classifier.classify(
            extract_features(self.backend, image, self.roi)
        )

    def most_redundant(self, state: str) -> str | None:
        """Return the reference most similar to another one of ``state``."""
        with self._lock:
            return self._most_redundant(state)

    def _checked_features(self, image: Image.Image) -> Features:
        features = extract_features(self.backend, image, self.roi)
        if not features.has_structure:
            raise BlankRegionError("the region has no structure to compare")
        return features

    def _check_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"unknown state {state!r}, expected one of {self.states}")

    def _evict(self, state: str) -> list[str]:
        evicted = []
        while sum(1 for label in self._labels.values() if label == state) > (
            self.max_per_state
        ):
            key = self._most_redundant(state)
            assert key is not None
            self._drop(key)
            evicted.append(key)
        return evicted

    def _drop(self, key: str) -> None:
        self._classifier.remove(key)
        self._vectors.pop(key, None)
        self._labels.pop(key, None)

    def _most_redundant(self, state: str) -> str | None:
        keys = [key for key, label in self._labels.items() if label == state]
        if len(keys) < 2:
            return None
        vectors = np.stack([self._vectors[key] for key in keys])
        similarity = vectors @ vectors.T
        np.fill_diagonal(similarity, -np.inf)
        return keys[int(similarity.max(axis=1).argmax())]
