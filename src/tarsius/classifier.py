"""Nearest-reference classifier with explicit "unsure" and "not ready" results."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .errors import BackendMismatchError
from .features import FeatureBackend, Features, Reference, Vector


class Status(StrEnum):
    """Why a result has or lacks a state."""

    CONFIDENT = "confident"
    # The best state does not beat the runner-up by the margin: ask the user.
    UNSURE = "unsure"
    # Fewer than two states have references: nothing to compare yet.
    NOT_READY = "not_ready"


@dataclass(frozen=True, slots=True)
class Result:
    """Outcome of one classification."""

    status: Status
    # Set only when confident.
    state: str | None = None
    # Best state even when unsure, e.g. to ask "is this <candidate>?".
    candidate: str | None = None
    margin: float = 0.0
    # Diagnostics only; the scale depends on the backend.
    scores: Mapping[str, float] = field(default_factory=dict)
    # Most similar reference key per state.
    nearest: Mapping[str, str] = field(default_factory=dict)


class _Index(NamedTuple):
    keys: tuple[str, ...]
    labels: npt.NDArray[np.str_]
    matrix: Vector


class Classifier:
    """
    k-nearest references per state, compared by similarity.

    A state wins only when its mean top-k similarity beats every other state
    by ``margin``. ``k`` shrinks to the smallest state's reference count, so
    adding references to one state never makes it lose. Safe to share between
    threads: changes swap in a new snapshot that lookups read once.
    """

    def __init__(self, backend_id: str, *, margin: float, k: int = 3) -> None:
        """Create an empty classifier for features of ``backend_id``."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if margin <= 0:
            raise ValueError("margin must be positive, a tie is never confident")
        self.backend_id = backend_id
        self.margin = margin
        self.k = k
        self._lock = threading.Lock()
        self._refs: dict[str, tuple[str, Vector]] = {}
        self._index: _Index | None = None

    @classmethod
    def for_backend(cls, backend: FeatureBackend, *, k: int = 3) -> Classifier:
        """Create a classifier with the backend's id and default margin."""
        return cls(backend.id, margin=backend.default_margin, k=k)

    def add(self, key: str, features: Features | Reference, *, state: str) -> None:
        """
        Add or replace the reference ``key``.

        Raises ValueError for a region without any structure, e.g. a blank
        frame from an offline camera, since it would match nothing.
        """
        self._check(features.backend_id)
        vector = features.vector
        if not np.linalg.norm(vector) > 0:
            raise ValueError("reference has no structure to compare")
        with self._lock:
            index = self._index
            if index is not None and index.matrix.shape[1] != vector.shape[0]:
                raise BackendMismatchError(
                    f"vector length {vector.shape[0]}, expected {index.matrix.shape[1]}"
                )
            self._refs[key] = (state, vector)
            self._index = self._build()

    def remove(self, key: str) -> None:
        """Drop the reference ``key`` if present."""
        with self._lock:
            if self._refs.pop(key, None) is not None:
                self._index = self._build()

    def clear(self) -> None:
        """Drop all references."""
        with self._lock:
            self._refs.clear()
            self._index = None

    def references(self) -> dict[str, str]:
        """Return the state of every reference key."""
        index = self._index
        if index is None:
            return {}
        return dict(zip(index.keys, index.labels.tolist(), strict=True))

    def counts(self) -> dict[str, int]:
        """Return the number of references per state."""
        index = self._index
        if index is None:
            return {}
        states, counts = np.unique(index.labels, return_counts=True)
        return dict(zip(states.tolist(), counts.tolist(), strict=True))

    @property
    def ready(self) -> bool:
        """Return True once at least two states have references."""
        return len(self.counts()) >= 2

    def __len__(self) -> int:
        """Return the number of references."""
        index = self._index
        return 0 if index is None else len(index.keys)

    def classify(self, features: Features) -> Result:
        """Classify ``features`` against the references."""
        self._check(features.backend_id)
        index = self._index
        if index is None:
            return Result(Status.NOT_READY)
        states, counts = np.unique(index.labels, return_counts=True)
        if len(states) < 2:
            return Result(Status.NOT_READY)
        similarity = (features.query @ index.matrix.T).max(axis=0)
        k = min(self.k, int(counts.min()))
        scores: dict[str, float] = {}
        nearest: dict[str, str] = {}
        for state in states.tolist():
            members = np.flatnonzero(index.labels == state)
            ranked = members[np.argsort(similarity[members])[::-1]]
            scores[state] = float(similarity[ranked[:k]].mean())
            nearest[state] = index.keys[ranked[0]]
        best, runner_up = sorted(scores, key=scores.__getitem__, reverse=True)[:2]
        margin = scores[best] - scores[runner_up]
        confident = margin >= self.margin
        return Result(
            Status.CONFIDENT if confident else Status.UNSURE,
            best if confident else None,
            best,
            margin,
            scores,
            nearest,
        )

    def _check(self, backend_id: str) -> None:
        if backend_id != self.backend_id:
            raise BackendMismatchError(
                f"features from {backend_id}, expected {self.backend_id}"
            )

    def _build(self) -> _Index | None:
        if not self._refs:
            return None
        keys = tuple(self._refs)
        return _Index(
            keys,
            np.array([self._refs[key][0] for key in keys]),
            np.stack([self._refs[key][1] for key in keys]),
        )
