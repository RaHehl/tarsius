"""Feature vectors and the backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from .roi import FULL_FRAME, Roi

type Vector = npt.NDArray[np.float32]


class FeatureBackend(Protocol):
    """Turns an image region into vectors that compare by dot product."""

    # Changes whenever extracted features change, so stored ones get rebuilt.
    id: str
    # Similarity gap between the two best states below which a result is unsure.
    default_margin: float

    def extract(self, image: Image.Image) -> tuple[Vector, Vector]:
        """
        Return ``(vector, query)`` for ``image``.

        ``vector`` is the unit-norm reference form. ``query`` holds one or more
        unit-norm rows; the similarity of a query to a reference is the best
        row, which lets a backend tolerate small camera shifts.
        """
        ...


# Arrays do not compare with ==, so identity equality.
@dataclass(frozen=True, slots=True, eq=False)
class Features:
    """Features of one image region."""

    backend_id: str
    vector: Vector
    query: Vector

    @property
    def has_structure(self) -> bool:
        """Return False for a region without anything to compare, e.g. a grey frame."""
        return bool(np.linalg.norm(self.vector) > 0)


def extract_features(
    backend: FeatureBackend, image: Image.Image, roi: Roi = FULL_FRAME
) -> Features:
    """Crop ``image`` to ``roi`` and extract its features."""
    vector, query = backend.extract(roi.crop(image))
    return Features(backend.id, vector, query)


def unit(array: npt.NDArray[np.floating]) -> Vector:
    """Return ``array`` flattened, zero-mean and scaled to unit length."""
    flat = array.astype(np.float32).ravel()
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 0 else flat
