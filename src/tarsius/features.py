"""Feature vectors and the backend protocol."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from .errors import FeaturesError
from .roi import FULL_FRAME, Roi

type Vector = npt.NDArray[np.float32]

# Bump when the stored layout changes; older data then fails to load.
_FORMAT = 1


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
class Reference:
    """The part of a confirmed frame a classifier keeps."""

    backend_id: str
    vector: Vector

    def to_bytes(self) -> bytes:
        """Serialize for a cache next to the original image."""
        buffer = BytesIO()
        np.savez(
            buffer,
            format=np.array(_FORMAT),
            backend_id=np.array(self.backend_id),
            vector=self.vector,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> Reference:
        """Load data written by :meth:`to_bytes`; raise FeaturesError if unusable."""
        try:
            with np.load(BytesIO(data), allow_pickle=False) as npz:
                if int(npz["format"]) != _FORMAT:
                    raise FeaturesError(f"unsupported format {int(npz['format'])}")
                return cls(str(npz["backend_id"]), npz["vector"].astype(np.float32))
        except FeaturesError:
            raise
        except (EOFError, OSError, KeyError, ValueError, zipfile.BadZipFile) as err:
            raise FeaturesError(f"unreadable features: {err}") from err


@dataclass(frozen=True, slots=True, eq=False)
class Features:
    """Features of one image region."""

    backend_id: str
    vector: Vector
    query: Vector

    def reference(self) -> Reference:
        """Return what a classifier keeps when this frame is confirmed."""
        return Reference(self.backend_id, self.vector)


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
