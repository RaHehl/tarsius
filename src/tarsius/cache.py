"""Features of a stored reference, kept so a reload can skip extraction."""

from __future__ import annotations

import hashlib
import json
import struct
from importlib.metadata import version

import numpy as np

from .features import Features, Vector
from .roi import Roi

MAGIC = b"tarsius-features"
FORMAT = 1
# Any release of tarsius, or of the decoder and inference library, may change
# how features come out of an image, so a cache from another one is extracted
# again.
_VERSIONS = {name: version(name) for name in ("tarsius", "pillow", "ncnn")}
_HEADER_SIZE = struct.Struct("<I")
# References are unit vectors; anything else was not written by dump().
_NORM_TOLERANCE = 1e-3


def _key(
    original: bytes, backend_id: str, roi: Roi, frame_side: int
) -> dict[str, object]:
    return {
        "format": FORMAT,
        "versions": _VERSIONS,
        "backend": backend_id,
        "roi": [roi.left, roi.top, roi.right, roi.bottom],
        "frame_side": frame_side,
        "original": hashlib.sha256(original).hexdigest(),
    }


def dump(features: Features, original: bytes, roi: Roi, frame_side: int) -> bytes:
    """Serialize the reference vector of ``original`` for these settings."""
    vector = features.vector.astype("<f4")
    header = _key(original, features.backend_id, roi, frame_side) | {
        "vector": len(vector)
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return b"".join((MAGIC, _HEADER_SIZE.pack(len(encoded)), encoded, vector.tobytes()))


def restore(
    data: bytes, original: bytes, backend_id: str, roi: Roi, frame_side: int
) -> Features | None:
    """
    Return the cached features, None unless they fit exactly these settings.

    Only the reference vector is kept, a reference never serves as a query.
    """
    vector = _read(data, _key(original, backend_id, roi, frame_side))
    if vector is None or not np.isfinite(vector).all():
        return None
    if abs(float(np.linalg.norm(vector)) - 1) > _NORM_TOLERANCE:
        return None
    return Features(backend_id, vector, vector[None])


def _read(data: bytes, expected: dict[str, object]) -> Vector | None:
    """Return the vector of a well-formed cache with this key."""
    try:
        data = bytes(data)
        if not data.startswith(MAGIC):
            return None
        offset = len(MAGIC)
        (size,) = _HEADER_SIZE.unpack_from(data, offset)
        offset += _HEADER_SIZE.size
        header = json.loads(data[offset : offset + size])
        offset += size
        if not isinstance(header, dict) or any(
            header.get(name) != value for name, value in expected.items()
        ):
            return None
        length = header["vector"]
        if (
            not isinstance(length, int)
            or length < 1
            or len(data) != offset + 4 * length
        ):
            return None
        return np.frombuffer(data, "<f4", length, offset).astype(np.float32)
    except ValueError, TypeError, KeyError, RecursionError, struct.error:
        return None
