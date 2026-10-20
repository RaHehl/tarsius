"""Features of a stored reference, kept so a reload can skip extraction."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import struct

import numpy as np

from .features import Features
from .roi import Roi

MAGIC = b"tarsius-features"
FORMAT = 1
# Any release may change how features come out of an image, so a cache from
# another one is extracted again.
_TARSIUS = version("tarsius")
_HEADER_SIZE = struct.Struct("<I")


def _key(original: bytes, backend_id: str, roi: Roi, frame_side: int) -> dict:
    return {
        "format": FORMAT,
        "tarsius": _TARSIUS,
        "backend": backend_id,
        "roi": [roi.left, roi.top, roi.right, roi.bottom],
        "frame_side": frame_side,
        "original": hashlib.sha256(original).hexdigest(),
    }


def dump(features: Features, original: bytes, roi: Roi, frame_side: int) -> bytes:
    """Serialize the features of ``original`` for these settings."""
    vector = features.vector.astype("<f4")
    query = features.query.astype("<f4")
    header = _key(original, features.backend_id, roi, frame_side) | {
        "vector": len(vector),
        "query": list(query.shape),
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return b"".join(
        (
            MAGIC,
            _HEADER_SIZE.pack(len(encoded)),
            encoded,
            vector.tobytes(),
            query.tobytes(),
        )
    )


def restore(
    data: bytes, original: bytes, backend_id: str, roi: Roi, frame_side: int
) -> Features | None:
    """Return the cached features, None unless they fit exactly these settings."""
    try:
        if not data.startswith(MAGIC):
            return None
        offset = len(MAGIC)
        (size,) = _HEADER_SIZE.unpack_from(data, offset)
        offset += _HEADER_SIZE.size
        header = json.loads(data[offset : offset + size])
        offset += size
        if not isinstance(header, dict):
            return None
        expected = _key(original, backend_id, roi, frame_side)
        if any(header.get(name) != value for name, value in expected.items()):
            return None
        length = header["vector"]
        rows, columns = header["query"]
        if length < 1 or rows < 1 or columns != length:
            return None
        if len(data) != offset + 4 * length * (1 + rows):
            return None
        vector = np.frombuffer(data, "<f4", length, offset)
        query = np.frombuffer(data, "<f4", rows * columns, offset + 4 * length)
    except (ValueError, TypeError, KeyError, struct.error):
        return None
    features = Features(
        backend_id,
        vector.astype(np.float32),
        query.astype(np.float32).reshape(rows, columns),
    )
    return features if features.has_structure else None
