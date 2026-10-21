"""Edge map backend: numpy only, needs a tight region."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ..features import Vector, unit

SIZE = (128, 96)
# Pixels the region may move between frames, e.g. after a camera bump.
SHIFT = 2


class EdgeBackend:
    """
    Gradient magnitude compared by normalized cross-correlation.

    Baseline without extra dependencies. Anything moved inside the region
    changes the edges, so the region must be tight, and it must keep some
    static structure (door, wall edges) for the empty state: a featureless
    region has nothing to compare.
    """

    id = "edges/1"
    default_margin = 0.02
    temperature = 0.077

    def extract(self, image: Image.Image) -> tuple[Vector, Vector]:
        """Return the centre crop and every shifted window of the edge map."""
        gray = image.convert("L").resize(SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(gray.filter(ImageFilter.GaussianBlur(1)), np.float32)
        grad_x = np.zeros_like(pixels)
        grad_y = np.zeros_like(pixels)
        grad_x[:, 1:-1] = pixels[:, 2:] - pixels[:, :-2]
        grad_y[1:-1, :] = pixels[2:, :] - pixels[:-2, :]
        edges = np.hypot(grad_x, grad_y)
        height, width = edges.shape
        inner = (slice(SHIFT, height - SHIFT), slice(SHIFT, width - SHIFT))
        windows = [
            unit(
                edges[
                    SHIFT + dy : height - SHIFT + dy,
                    SHIFT + dx : width - SHIFT + dx,
                ]
            )
            for dy in range(-SHIFT, SHIFT + 1)
            for dx in range(-SHIFT, SHIFT + 1)
        ]
        return unit(edges[inner]), np.stack(windows)
