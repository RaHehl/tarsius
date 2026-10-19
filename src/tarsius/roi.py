"""Region of interest inside a camera image."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import astuple, dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class Roi:
    """Rectangle as fractions of the image size, so it survives resolution changes."""

    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0

    def __post_init__(self) -> None:
        """Reject empty or out-of-frame rectangles."""
        if not (0.0 <= self.left < self.right <= 1.0):
            raise ValueError(f"invalid horizontal range {self.left}..{self.right}")
        if not (0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError(f"invalid vertical range {self.top}..{self.bottom}")

    @classmethod
    def from_cells(
        cls, cells: Iterable[tuple[int, int]], *, columns: int, rows: int
    ) -> Roi:
        """Return the bounding box of grid cells given as ``(column, row)``."""
        picked = list(cells)
        if not picked:
            raise ValueError("no cells selected")
        for column, row in picked:
            if not (0 <= column < columns and 0 <= row < rows):
                raise ValueError(f"cell {column},{row} outside a {columns}x{rows} grid")
        return cls(
            min(column for column, _ in picked) / columns,
            min(row for _, row in picked) / rows,
            (max(column for column, _ in picked) + 1) / columns,
            (max(row for _, row in picked) + 1) / rows,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(left, top, right, bottom)``, e.g. for config storage."""
        return astuple(self)

    def crop(self, image: Image.Image) -> Image.Image:
        """Return the region of ``image``, at least one pixel in each direction."""
        width, height = image.size
        left = min(round(self.left * width), width - 1)
        top = min(round(self.top * height), height - 1)
        right = max(round(self.right * width), left + 1)
        bottom = max(round(self.bottom * height), top + 1)
        return image.crop((left, top, right, bottom))


FULL_FRAME = Roi()
