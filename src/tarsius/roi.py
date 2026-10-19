"""Region of interest inside a camera image."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import astuple, dataclass

from PIL import Image

from .errors import RoiError


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
            raise RoiError(
                "horizontal", f"invalid horizontal range {self.left}..{self.right}"
            )
        if not (0.0 <= self.top < self.bottom <= 1.0):
            raise RoiError(
                "vertical", f"invalid vertical range {self.top}..{self.bottom}"
            )

    @classmethod
    def from_percent(cls, left: float, top: float, right: float, bottom: float) -> Roi:
        """Create a region from percentages of the frame."""
        return cls(left / 100, top / 100, right / 100, bottom / 100)

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

    def as_percent(self) -> tuple[float, float, float, float]:
        """Return ``(left, top, right, bottom)`` in percent."""
        return (self.left * 100, self.top * 100, self.right * 100, self.bottom * 100)

    def box(self, size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return the pixel box for an image size, at least one pixel each way."""
        width, height = size
        left = min(round(self.left * width), width - 1)
        top = min(round(self.top * height), height - 1)
        right = max(round(self.right * width), left + 1)
        bottom = max(round(self.bottom * height), top + 1)
        return left, top, right, bottom

    def crop(self, image: Image.Image) -> Image.Image:
        """Return the region of ``image``."""
        return image.crop(self.box(image.size))


FULL_FRAME = Roi()
