"""Exceptions raised by tarsius."""

from __future__ import annotations

from typing import Literal


class TarsiusError(Exception):
    """Base class for all tarsius errors."""


class ImageError(TarsiusError):
    """The image could not be decoded, e.g. a camera returned an error page."""


class BackendError(TarsiusError):
    """A feature backend could not load its model or process an image."""


class BackendMismatchError(TarsiusError, ValueError):
    """Features come from another backend than the classifier expects."""


class BlankRegionError(TarsiusError, ValueError):
    """The region has nothing to compare, e.g. a grey frame of an offline camera."""


class RoiError(TarsiusError, ValueError):
    """A region is empty or reaches outside the frame."""

    def __init__(self, axis: Literal["horizontal", "vertical"], message: str) -> None:
        """Remember which axis is wrong, so a UI can point at it."""
        super().__init__(message)
        self.axis = axis
