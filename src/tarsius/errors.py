"""Exceptions raised by tarsius."""

from __future__ import annotations


class TarsiusError(Exception):
    """Base class for all tarsius errors."""


class ImageError(TarsiusError):
    """The image could not be decoded, e.g. a camera returned an error page."""


class BackendError(TarsiusError):
    """A feature backend could not load its model or process an image."""


class BackendMismatchError(TarsiusError, ValueError):
    """Features come from another backend than the classifier expects."""


class FeaturesError(TarsiusError):
    """Stored features are corrupt or written by an incompatible version."""
