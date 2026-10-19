"""Local, training-free state detection for a region of a camera image."""

from importlib.metadata import version

from .backends.edges import EdgeBackend
from .backends.ncnn import NcnnBackend
from .classifier import Classifier, Result, Status
from .errors import (
    BackendError,
    BackendMismatchError,
    FeaturesError,
    ImageError,
    TarsiusError,
)
from .features import FeatureBackend, Features, Reference, extract_features
from .image import load_image
from .roi import FULL_FRAME, Roi
from .vote import majority

__version__ = version("tarsius")

__all__ = [
    "FULL_FRAME",
    "BackendError",
    "BackendMismatchError",
    "Classifier",
    "EdgeBackend",
    "FeatureBackend",
    "Features",
    "FeaturesError",
    "ImageError",
    "NcnnBackend",
    "Reference",
    "Result",
    "Roi",
    "Status",
    "TarsiusError",
    "extract_features",
    "load_image",
    "majority",
]
