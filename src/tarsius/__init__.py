"""Local, training-free state detection for a region of a camera image."""

from importlib.metadata import version

from .backends.edges import EdgeBackend
from .backends.ncnn import NcnnBackend
from .classifier import Classifier, Result, Status
from .detector import LoadReport, Prepared, SkipReason, StateDetector
from .errors import (
    BackendError,
    BackendMismatchError,
    BlankRegionError,
    ImageError,
    RoiError,
    TarsiusError,
)
from .features import FeatureBackend, Features, extract_features
from .image import FRAME_SIDE, load_image
from .preview import outline
from .roi import FULL_FRAME, Roi
from .vote import majority

__version__ = version("tarsius")

__all__ = [
    "FRAME_SIDE",
    "FULL_FRAME",
    "BackendError",
    "BackendMismatchError",
    "BlankRegionError",
    "Classifier",
    "EdgeBackend",
    "FeatureBackend",
    "Features",
    "ImageError",
    "LoadReport",
    "NcnnBackend",
    "Prepared",
    "Result",
    "Roi",
    "RoiError",
    "SkipReason",
    "StateDetector",
    "Status",
    "TarsiusError",
    "extract_features",
    "load_image",
    "majority",
    "outline",
]
