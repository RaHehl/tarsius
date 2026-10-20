"""SqueezeNet embedding backend on ncnn."""

from __future__ import annotations

import threading
from importlib.resources import as_file, files

import ncnn
import numpy as np
from PIL import Image

from ..errors import BackendError
from ..features import Vector

_INPUT = 227
_MEAN = [104.0, 117.0, 123.0]
# Last feature map before the ImageNet classifier, 512 channels.
_BLOB = "fire9/concat_drop9"


class NcnnBackend:
    """
    ImageNet SqueezeNet 1.1 features, averaged over the feature map.

    No training: the network only describes the region, the classifier
    compares those descriptions. Robust to clutter and lighting, so a rough
    region is enough.

    Loading costs about 10 MB, the first extraction another 10 MB of ncnn
    buffers. The backend keeps no state between calls, so one instance can
    serve every detector in a process, also from several threads. ncnn keeps
    buffers for every inference that ran at the same time, so inferences
    take turns; the GIL is released while they run.
    """

    # Bump with any change to the model, ncnn options or preprocessing.
    id = "squeezenet11/2"
    default_margin = 0.02
    temperature = 0.019

    def __init__(self, *, threads: int = 1) -> None:
        """Load the bundled model; blocking file I/O, raises BackendError."""
        self._lock = threading.Lock()
        self._net = ncnn.Net()
        opt = self._net.opt
        opt.num_threads = threads
        # fp16 differs between CPUs; keep features comparable when
        # references move to other hardware, e.g. restored from a backup.
        opt.use_fp16_arithmetic = False
        opt.use_fp16_storage = False
        opt.use_fp16_packed = False
        opt.use_vulkan_compute = False
        models = files("tarsius") / "models"
        param = models / "squeezenet_v1.1.param"
        model = models / "squeezenet_v1.1.bin"
        if not (param.is_file() and model.is_file()):
            raise BackendError("bundled SqueezeNet model is missing")
        with as_file(param) as param_path, as_file(model) as model_path:
            if self._net.load_param(str(param_path)) or self._net.load_model(
                str(model_path)
            ):
                raise BackendError("ncnn could not load the SqueezeNet model")

    def extract(self, image: Image.Image) -> tuple[Vector, Vector]:
        """Return the unit-norm embedding as both reference and query."""
        # PIL antialiases when shrinking, ncnn's own resize does not.
        resized = image.convert("RGB").resize(
            (_INPUT, _INPUT), Image.Resampling.BILINEAR
        )
        bgr = np.ascontiguousarray(np.asarray(resized)[..., ::-1])
        mat = ncnn.Mat.from_pixels(bgr, ncnn.Mat.PixelType.PIXEL_BGR, _INPUT, _INPUT)
        mat.substract_mean_normalize(_MEAN, [])
        with self._lock:
            extractor = self._net.create_extractor()
            extractor.input("data", mat)
            ret, out = extractor.extract(_BLOB)
            if ret:  # pragma: no cover - ncnn reports no recoverable cause
                raise BackendError(f"ncnn extraction failed with {ret}")
            maps = np.array(out, dtype=np.float32)
        embedding = maps.reshape(maps.shape[0], -1).mean(axis=1)
        norm = float(np.linalg.norm(embedding))
        vector = embedding / norm if norm > 0 else embedding
        return vector, vector[None]
