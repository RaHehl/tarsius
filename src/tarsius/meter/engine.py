"""
Numpy inference for the bundled wheel and dial readers.

A model file is an ``.npz`` archive with the architecture as JSON and float32
weights, exported from Keras with batch normalization folded into the
convolutions. The op set is what the readers use: rescale, 3x3 convolution
with same padding, 2x2 max pooling, flatten, dense, plus add, concat and
nearest upsampling for skip connections.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from ..errors import BackendError

type Array = npt.NDArray[np.float32]

# Classes of the value head: class k is the position k / 10.
VALUE_CLASSES = 100
ORIENT_CLASSES = 2


class Fault(StrEnum):
    """Why a crop cannot be read, from the reader's fault head."""

    OK = "ok"
    FOCUS = "focus"
    GLARE = "glare"
    DARK = "dark"
    # The region drifted, or holds something that is not a wheel at all.
    OFFSET = "offset"


_FAULTS = tuple(Fault)

# Grey spread below which a crop holds no digit or pointer: a covered wheel
# or an empty window. Wheels on the maintainer's meter keep 20 or more.
BLANK_STDDEV = 8.0
# Grey levels that count as clipped white or black.
SATURATED = 250.0
DARK = 8.0
# Share of clipped white above which a crop is glare. White dial faces clip
# more in normal light than wheels do, up to 20 % on the maintainer's meter.
WHEEL_GLARE_SHARE = 0.4
DIAL_GLARE_SHARE = 0.85
DARK_SHARE = 0.75


@dataclass(frozen=True, slots=True)
class Reading:
    """One wheel or dial, read."""

    # Continuous position 0.0 to below 10.0; a wheel between 3 and 4 reads 3.4.
    value: float
    # 0 to 1, margin of the best digit against the best competing one.
    confidence: float
    # From the raw crop's exposure, else the fault head; OK without one.
    fault: Fault = Fault.OK
    # Probability that the crop is turned 180 degrees, 0 without that head.
    upside_down: float = 0.0


def _conv2d_same(x: Array, w: Array, b: Array) -> Array:
    """Stride-1 same-padding convolution, ``(H, W, Cin) -> (H, W, Cout)``."""
    kh, kw, cin, cout = w.shape
    padded = np.pad(x, ((kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
    windows = sliding_window_view(padded, (kh, kw, cin))[:, :, 0]
    h, wd = windows.shape[:2]
    out = windows.reshape(h * wd, kh * kw * cin) @ w.reshape(kh * kw * cin, cout)
    return (out.reshape(h, wd, cout) + b).astype(np.float32)


def _maxpool2(x: Array) -> Array:
    h, w, c = x.shape
    x = x[: h - h % 2, : w - w % 2]
    pooled: Array = x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))
    return pooled


def softmax(logits: Array) -> Array:
    """Return the softmax of a logit vector."""
    e = np.exp(logits - logits.max())
    out: Array = (e / e.sum()).astype(np.float32)
    return out


def decode(probs: npt.NDArray[np.floating]) -> tuple[float, float]:
    """
    Return value and confidence from a distribution over the value classes.

    The value is refined below the class grid by a circular centre of mass
    over the peak and its neighbours, since a wheel position is continuous.
    Confidence compares the peak with the best mass more than two classes
    away, so an honest reading between two neighbouring classes stays sure.
    """
    k = probs.shape[-1]
    cls = int(np.argmax(probs))
    lo, hi = (cls - 1) % k, (cls + 1) % k
    w = np.array([probs[lo], probs[cls], probs[hi]], dtype=np.float64)
    offset = float((w[2] - w[0]) / max(w.sum(), 1e-9))
    value = ((cls + offset) / (k / 10.0)) % 10.0
    band = np.zeros(k, dtype=bool)
    for d in range(-2, 3):
        band[(cls + d) % k] = True
    own = float(w.sum())
    rival = float(probs[~band].max())
    return value, min(1.0, own / (own + rival * 3.0 + 1e-9))


def exposure_fault(image: Array, *, glare_share: float = WHEEL_GLARE_SHARE) -> Fault:
    """
    Return GLARE or DARK when a raw ``(H, W, 3)`` crop cannot show a digit.

    Checked before the network normalizes exposure: stretched to the same
    contrast, a blank or clipped crop looks like any other and can read as a
    confident digit.
    """
    grey = image.mean(axis=2)
    if float(grey.std()) < BLANK_STDDEV:
        return Fault.GLARE if float(grey.mean()) >= 128.0 else Fault.DARK
    if float((grey >= SATURATED).mean()) > glare_share:
        return Fault.GLARE
    if float((grey <= DARK).mean()) > DARK_SHARE:
        return Fault.DARK
    return Fault.OK


def _views(image: Array, count: int) -> Iterator[Array]:
    """Yield the crop and up to four one-pixel shifts of it."""
    yield image
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))[: count - 1]:
        shifted = np.roll(image, (dy, dx), axis=(0, 1))
        # Repeat the edge instead of wrapping the opposite one in.
        if dy:
            edge = 0 if dy > 0 else -1
            shifted[edge, :] = image[edge, :]
        if dx:
            edge = 0 if dx > 0 else -1
            shifted[:, edge] = image[:, edge]
        yield shifted


@dataclass(frozen=True, slots=True, eq=False)
class Model:
    """A loaded reader; stateless, so safe to share between threads."""

    layers: tuple[dict[str, Any], ...]
    arrays: dict[str, Array]
    # Input height and width.
    input_hw: tuple[int, int]

    @classmethod
    def load(cls, source: str | Path | IO[bytes]) -> Model:
        """Load an exported ``.npz`` model; raises BackendError if unusable."""
        try:
            with np.load(source) as data:
                arch = json.loads(bytes(data["__arch__"]).decode())
                arrays = {
                    k: data[k].astype(np.float32) for k in data.files if k != "__arch__"
                }
            hw = arrays.pop("__input_hw__")
            model = cls(tuple(arch), arrays, (int(hw[0]), int(hw[1])))
            model.logits(np.zeros((*model.input_hw, 3), np.float32))
        except (OSError, KeyError, ValueError, TypeError) as err:
            raise BackendError(f"cannot load reader model: {err}") from err
        return model

    def logits(self, image: Array) -> Array:
        """Run the network on one ``(H, W, 3)`` crop in raw 0-255 values."""
        x = image.astype(np.float32)
        # Exposure and white balance belong to the moment, not to the digit.
        grey = x.mean(axis=2)
        x = np.clip((x - grey.mean()) / (grey.std() + 1e-6) * 45.0 + 128.0, 0, 255)
        saved: dict[str, Array] = {}
        for spec in self.layers:
            if source := spec.get("from"):
                x = saved[source]
            x = self._apply(spec, x, saved)
            if spec["op"] == "relu" or spec.get("relu"):
                x = np.maximum(x, 0.0)
            if name := spec.get("save"):
                saved[name] = x
        return x

    def _apply(self, spec: dict[str, Any], x: Array, saved: dict[str, Array]) -> Array:
        match spec["op"]:
            case "rescale":
                y = x * np.float32(spec["scale"])
            case "conv":
                y = _conv2d_same(x, self.arrays[spec["w"]], self.arrays[spec["b"]])
            case "maxpool":
                y = _maxpool2(x)
            case "flatten":
                y = x.reshape(-1)
            case "dense":
                y = x @ self.arrays[spec["w"]] + self.arrays[spec["b"]]
            case "add":
                y = x + saved[spec["other"]]
            case "concat":
                y = np.concatenate([x, saved[spec["other"]]], axis=-1)
            case "upsample":
                k = int(spec.get("factor", 2))
                y = np.repeat(np.repeat(x, k, axis=0), k, axis=1)
            case "relu":
                y = x
            case op:
                raise ValueError(f"unknown op {op!r}")
        return y

    def inspect(
        self, image: Array, *, views: int = 1, glare_share: float = WHEEL_GLARE_SHARE
    ) -> Reading:
        """
        Read one crop, resized to ``input_hw`` by the caller.

        ``views`` above one averages the distributions of the crop and up to
        four one-pixel shifts of it, which removes some gross errors. A crop
        that :func:`exposure_fault` finds blank, dark or clipped beyond
        ``glare_share`` gets that fault whatever the fault head says.
        """
        value_sum: Array | None = None
        fault_sum: Array | None = None
        orient_sum: Array | None = None
        count = max(1, min(views, 5))
        for view in _views(image, count):
            out = self.logits(view)
            value = softmax(out[:VALUE_CLASSES])
            value_sum = value if value_sum is None else value_sum + value
            if out.shape[0] >= VALUE_CLASSES + len(_FAULTS):
                faults = softmax(out[VALUE_CLASSES : VALUE_CLASSES + len(_FAULTS)])
                fault_sum = faults if fault_sum is None else fault_sum + faults
            if out.shape[0] == VALUE_CLASSES + len(_FAULTS) + ORIENT_CLASSES:
                orient = softmax(out[VALUE_CLASSES + len(_FAULTS) :])
                orient_sum = orient if orient_sum is None else orient_sum + orient
        assert value_sum is not None
        value_pos, confidence = decode(value_sum / count)
        fault = exposure_fault(image, glare_share=glare_share)
        if fault is Fault.OK and fault_sum is not None:
            fault = _FAULTS[int(np.argmax(fault_sum))]
        flip = float(orient_sum[1] / count) if orient_sum is not None else 0.0
        return Reading(value_pos, confidence, fault, flip)


def circular_distance(a: float, b: float) -> float:
    """Return the distance of two positions on a ten-digit wheel."""
    d = math.fabs(a - b) % 10.0
    return min(d, 10.0 - d)
