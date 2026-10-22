"""
ALPHA: suggest where the counter window and the dials of a meter are.

This module is alpha. Its results, API and models may change in any
release, and it emits :class:`AlphaWarning` on first use. It only suggests
regions for a user to confirm and never reads a value itself.

Two small segmentation networks on ncnn, trained on synthetic faces and the
Dryad word-wheel water meter photographs: one marks the counter window, one
marks the pointer dials. The window's rotated rectangle gives the rotation
that makes it horizontal, the dials are found as discs in the dial mask.

Known weakness: trained mostly on one population of meters, it can mark a
printed serial number instead of the counter. On a European meter with a
vertical serial number on the housing ring, earlier models marked the serial
number on both photographs tried; the bundled one found the window on 33 of
35 frames of that meter and the serial number on the other two. Show the
suggestion to the user.

Takes about 50 ms on one thread of an i5-8400T. Estimated from the
operation count, a Home Assistant Green needs up to 1 s for the window and
3.7 s for the dials, so run it at setup or once a day, not per reading.
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass
from importlib.resources import as_file, files

import ncnn
import numpy as np
import numpy.typing as npt
from PIL import Image

from ..errors import BackendError
from ..roi import Roi
from .frame import decode_frame
from .layout import Dial, MeterLayout

__alpha__ = True

_INPUT = 256
_WINDOW_MODEL = "finder"
_DIAL_MODEL = "dialfinder"
# Side of the isotropic grid the dial mask is analysed on.
_DIAL_GRID = 128
# Dial region side as a multiple of the radius to the edge of the mask.
_DIAL_SIDE = 2.0

type Mask = npt.NDArray[np.bool_]
type Floats = npt.NDArray[np.float32]


class AlphaWarning(FutureWarning):
    """The finder is alpha: results, API and models may change."""


@dataclass(frozen=True, slots=True)
class Suggestion:
    """
    Suggested regions, all in the frame turned by ``rotation``.

    The window may also be turned 180 degrees; ``MeterReader.calibrate``
    finds that out.
    """

    # Degrees counter-clockwise that make the window horizontal.
    rotation: float = 0.0
    # The counter window, or None when nothing was found.
    digits: Roi | None = None
    # Mean window probability over the marked area, 0 to 1.
    digits_score: float = 0.0
    # Dials ordered clockwise from twelve o'clock around their centre; the
    # usual order of x0.1, x0.01, ... but a guess, let the user check it.
    dials: tuple[Roi, ...] = ()
    dial_scores: tuple[float, ...] = ()

    def layout(self, wheels: int, *, decimal_wheels: int = 0) -> MeterLayout:
        """Return a layout from the suggestion; raises ValueError without window."""
        if self.digits is None:
            raise ValueError("no counter window was found")
        return MeterLayout(
            digits=self.digits,
            wheels=wheels,
            dials=tuple(Dial(roi) for roi in self.dials),
            decimal_wheels=decimal_wheels,
            rotation=self.rotation,
        )


class _Net:
    """One ncnn network, loaded on first use; inferences take turns."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()
        self._net: ncnn.Net | None = None

    def logits(self, frame: Image.Image) -> Floats:
        """Return the output logits for a frame squeezed to the input size."""
        pixels = np.ascontiguousarray(
            np.asarray(frame.resize((_INPUT, _INPUT), Image.Resampling.BILINEAR))
        )
        mat = ncnn.Mat.from_pixels(pixels, ncnn.Mat.PixelType.PIXEL_RGB, _INPUT, _INPUT)
        with self._lock:
            net = self._net or self._load()
            extractor = net.create_extractor()
            extractor.input("input_layer", mat)
            ret, out = extractor.extract("logit")
            if ret:  # pragma: no cover - ncnn reports no recoverable cause
                raise BackendError(f"ncnn extraction failed with {ret}")
            result: Floats = np.array(out, dtype=np.float32)[0]
        return result

    def _load(self) -> ncnn.Net:
        net = ncnn.Net()
        opt = net.opt
        opt.num_threads = 1
        # fp16 arithmetic differs between CPUs, keep suggestions reproducible.
        opt.use_fp16_arithmetic = False
        opt.use_fp16_storage = False
        opt.use_fp16_packed = False
        opt.use_vulkan_compute = False
        models = files("tarsius.meter") / "models"
        param = models / f"{self._name}.param"
        model = models / f"{self._name}.bin"
        if not (param.is_file() and model.is_file()):
            raise BackendError(f"bundled finder model {self._name} is missing")
        with as_file(param) as param_path, as_file(model) as model_path:
            if net.load_param(str(param_path)) or net.load_model(str(model_path)):
                raise BackendError(f"ncnn could not load the {self._name} model")
        self._net = net
        return net


_NETS = {name: _Net(name) for name in (_WINDOW_MODEL, _DIAL_MODEL)}
_warned = threading.Event()


def suggest(image: bytes | Image.Image, *, dials: bool = True) -> Suggestion:
    """
    ALPHA: suggest the counter window and dials of the meter in a frame.

    Raises ImageError for undecodable data, BlankFrameError for a frame that
    shows nothing and BackendError if a model cannot load. Blocks for
    seconds on small hardware; call it in an executor.
    """
    if not _warned.is_set():
        _warned.set()
        warnings.warn(
            "tarsius.meter.finder is alpha: suggestions, API and models may change",
            AlphaWarning,
            stacklevel=2,
        )
    frame = decode_frame(image)
    rotation, digits, digits_score = _window(frame)
    found: list[tuple[Roi, float]] = []
    if dials:
        found = _dials(frame, rotation)
    return Suggestion(
        rotation,
        digits,
        digits_score,
        tuple(roi for roi, _ in found),
        tuple(score for _, score in found),
    )


def _window(frame: Image.Image) -> tuple[float, Roi | None, float]:
    """Return rotation, window region and score from the window mask."""
    logits = _NETS[_WINDOW_MODEL].logits(frame)
    blob = _largest(logits > 0)
    if blob.sum() < 8:
        return 0.0, None, 0.0
    score = float(np.mean(1 / (1 + np.exp(-logits[blob]))))
    width, height = frame.size
    rows, cols = logits.shape
    ys, xs = np.nonzero(blob)
    # Pixel centres in frame coordinates, so a squeezed input does not skew.
    px = (xs + 0.5) * width / cols
    py = (ys + 0.5) * height / rows
    cx, cy = float(px.mean()), float(py.mean())
    cov = np.cov(np.stack([px - cx, py - cy]))
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    # A strip has a direction only modulo 180 degrees.
    angle = (math.atan2(major[1], major[0]) + math.pi / 2) % math.pi - math.pi / 2
    u = (px - cx) * math.cos(angle) + (py - cy) * math.sin(angle)
    v = -(px - cx) * math.sin(angle) + (py - cy) * math.cos(angle)
    # Half a mask cell beyond the outermost centres.
    half_u = float(np.abs(u).max()) + width / cols / 2
    half_v = float(np.abs(v).max()) + height / rows / 2
    rotation = math.degrees(angle)
    rx, ry = _to_rotated(cx, cy, rotation, frame.size)
    roi = _clamped(rx - half_u, ry - half_v, rx + half_u, ry + half_v, frame.size)
    return rotation, roi, score


def _dials(frame: Image.Image, rotation: float) -> list[tuple[Roi, float]]:
    """Return dial regions in the rotated frame, clockwise from twelve."""
    logits = _NETS[_DIAL_MODEL].logits(frame.convert("L").convert("RGB"))
    width, height = frame.size
    scale = _DIAL_GRID / max(width, height)
    grid = (max(1, round(width * scale)), max(1, round(height * scale)))
    # Resample to square cells, so discs are round for the distance transform.
    probs = 1 / (1 + np.exp(-logits))
    square = np.asarray(
        Image.fromarray(probs.astype(np.float32), mode="F").resize(
            grid, Image.Resampling.BILINEAR
        ),
        np.float32,
    )
    discs = _discs(square > 0.5)
    found = []
    for gx, gy, radius in discs:
        x, y, r = (gx + 0.5) / scale, (gy + 0.5) / scale, (radius + 0.5) / scale
        yy, xx = np.ogrid[: square.shape[0], : square.shape[1]]
        inside = (xx - gx) ** 2 + (yy - gy) ** 2 <= radius**2
        rx, ry = _to_rotated(x, y, rotation, frame.size)
        half = r * _DIAL_SIDE / 2
        roi = _clamped(rx - half, ry - half, rx + half, ry + half, frame.size)
        if roi is not None:
            found.append((rx, ry, roi, float(square[inside].mean())))
    if not found:
        return []
    mx = float(np.mean([f[0] for f in found]))
    my = float(np.mean([f[1] for f in found]))
    # Clockwise on screen from twelve o'clock, y grows downwards.
    found.sort(key=lambda f: math.atan2(f[0] - mx, my - f[1]) % (2 * math.pi))
    return [(roi, score) for _, _, roi, score in found]


def _discs(mask: Mask) -> list[tuple[int, int, float]]:
    """
    Return ``(x, y, radius)`` of the separate discs in a mask, in grid cells.

    Peaks of the distance transform are disc centres, also where two dials
    touch. Text and thin strokes give small peaks and are dropped against
    the largest disc. Blobs that spill far beyond their inscribed disc are
    not dials either: an isolated disc leaves a ring around it nearly empty,
    a touching neighbour fills only a sector of it. Nor are bars, whose
    distance transform has a ridge where a disc has a single peak.
    """
    depth = _distance(mask)
    peaks: list[tuple[int, int, float]] = []
    work = depth.copy()
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    while (best := float(work.max())) >= 3:
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        near = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        work[near <= 1.5 * best] = 0
        if mask[(near > 1.3 * best) & (near <= 1.7 * best)].mean() >= 0.5:
            continue
        # A disc peaks in a point, a bar or a line of text in a ridge.
        if ((depth >= best - 1) & (near <= best)).sum() > 5 * best:
            continue
        # The edge is where the mask stops filling a thin ring.
        radius = best
        while radius < 2 * best and mask[np.abs(near - radius) <= 0.5].mean() >= 0.5:
            radius += 0.5
        peaks.append((int(x), int(y), radius))
    if not peaks:
        return []
    largest = max(r for _, _, r in peaks)
    height, width = mask.shape
    # A dial cut by the frame edge cannot be read, keep whole ones only.
    return [
        (x, y, r)
        for x, y, r in peaks
        if r >= 0.6 * largest
        and x - r >= -1
        and x + r <= width + 1
        and y - r >= -1
        and y + r <= height + 1
    ]


def _distance(mask: Mask) -> Floats:
    """Return an approximate distance to the nearest background cell."""
    depth = np.zeros(mask.shape, np.float32)
    current = mask.copy()
    step = 0
    while current.any():
        step += 1
        depth[current] = step
        padded = np.pad(current, 1)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        if step % 2 == 0:
            # Alternating cross and square erosion approximates a circle.
            eroded &= padded[:-2, :-2] & padded[:-2, 2:] & padded[2:, :-2]
            eroded &= padded[2:, 2:]
        current = eroded
    return depth


def _largest(mask: Mask) -> Mask:
    """Return the largest 4-connected component of a mask."""
    if not mask.any():
        return mask
    size = mask.size
    labels = np.where(mask, np.arange(size).reshape(mask.shape), size)
    while True:
        padded = np.pad(labels, 1, constant_values=size)
        spread = np.minimum.reduce(
            [
                labels,
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ]
        )
        spread = np.where(mask, spread, size)
        if np.array_equal(spread, labels):
            break
        labels = spread
    counts = np.bincount(labels[mask])
    largest: Mask = labels == int(np.argmax(counts))
    return largest


def _to_rotated(
    x: float, y: float, rotation: float, size: tuple[int, int]
) -> tuple[float, float]:
    """Map a frame point into the frame turned by ``rotation`` degrees."""
    angle = math.radians(rotation)
    cx, cy = size[0] / 2, size[1] / 2
    dx, dy = x - cx, y - cy
    # Image.rotate turns counter-clockwise on screen, where y grows down.
    return (
        cx + dx * math.cos(angle) + dy * math.sin(angle),
        cy - dx * math.sin(angle) + dy * math.cos(angle),
    )


def _clamped(
    left: float, top: float, right: float, bottom: float, size: tuple[int, int]
) -> Roi | None:
    """Return the box as a region clamped to the frame, None if nothing is left."""
    width, height = size
    box = (
        max(0.0, left / width),
        max(0.0, top / height),
        min(1.0, right / width),
        min(1.0, bottom / height),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return Roi(*box)
