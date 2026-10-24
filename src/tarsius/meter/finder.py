"""
ALPHA: suggest where the counter window and the dials of a meter are.

This module is alpha. Its results, API and models may change in any
release, and it emits :class:`AlphaWarning` on first use. It only suggests
regions for a user to confirm and never reads a value itself.

Two small segmentation networks on ncnn, trained on synthetic faces and the
Dryad word-wheel water meter photographs: one marks the counter window, one
marks the pointer dials. The window's rotated rectangle gives the rotation
that makes it horizontal. The dial network is not rotation invariant, so it
looks at the frame turned level; the dials are found as discs in its mask,
and each box is then moved onto the dial's axis and sized to its ring.

Known weakness: trained mostly on one population of meters, it also marks
a printed serial number. On a European meter with a vertical serial number
on the housing ring both are strong blobs; the window is the one the
network is surer of, so that one is kept rather than the larger one. That
found the window on all 735 frames of that meter, also desaturated. Show
the suggestion to the user.

Takes about 90 ms on one thread of an i5-8400T, 40 ms of it for turning
the frame and finding the axes. Estimated from the operation count, a Home
Assistant Green needs up to 1 s for the window and 3.7 s for the dial
network, so run it at setup or once a day, not per reading.
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
# Axis votes must peak this many times over their mean to move a dial box.
# Dials found on a real meter scored at least 5.8 (median 9.7), other spots
# of its face 2.7 to 4.5.
_AXIS_SHARPNESS = 4.0
# Dial region half side as a multiple of the scale ring's radius.
_RING_MARGIN = 1.03
# Share of the strongest edges in a dial's surroundings that vote for its axis.
_AXIS_EDGES = 0.18
# Dials larger than this radius in pixels are voted on scaled down.
_AXIS_RADIUS = 64

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
    probs: Floats = (1 / (1 + np.exp(-logits))).astype(np.float32)
    blob = _strongest(logits > 0, probs)
    if not blob.any():
        return 0.0, None, 0.0
    score = float(np.mean(probs[blob]))
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
    # The dial network is not rotation invariant, it looks at the level frame.
    level = frame.rotate(rotation, Image.Resampling.BILINEAR) if rotation else frame
    grey = level.convert("L")
    logits = _NETS[_DIAL_MODEL].logits(grey.convert("RGB"))
    levels: Floats = np.asarray(grey, np.float32)
    width, height = level.size
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
        x, y, half = _on_axis(levels, x, y, r * _DIAL_SIDE / 2)
        roi = _clamped(x - half, y - half, x + half, y + half, level.size)
        if roi is not None:
            found.append((x, y, roi, float(square[inside].mean())))
    if not found:
        return []
    mx = float(np.mean([f[0] for f in found]))
    my = float(np.mean([f[1] for f in found]))
    # Clockwise on screen from twelve o'clock, y grows downwards.
    found.sort(key=lambda f: math.atan2(f[0] - mx, my - f[1]) % (2 * math.pi))
    return [(roi, score) for _, _, roi, score in found]


def _on_axis(
    levels: Floats, x: float, y: float, half: float
) -> tuple[float, float, float]:
    """
    Return centre and half side of a dial box, moved onto the dial's axis.

    The mask runs into the lettering beside a dial and has dents, so the
    centre of its largest disc lies beside the axis, about a tenth of the
    diameter on a real meter. Where the axis vote is not clear the box stays.
    """
    ax, ay, ring, sharpness = _recentre(levels, x, y, half)
    if sharpness < _AXIS_SHARPNESS:
        return x, y, half
    return ax, ay, ring * _RING_MARGIN


def _recentre(
    levels: Floats, x: float, y: float, radius: float
) -> tuple[float, float, float, float]:
    """
    Return axis, ring radius and vote sharpness of a dial roughly at ``x, y``.

    A dial face is radially symmetric: ring, ticks and numerals lie on
    circles about the axis, so the gradient of every edge on them points
    towards the axis or away from it. Each strong edge votes along its
    gradient both ways, over the plausible radii, and the votes pile up on
    the axis. Neither colour nor brightness is used: a red hub is the
    nearest feature, and a ring may be darker or lighter than its ground.
    The sharpness is the smoothed vote peak over its mean.
    """
    height, width = levels.shape
    half = max(8.0, radius * 1.6)
    left, top = max(0, round(x - half)), max(0, round(y - half))
    right, bottom = min(width, round(x + half)), min(height, round(y + half))
    if right - left < 12 or bottom - top < 12:
        return x, y, radius, 0.0
    patch: Floats = levels[top:bottom, left:right]
    # Large dials are voted on at a bounded size, the time grows with it.
    scale = min(1.0, _AXIS_RADIUS / radius)
    if scale < 1.0:
        size = (round(patch.shape[1] * scale), round(patch.shape[0] * scale))
        patch = np.asarray(
            Image.fromarray(patch, mode="F").resize(size, Image.Resampling.BILINEAR),
            np.float32,
        )
    r = radius * scale
    gx, gy = _sobel(patch)
    strength = np.hypot(gx, gy)
    limit = max(float(np.quantile(strength, 1 - _AXIS_EDGES)), 1e-6)
    ys, xs = np.nonzero(strength > limit)
    if len(xs) < 24:
        return x, y, radius, 0.0
    weight = strength[ys, xs]
    ux, uy = gx[ys, xs] / weight, gy[ys, xs] / weight
    radii = np.arange(0.6 * r, 1.15 * r + 1e-6, max(1.0, r * 0.06))
    steps = np.concatenate([radii, -radii])[:, None]
    vx = np.rint(xs + steps * ux).astype(np.intp)
    vy = np.rint(ys + steps * uy).astype(np.intp)
    rows, cols = patch.shape
    ok = (vx >= 0) & (vx < cols) & (vy >= 0) & (vy < rows)
    votes = np.bincount(
        vy[ok] * cols + vx[ok],
        weights=np.broadcast_to(weight, vx.shape)[ok],
        minlength=rows * cols,
    ).reshape(rows, cols)
    # The votes scatter about the axis; a box filter makes them one peak.
    pile = _box_sum(votes, max(1, round(r * 0.12)))
    iy, ix = np.unravel_index(int(np.argmax(pile)), pile.shape)
    sharpness = float(pile.max()) / (float(pile.mean()) + 1e-9)
    # The ring is the circle about the axis with the strongest edges.
    edges = np.arange(0.35 * r, 1.3 * r, max(1.0, r * 0.05))
    yy, xx = np.mgrid[0:rows, 0:cols]
    band = np.digitize(np.hypot(xx - ix, yy - iy), edges) - 1
    inner = (band >= 0) & (band < len(edges) - 1)
    sums = np.bincount(band[inner], strength[inner], minlength=len(edges) - 1)
    counts = np.bincount(band[inner], minlength=len(edges) - 1)
    profile = sums / np.maximum(counts, 1)
    ring = float(edges[int(np.argmax(profile))]) + r * 0.025
    return (
        left + (float(ix) + 0.5) / scale,
        top + (float(iy) + 0.5) / scale,
        ring / scale,
        sharpness,
    )


def _sobel(levels: Floats) -> tuple[Floats, Floats]:
    """Return the 3x3 Sobel gradients, zero on the border."""
    gx = np.zeros_like(levels)
    gy = np.zeros_like(levels)
    across = levels[:-2] + 2 * levels[1:-1] + levels[2:]
    gx[1:-1, 1:-1] = across[:, 2:] - across[:, :-2]
    down = levels[:, :-2] + 2 * levels[:, 1:-1] + levels[:, 2:]
    gy[1:-1, 1:-1] = down[2:] - down[:-2]
    return gx, gy


def _box_sum(values: Floats, k: int) -> Floats:
    """Return the sum over the (2k+1)-square around each cell, clipped at edges."""
    rows, cols = values.shape
    cum = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    lo_y = np.clip(np.arange(rows) - k, 0, rows)
    hi_y = np.clip(np.arange(rows) + k + 1, 0, rows)
    lo_x = np.clip(np.arange(cols) - k, 0, cols)
    hi_x = np.clip(np.arange(cols) + k + 1, 0, cols)
    total: Floats = (
        cum[np.ix_(hi_y, hi_x)]
        - cum[np.ix_(lo_y, hi_x)]
        - cum[np.ix_(hi_y, lo_x)]
        + cum[np.ix_(lo_y, lo_x)]
    )
    return total


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


# Components smaller than this many mask cells are noise, not a window.
MIN_CELLS = 8


def _strongest(mask: Mask, probs: Floats) -> Mask:
    """
    Return the 4-connected component the network is surest of, by mean.

    On a European meter the vertical serial number is a second strong
    blob, and which of the two is larger depends on light and colour: on
    desaturated frames the window was the larger one only a third of the
    time. The window had the higher mean in all frames. Area breaks ties;
    components under MIN_CELLS cells are left out.
    """
    if not mask.any():
        return mask
    labels = _labels(mask)
    counts = np.bincount(labels[mask])
    sums = np.bincount(labels[mask], weights=probs[mask])
    best, key = -1, (0.0, 0)
    for label in np.flatnonzero(counts >= MIN_CELLS):
        candidate = (float(sums[label] / counts[label]), int(counts[label]))
        if candidate > key:
            best, key = int(label), candidate
    strongest: Mask = labels == best
    return strongest


def _labels(mask: Mask) -> npt.NDArray[np.int64]:
    """Label the 4-connected components of a mask; background gets mask.size."""
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
            return spread
        labels = spread


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
