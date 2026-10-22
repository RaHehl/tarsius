"""
Run a converted finder on ncnn and compare with saved Keras logits.

The ncnn half of meter_finder_parity.py, which starts it in a separate
process: ncnn and TensorFlow corrupt each other's heap in one process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ncnn
import numpy as np


def run_ncnn(stem: Path, data: Path) -> None:
    net = ncnn.Net()
    net.opt.use_fp16_arithmetic = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    if net.load_param(str(stem.with_suffix(".param"))) or net.load_model(
        str(stem.with_suffix(".bin"))
    ):
        raise SystemExit("ncnn could not load the model")
    with np.load(data) as arrays:
        images, reference = arrays["images"], arrays["reference"]
    worst = 0.0
    flipped = 0
    for pixels, ref in zip(images, reference, strict=True):
        h, w = pixels.shape[:2]
        mat = ncnn.Mat.from_pixels(
            np.ascontiguousarray(pixels), ncnn.Mat.PixelType.PIXEL_RGB, w, h
        )
        extractor = net.create_extractor()
        extractor.input("input_layer", mat)
        _, out = extractor.extract("logit")
        got = np.array(out)[0]
        worst = max(worst, float(np.abs(got - ref).max()))
        flipped += int(((got > 0) != (ref > 0)).sum())
    print(
        f"{len(images)} images, max abs logit diff {worst:.4g}, mask pixels "
        f"differing {flipped} of {reference.size} ({flipped / reference.size:.2e})"
    )


if __name__ == "__main__":
    run_ncnn(Path(sys.argv[1]), Path(sys.argv[2]))
