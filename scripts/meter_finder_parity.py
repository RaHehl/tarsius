"""
Compare a converted finder on ncnn with the Keras original.

Needs TensorFlow, which has no wheels for the Python tarsius targets, so it
runs apart from the test suite:

    uv run --python 3.12 --with tensorflow-cpu==2.21.0 --with keras==3.15.1 \
        --with ncnn==1.0.20261020 --with pillow python \
        scripts/meter_finder_parity.py finder.keras \
        src/tarsius/meter/models/finder [photo.jpg ...]

Feeds synthetic scenes and any photographs given, and reports the largest
logit difference and how many mask pixels (probability above 0.5) differ.
ncnn runs in a child process (meter_finder_ncnn.py): loaded next to
TensorFlow it corrupts the heap.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageDraw


def scenes(count: int, size: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(count):
        image = Image.new(
            "RGB", (size, size), tuple(int(v) for v in rng.integers(0, 255, 3))
        )
        draw = ImageDraw.Draw(image)
        for _ in range(12):
            x, y = rng.integers(0, size, 2)
            r = int(rng.integers(4, size // 4))
            fill = tuple(int(v) for v in rng.integers(0, 255, 3))
            if i % 2:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(0, 0, 0))
            else:
                draw.rectangle((x, y, x + 3 * r, y + r), fill=fill)
                draw.text(
                    (x + 2, y + 2), str(int(rng.integers(0, 99999))), fill=(0, 0, 0)
                )
        pixels = np.asarray(image, np.float32) + rng.normal(0, 8, (size, size, 3))
        out.append(np.clip(pixels, 0, 255))
    return out


def main() -> None:
    keras_path, ncnn_stem = Path(sys.argv[1]), Path(sys.argv[2])
    model = keras.saving.load_model(keras_path, compile=False)
    size = int(model.input_shape[1])
    images = scenes(24, size)
    for photo in sys.argv[3:]:
        im = (
            Image.open(photo)
            .convert("RGB")
            .resize((size, size), Image.Resampling.BILINEAR)
        )
        images.append(np.asarray(im, np.float32))
    # ncnn gets 8-bit pixels, so Keras must too.
    batch = np.stack([np.round(i).astype(np.uint8) for i in images])
    reference = model.predict(batch.astype(np.float32), verbose=0)[..., 0]
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "reference.npz"
        np.savez(data, images=batch, reference=reference)
        subprocess.run(
            [
                sys.executable,
                Path(__file__).with_name("meter_finder_ncnn.py"),
                ncnn_stem,
                data,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
