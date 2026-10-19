"""Synthetic garage scenes: a wall with texture, optionally a car, clutter around it."""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

SIZE = (320, 180)


def scene(
    *,
    car: bool,
    seed: int,
    night: bool = False,
    clutter: bool = True,
    clutter_seed: int | None = None,
    shift: tuple[int, int] | None = None,
) -> Image.Image:
    """
    Render one frame; ``seed`` varies shift, brightness and noise.

    ``clutter_seed`` sets the clutter outside the car spot on its own, so
    tests can keep it fixed while the car comes and goes.
    """
    rng = random.Random(seed)
    image = Image.new("RGB", SIZE, (150, 140, 120))
    draw = ImageDraw.Draw(image)
    dx, dy = shift or (rng.randint(-2, 2), rng.randint(-2, 2))
    # Garage door panels at the back.
    for y in range(20, 120, 20):
        draw.line((60 + dx, y + dy, 260 + dx, y + dy), fill=(110, 105, 95), width=2)
    if car:
        draw.rounded_rectangle(
            (100 + dx, 50 + dy, 220 + dx, 150 + dy), 18, fill=(40, 40, 70)
        )
        draw.rectangle((120 + dx, 60 + dy, 200 + dx, 90 + dy), fill=(180, 200, 220))
    if clutter:
        clutter_rng = random.Random(seed if clutter_seed is None else clutter_seed)
        # Heavy things that get moved around outside the car spot.
        for _ in range(12):
            x = clutter_rng.choice(
                [clutter_rng.randint(0, 70), clutter_rng.randint(250, 310)]
            )
            draw.line(
                (x, clutter_rng.randint(0, 180), x + clutter_rng.randint(-40, 40), 180),
                fill=(clutter_rng.randint(0, 255), 60, 60),
                width=8,
            )
    pixels = np.asarray(image, dtype=np.float32) * rng.uniform(0.7, 1.3)
    pixels += np.random.default_rng(seed).normal(0, 4, pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return image.convert("L").convert("RGB") if night else image


@pytest.fixture
def make_scene():
    return scene
