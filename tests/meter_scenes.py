"""Synthetic meter frames: a counter window with drawn digits and pointer dials."""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tarsius.meter import Dial, MeterLayout
from tarsius.roi import Roi

SIZE = (640, 480)
WINDOW = (150, 60, 470, 140)
DIALS = ((210, 300, 60), (430, 300, 60))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def wheel_strip(
    digits: str, *, size: tuple[int, int] = (320, 80), roll: float = 0.0
) -> Image.Image:
    """Return a black window with white digits; ``roll`` turns the last wheel."""
    width, height = size
    strip = Image.new("RGB", size, (25, 25, 25))
    pitch = width / len(digits)
    font = _font(int(height * 0.8))
    for i, char in enumerate(digits):
        cell = Image.new("RGB", (int(pitch), height * 3), (25, 25, 25))
        draw = ImageDraw.Draw(cell)
        # The digit and its successor on the drum, for a rolling last wheel.
        for row, value in enumerate((int(char), (int(char) + 1) % 10)):
            draw.text(
                (pitch / 2, height * (1.5 + row)),
                str(value),
                fill=(235, 235, 235),
                font=font,
                anchor="mm",
            )
        shift = roll if i == len(digits) - 1 else 0.0
        top = int(height + shift * height)
        strip.paste(cell.crop((0, top, int(pitch), top + height)), (int(i * pitch), 0))
        if i:
            ImageDraw.Draw(strip).line(
                (i * pitch, 0, i * pitch, height), fill=(90, 90, 90), width=2
            )
    return strip


def dial_face(position: float, radius: int) -> Image.Image:
    """Return a white dial with ten marks and a red pointer at ``position``."""
    side = radius * 2
    face = Image.new("RGB", (side, side), (240, 240, 235))
    draw = ImageDraw.Draw(face)
    draw.ellipse((2, 2, side - 3, side - 3), outline=(30, 30, 30), width=3)
    font = _font(int(radius * 0.28))
    for k in range(10):
        a = math.radians(k * 36)
        x = radius + math.sin(a) * radius * 0.75
        y = radius - math.cos(a) * radius * 0.75
        draw.text((x, y), str(k), fill=(20, 20, 20), font=font, anchor="mm")
    a = math.radians(position * 36)
    tip = (radius + math.sin(a) * radius * 0.62, radius - math.cos(a) * radius * 0.62)
    left = (radius + math.cos(a) * radius * 0.16, radius + math.sin(a) * radius * 0.16)
    right = (radius - math.cos(a) * radius * 0.16, radius - math.sin(a) * radius * 0.16)
    draw.polygon((tip, left, right), fill=(200, 40, 40))
    hub = radius * 0.2
    draw.ellipse(
        (radius - hub, radius - hub, radius + hub, radius + hub), fill=(200, 40, 40)
    )
    return face


def meter_frame(
    digits: str = "00628",
    dials: tuple[float, ...] = (8.8, 8.4),
    *,
    roll: float = 0.0,
    rotation: float = 0.0,
    seed: int = 0,
) -> Image.Image:
    """Return a meter face; ``rotation`` turns it clockwise like a tilted camera."""
    rng = random.Random(seed)
    frame = Image.new("RGB", SIZE, (205, 205, 200))
    draw = ImageDraw.Draw(frame)
    draw.ellipse((40, -60, 600, 500), fill=(235, 235, 230), outline=(120, 120, 120))
    left, top, right, bottom = WINDOW
    draw.rectangle((left - 8, top - 8, right + 8, bottom + 8), fill=(60, 60, 60))
    frame.paste(
        wheel_strip(digits, size=(right - left, bottom - top), roll=roll), (left, top)
    )
    draw.text((480, 80), "m3", fill=(30, 30, 30), font=_font(36))
    for (cx, cy, r), position in zip(DIALS, dials, strict=False):
        frame.paste(dial_face(position, r), (cx - r, cy - r))
    if rotation:
        frame = frame.rotate(
            -rotation, Image.Resampling.BICUBIC, fillcolor=(205, 205, 200)
        )
    pixels = np.asarray(frame, np.float32) * rng.uniform(0.85, 1.15)
    pixels += np.random.default_rng(seed).normal(0, 3, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def layout(wheels: int = 5, dials: int = 2, rotation: float = 0.0) -> MeterLayout:
    """Return the layout of :func:`meter_frame`, read back with ``rotation``."""
    width, height = SIZE
    left, top, right, bottom = WINDOW
    return MeterLayout(
        digits=Roi(left / width, top / height, right / width, bottom / height),
        wheels=wheels,
        dials=tuple(
            Dial(
                Roi(
                    (cx - r) / width,
                    (cy - r) / height,
                    (cx + r) / width,
                    (cy + r) / height,
                )
            )
            for cx, cy, r in DIALS[:dials]
        ),
        rotation=rotation,
    )
