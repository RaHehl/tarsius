from __future__ import annotations

import threading
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from meter_scenes import DIALS, SIZE, WINDOW, dial_face, layout, meter_frame
from PIL import Image, ImageDraw, ImageFilter

from tarsius import BackendError, BlankFrameError, Roi
from tarsius.meter import MeterLayout, MeterReader, finder
from tarsius.meter.finder import AlphaWarning, Suggestion, suggest
from tarsius.meter.frame import crop


def _iou(a: Roi, b: Roi) -> float:
    left, top = max(a.left, b.left), max(a.top, b.top)
    right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
    inter = max(0.0, right - left) * max(0.0, bottom - top)

    def area(r: Roi) -> float:
        return (r.right - r.left) * (r.bottom - r.top)

    return inter / (area(a) + area(b) - inter)


@pytest.fixture(autouse=True)
def quiet_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(finder, "_warned", threading.Event())
    warnings.simplefilter("ignore", AlphaWarning)


def test_marked_alpha() -> None:
    assert finder.__alpha__ is True
    assert "ALPHA" in (finder.__doc__ or "")
    assert issubclass(AlphaWarning, FutureWarning)


def test_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = meter_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suggest(frame, dials=False)
        suggest(frame, dials=False)
    assert [w.category for w in caught] == [AlphaWarning]


def test_suggests_window_and_dials_of_a_tilted_meter() -> None:
    frame = meter_frame("00627", (8.84, 8.4), roll=0.884, rotation=20)
    suggestion = suggest(frame)
    expected = layout(rotation=20)
    assert suggestion.rotation == pytest.approx(20, abs=3)
    assert suggestion.digits is not None
    assert _iou(suggestion.digits, expected.digits) > 0.6
    assert suggestion.digits_score > 0.5
    assert len(suggestion.dials) == 2
    assert len(suggestion.dial_scores) == 2
    # Clockwise from twelve: the right dial comes first here.
    found = sorted(suggestion.dials, key=lambda roi: roi.left)
    assert suggestion.dials[0] == found[1]
    for roi, dial in zip(found, expected.dials, strict=True):
        assert _iou(roi, dial.roi) > 0.5
    # With the dials in their real order the suggestion reads the meter.
    suggested = suggestion.layout(5)
    reading = MeterReader().read(
        frame,
        MeterLayout(
            suggested.digits,
            5,
            tuple(reversed(suggested.dials)),
            rotation=suggested.rotation,
        ),
    )
    assert reading.value == pytest.approx(627.884, abs=0.02)


def test_without_dials() -> None:
    suggestion = suggest(meter_frame(), dials=False)
    assert suggestion.dials == ()
    assert isinstance(suggestion.layout(5), MeterLayout)


def test_nothing_found() -> None:
    plain = Image.new("RGB", (320, 240), (200, 200, 200))
    ImageDraw.Draw(plain).line((0, 0, 320, 240), fill=(0, 0, 0), width=3)
    suggestion = suggest(plain)
    assert suggestion.digits is None
    with pytest.raises(ValueError, match="no counter window"):
        suggestion.layout(5)


def test_empty_suggestion() -> None:
    with pytest.raises(ValueError, match="no counter window"):
        Suggestion().layout(4)


def test_blank_frame() -> None:
    with pytest.raises(BlankFrameError):
        suggest(Image.new("RGB", (64, 48), (90, 90, 90)))


def test_rotated_points_match_rotated_crops() -> None:
    frame = Image.new("RGB", (200, 100))
    ImageDraw.Draw(frame).rectangle((150, 20, 154, 24), fill=(255, 255, 255))
    x, y = finder._to_rotated(152.5, 22.5, 30.0, frame.size)
    turned = np.asarray(crop(frame, Roi(), 30.0).convert("L"))
    ys, xs = np.nonzero(turned > 128)
    assert (xs.mean(), ys.mean()) == pytest.approx((x, y), abs=1.0)


def test_strongest_component_wins_over_a_larger_one() -> None:
    # A large, weaker blob (a serial number) and a smaller, surer window.
    probs = np.zeros((20, 20), np.float32)
    probs[1:13, 1:5] = 0.8
    probs[15:18, 8:18] = 0.95
    chosen = finder._strongest(probs > 0.5, probs)
    assert chosen.sum() == 30
    assert chosen[16, 10]


def test_strongest_component_breaks_ties_by_area() -> None:
    probs = np.zeros((20, 20), np.float32)
    probs[1:4, 1:4] = 0.9
    probs[10:14, 10:14] = 0.9
    assert finder._strongest(probs > 0.5, probs).sum() == 16


def test_tiny_components_are_noise() -> None:
    probs = np.zeros((10, 10), np.float32)
    probs[1:3, 1:3] = 0.99
    assert not finder._strongest(probs > 0.5, probs).any()
    assert not finder._strongest(np.zeros((4, 4), bool), probs[:4, :4]).any()


def test_touching_discs_are_separated() -> None:
    yy, xx = np.ogrid[:60, :100]
    mask = ((xx - 30) ** 2 + (yy - 30) ** 2 <= 144) | (
        (xx - 54) ** 2 + (yy - 30) ** 2 <= 144
    )
    # A thin stroke, like text, is not a dial.
    mask[50:52, 5:95] = True
    discs = finder._discs(mask)
    assert sorted((x, y) for x, y, _ in discs) == [(30, 30), (54, 30)]
    assert all(r == pytest.approx(12.5, abs=1.5) for _, _, r in discs)


def test_strokes_and_crossings_are_not_dials() -> None:
    yy, xx = np.ogrid[:100, :160]
    mask = (xx - 30) ** 2 + (yy - 30) ** 2 <= 144
    # Strokes thinner than half the largest disc, like printed text.
    mask[70:74, 10:150] = True
    mask[80:90, 10:150] = True
    # A crossing: its centre fills the whole ring around it.
    mask[20:44, 90:150] = True
    mask[2:62, 108:132] = True
    assert [(x, y) for x, y, _ in finder._discs(mask)] == [(30, 30)]


def test_box_outside_the_frame() -> None:
    assert finder._clamped(-10, -10, -5, -5, (100, 100)) is None


def test_missing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(finder, "files", lambda _package: tmp_path)
    monkeypatch.setattr(finder, "_NETS", {n: finder._Net(n) for n in finder._NETS})
    with pytest.raises(BackendError, match="missing"):
        suggest(meter_frame())


def test_unloadable_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    for name in ("finder.param", "finder.bin"):
        (tmp_path / "models" / name).write_text("garbage")
    monkeypatch.setattr(finder, "files", lambda _package: tmp_path)
    monkeypatch.setattr(finder, "_NETS", {n: finder._Net(n) for n in finder._NETS})
    with pytest.raises(BackendError, match="could not load"):
        suggest(meter_frame())


def _dial_scene(
    radius: int, centre: tuple[int, int], size: tuple[int, int] = (320, 240)
) -> npt.NDArray[np.float32]:
    frame = Image.new("RGB", size, (190, 190, 185))
    frame.paste(dial_face(3.3, radius), (centre[0] - radius, centre[1] - radius))
    return np.asarray(frame.convert("L"), np.float32)


def _disc_scene(radius: int, fill: int, ground: int) -> npt.NDArray[np.float32]:
    frame = Image.new("L", (360, 300), ground)
    ImageDraw.Draw(frame).ellipse(
        (180 - radius, 150 - radius, 180 + radius, 150 + radius), fill=fill
    )
    return np.asarray(frame, np.float32)


@pytest.mark.parametrize("radius", [40, 100])
def test_recentre_finds_the_axis_of_an_offset_box(radius: int) -> None:
    levels = _dial_scene(radius, (160, 120), (360, 300))
    x, y, ring, sharpness = finder._recentre(
        levels, 160 + 0.2 * radius, 120 - 0.15 * radius, radius
    )
    assert (x, y) == pytest.approx((160, 120), abs=2)
    # The scale ring is drawn just inside the face.
    assert ring == pytest.approx(radius - 3, abs=0.05 * radius)
    assert sharpness > 13


@pytest.mark.parametrize(("fill", "ground"), [(60, 200), (200, 60)])
@pytest.mark.parametrize("radius", [36, 44])
def test_recentre_votes_both_ways_along_the_gradient(
    fill: int, ground: int, radius: int
) -> None:
    # A darker and a lighter disc, a little inside and outside the ring
    # radius the mask suggests.
    x, y, ring, sharpness = finder._recentre(
        _disc_scene(40, fill, ground), 184, 146, radius
    )
    assert (x, y) == pytest.approx((180, 150), abs=1)
    assert ring == pytest.approx(40, abs=2)
    assert sharpness > 15


def test_recentre_votes_through_noise() -> None:
    levels = _dial_scene(50, (180, 150), (360, 300))
    levels += np.random.default_rng(2).normal(0, 40, levels.shape).astype(np.float32)
    x, y, _, sharpness = finder._recentre(levels, 188, 144, 50)
    assert (x, y) == pytest.approx((180, 150), abs=3)
    # Only the strongest edges vote, the noise's would flatten the peak.
    assert sharpness > 8


def test_on_axis_moves_the_box_to_the_ring() -> None:
    levels = _dial_scene(50, (150, 110))
    x, y, half = finder._on_axis(levels, 158, 103, 54)
    assert (x, y) == pytest.approx((150, 110), abs=1.5)
    # Half side 1.2 times the scale ring, drawn 3 pixels inside the face.
    assert half == pytest.approx(47 * 1.2, abs=2.5)


def test_on_axis_shrinks_a_box_at_the_frame_edge() -> None:
    # The ring's box would reach 7 pixels beyond the left edge.
    levels = _dial_scene(40, (40, 150), (360, 300))
    x, y, half = finder._on_axis(levels, 46, 150, 40)
    assert (x, y) == pytest.approx((40, 150), abs=1)
    # Square and still centred on the axis, not cut on one side.
    assert half == pytest.approx(x)


def test_on_axis_keeps_the_box_without_a_clear_vote() -> None:
    rng = np.random.default_rng(1)
    noise = rng.uniform(0, 255, (240, 320)).astype(np.float32)
    assert finder._recentre(noise, 160, 120, 50)[3] < 3
    assert finder._on_axis(noise, 160, 120, 50) == (160, 120, 50)
    # Nothing to vote with: a flat patch, or one cut off by the frame.
    flat = np.full((240, 320), 128, np.float32)
    assert finder._recentre(flat, 160, 120, 50) == (160, 120, 50, 0.0)
    assert finder._recentre(noise, 2, 2, 4) == (2, 2, 4, 0.0)


@pytest.mark.parametrize(("radius", "low", "high"), [(30, 4.0, 5.0), (35, 5.0, 5.5)])
def test_on_axis_keeps_the_box_for_a_weak_vote(
    radius: int, low: float, high: float
) -> None:
    # A mask disc of 0.6 or 0.7 times the dial's radius: the vote peaks off
    # the axis and only a little, as on spots beside the dials of a real meter.
    levels = _dial_scene(50, (180, 150), (360, 300))
    x, y, _, sharpness = finder._recentre(levels, 185, 146, radius)
    assert abs(x - 180) + abs(y - 150) > 10
    assert low < sharpness < high
    assert finder._on_axis(levels, 185, 146, radius) == (185, 146, radius)


def test_on_axis_keeps_the_box_when_the_ring_is_far_inside_the_mask() -> None:
    # On a tiny dial the numerals outweigh the ring, which comes out at
    # about half the mask's radius. On a real meter it was at least 0.82.
    levels = _dial_scene(12, (180, 150), (360, 300))
    _, _, ring, sharpness = finder._recentre(levels, 182.4, 150, 12)
    assert ring < 0.6 * 12
    assert sharpness > 5
    assert finder._on_axis(levels, 182.4, 150, 12) == (182.4, 150, 12)


def test_dials_are_segmented_in_the_level_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = meter_frame(rotation=30)
    seen: list[Image.Image] = []
    net = finder._NETS[finder._DIAL_MODEL]
    logits = net.logits

    def spy(image: Image.Image) -> npt.NDArray[np.float32]:
        seen.append(image)
        return logits(image)

    monkeypatch.setattr(net, "logits", spy)
    finder._dials(frame, 30.0)
    finder._dials(frame, 0.0)
    level = frame.rotate(30.0, Image.Resampling.BILINEAR).convert("L")
    assert np.array_equal(np.asarray(seen[0]), np.asarray(level.convert("RGB")))
    assert np.array_equal(
        np.asarray(seen[1]), np.asarray(frame.convert("L").convert("RGB"))
    )


@pytest.mark.parametrize("rotation", [-30, 25, 45])
def test_dial_boxes_sit_on_the_axes_of_a_tilted_meter(rotation: float) -> None:
    suggestion = suggest(meter_frame(rotation=rotation))
    width, height = SIZE
    assert len(suggestion.dials) == 2
    for roi, (cx, cy, radius) in zip(
        sorted(suggestion.dials, key=lambda r: r.left), DIALS, strict=True
    ):
        # Turned back by the measured rotation, which may be off a little.
        x, y = finder._to_rotated(cx, cy, suggestion.rotation - rotation, SIZE)
        assert (roi.left + roi.right) / 2 * width == pytest.approx(x, abs=2.5)
        assert (roi.top + roi.bottom) / 2 * height == pytest.approx(y, abs=2.5)
        # Sized to the scale ring, drawn just inside the face.
        side = 2 * (radius - 3) * 1.2
        assert (roi.right - roi.left) * width == pytest.approx(side, rel=0.04)


def _window_scene(
    ground: int = 215,
    window: tuple[int, int, int, int] = (200, 150, 420, 220),
    tilt: float = 0.0,
) -> Image.Image:
    """Return a dark wheel strip on a lighter face, with some print around it."""
    frame = Image.new("RGB", (640, 400), (ground, ground, ground - 5))
    left, top, right, bottom = window
    draw = ImageDraw.Draw(frame)
    draw.rectangle((left, top, right - 1, bottom - 1), fill=(30, 30, 30))
    pitch = (right - left) / 5
    for i, digit in enumerate("00628"):
        draw.text(
            (left + (i + 0.5) * pitch, (top + bottom) / 2),
            digit,
            fill=(230, 230, 230),
            font_size=(bottom - top) // 2,
            anchor="mm",
        )
    draw.text((430, 170), "m3", fill=(40, 40, 40), font_size=30)
    draw.text((210, 240), "Qn 2.5 m3/h", fill=(40, 40, 40), font_size=18)
    # Turned clockwise by tilt, so turning by tilt levels it again.
    frame = frame.rotate(-tilt, Image.Resampling.BICUBIC, fillcolor=(ground,) * 3)
    # A camera does not draw edges a pixel sharp.
    frame = frame.filter(ImageFilter.GaussianBlur(1.5))
    pixels = np.asarray(frame, np.float32)
    pixels += np.random.default_rng(0).normal(0, 3, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


# The finder's box on a real meter: too wide, mostly to the right, and tall.
_LOOSE = (196.0, 144.0, 432.0, 224.0)


def test_dark_window_is_tightened() -> None:
    rotation, box = finder._refined(_window_scene(), 0.0, _LOOSE)
    assert rotation == 0.0
    assert box == pytest.approx((200, 150, 420, 220), abs=1.5)


def test_suggested_window_is_tight() -> None:
    suggestion = suggest(meter_frame(rotation=20), dials=False)
    assert suggestion.digits is not None
    # The window with its dark frame, 8 pixels around the strip.
    left, top, right, bottom = WINDOW
    width, height = SIZE
    tight = Roi(
        (left - 8) / width,
        (top - 8) / height,
        (right + 8) / width,
        (bottom + 8) / height,
    )
    assert _iou(suggestion.digits, tight) > 0.95


def test_light_window_keeps_the_finder_box() -> None:
    light = Image.fromarray(255 - np.asarray(_window_scene(ground=225)))
    assert finder._tightened(light, 0.0, _LOOSE) is None
    assert finder._refined(light, 0.0, _LOOSE) == (0.0, _LOOSE)


def test_clutter_keeps_the_finder_box() -> None:
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 255, (40, 64), dtype=np.uint8)
    clutter = Image.fromarray(blocks).resize((640, 400), Image.Resampling.NEAREST)
    assert finder._refined(clutter.convert("RGB"), 0.0, _LOOSE) == (0.0, _LOOSE)


def test_tightening_needs_a_dark_middle() -> None:
    plain = Image.new("RGB", (640, 400), (200, 200, 200))
    assert finder._tightened(plain, 0.0, _LOOSE) is None
    # Dark rows, but light in the middle of them.
    ImageDraw.Draw(plain).rectangle((200, 150, 420, 220), fill=(30, 30, 30))
    ImageDraw.Draw(plain).rectangle((295, 150, 325, 220), fill=(200, 200, 200))
    assert finder._tightened(plain, 0.0, _LOOSE) is None
    # Nothing, or only the surroundings, of the box inside the frame.
    assert finder._tightened(plain, 0.0, (-500, -500, -400, -400)) is None
    assert finder._tightened(plain, 0.0, (-50, -50, -10, -10)) is None


@pytest.mark.parametrize("tilt", [-2.0, 1.2])
def test_residual_tilt_is_corrected(tilt: float) -> None:
    # Turned clockwise by tilt, so turning by tilt levels it again.
    frame = _window_scene().rotate(
        -tilt, Image.Resampling.BICUBIC, fillcolor=(215,) * 3
    )
    rotation, box = finder._refined(frame, 0.0, _LOOSE)
    assert rotation == pytest.approx(tilt, abs=0.15)
    assert box == pytest.approx((200, 150, 420, 220), abs=2)


@pytest.mark.parametrize("tilt", [0.1, 4.0])
def test_small_or_out_of_range_tilt_is_left(tilt: float) -> None:
    frame = _window_scene().rotate(
        -tilt, Image.Resampling.BICUBIC, fillcolor=(215,) * 3
    )
    tight = finder._tightened(frame, 0.0, _LOOSE)
    assert tight is not None
    assert finder._tilt(frame, 0.0, tight) == 0.0


def test_tilted_patch_matches_rotated_crop() -> None:
    frame = meter_frame(rotation=10).convert("L")
    centre = finder._to_rotated(300.0, 100.0, -10.0, frame.size)
    patch = finder._around(frame, centre, 10.0, (80, 40))
    expected = crop(frame, Roi(260 / 640, 80 / 480, 340 / 640, 120 / 480), 10.0)
    difference = np.abs(patch - np.asarray(expected, np.float32))
    assert float(difference.mean()) < 2
