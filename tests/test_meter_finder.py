from __future__ import annotations

import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
from meter_scenes import layout, meter_frame
from PIL import Image, ImageDraw

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
