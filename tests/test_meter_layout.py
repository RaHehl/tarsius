from __future__ import annotations

import json
from typing import Any

import pytest

from tarsius import MeterDataError, Roi, RoiError
from tarsius.meter import Calibration, Dial, MeterLayout, Raster
from tarsius.meter.engine import Exposure

WINDOW = Roi.from_percent(45, 13, 78, 28)
DIAL = [0.1, 0.1, 0.2, 0.2]


def test_round_trip_through_json() -> None:
    layout = MeterLayout(
        digits=WINDOW,
        wheels=5,
        dials=(Dial(Roi.from_percent(74, 46, 92, 70)), Dial(Roi(), True)),
        decimal_wheels=1,
        rotation=-25.0,
    )
    restored = MeterLayout.from_dict(json.loads(json.dumps(layout.as_dict())))
    assert restored == layout
    assert hash(restored) == hash(layout)
    assert layout.decimals == 3


def test_dials_become_a_tuple() -> None:
    layout = MeterLayout(WINDOW, 4, [Dial(Roi())])  # type: ignore[arg-type]
    assert layout.dials == (Dial(Roi()),)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wheels": 0}, "wheels"),
        ({"wheels": 13}, "wheels"),
        ({"decimal_wheels": 6}, "decimal_wheels"),
        ({"decimal_wheels": -1}, "decimal_wheels"),
        ({"dials": (Dial(Roi()),) * 9}, "dials"),
        ({"rotation": 400.0}, "rotation"),
    ],
)
def test_invalid(kwargs: dict[str, object], match: str) -> None:
    settings: dict[str, object] = {"digits": WINDOW, "wheels": 5} | kwargs
    with pytest.raises(ValueError, match=match):
        MeterLayout(**settings)  # type: ignore[arg-type]


def test_invalid_dict() -> None:
    with pytest.raises(ValueError, match="invalid meter layout"):
        MeterLayout.from_dict({"wheels": 5})


def test_region_outside_the_frame() -> None:
    data = MeterLayout(WINDOW, 5).as_dict() | {"digits": [0.5, 0.1, 1.2, 0.3]}
    with pytest.raises(RoiError):
        MeterLayout.from_dict(data)


def test_calibration_round_trip() -> None:
    calibration = Calibration(Raster(-0.05, 0.86, 0.72), True, -12.5, 0.91, 0.8)
    restored = Calibration.from_dict(json.loads(json.dumps(calibration.as_dict())))
    assert restored == calibration


def test_invalid_calibration() -> None:
    with pytest.raises(ValueError, match="invalid calibration"):
        Calibration.from_dict({"offset": 0.0})


def test_period_counts_whole_wheels() -> None:
    assert MeterLayout(WINDOW, 5).period == 100000.0
    assert MeterLayout(WINDOW, 8, decimal_wheels=3).period == 100000.0
    assert MeterLayout(WINDOW, 5, (Dial(Roi()),)).period == 100000.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"wheels": 5.9}, "wheels must be an integer"),
        ({"wheels": "5"}, "wheels must be an integer"),
        ({"wheels": True}, "wheels must be an integer"),
        ({"decimal_wheels": 1.0}, "decimal_wheels must be an integer"),
        (
            {"dials": [{"roi": [0.1, 0.1, 0.2, 0.2], "counterclockwise": "false"}]},
            "counterclockwise",
        ),
        (
            {"dials": [{"roi": [0.1, 0.1, 0.2, 0.2], "counterclockwise": 0}]},
            "counterclockwise",
        ),
        ({"dials": [5]}, "dial must be a mapping"),
        ({"dials": [{}]}, "missing"),
        ({"dials": None}, "dials must be a list"),
        ({"dials": "abc"}, "dials must be a list"),
        ({"digits": [0.1, 0.1, 0.5, 0.3, 0.9]}, "four numbers"),
        ({"digits": "0.1"}, "four numbers"),
        ({"digits": [0, 0, 1, float("nan")]}, "finite"),
        ({"digits": [0, 0, 1, "1"]}, "must be a number"),
        ({"rotation": "inf"}, "rotation must be a number"),
        ({"rotation": float("inf")}, "finite"),
    ],
)
def test_dict_values_are_not_coerced(change: dict[str, Any], match: str) -> None:
    data = MeterLayout(WINDOW, 5).as_dict() | change
    with pytest.raises(MeterDataError, match=f"invalid meter layout: .*{match}"):
        MeterLayout.from_dict(data)


@pytest.mark.parametrize("data", ["abc", None, 5])
def test_layout_dict_must_be_a_mapping(data: Any) -> None:
    with pytest.raises(MeterDataError, match="must be a mapping"):
        MeterLayout.from_dict(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"digits": (0.1, 0.1, 0.5, 0.3)}, "digits must be a Roi"),
        ({"dials": ((0.1, 0.1, 0.2, 0.2),)}, "Dial objects"),
        ({"rotation": float("nan")}, "rotation"),
    ],
)
def test_invalid_layout_objects(kwargs: dict[str, Any], match: str) -> None:
    settings: dict[str, Any] = {"digits": WINDOW, "wheels": 5} | kwargs
    with pytest.raises(MeterDataError, match=match):
        MeterLayout(**settings)


def test_invalid_dial() -> None:
    with pytest.raises(MeterDataError, match="dial region must be a Roi"):
        Dial((0.1, 0.1, 0.2, 0.2))  # type: ignore[arg-type]
    with pytest.raises(MeterDataError, match="counterclockwise"):
        Dial(Roi(), "yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"offset": float("nan")}, "offset must be finite"),
        ({"offset": 0.6}, "offset must be"),
        ({"width": -5}, "width must be"),
        ({"width": 1e9}, "width must be"),
        ({"height": 0}, "height must be"),
        ({"height": "1"}, "height must be a number"),
        ({"upside_down": "no"}, "upside_down"),
        ({"upside_down": 0}, "upside_down"),
        ({"score": 2.0}, "score"),
        ({"agreement": -0.1}, "agreement"),
        ({"orientation_margin": float("inf")}, "orientation_margin"),
    ],
)
def test_calibration_values_are_checked(change: dict[str, Any], match: str) -> None:
    data = Calibration().as_dict() | change
    with pytest.raises(MeterDataError, match=f"invalid calibration: {match}"):
        Calibration.from_dict(data)


def test_calibration_dict_must_be_a_mapping() -> None:
    with pytest.raises(MeterDataError, match="must be a mapping"):
        Calibration.from_dict(5)  # type: ignore[arg-type]


def test_calibration_needs_a_raster() -> None:
    with pytest.raises(MeterDataError, match="raster must be a Raster"):
        Calibration((0.0, 1.0, 1.0))  # type: ignore[arg-type]


def test_calibration_with_exposure_round_trip() -> None:
    calibration = Calibration(
        exposure=(
            Exposure(20.0, 0.0, 0.1, 0.3, 90.0),
            Exposure(30.0, 0.2, 0.0, 0.9, 210.0),
        )
    )
    restored = Calibration.from_dict(json.loads(json.dumps(calibration.as_dict())))
    assert restored == calibration


@pytest.mark.parametrize(
    ("exposure", "match"),
    [
        ("abc", "exposure must be a list"),
        (["abc"], "lists of 5 numbers"),
        ([[1, 2, 3]], "lists of 5 numbers"),
        ([[1, 0.5, 0, 2.0, 100]], "light"),
        ([[-1, 0.5, 0, 0.5, 100]], "spread"),
    ],
)
def test_calibration_exposure_is_checked(exposure: Any, match: str) -> None:
    data = Calibration().as_dict() | {"exposure": exposure}
    with pytest.raises(MeterDataError, match=match):
        Calibration.from_dict(data)


def test_calibration_exposure_items_are_exposures() -> None:
    with pytest.raises(MeterDataError, match="exposure must be Exposure"):
        Calibration(exposure=([1, 2, 3, 4, 5],))  # type: ignore[arg-type]
