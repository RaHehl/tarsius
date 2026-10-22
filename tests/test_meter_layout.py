from __future__ import annotations

import json

import pytest

from tarsius import Roi, RoiError
from tarsius.meter import Calibration, Dial, MeterLayout, Raster

WINDOW = Roi.from_percent(45, 13, 78, 28)


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
