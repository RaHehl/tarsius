<p align="center">
  <img src="https://raw.githubusercontent.com/RaHehl/tarsius/main/docs/logo.png" alt="tarsius logo" width="160">
</p>

<h1 align="center">tarsius</h1>

<p align="center">
  <a href="https://github.com/RaHehl/tarsius/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/RaHehl/tarsius/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://codecov.io/gh/RaHehl/tarsius"><img src="https://codecov.io/gh/RaHehl/tarsius/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/tarsius/"><img src="https://img.shields.io/pypi/v/tarsius" alt="PyPI"></a>
  <a href="https://pypi.org/project/tarsius/"><img src="https://img.shields.io/pypi/pyversions/tarsius" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/j178/prek"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek"></a>
</p>

Local image recognition for regions of a camera image:

- **State detection:** is the car in the garage, is the gate open, is the mailbox flag up.
- **Meter reading:** water, gas and electricity meters with number wheels and pointer dials.

Everything runs on your machine, without a cloud service and without training on your frames.

Tarsiers are small nocturnal primates whose eyes are each about as large as their brain: lots of eye, little brain, very good at seeing in the dark. That is the idea here too.

## Install

```sh
pip install tarsius
```

Python 3.14 or newer. Works on Alpine (musl), so also in Home Assistant OS.

## Detect a state

```python
from tarsius import Roi, StateDetector

detector = StateDetector(Roi.from_percent(10, 10, 60, 50), ("on", "off"))

# Teach a few frames per state.
detector.add("ref-1", detector.prepare(car_jpeg), state="on")
detector.add("ref-2", detector.prepare(empty_jpeg), state="off")

result = detector.classify(new_jpeg)
result.status  # CONFIDENT, UNSURE or NOT_READY
result.state  # "on" or "off" when confident
result.candidate  # best guess, also when unsure
result.probabilities  # {"on": 0.97, "off": 0.03}
```

- You confirm a few frames per state. tarsius compares new frames against them and answers with a state, or "unsure" with its best guess.
- Store `prepared.original` when you teach, and give it back with `detector.load(...)` after a restart.
- A grey frame from an offline camera raises an error, an empty region answers "unsure".
- All methods block. Call them in an executor; they are safe to use from several threads.

Two backends compare the frames:

- `NcnnBackend` (default): image features on [ncnn](https://github.com/Tencent/ncnn). Tolerates clutter and changing light, a rough region is enough.
- `EdgeBackend`: numpy only, a baseline that needs a tight region.

### Results

One household, two garage cameras, 24 days, day and IR night. The table shows the harder camera (dark car, clutter that gets moved); the other one scores 100 % with both backends.

| Backend | Region | 2 confirmed segments per state (about 10 frames) | 5 per state (about 25 frames) |
|---|---|---|---|
| ncnn | tight | 0 % wrong, 0 % unsure | 0 % wrong, 0 % unsure |
| ncnn | wide, with clutter | 0 % wrong, 0 % unsure | 0 % wrong, 0 % unsure |
| edges | tight | 1.8 % wrong, 3.5 % unsure | 0.1 % wrong, 1.7 % unsure |
| edges | wide, with clutter | 27 % wrong, 7.4 % unsure | 20 % wrong, 8.8 % unsure |

About 36 ms per 4 MP frame with ncnn on one thread of an Intel i5-8400T. A small sample, not a guarantee: `scripts/history_split.py` runs the same test on your own recordings.

[docs/state.md](docs/state.md) covers probabilities, stored references and errors in detail.

## Read a meter

```python
from tarsius import Roi
from tarsius.meter import Dial, MeterLayout, MeterReader, Plausibility

layout = MeterLayout(
    digits=Roi.from_percent(45, 13, 78, 28),  # the counter window, all wheels
    wheels=5,
    dials=(  # most significant first: x0.1, x0.01
        Dial(Roi.from_percent(74, 46, 92, 70)),
        Dial(Roi.from_percent(62, 69, 80, 93)),
    ),
    rotation=-25.0,  # degrees counter-clockwise that make the window level
)
reader = MeterReader()
calibration = reader.calibrate([jpeg_1, jpeg_2], layout)  # once, store as_dict()
reading = reader.read(jpeg, layout, calibration)
reading.value  # 627.8846, or None when rejected
reading.candidate  # the composed value, also when rejected
reading.rejection  # None, FAULT, LOW_CONFIDENCE, INCONSISTENT or AMBIGUOUS

guard = Plausibility(max_rate=4.0, tolerance=0.0002, period=layout.period)
guard.accept(627.88, at=confirmed_at)  # a value the user confirmed, aware time
reading = guard.review(reading, at=now)  # may add DECREASED, TOO_FAST, ...
store(guard.state.as_dict())  # and pass state=PlausibilityState.from_dict(...)
```

- Wheels and dials are combined with the carry rule, so a wheel rolling between two digits reads right.
- A reading is rejected with a reason when a wheel is covered or glared, when the reader is unsure, or when the wheels contradict each other.
- `Plausibility` checks each reading against the last value a user confirmed: it never goes down, rises at most `max_rate` per hour, and asks the user when readings keep disagreeing.
- Run `calibrate()` once at setup; a reading then takes about 18 ms.

On one water meter with five wheels and four dials, 99.7 % of 696 frames read within 0.01 m³. That is one meter, not a guarantee for others.

The models were trained by the author on generated meters and the Word-Wheel Water Meter Dataset (Zhao et al., Dryad, CC0-1.0, [doi:10.5061/dryad.7d7wm3860](https://doi.org/10.5061/dryad.7d7wm3860)), with thanks to its authors.

[docs/meter.md](docs/meter.md) covers the carry rule, the checks and calibration in detail.

### Suggest the regions (alpha)

```python
from tarsius.meter.finder import suggest

suggestion = suggest(jpeg)  # rotation, digits, dials, scores
layout = suggestion.layout(wheels=5)  # check it with the user
```

On a straight-on photo of one meter it usually finds the counter and the dials, but it can miss them or mark a serial number instead. Always let the user check the suggestion. **Alpha:** results and API may change in any release.

## Development

```sh
scripts/build_model.sh   # once, builds the ncnn model into src/tarsius/models
uv sync
uv run pytest
uvx prek run --all-files
```

The SqueezeNet model is not stored in git. `scripts/build_model.sh` converts the original SqueezeNet 1.1 release with the ncnn tools. Inputs and outputs are checked against pinned sha256 sums, so CI and a local run produce the same files. It needs x86_64 Linux with `curl`, `unzip`, `libgomp1` and `libprotobuf32t64` (Ubuntu 24.04).

The meter models are stored in git with their sha256 sums, since their training outputs are not public. `scripts/build_meter_models.sh` reproduces them from those outputs and `scripts/meter_finder_parity.py` compares the converted finders with Keras, see `src/tarsius/meter/models/MODELS.md`.

Releases: every PR gets a label, Release Drafter collects them into a draft, publishing the GitHub release builds and uploads to PyPI.

## License

Apache-2.0. The bundled SqueezeNet model is BSD-2-Clause, see NOTICE. The meter models are Apache-2.0.
