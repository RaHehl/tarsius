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

Local, training-free state detection for a region of a camera image: is the car in the garage, is the mailbox flag up, is the gate open.

You confirm a few frames per state, tarsius compares new frames against them and answers with a state or "unsure". Nothing leaves the machine and nothing gets trained.

## Why tarsius

Tarsiers are small nocturnal primates whose eyes are each about as large as their brain. Lots of eye, little brain, and very good at seeing in the dark. That is the idea here too: no training and no model to learn, just careful looking, including at IR night frames.

```python
from tarsius import Roi, StateDetector

# The detector picks the backend for the task; detector.backend_id tells which.
detector = StateDetector(Roi.from_percent(10, 10, 60, 50), ("on", "off"))

# Teaching: check the frame, store prepared.original and prepared.cache, add it.
prepared = detector.prepare(car_jpeg)
evicted = detector.add("ref-1", prepared, state="on")  # keys dropped by the cap
detector.add("ref-2", detector.prepare(empty_jpeg), state="off")

result = detector.classify(new_jpeg)
result.status  # Status.CONFIDENT, Status.UNSURE or Status.NOT_READY
result.state  # "on" or "off" when confident
result.candidate  # best guess, also when unsure
result.probabilities  # {"on": 0.97, "off": 0.03}, estimated from the scores
```

`probabilities` is a softmax of the per-state scores with a temperature per backend, fitted by `scripts/calibrate.py` on the week before the test week of the history below. Only the harder camera had both states in that week, so the fit rests on its 142 frames. The test week has 284 frames, each classified with 15 sets of references (1, 2 and 5 segments per state, 5 seeds each). Of those answers ncnn gives 0.9 or more to 97 %, all of them right, with a calibration error of 0.009. edges gives 0.9 or more to 79 %. With one segment per state the estimates are slightly too sure, ncnn erred at up to 0.85. With more references they are rather too cautious, edges clearly. The status comes from the margin, not from the probabilities: with two states, unsure means below 0.67 for ncnn. A blank region gets the same probability for every state, a detector that is not ready gives none.

After a restart, hand the stored originals back with `detector.load([(key, original, state), ...], caches={key: cache, ...})`. A cache spares the extraction, about 14 ms per reference on an i5-8400T, one reference is still extracted to check that the cached ones fit the backend. A cache made for another region, `frame_side`, backend or release of tarsius, Pillow or ncnn is ignored and the reference is extracted again. A cache that is damaged or does not hold a unit vector is ignored too. `LoadReport.caches` holds only the new caches: store them, keep the others and drop those of skipped and evicted keys. Development builds all report version 0.0.0, so clear stored caches after changing tarsius code there. Unusable references (blank, undecodable, unknown state) are skipped and reported in the returned `LoadReport` instead of failing. `prepare()` and `classify()` raise `ImageError` for undecodable data and `BlankFrameError` when the frame shows nothing, for example a grey frame from an offline camera. `prepare()` also raises `BlankRegionError` (the base class) when the region has nothing to compare, `classify()` answers such a region as unsure, without a candidate. `tarsius.outline(frame, roi)` draws the region on a frame with the same box rule as the comparison.

All methods block, and so does creating the first detector, which loads the model; call them in an executor. They are safe to call from several threads: changes are serialized and `classify()` never waits for them, only inferences on the shared backend take turns.

## Backends

- `NcnnBackend`: SqueezeNet 1.1 ImageNet features on [ncnn](https://github.com/Tencent/ncnn). Tolerates clutter and lighting changes, a rough region is enough.
- `EdgeBackend`: numpy only, a baseline. Needs a tight region that still shows some static structure when the object is gone.

`StateDetector` uses `NcnnBackend` unless you pass `backend=`. It loads one instance per process (about 20 MB) and shares it, inferences take turns on it. ncnn is a regular dependency. Since ncnn 1.0.20261020, `opencv-python` is an optional extra of ncnn, so tarsius installs on Alpine (musl) systems such as Home Assistant OS. Checked in `python:3.14-alpine` on x86_64 and aarch64. Both backends support Python 3.14 and 3.15.

## Results so far

One household, two garage cameras, 24 days of recorded history, day and IR night.
References come from older frames, the test is the newest week.
The table is the harder camera: a dark car and clutter that gets moved around in front of it (459 frames, 164 in the test week).
The other camera scores 100 % with both backends.

| Backend | Region | 2 confirmed segments per state (about 10 frames) | 5 per state (about 25 frames) |
|---|---|---|---|
| ncnn | tight | 0 % wrong, 0 % unsure | 0 % wrong, 0 % unsure |
| ncnn | wide, with clutter | 0 % wrong, 0 % unsure | 0 % wrong, 0 % unsure |
| edges | tight | 1.8 % wrong, 3.5 % unsure | 0.1 % wrong, 1.7 % unsure |
| edges | wide, with clutter | 27 % wrong, 7.4 % unsure | 20 % wrong, 8.8 % unsure |

About 30 ms per 4 MP frame with edges and 36 ms with ncnn, including JPEG decoding, on one thread of an Intel i5-8400T.
A small sample, not a guarantee. `scripts/history_split.py` runs the same evaluation with `StateDetector` on your own recordings. It also reports how often the detector would ask when unsure answers are confirmed.

## Reading meters

`tarsius.meter` reads utility meters with counter wheels and optional pointer dials. You give the regions once, it reads a value from each frame and says why when it will not.

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
reading.rejection  # None, FAULT, LOW_CONFIDENCE or INCONSISTENT

guard = Plausibility(max_rate=5.0, tolerance=0.0002)  # units per hour
reading = guard.review(reading, at=now)  # adds DECREASED or TOO_FAST
store(guard.state.as_dict())  # and pass state=PlausibilityState.from_dict(...)
```

Regions are fractions of the frame after turning it by `rotation` about its centre, like `PIL.Image.rotate`. The window is cut into `wheels` equal cells; `decimal_wheels` marks the rightmost ones as fractions. Dials run most significant first, `Dial(..., counterclockwise=True)` for those that turn the other way. `MeterLayout`, `Calibration` and `PlausibilityState` convert to and from JSON-compatible dicts.

Wheels and dials are composed with the carry rule, not concatenated: a wheel at 1.7 shows 1 or 2 depending on the wheels below it, so each wheel is resolved against what the lower ones contribute, and an upper wheel that jumps instead of rolling is recognized too. How badly the positions contradict each other is `reading.residual`, above `max_residual` (0.4 digits) the reading is rejected as inconsistent. The last wheel or dial keeps its fraction: a rolling last wheel reads 601.6, not 602.

`calibrate()` decides whether the meter is mounted upside down and searches the cell raster (offset, cell width and height) on one or more frames. It costs about 45 reads, 0.36 s per frame on the i5-8400T; run it at setup and after the camera moved. `calibration.agreement` is the share of the search grid that reads the same whole number, low values mean the window needs fixing. Without a calibration `read()` cuts equal upright cells.

The readers are small numpy networks: 475k parameters for a wheel (32×20 pixels), 168k for a dial. About 18 ms per read of five wheels and four dials on one thread of an i5-8400T, including JPEG decoding. `MeterReader(views=3)` also reads each crop shifted by a pixel and averages, which costs three times as much. The methods block and are safe to call from several threads; the readers load on first use (2.6 MB) and are shared per process.

Results so far, one water meter (Corona MWI: five wheels, four dials), 696 frames of 640×480 over two days, compared with the value an existing reader on the same camera reported. Regions taken from that reader, calibrated on ten frames:

| | |
|---|---|
| within 0.01 m³ | 99.7 % (the two others lie between two reference rounds) |
| within 0.001 m³ | 97.1 % |
| whole m³ right | 100 % |
| rejected with the defaults | 1.7 %, all as inconsistent |
| rejected by `Plausibility` | none |

One meter, so this says the pipeline works on it, not how well it carries over to other meters.

The models were trained by the author on generated meters and the training split of the Word-Wheel Water Meter Dataset (Zhao et al., Dryad, CC0-1.0, [doi:10.5061/dryad.7d7wm3860](https://doi.org/10.5061/dryad.7d7wm3860)), with thanks to its authors. See `src/tarsius/meter/models/MODELS.md`.

## Development

```sh
scripts/build_model.sh   # once, builds the ncnn model into src/tarsius/models
uv sync
uv run pytest
uvx prek run --all-files
```

The SqueezeNet model is not stored in git. `scripts/build_model.sh` converts the original SqueezeNet 1.1 release with the ncnn tools. Inputs and outputs are checked against pinned sha256 sums, so CI and a local run produce the same files. It needs x86_64 Linux with `curl`, `unzip`, `libgomp1` and `libprotobuf32t64` (Ubuntu 24.04).

The meter models are stored in git with their sha256 sums, since their training outputs are not public. See `src/tarsius/meter/models/MODELS.md`.

Releases: every PR gets a label, Release Drafter collects them into a draft, publishing the GitHub release builds and uploads to PyPI.

## License

Apache-2.0. The bundled SqueezeNet model is BSD-2-Clause, see NOTICE. The meter models are Apache-2.0.
