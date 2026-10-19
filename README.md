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
from tarsius import EdgeBackend, Roi, StateDetector

detector = StateDetector(EdgeBackend(), Roi.from_percent(10, 10, 60, 50), ("on", "off"))

# Teaching: check the frame, store prepared.original, add it.
prepared = detector.prepare(car_jpeg)
evicted = detector.add("ref-1", prepared, state="on")  # keys dropped by the cap
detector.add("ref-2", detector.prepare(empty_jpeg), state="off")

result = detector.classify(new_jpeg)
result.status  # Status.CONFIDENT, Status.UNSURE or Status.NOT_READY
result.state  # "on" or "off" when confident
result.candidate  # best guess, also when unsure
```

After a restart, hand the stored originals back with `detector.load([(key, original, state), ...])`. References are always re-extracted with the current region and backend; unusable ones (blank, undecodable, unknown state) are skipped and reported in the returned `LoadReport` instead of failing. `prepare()` and `classify()` raise `ImageError` for undecodable data and `BlankFrameError` when the frame shows nothing, for example a grey frame from an offline camera. `prepare()` also raises `BlankRegionError` (the base class) when the region has nothing to compare, `classify()` answers such a region as unsure, without a candidate. `tarsius.outline(frame, roi)` draws the region on a frame with the same box rule as the comparison.

All methods block; call them in an executor. They are safe to call from several threads: changes are serialized and `classify()` never waits for them.

## Backends

- `NcnnBackend`: SqueezeNet 1.1 ImageNet features on [ncnn](https://github.com/Tencent/ncnn), install with `tarsius[ncnn]`. Tolerates clutter and lighting changes, a rough region is enough.
- `EdgeBackend`: numpy only, a baseline. Needs a tight region that still shows some static structure when the object is gone.

`NcnnBackend` is the intended default. It stays an optional extra until ncnn ships a release that no longer requires `opencv-python`, which has no musllinux wheels. Without that, tarsius cannot install on Alpine (musl) systems such as Home Assistant OS, whose Core container is Alpine based. ncnn master already made opencv optional ([Tencent/ncnn@0ca5a4b](https://github.com/Tencent/ncnn/commit/0ca5a4b6d)), so the next ncnn release should unblock it. ncnn has no wheels for Python 3.15 yet, and building it from the source package fails, so the `ncnn` extra needs Python 3.14 until the next ncnn release. The rest of tarsius supports 3.14 and 3.15.

## Results so far

One household, two garage cameras, 24 days of recorded history, day and IR night.
References come from older frames, the test is the newest week.
The table is the harder camera: a dark car and clutter that gets moved around in front of it (459 frames, 164 in the test week).
The other camera scores 100 % with both backends.

| Backend | Region | 2 confirmed segments per state | 5 per state |
|---|---|---|---|
| ncnn | tight | 0 % wrong, 0 % unsure | 0 % wrong, 0 % unsure |
| ncnn | wide, with clutter | 0 % wrong, 0 % unsure | 0 % wrong, 0 % unsure |
| edges | tight | 1.8 % wrong, 3.5 % unsure | 0.1 % wrong, 1.8 % unsure |
| edges | wide, with clutter | 27 % wrong, 7.4 % unsure | 20 % wrong, 8.8 % unsure |

About 20 ms per 4 MP frame including JPEG decoding, for either backend, on an x86 desktop with one thread.
A small sample, not a guarantee. `scripts/history_split.py` runs the same evaluation on your own recordings.

## Development

```sh
scripts/build_model.sh   # once, builds the ncnn model into src/tarsius/models
uv sync --extra ncnn
uv run pytest
uvx prek run --all-files
```

The ncnn model is not stored in git. `scripts/build_model.sh` converts the original SqueezeNet 1.1 release with the ncnn tools. Inputs and outputs are checked against pinned sha256 sums, so CI and a local run produce the same files. It needs x86_64 Linux with `curl`, `unzip`, `libgomp1` and `libprotobuf32t64` (Ubuntu 24.04).

Releases: every PR gets a label, Release Drafter collects them into a draft, publishing the GitHub release builds and uploads to PyPI.

## License

Apache-2.0. The bundled SqueezeNet model is BSD-2-Clause, see NOTICE.
