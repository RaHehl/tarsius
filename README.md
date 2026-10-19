# tarsius

Local, training-free state detection for a region of a camera image: is the car in the garage, is the mailbox flag up, is the gate open.

You confirm a few frames per state, tarsius compares new frames against them and answers with a state or "unsure". Nothing leaves the machine and nothing gets trained.

```python
from tarsius import Classifier, NcnnBackend, Roi, extract_features, load_image

backend = NcnnBackend()
spot = Roi(0.10, 0.10, 0.60, 0.50)
classifier = Classifier.for_backend(backend)
car = extract_features(backend, load_image(car_jpeg), spot)
empty = extract_features(backend, load_image(empty_jpeg), spot)
classifier.add("ref-1", car, state="occupied")
classifier.add("ref-2", empty, state="free")

result = classifier.classify(extract_features(backend, load_image(new_jpeg), spot))
result.status  # Status.CONFIDENT, Status.UNSURE or Status.NOT_READY
result.state  # "occupied" or "free" when confident
result.candidate  # best guess, also when unsure
```

Store `features.reference().to_bytes()` next to each confirmed frame and restore it with `Reference.from_bytes()`. On `FeaturesError`, extract again from the stored frame.

## Backends

- `NcnnBackend`: SqueezeNet 1.1 ImageNet features on [ncnn](https://github.com/Tencent/ncnn), install with `tarsius[ncnn]`. Tolerates clutter and lighting changes, a rough region is enough.
- `EdgeBackend`: numpy only, a baseline. Needs a tight region that still shows some static structure when the object is gone.

`NcnnBackend` is the intended default. It stays an optional extra until ncnn ships a release that no longer requires `opencv-python`, which has no musllinux wheels. Without that, tarsius cannot install on Alpine (musl) systems such as Home Assistant OS, whose Core container is Alpine based. ncnn master already made opencv optional ([Tencent/ncnn@0ca5a4b](https://github.com/Tencent/ncnn/commit/0ca5a4b6d)), so the next ncnn release should unblock it.

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
