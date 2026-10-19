"""
Accuracy on recorded history: references from older frames, test on the newest.

Data stays outside the repo. TARSIUS_DATA points to a directory with
raw/<camera>/<YYYYmmddTHHMMSSZ>.jpg, segments.json (frames grouped by closed-door
segment), labels.json (state per segment, "ignore" to skip one) and rois.json
(camera -> [l, t, r, b]). Usage: history_split.py <camera> [<camera> ...]
"""

import json
import os
import pathlib
import random
import sys
import time

from tarsius import (
    Classifier,
    EdgeBackend,
    NcnnBackend,
    Roi,
    Status,
    extract_features,
    load_image,
)

DATA = pathlib.Path(os.environ["TARSIUS_DATA"])
SPLIT = os.environ.get("TARSIUS_SPLIT", "20261011")


def load(camera, backend):
    segments = json.loads((DATA / "segments.json").read_text())[camera]
    labels = json.loads((DATA / "labels.json").read_text())[camera]
    roi = Roi(*json.loads((DATA / "rois.json").read_text())[camera])
    rows, spent = [], 0.0
    for index, segment in enumerate(segments):
        state = labels[str(index)]
        if state == "ignore":
            continue
        for name in segment["images"]:
            start = time.perf_counter()
            image = load_image((DATA / "raw" / camera / name).read_bytes())
            features = extract_features(backend, image, roi)
            spent += time.perf_counter() - start
            rows.append((index, name, state, features))
    return rows, spent / len(rows)


def run(rows, backend, budget, seed):
    """Confirm `budget` older segments per state, then go through the newest frames."""
    older = {}
    for index, name, state, _ in rows:
        if name < SPLIT:
            older.setdefault(state, set()).add(index)
    rng = random.Random(seed)
    chosen = set()
    for ids in older.values():
        chosen |= set(rng.sample(sorted(ids), min(budget, len(ids))))
    classifier = Classifier.for_backend(backend)
    for index, name, state, features in rows:
        if index in chosen:
            classifier.add(name, features, state=state)
    stats = {"n": 0, "wrong": 0, "unsure": 0, "questions": 0}
    asked = set()
    for index, name, state, features in rows:
        if name < SPLIT:
            continue
        result = classifier.classify(features)
        stats["n"] += 1
        if result.status is not Status.CONFIDENT:
            stats["unsure"] += 1
            # Active labelling: one question per segment, the answer is a reference.
            if index not in asked:
                stats["questions"] += 1
                asked.add(index)
            classifier.add(name, features, state=state)
        elif result.state != state:
            stats["wrong"] += 1
    return stats


def main():
    backends = {"edges": EdgeBackend, "ncnn": NcnnBackend}
    for camera in sys.argv[1:]:
        for name, factory in backends.items():
            backend = factory()
            rows, per_frame = load(camera, backend)
            line = [f"{camera:7} {name:6} {per_frame * 1000:5.0f} ms/frame"]
            for budget in (1, 2, 5):
                total = {"n": 0, "wrong": 0, "unsure": 0, "questions": 0}
                for seed in range(5):
                    for key, value in run(rows, backend, budget, seed).items():
                        total[key] += value
                line.append(
                    f"start {budget}: wrong {total['wrong'] / total['n'] * 100:4.1f} %"
                    f" questions {total['questions'] / 5:.1f}"
                )
            print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
