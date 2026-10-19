"""
Accuracy on recorded history: references from older frames, test on the newest.

Runs StateDetector the way an application does: originals are prepared and
taught, new frames are classified from their bytes.

Data stays outside the repo. TARSIUS_DATA points to a directory with
raw/<camera>/<YYYYmmddTHHMMSSZ>.jpg, segments.json (frames grouped by closed-door
segment), labels.json (state per segment, "ignore" to skip one) and rois.json
(camera -> [l, t, r, b], or the file in TARSIUS_ROIS).
Usage: history_split.py <camera> [<camera> ...]
"""

import contextlib
import functools
import json
import os
import pathlib
import random
import sys
import time

from tarsius import (
    BlankRegionError,
    EdgeBackend,
    NcnnBackend,
    Roi,
    StateDetector,
    Status,
)

DATA = pathlib.Path(os.environ["TARSIUS_DATA"])
SPLIT = os.environ.get("TARSIUS_SPLIT", "20261011")
SEEDS = 5


def load(camera):
    segments = json.loads((DATA / "segments.json").read_text())[camera]
    labels = json.loads((DATA / "labels.json").read_text())[camera]
    rows = []
    for index, segment in enumerate(segments):
        state = labels[str(index)]
        if state == "ignore":
            continue
        rows.extend(
            (index, name, state, (DATA / "raw" / camera / name).read_bytes())
            for name in segment["images"]
        )
    return rows


def run(rows, prepared, make_detector, budget, seed, *, ask):
    """
    Teach `budget` older segments per state, then classify the newest frames.

    With `ask`, every unsure frame is answered and taught, one question per
    segment, like confirming in a dialog.
    """
    older = {}
    for index, name, state, _ in rows:
        if name < SPLIT:
            older.setdefault(state, set()).add(index)
    rng = random.Random(seed)
    chosen = set()
    for ids in older.values():
        chosen |= set(rng.sample(sorted(ids), min(budget, len(ids))))
    detector = make_detector()
    for index, name, state, _ in rows:
        if index in chosen and name in prepared:
            detector.add(name, prepared[name], state=state)
    stats = {
        "seconds": 0.0,
        "n": 0,
        "wrong": 0,
        "unknown": 0,
        "questions": 0,
        "taught": len(detector.references()),
    }
    asked = set()
    for index, name, state, data in rows:
        if name < SPLIT:
            continue
        start = time.perf_counter()
        try:
            result = detector.classify(data)
        except BlankRegionError:
            result = None
        stats["seconds"] += time.perf_counter() - start
        stats["n"] += 1
        if result is None or result.status is not Status.CONFIDENT:
            stats["unknown"] += 1
            if ask and name in prepared:
                if index not in asked:
                    stats["questions"] += 1
                    asked.add(index)
                detector.add(name, prepared[name], state=state)
        elif result.state != state:
            stats["wrong"] += 1
    return stats


def main():
    rois_file = os.environ.get("TARSIUS_ROIS", DATA / "rois.json")
    rois = json.loads(pathlib.Path(rois_file).read_text())
    for camera in sys.argv[1:]:
        rows = load(camera)
        states = sorted({state for _, _, state, _ in rows})
        roi = Roi(*rois[camera])
        for name, factory in {"edges": EdgeBackend, "ncnn": NcnnBackend}.items():
            make_detector = functools.partial(StateDetector, factory(), roi, states)
            preparer = make_detector()
            prepared = {}
            for _, frame, _, data in rows:
                with contextlib.suppress(BlankRegionError):
                    prepared[frame] = preparer.prepare(data)
            print(f"{camera} {name}")
            for budget in (1, 2, 5):
                fixed = dict.fromkeys(
                    ("seconds", "n", "wrong", "unknown", "questions", "taught"), 0
                )
                asking = dict(fixed)
                for seed in range(SEEDS):
                    for total, ask in ((fixed, False), (asking, True)):
                        for key, value in run(
                            rows, prepared, make_detector, budget, seed, ask=ask
                        ).items():
                            total[key] += value
                per_state = fixed["taught"] / SEEDS / len(states)
                print(
                    f"  {budget} segments per state (~{per_state:.0f} frames each):"
                    f" wrong {fixed['wrong'] / fixed['n'] * 100:4.1f} %,"
                    f" unknown {fixed['unknown'] / fixed['n'] * 100:4.1f} %"
                    f" | asking: wrong {asking['wrong'] / asking['n'] * 100:4.1f} %,"
                    f" questions {asking['questions'] / SEEDS:.1f}"
                    f" | {fixed['seconds'] / fixed['n'] * 1000:.0f} ms per frame",
                    flush=True,
                )


if __name__ == "__main__":
    main()
