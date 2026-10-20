"""
Fit a backend's temperature on recorded history and report how well it fits.

Uses the same data and split as history_split.py: references from older
segments (1, 2 and 5 per state, several seeds), scores of the newest frames.
The temperature minimizes the negative log likelihood of the true states.
Usage: calibrate.py <camera> [<camera> ...]
"""

import contextlib
import functools
import json
import os
import pathlib
import random
import sys

import numpy as np
from history_split import DATA, SEEDS, SPLIT, load

from tarsius import BlankRegionError, EdgeBackend, NcnnBackend, Roi, StateDetector

BUDGETS = (1, 2, 5)


def collect(rows, prepared, make_detector):
    """Return (scores, truth) of the newest frames for every budget and seed."""
    scores, truth = [], []
    for budget in BUDGETS:
        for seed in range(SEEDS):
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
            for _, name, state, data in rows:
                if name < SPLIT:
                    continue
                try:
                    result = detector.classify(data)
                except BlankRegionError:
                    continue
                states = sorted(result.scores)
                scores.append([result.scores[s] for s in states])
                truth.append(states.index(state))
    return np.array(scores), np.array(truth)


def softmax(scores, temperature):
    z = scores / temperature
    z -= z.max(axis=1, keepdims=True)
    weights = np.exp(z)
    return weights / weights.sum(axis=1, keepdims=True)


def fit(scores, truth):
    """Return the temperature with the lowest negative log likelihood."""
    rows = np.arange(len(truth))
    candidates = np.geomspace(1e-4, 1, 400)
    losses = [
        -np.mean(np.log(softmax(scores, t)[rows, truth] + 1e-12)) for t in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def calibration_error(probabilities, truth, bins=10):
    """Expected calibration error over equal-width confidence bins."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    error = 0.0
    for low in np.linspace(0, 1, bins, endpoint=False):
        mask = (confidence > low) & (confidence <= low + 1 / bins)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return error


def main():
    rois_file = os.environ.get("TARSIUS_ROIS", DATA / "rois.json")
    rois = json.loads(pathlib.Path(rois_file).read_text())
    for name, factory in {"edges": EdgeBackend, "ncnn": NcnnBackend}.items():
        backend = factory()
        all_scores, all_truth = [], []
        for camera in sys.argv[1:]:
            rows = load(camera)
            states = sorted({state for _, _, state, _ in rows})
            make_detector = functools.partial(
                StateDetector, Roi(*rois[camera]), states, backend=backend
            )
            preparer = make_detector()
            prepared = {}
            for _, frame, _, data in rows:
                with contextlib.suppress(BlankRegionError):
                    prepared[frame] = preparer.prepare(data)
            scores, truth = collect(rows, prepared, make_detector)
            all_scores.append(scores)
            all_truth.append(truth)
        scores = np.concatenate(all_scores)
        truth = np.concatenate(all_truth)
        temperature = fit(scores, truth)
        probabilities = softmax(scores, temperature)
        correct = probabilities.argmax(axis=1) == truth
        high = probabilities.max(axis=1) >= 0.9
        print(
            f"{name}: temperature {temperature:.3f} (backend uses"
            f" {backend.temperature}), {len(truth)} frames,"
            f" calibration error {calibration_error(probabilities, truth):.3f},"
            f" {high.mean() * 100:.1f} % at 0.9 or more, of which"
            f" {correct[high].mean() * 100:.1f} % right",
            flush=True,
        )


if __name__ == "__main__":
    main()
