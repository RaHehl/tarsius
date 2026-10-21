"""
Fit a backend's temperature on recorded history and check it on newer frames.

Same data as history_split.py. The temperature is fitted on the frames between
TARSIUS_FIT_SPLIT and TARSIUS_SPLIT, with references from before the first,
and checked on the frames after TARSIUS_SPLIT with references from before it.
References are 1, 2 and 5 segments per state over several seeds, so a frame
counts once per budget and seed. A camera without every state before the
split cannot be taught there and only counts in the check. The fit
minimizes the negative log likelihood of the true states.
Usage: calibrate.py <camera> [<camera> ...]
"""

import contextlib
import functools
import json
import os
import pathlib
import sys

import numpy as np
from history_split import DATA, SEEDS, SPLIT, load, teach

from tarsius import BlankRegionError, EdgeBackend, NcnnBackend, Roi, StateDetector

FIT_SPLIT = os.environ.get("TARSIUS_FIT_SPLIT", "20261003")
BUDGETS = (1, 2, 5)
CANDIDATES = np.geomspace(1e-4, 1, 400)


def collect(rows, prepared, make_detector, *, before, until=None):
    """Return (scores, truth) of the frames from `before` to `until`."""
    samples = []
    for budget in BUDGETS:
        for seed in range(SEEDS):
            detector = teach(rows, prepared, make_detector, budget, seed, before=before)
            if not detector.ready:
                continue
            for _, name, state, data in rows:
                if name < before or (until is not None and name >= until):
                    continue
                try:
                    result = detector.classify(data)
                except BlankRegionError:
                    continue
                states = sorted(result.scores)
                samples.append(
                    (np.array([result.scores[s] for s in states]), states.index(state))
                )
    return samples


def probabilities(scores, temperature):
    weights = np.exp((scores - scores.max()) / temperature)
    return weights / weights.sum()


def nll(samples, temperature):
    return -np.mean(
        [np.log(probabilities(s, temperature)[t] + 1e-12) for s, t in samples]
    )


def fit(samples):
    """Return the temperature with the lowest NLL, None at the edge of the grid."""
    best = int(np.argmin([nll(samples, t) for t in CANDIDATES]))
    return None if best in (0, len(CANDIDATES) - 1) else float(CANDIDATES[best])


def report(samples, temperature):
    """Return calibration error, share at 0.9 or more and accuracy there."""
    confidence, correct = [], []
    for scores, truth in samples:
        p = probabilities(scores, temperature)
        confidence.append(p.max())
        correct.append(p.argmax() == truth)
    confidence, correct = np.array(confidence), np.array(correct)
    error = 0.0
    for low in np.linspace(0, 1, 10, endpoint=False):
        mask = (confidence > low) & (confidence <= low + 0.1)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    high = confidence >= 0.9
    return error, high.mean(), correct[high].mean() if high.any() else float("nan")


def main():
    rois_file = os.environ.get("TARSIUS_ROIS", DATA / "rois.json")
    rois = json.loads(pathlib.Path(rois_file).read_text())
    for name, factory in {"edges": EdgeBackend, "ncnn": NcnnBackend}.items():
        backend = factory()
        fitting, checking = [], []
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
            fitting += collect(
                rows, prepared, make_detector, before=FIT_SPLIT, until=SPLIT
            )
            checking += collect(rows, prepared, make_detector, before=SPLIT)
        temperature = fit(fitting) if fitting else None
        if temperature is None:
            reason = "nothing to fit" if not fitting else "the fit hit the grid edge"
            print(f"{name}: no temperature, {reason}")
            continue
        error, high, right = report(checking, temperature)
        print(
            f"{name}: temperature {temperature:.3f} (backend uses"
            f" {backend.temperature}) from {len(fitting)} samples, checked on"
            f" {len(checking)}: calibration error {error:.3f}, {high * 100:.1f} %"
            f" at 0.9 or more, {right * 100:.2f} % of those right",
            flush=True,
        )


if __name__ == "__main__":
    main()
