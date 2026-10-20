"""Combine several frames into one decision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .classifier import Result, Status


def majority(results: Sequence[Result], *, min_share: float = 0.5) -> Result:
    """
    Combine results of several frames, e.g. a burst after a trigger.

    A state wins when more than ``min_share`` of all frames picked it
    confidently. Unsure and not ready frames count against every state, so a
    burst of unsure frames stays unsure instead of letting one confident frame
    win. Not ready when at least half of the frames were.

    ``scores`` hold the confident share of every state seen in the results,
    ``candidate`` is the state most frames leaned to, confident or not, ties
    going to the earliest frame, ``margin`` is the confident share of the
    reported state or candidate, and ``probabilities`` are the mean of the
    frames' probabilities.
    """
    if not 0.5 <= min_share < 1:
        raise ValueError("min_share must be in [0.5, 1)")
    not_ready = sum(result.status is Status.NOT_READY for result in results)
    if 2 * not_ready >= len(results):
        return Result(Status.NOT_READY)
    shares = {state: 0.0 for result in results for state in result.scores}
    votes = Counter(result.state for result in results if result.state is not None)
    for state, count in votes.items():
        shares[state] = count / len(results)
    probabilities = _mean_probabilities(results)
    if votes:
        best, count = votes.most_common(1)[0]
        if count / len(results) > min_share:
            return Result(
                Status.CONFIDENT,
                best,
                best,
                shares[best],
                shares,
                probabilities=probabilities,
            )
    leanings = Counter(
        result.state or result.candidate
        for result in results
        if result.state or result.candidate
    )
    candidate = leanings.most_common(1)[0][0] if leanings else None
    margin = shares.get(candidate, 0.0) if candidate is not None else 0.0
    return Result(
        Status.UNSURE, None, candidate, margin, shares, probabilities=probabilities
    )


def _mean_probabilities(results: Sequence[Result]) -> dict[str, float]:
    with_probabilities = [result for result in results if result.probabilities]
    totals = {
        state: 0.0 for result in with_probabilities for state in result.probabilities
    }
    for result in with_probabilities:
        for state, probability in result.probabilities.items():
            totals[state] += probability / len(with_probabilities)
    return totals
