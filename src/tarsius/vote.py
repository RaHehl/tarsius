"""Combine several frames into one decision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .classifier import Result, Status


def majority(results: Sequence[Result], *, min_share: float = 0.5) -> Result:
    """
    Combine results of several frames, e.g. a burst after a trigger.

    A state wins when more than ``min_share`` of all frames picked it.
    Unsure frames count against every state, so a burst of unsure frames
    stays unsure instead of letting one confident frame win; the most voted
    state is still the candidate. ``scores`` hold the share of each state and
    ``margin`` the winner's share. Not ready only when every frame was.
    """
    if not 0.5 <= min_share < 1:
        raise ValueError("min_share must be in [0.5, 1)")
    if all(result.status is Status.NOT_READY for result in results):
        return Result(Status.NOT_READY)
    votes = Counter(result.state for result in results if result.state is not None)
    candidates = Counter(
        result.candidate for result in results if result.candidate is not None
    )
    shares = {state: count / len(results) for state, count in votes.items()}
    if votes:
        best, count = votes.most_common(1)[0]
        share = count / len(results)
        if share > min_share:
            return Result(Status.CONFIDENT, best, best, share, shares)
    candidate = candidates.most_common(1)[0][0] if candidates else None
    return Result(
        Status.UNSURE, None, candidate, max(shares.values(), default=0.0), shares
    )
