"""Combine several frames into one decision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .classifier import Result


def majority(results: Sequence[Result], *, min_share: float = 0.5) -> str | None:
    """
    Return the state picked by more than ``min_share`` of all frames.

    Unsure and not-ready frames count against every state, so a burst of
    unsure frames keeps the decision open instead of letting one confident
    frame win. ``min_share`` of at least 0.5 makes the winner unique.
    """
    if not 0.5 <= min_share < 1:
        raise ValueError("min_share must be in [0.5, 1)")
    if not results:
        return None
    votes = Counter(result.state for result in results if result.state is not None)
    if not votes:
        return None
    state, count = votes.most_common(1)[0]
    return state if count / len(results) > min_share else None
