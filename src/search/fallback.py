"""Safe deterministic response formatting for provider failures."""

from __future__ import annotations

from .schemas import Recommendation, StoryCandidate


def recommendation_from_candidate(candidate: StoryCandidate, reason: str | None = None) -> Recommendation:
    explanation = reason or "Its story sounds like it could make a lovely next read for you."
    return Recommendation(
        title=candidate.title,
        line=f"{candidate.title} — {explanation}",
    )
