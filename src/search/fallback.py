"""Safe deterministic response formatting for provider failures."""

from __future__ import annotations

from .schemas import Recommendation, StoryCandidate


def recommendation_from_candidate(candidate: StoryCandidate, reason: str | None = None) -> Recommendation:
    explanation = reason or "It has a story idea that matches what you asked for."
    return Recommendation(
        title=candidate.title,
        line=f"{candidate.title} — {explanation}",
    )
