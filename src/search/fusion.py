"""Deterministic reciprocal-rank fusion and story-level grouping."""

from __future__ import annotations

from collections import defaultdict

from .schemas import ChunkMatch, StoryCandidate


def fuse_chunks(dense: list[ChunkMatch], sparse: list[ChunkMatch], k: int) -> list[StoryCandidate]:
    contributions: dict[str, float] = defaultdict(float)
    matches: dict[str, ChunkMatch] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for result_list in (dense, sparse):
        for match in result_list:
            contributions[match.record_id] += 1 / (k + match.rank)
            matches.setdefault(match.record_id, match)
            ranks[match.record_id][match.source] = match.rank

    by_story: dict[str, StoryCandidate] = {}
    for record_id, score in contributions.items():
        match = matches[record_id]
        meta = match.metadata
        story_id = str(meta["story_id"])
        candidate = StoryCandidate(
            story_id=story_id,
            title=str(meta["display_title"]),
            author=str(meta["author_credit_raw"]),
            illustrator=str(meta["illustrator_credit_raw"]),
            page_count=int(meta["page_count"]),
            character_count=int(meta["character_count"]) if "character_count" in meta else None,
            page_start=int(meta["page_start"]),
            page_end=int(meta["page_end"]),
            chunk_text=str(meta["chunk_text"]),
            rrf_score=score,
            dense_rank=ranks[record_id].get("dense"),
            sparse_rank=ranks[record_id].get("sparse"),
        )
        current = by_story.get(story_id)
        if current is None or _sort_key(candidate) < _sort_key(current):
            by_story[story_id] = candidate
    return sorted(by_story.values(), key=_sort_key)


def _sort_key(candidate: StoryCandidate) -> tuple:
    return (
        -candidate.rrf_score,
        candidate.dense_rank if candidate.dense_rank is not None else 10**9,
        candidate.sparse_rank if candidate.sparse_rank is not None else 10**9,
        candidate.story_id,
    )


def rerank_order(candidates: list[StoryCandidate]) -> list[StoryCandidate]:
    return sorted(
        candidates,
        key=lambda item: (-(item.rerank_score or float("-inf")),) + _sort_key(item),
    )
