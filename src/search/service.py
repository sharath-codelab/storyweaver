"""Orchestrates safe recommendation generation."""

from __future__ import annotations

from .fallback import recommendation_from_candidate
from .fusion import fuse_chunks, rerank_order
from .groq_client import GroqService
from .pinecone_client import PineconeSearch
from .schemas import QueryAnalysis, RecommendationRequest, RecommendationResponse

LENGTH_RANGES = {
    "very_short": (0, 1000), "short": (1000, 5000), "medium": (5000, 12000), "long": (12000, 25001),
}


class RecommendationService:
    def __init__(self, groq: GroqService, pinecone: PineconeSearch, rrf_k: int, rerank_candidate_count: int):
        self.groq = groq
        self.pinecone = pinecone
        self.rrf_k = rrf_k
        self.rerank_candidate_count = rerank_candidate_count

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        analysis, analysis_fallback = self._analysis(request.input)
        count_range = self._length_range(analysis)
        metadata_filter = self._filter("en", count_range)
        dense, sparse = self.pinecone.retrieve(analysis.search_query, metadata_filter)
        candidates = fuse_chunks(dense, sparse, self.rrf_k)
        relaxed = False
        if count_range and len(candidates) < 5:
            dense, sparse = self.pinecone.retrieve(analysis.search_query, self._filter("en", None))
            candidates = fuse_chunks(dense, sparse, self.rrf_k)
            relaxed = True
        candidates = candidates[: self.rerank_candidate_count]
        rerank_applied = False
        try:
            scores = self.pinecone.rerank(request.input, candidates)
            candidates = [candidate.model_copy(update={"rerank_score": scores.get(candidate.story_id)}) for candidate in candidates]
            candidates = rerank_order(candidates)
            rerank_applied = True
        except Exception:
            pass
        top_five = candidates[:5]
        recommendations = self._write(request.input, top_five, 2)
        if not recommendations:
            return RecommendationResponse(output="Mujhe abhi aapke liye sahi kahani nahi mili. Kripya ek aur idea bataiye!")
        return RecommendationResponse(output="\n".join(f"• {item.line}" for item in recommendations))

    def _analysis(self, query: str) -> tuple[QueryAnalysis, bool]:
        try:
            return self.groq.analyse(query), False
        except Exception:
            return QueryAnalysis(search_query=query), True

    @staticmethod
    def _length_range(analysis: QueryAnalysis) -> tuple[int, int] | None:
        if analysis.length_confidence not in {"medium", "high"} or analysis.length_preference is None:
            return None
        return LENGTH_RANGES[analysis.length_preference]

    @staticmethod
    def _filter(language: str, count_range: tuple[int, int] | None) -> dict:
        conditions = [{"language": {"$eq": language}}]
        if count_range:
            conditions.extend(({"character_count": {"$gte": count_range[0]}}, {"character_count": {"$lt": count_range[1]}}))
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _write(self, query: str, candidates: list, limit: int):
        if not candidates:
            return []
        try:
            written = self.groq.write(query, candidates, limit)
            by_id = {item.story_id: item for item in candidates}
            selected = []
            for choice in written.selections:
                candidate = by_id.get(choice.story_id)
                if candidate and all(item.story_id != choice.story_id for item in selected):
                    selected.append(recommendation_from_candidate(candidate, choice.why_recommended))
                if len(selected) == limit:
                    break
            if selected:
                return selected
        except Exception:
            pass
        return [recommendation_from_candidate(item) for item in candidates[:limit]]
