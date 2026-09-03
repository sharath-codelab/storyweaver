"""Orchestrates safe recommendation generation."""

from __future__ import annotations

from .fallback import recommendation_from_candidate
from .fusion import fuse_chunks, rerank_order
from .groq_client import GroqService
from .pinecone_client import PineconeSearch
from .schemas import (
    DebugRecommendationPayload,
    DebugRecommendationResponse,
    QueryAnalysis,
    RecommendationRequest,
    RecommendationResponse,
)

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
        message, _ = self._recommend(request, include_debug=False)
        return RecommendationResponse(response=message)

    def recommend_debug(self, request: RecommendationRequest) -> DebugRecommendationResponse:
        message, trace = self._recommend(request, include_debug=True)
        return DebugRecommendationResponse(response=DebugRecommendationPayload(message=message, **trace))

    def _recommend(self, request: RecommendationRequest, include_debug: bool) -> tuple[str, dict]:
        analysis, analysis_fallback = self._analysis(request.input)
        count_range = self._length_range(analysis)
        metadata_filter = self._filter("en", count_range)
        dense, sparse = self.pinecone.retrieve(analysis.search_query, metadata_filter)
        candidates = fuse_chunks(dense, sparse, self.rrf_k)
        initial_candidates = candidates
        attempts = [{
            "metadata_filter": metadata_filter,
            "dense_matches": dense,
            "sparse_matches": sparse,
        }]
        if count_range and len(candidates) < 5:
            relaxed_filter = self._filter("en", None)
            dense, sparse = self.pinecone.retrieve(analysis.search_query, relaxed_filter)
            candidates = fuse_chunks(dense, sparse, self.rrf_k)
            attempts.append({
                "metadata_filter": relaxed_filter,
                "dense_matches": dense,
                "sparse_matches": sparse,
            })
        fused_candidates = candidates
        candidates = candidates[: self.rerank_candidate_count]
        rerank_candidates = candidates
        rerank_applied = False
        rerank_documents: list[dict[str, str]] = []
        documents_for_rerank: list[dict[str, str]] | None = None
        if include_debug:
            try:
                rerank_documents = self._rerank_documents(candidates)
                documents_for_rerank = rerank_documents
            except Exception:
                # The actual rerank still gets its usual chance below. If building
                # debug-only documents fails, retain the service's fallback path.
                pass
        try:
            if documents_for_rerank is None:
                scores = self.pinecone.rerank(request.input, candidates)
            else:
                scores = self.pinecone.rerank(request.input, candidates, documents_for_rerank)
            candidates = [candidate.model_copy(update={"rerank_score": scores.get(candidate.story_id)}) for candidate in candidates]
            candidates = rerank_order(candidates)
            rerank_applied = True
        except Exception:
            scores = {}
        top_five = candidates[:5]
        rendered, writing_result, writing_fallback = self._write(request.input, top_five, 2)
        if rendered is None:
            message = "Mujhe abhi aapke liye sahi kahani nahi mili, dost. Kripya ek aur kahani ka idea bataiye!"
            return message, self._debug_trace(
                analysis, analysis_fallback, attempts, initial_candidates, fused_candidates, rerank_candidates,
                rerank_documents, rerank_applied, scores, candidates, top_five, writing_result, writing_fallback,
                [],
            ) if include_debug else {}
        introduction, lines = rendered
        message = introduction + "\n\n" + "\n".join(f"• {item.line}" for item in lines)
        return message, self._debug_trace(
            analysis, analysis_fallback, attempts, initial_candidates, fused_candidates, rerank_candidates,
            rerank_documents, rerank_applied, scores, candidates, top_five, writing_result, writing_fallback,
            lines,
        ) if include_debug else {}

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
            return None, None, False
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
                return (written.introduction, selected), written, False
        except Exception:
            written = None
        return (
            "What a lovely idea! I found these stories that may be just right for you:",
            [recommendation_from_candidate(item) for item in candidates[:limit]],
        ), written, True

    def _rerank_documents(self, candidates: list) -> list[dict[str, str]]:
        """Read the exact rerank payload when the configured backend exposes it."""
        build_documents = getattr(self.pinecone, "rerank_documents", None)
        return build_documents(candidates) if callable(build_documents) else []

    @staticmethod
    def _debug_trace(
        analysis, analysis_fallback, attempts, initial_candidates, fused_candidates, rerank_candidates,
        rerank_documents, rerank_applied, scores, ordered_candidates, top_five, writing_result,
        writing_fallback, recommendations,
    ) -> dict:
        return {
            "analysis": analysis,
            "analysis_fallback": analysis_fallback,
            "retrieval": {
                "attempts": attempts,
                "length_filter_applied": len(attempts[0]["metadata_filter"].get("$and", [])) > 1,
                "filter_relaxed": len(attempts) > 1,
            },
            "fusion": {
                "initial_candidates": initial_candidates,
                "final_candidates": fused_candidates,
                "rerank_candidates": rerank_candidates,
            },
            "reranking": {
                "applied": rerank_applied,
                "documents": rerank_documents,
                "scores": scores,
                "ordered_candidates": ordered_candidates,
            },
            "finalization": {
                "top_five": top_five,
                "writing_result": writing_result,
                "writing_fallback": writing_fallback,
                "recommendations": recommendations,
            },
        }
