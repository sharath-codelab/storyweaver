"""Pinecone embedding, first-stage retrieval, and reranking operations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.ingest.embeddings import _get
from src.ingest.pinecone_store import PineconeStore

from .config import SearchSettings
from .schemas import ChunkMatch, StoryCandidate


class PineconeSearch:
    def __init__(self, settings: SearchSettings):
        self.settings = settings
        self.store = PineconeStore(settings.ingestion)
        self.index = self.store.connect_existing()
        self.client = self.store.client

    def retrieve(self, query: str, metadata_filter: dict | None) -> tuple[list[ChunkMatch], list[ChunkMatch]]:
        dense = list(self.client.inference.embed(
            model=self.settings.ingestion.dense_embedding_model,
            inputs=[query], parameters={"input_type": "query", "truncate": "END"},
        ))[0]
        sparse = list(self.client.inference.embed(
            model=self.settings.ingestion.sparse_embedding_model,
            inputs=[query], parameters={"input_type": "query", "truncate": "END"},
        ))[0]
        dense_values = list(_get(dense, "values"))
        sparse_vector = {"indices": list(_get(sparse, "sparse_indices")), "values": list(_get(sparse, "sparse_values"))}
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self.index.query, vector=dense_values, top_k=self.settings.dense_top_k,
                                           filter=metadata_filter, include_metadata=True, include_values=False,
                                           namespace=self.settings.ingestion.pinecone_namespace)
            # Pinecone's single-index hybrid design still requires a dense vector.
            # A zero vector makes this request sparse-only under dotproduct scoring.
            sparse_future = executor.submit(self.index.query, vector=[0.0] * self.settings.ingestion.dense_dimension,
                                            sparse_vector=sparse_vector, top_k=self.settings.sparse_top_k,
                                            filter=metadata_filter, include_metadata=True, include_values=False,
                                            namespace=self.settings.ingestion.pinecone_namespace)
            return self._matches(dense_future.result(), "dense"), self._matches(sparse_future.result(), "sparse")

    def rerank(self, query: str, candidates: list[StoryCandidate]) -> dict[str, float]:
        documents = [{"id": candidate.story_id, "text": self._rerank_text(candidate)} for candidate in candidates]
        result = self.client.inference.rerank(
            model=self.settings.pinecone_rerank_model,
            query=query,
            documents=documents,
            return_documents=False,
            parameters={"truncate": "END"},
        )
        scores: dict[str, float] = {}
        result_items = _get(result, "data") if hasattr(result, "data") or (isinstance(result, dict) and "data" in result) else result
        for item in result_items:
            index = _get(item, "index")
            score = float(_get(item, "score"))
            scores[documents[index]["id"]] = score
        return scores

    @staticmethod
    def _matches(response: Any, source: str) -> list[ChunkMatch]:
        raw_matches = _get(response, "matches")
        return [
            ChunkMatch(record_id=_get(match, "id"), source=source, rank=rank, score=float(_get(match, "score")),
                       metadata=dict(_get(match, "metadata")))
            for rank, match in enumerate(raw_matches, start=1)
        ]

    @staticmethod
    def _rerank_text(candidate: StoryCandidate) -> str:
        # Conservatively restrict content before the reranker performs its own model-specific truncation.
        content = candidate.chunk_text[:6000]
        if "." in content:
            content = content.rsplit(".", 1)[0] + "."
        return "\n".join((
            f"title: {candidate.title}", f"author: {candidate.author}",
            f"illustrator: {candidate.illustrator}",
            f"pages: {candidate.page_start}-{candidate.page_end} of {candidate.page_count}",
            f"content: {content}",
        ))
