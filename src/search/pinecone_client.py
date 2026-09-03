"""Pinecone embedding, first-stage retrieval, and reranking operations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.ingest.embeddings import _get
from src.ingest.parse_stories import CorpusValidationError, parse_story
from src.ingest.pinecone_store import PineconeStore

from .config import SearchSettings
from .schemas import ChunkMatch, StoryCandidate


class PineconeSearch:
    def __init__(self, settings: SearchSettings):
        self.settings = settings
        self.store = PineconeStore(settings.ingestion)
        self.index = self.store.connect_existing()
        self.client = self.store.client
        self._story_text_cache: dict[str, str | None] = {}

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

    def rerank(
        self, query: str, candidates: list[StoryCandidate], documents: list[dict[str, str]] | None = None,
    ) -> dict[str, float]:
        documents = documents if documents is not None else self.rerank_documents(candidates)
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

    def rerank_documents(self, candidates: list[StoryCandidate]) -> list[dict[str, str]]:
        """Build the exact documents supplied to the reranker.

        Kept public so the debug endpoint can report the actual rerank input
        without changing the production reranking behavior.
        """
        return [
            {"id": candidate.story_id, "text": self._rerank_text(candidate, self._full_story_text(candidate.story_id))}
            for candidate in candidates
        ]

    @staticmethod
    def _matches(response: Any, source: str) -> list[ChunkMatch]:
        raw_matches = _get(response, "matches")
        return [
            ChunkMatch(record_id=_get(match, "id"), source=source, rank=rank, score=float(_get(match, "score")),
                       metadata=dict(_get(match, "metadata")))
            for rank, match in enumerate(raw_matches, start=1)
        ]

    def _full_story_text(self, story_id: str) -> str | None:
        """Load canonical story content for a rerank candidate, caching misses too."""
        if story_id in self._story_text_cache:
            return self._story_text_cache[story_id]

        stories_dir = self.settings.ingestion.stories_dir.resolve()
        story_path = (stories_dir / f"{story_id}.md").resolve()
        content: str | None = None
        # story_id originates in indexed metadata, so keep it from escaping the corpus
        # even if an invalid record is present in the index.
        if story_path.parent == stories_dir:
            try:
                story = parse_story(story_path)
                if story.story_id == story_id:
                    content = "\n\n".join(story.pages)
            except (CorpusValidationError, OSError):
                # Preserve search availability when the source corpus and index drift.
                # The retrieved chunk remains a grounded fallback below.
                pass
        self._story_text_cache[story_id] = content
        return content

    @staticmethod
    def _rerank_text(candidate: StoryCandidate, full_story_text: str | None = None) -> str:
        content = full_story_text or candidate.chunk_text
        page_start = 1 if full_story_text else candidate.page_start
        page_end = candidate.page_count if full_story_text else candidate.page_end
        return "\n".join((
            f"title: {candidate.title}", f"author: {candidate.author}",
            f"illustrator: {candidate.illustrator}",
            f"pages: {page_start}-{page_end} of {candidate.page_count}",
            f"content: {content}",
        ))
