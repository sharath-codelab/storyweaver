"""Pinecone index lifecycle and record persistence."""

from __future__ import annotations

import time
from typing import Any, Iterable

from .config import Settings


class PineconeStore:
    def __init__(self, settings: Settings):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as error:
            raise RuntimeError("Install dependencies with: python -m pip install -r requirements.txt") from error
        self.settings = settings
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self._serverless_spec = ServerlessSpec
        self.index: Any | None = None

    def ensure_index(self) -> Any:
        if not self.client.has_index(self.settings.pinecone_index_name):
            self.client.create_index(
                name=self.settings.pinecone_index_name,
                vector_type="dense",
                dimension=self.settings.dense_dimension,
                metric="dotproduct",
                spec=self._serverless_spec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
                deletion_protection="disabled",
                tags={"project": "swv2", "purpose": "story-ingestion"},
            )
        return self._connect_existing()

    def connect_existing(self) -> Any:
        """Connect to an existing compatible index without creating resources."""
        if not self.client.has_index(self.settings.pinecone_index_name):
            raise RuntimeError(f"Pinecone index does not exist: {self.settings.pinecone_index_name}")
        return self._connect_existing()

    def _connect_existing(self) -> Any:
        description = self.client.describe_index(self.settings.pinecone_index_name)
        metric = _get(description, "metric")
        dimension = _get(description, "dimension")
        if metric != "dotproduct" or dimension != self.settings.dense_dimension:
            raise RuntimeError(
                f"Existing index is incompatible (metric={metric}, dimension={dimension}); "
                "refusing to write"
            )
        for _ in range(60):
            status = _get(description, "status")
            if _get(status, "ready"):
                break
            time.sleep(2)
            description = self.client.describe_index(self.settings.pinecone_index_name)
        else:
            raise TimeoutError("Pinecone index did not become ready within 120 seconds")
        host = _get(description, "host")
        self.index = self.client.Index(host=host)
        return self.index

    def delete_story(self, story_id: str) -> None:
        self._require_index().delete(
            namespace=self.settings.pinecone_namespace,
            filter={"story_id": {"$eq": story_id}},
        )

    def upsert_records(self, records: Iterable[dict[str, Any]]) -> int:
        record_list = list(records)
        for offset in range(0, len(record_list), self.settings.upsert_batch_size):
            self._require_index().upsert(
                vectors=record_list[offset : offset + self.settings.upsert_batch_size],
                namespace=self.settings.pinecone_namespace,
            )
        return len(record_list)

    def namespace_count(self) -> int:
        stats = self._require_index().describe_index_stats()
        namespaces = _get(stats, "namespaces")
        value = namespaces.get(self.settings.pinecone_namespace)
        if value is None:
            return 0
        return _get(value, "vector_count")

    def _require_index(self) -> Any:
        if self.index is None:
            raise RuntimeError("Call ensure_index before data operations")
        return self.index


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value[key]
    return getattr(value, key)
