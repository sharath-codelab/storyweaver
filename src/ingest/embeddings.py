"""Pinecone-hosted dense and sparse embedding generation."""

from __future__ import annotations

from typing import Any

from .config import Settings
from .models import Chunk


def _get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)


class PineconeEmbedder:
    def __init__(self, client: Any, settings: Settings):
        self.client = client
        self.settings = settings

    def embed_chunks(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for offset in range(0, len(chunks), self.settings.embed_batch_size):
            batch = chunks[offset : offset + self.settings.embed_batch_size]
            texts = [chunk.embedding_text for chunk in batch]
            dense = self.client.inference.embed(
                model=self.settings.dense_embedding_model,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            sparse = self.client.inference.embed(
                model=self.settings.sparse_embedding_model,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            dense_items = list(dense)
            sparse_items = list(sparse)
            if len(dense_items) != len(batch) or len(sparse_items) != len(batch):
                raise RuntimeError("Embedding provider returned an unexpected number of vectors")
            for chunk, dense_item, sparse_item in zip(batch, dense_items, sparse_items):
                values = list(_get(dense_item, "values"))
                indices = list(_get(sparse_item, "sparse_indices"))
                sparse_values = list(_get(sparse_item, "sparse_values"))
                if len(values) != self.settings.dense_dimension:
                    raise RuntimeError(
                        f"{chunk.id}: dense dimension {len(values)} does not match configured "
                        f"dimension {self.settings.dense_dimension}"
                    )
                if not indices or len(indices) != len(sparse_values):
                    raise RuntimeError(f"{chunk.id}: invalid sparse embedding")
                records.append(
                    {
                        "id": chunk.id,
                        "values": values,
                        "sparse_values": {"indices": indices, "values": sparse_values},
                        "metadata": chunk.metadata,
                    }
                )
        return records
