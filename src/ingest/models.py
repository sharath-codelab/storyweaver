"""In-memory domain objects used by parsing, chunking, and ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Story:
    story_id: str
    source_filename: str
    display_title: str
    filename_title: str
    page_count: int
    character_count: int
    pages: tuple[str, ...]
    author_credit_raw: str
    illustrator_credit_raw: str
    authors: tuple[str, ...]
    illustrators: tuple[str, ...]
    license: str | None
    language: str | None
    translation: str | None
    content_sha256: str


@dataclass(frozen=True)
class Chunk:
    id: str
    story_id: str
    chunk_number: int
    page_start: int
    page_end: int
    text: str
    embedding_text: str
    metadata: dict[str, Any]

    def manifest_record(self) -> dict[str, Any]:
        return {"id": self.id, "embedding_text": self.embedding_text, "metadata": self.metadata}


def jsonable(value: Any) -> Any:
    """Convert dataclass-heavy values into JSON-compatible data."""
    return asdict(value) if hasattr(value, "__dataclass_fields__") else value
