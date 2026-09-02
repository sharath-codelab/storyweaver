"""Configuration loading for the ingestion command."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when required ingestion configuration is absent or invalid."""


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the shell environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    project_root: Path
    stories_dir: Path
    artifacts_dir: Path
    pinecone_api_key: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_index_name: str
    pinecone_namespace: str
    dense_embedding_model: str
    sparse_embedding_model: str
    dense_dimension: int
    chunk_target_tokens: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    embed_batch_size: int
    upsert_batch_size: int

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        load_dotenv(project_root / ".env")
        target = _positive_int("CHUNK_TARGET_TOKENS", 550)
        maximum = _positive_int("CHUNK_MAX_TOKENS", 700)
        if target > maximum:
            raise ConfigurationError("CHUNK_TARGET_TOKENS cannot exceed CHUNK_MAX_TOKENS")
        return cls(
            project_root=project_root,
            stories_dir=project_root / os.getenv("STORIES_DIR", "stories"),
            artifacts_dir=project_root / os.getenv("INGEST_ARTIFACTS_DIR", ".local/ingestion"),
            pinecone_api_key=_required("PINECONE_API_KEY"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "swv2"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "stories"),
            dense_embedding_model=os.getenv("DENSE_EMBEDDING_MODEL", "llama-text-embed-v2"),
            sparse_embedding_model=os.getenv("SPARSE_EMBEDDING_MODEL", "pinecone-sparse-english-v0"),
            dense_dimension=_positive_int("DENSE_EMBEDDING_DIMENSION", 1024),
            chunk_target_tokens=target,
            chunk_max_tokens=maximum,
            chunk_overlap_tokens=_positive_int("CHUNK_OVERLAP_TOKENS", 75),
            embed_batch_size=_positive_int("EMBED_BATCH_SIZE", 64),
            upsert_batch_size=_positive_int("UPSERT_BATCH_SIZE", 100),
        )
