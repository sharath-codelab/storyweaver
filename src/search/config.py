"""Configuration for the search API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.ingest.config import ConfigurationError, Settings, load_dotenv


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class SearchSettings:
    ingestion: Settings
    groq_api_key: str
    groq_analysis_model: str
    groq_writing_model: str
    pinecone_rerank_model: str
    dense_top_k: int
    sparse_top_k: int
    rerank_candidate_count: int
    rrf_k: int
    calibrate_api_key: str | None
    calibrate_agent_id: str | None
    calibrate_trace_timeout_seconds: float

    @classmethod
    def from_environment(cls, project_root: Path) -> "SearchSettings":
        load_dotenv(project_root / ".env")
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_key:
            raise ConfigurationError("Missing required environment variable: GROQ_API_KEY")
        return cls(
            ingestion=Settings.from_environment(project_root),
            groq_api_key=groq_key,
            groq_analysis_model=os.getenv("GROQ_ANALYSIS_MODEL", "openai/gpt-oss-20b"),
            groq_writing_model=os.getenv("GROQ_WRITING_MODEL", "openai/gpt-oss-20b"),
            pinecone_rerank_model=os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3"),
            dense_top_k=_positive_int("DENSE_TOP_K", 50),
            sparse_top_k=_positive_int("SPARSE_TOP_K", 50),
            rerank_candidate_count=_positive_int("RERANK_CANDIDATE_COUNT", 20),
            rrf_k=_positive_int("RRF_K", 60),
            calibrate_api_key=os.getenv("CALIBRATE_API_KEY", "").strip() or None,
            calibrate_agent_id=os.getenv("CALIBRATE_AGENT_ID", "").strip() or None,
            calibrate_trace_timeout_seconds=_positive_float("CALIBRATE_TRACE_TIMEOUT_SECONDS", 2.0),
        )
