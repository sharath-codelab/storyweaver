"""Validated HTTP and internal models for story recommendations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Confidence = Literal["none", "low", "medium", "high"]
LengthPreference = Literal["very_short", "short", "medium", "long"]


class RecommendationRequest(BaseModel):
    input: str = Field(min_length=1, max_length=500)

    @field_validator("input")
    @classmethod
    def strip_text(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("must not be blank")
        return result


class AgeRange(BaseModel):
    min: int = Field(ge=0, le=18)
    max: int = Field(ge=0, le=18)


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_query: str = Field(min_length=1, max_length=500)
    age_range: AgeRange | None = None
    age_confidence: Confidence = "none"
    length_preference: LengthPreference | None = None
    length_confidence: Confidence = "none"

    @field_validator("search_query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("search_query must not be blank")
        return value.strip()


class Recommendation(BaseModel):
    title: str
    line: str = Field(min_length=1, max_length=600)


class RecommendationResponse(BaseModel):
    response: str


class ChatMessage(BaseModel):
    """One client-held message in a story-chat transcript."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=500)

    @field_validator("content")
    @classmethod
    def strip_text(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("must not be blank")
        return result


class ChatRequest(BaseModel):
    """A complete client-held transcript ending in the new user prompt."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=20)

    @field_validator("messages")
    @classmethod
    def final_message_must_be_user(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if value[-1].role != "user":
            raise ValueError("the final message must have role 'user'")
        return value


class ChatResponse(BaseModel):
    response: str


class ChatAgentOutput(BaseModel):
    """The only response shape the librarian agent may return to FastAPI."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["clarification", "recommendations"]
    response: str = Field(min_length=1, max_length=1600)
    # The endpoint keeps this internal, but the runner uses it to prove that
    # every recommendation came from a search tool result.
    selected_story_ids: list[str] = Field(default_factory=list, max_length=2)


class ChatStoryCandidate(BaseModel):
    """Grounded, compact story data that is safe to provide to the agent."""

    story_id: str
    title: str
    author: str
    illustrator: str
    page_count: int
    excerpt: str = Field(min_length=1, max_length=1200)


class DebugRecommendationPayload(BaseModel):
    """Full, intentionally verbose trace of one recommendation request."""

    message: str
    analysis: QueryAnalysis
    analysis_fallback: bool
    retrieval: dict[str, Any]
    fusion: dict[str, Any]
    reranking: dict[str, Any]
    finalization: dict[str, Any]


class DebugRecommendationResponse(BaseModel):
    response: DebugRecommendationPayload


class ChunkMatch(BaseModel):
    record_id: str
    source: Literal["dense", "sparse"]
    rank: int
    score: float
    metadata: dict


class StoryCandidate(BaseModel):
    story_id: str
    title: str
    author: str
    illustrator: str
    page_count: int
    character_count: int | None
    page_start: int
    page_end: int
    chunk_text: str
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class WritingChoice(BaseModel):
    story_id: str
    why_recommended: str = Field(min_length=1, max_length=500)


class WritingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    introduction: str = Field(min_length=1, max_length=300)
    selections: list[WritingChoice]
