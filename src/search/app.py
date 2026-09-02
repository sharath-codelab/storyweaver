"""FastAPI application factory."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

from .config import SearchSettings
from .groq_client import GroqService
from .pinecone_client import PineconeSearch
from .schemas import RecommendationRequest, RecommendationResponse
from .service import RecommendationService


def create_app(project_root: Path | None = None) -> FastAPI:
    root = project_root or Path.cwd()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = SearchSettings.from_environment(root)
        app.state.service = RecommendationService(
            GroqService(settings), PineconeSearch(settings), settings.rrf_k, settings.rerank_candidate_count
        )
        yield

    app = FastAPI(title="SWV2 Story Recommendations", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/story-recommendations", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest, raw_request: Request) -> RecommendationResponse:
        return await run_in_threadpool(raw_request.app.state.service.recommend, request)

    return app
