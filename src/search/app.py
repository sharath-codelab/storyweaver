"""FastAPI application factory."""

from __future__ import annotations

import uuid
from secrets import compare_digest
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .calibrate_client import CalibrateTraceClient
from .config import SearchSettings
from .groq_client import GroqService
from .librarian import client_message_history, create_librarian_agent, run_librarian_turn
from .pinecone_client import PineconeSearch
from .schemas import ChatAgentOutput, ChatRequest, ChatResponse, DebugRecommendationResponse, RecommendationRequest, RecommendationResponse
from .service import RecommendationService


chat_bearer = HTTPBearer(auto_error=False)


def create_app(project_root: Path | None = None) -> FastAPI:
    root = project_root or Path.cwd()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = SearchSettings.from_environment(root)
        app.state.service = RecommendationService(
            GroqService(settings), PineconeSearch(settings), settings.rrf_k, settings.rerank_candidate_count
        )
        app.state.trace_client = CalibrateTraceClient.from_settings(settings)
        app.state.settings = settings
        app.state.librarian_agent = create_librarian_agent(settings)
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
    async def recommend(
        request: RecommendationRequest, raw_request: Request, background_tasks: BackgroundTasks,
    ) -> RecommendationResponse:
        debug_result = await run_in_threadpool(raw_request.app.state.service.recommend_debug, request)
        _queue_trace(background_tasks, raw_request.app.state.trace_client, request.input, debug_result)
        return RecommendationResponse(response=debug_result.response.message)

    @app.post("/v1/story-recommendations/debug", response_model=DebugRecommendationResponse)
    async def recommend_debug(
        request: RecommendationRequest, raw_request: Request, background_tasks: BackgroundTasks,
    ) -> DebugRecommendationResponse:
        """Unprotected diagnostic route; response includes complete rerank story text."""
        debug_result = await run_in_threadpool(raw_request.app.state.service.recommend_debug, request)
        _queue_trace(background_tasks, raw_request.app.state.trace_client, request.input, debug_result)
        return debug_result

    @app.post("/v1/story-chat", response_model=ChatResponse)
    async def story_chat(
        request: ChatRequest,
        raw_request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(chat_bearer),
    ) -> ChatResponse:
        _require_chat_token(credentials, raw_request.app.state.settings.chat_api_token)
        prior_messages = client_message_history(request.messages[:-1])
        try:
            result, _ = await run_in_threadpool(
                run_librarian_turn,
                raw_request.app.state.librarian_agent,
                raw_request.app.state.service,
                request.messages[-1].content,
                prior_messages,
            )
        except Exception:
            # A model or tool outage must not leak internals or turn a child's
            # chat message into an HTTP 500 response.
            result = ChatAgentOutput(
                type="clarification",
                response="I couldn't look through the stories just now. Could you try again in a moment?",
            )
        return ChatResponse(response=result.response)

    return app


def _queue_trace(
    background_tasks: BackgroundTasks,
    trace_client: CalibrateTraceClient | None,
    user_input: str,
    debug_result: DebugRecommendationResponse,
) -> None:
    if trace_client is not None:
        background_tasks.add_task(trace_client.send, user_input, debug_result.model_dump(mode="json"))


def _require_chat_token(credentials: HTTPAuthorizationCredentials | None, expected_token: str) -> None:
    """Reject missing or non-matching bearer credentials without leaking detail."""
    if credentials is None or credentials.scheme.lower() != "bearer" or not compare_digest(credentials.credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
