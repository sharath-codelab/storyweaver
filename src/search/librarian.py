"""The small PydanticAI orchestration layer for conversational story search.

The model decides *when* to search. It never talks to Pinecone directly: the
single function tool below is the only route to the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from .config import SearchSettings
from .schemas import ChatAgentOutput, ChatMessage, ChatStoryCandidate

if TYPE_CHECKING:
    from .service import RecommendationService


INSTRUCTIONS = """You are a warm children's librarian.

Help children find stories from the supplied story collection. For greetings,
vague requests, irrelevant messages, or unresolved follow-ups, return a short,
friendly clarification question with type "clarification" and do not call a
tool.

For every usable story request, call search_stories before answering. The tool
returns the only stories you may recommend. Recommend at most two stories and
base each reason only on the tool's title and excerpt. Never invent story
details, IDs, links, rankings, or facts outside the tool result. Return type
"recommendations" after a search and put the one or two chosen tool story IDs
in selected_story_ids. For a clarification, selected_story_ids must be empty.
If a search returns no stories, return a friendly clarification asking for a
different story idea instead of naming a story.

Use previous conversation and tool messages to resolve follow-ups such as
"make it shorter". If no prior request makes a short follow-up meaningful, ask
one clarification question instead."""


@dataclass
class LibrarianDependencies:
    """Per-run state injected into tools; it is never shown to the model."""

    service: RecommendationService
    search_was_called: bool = False
    allowed_story_ids: set[str] | None = None


def create_librarian_agent(settings: SearchSettings) -> Agent[LibrarianDependencies, ChatAgentOutput]:
    """Build the one-agent orchestrator used by the chat endpoint.

    The current deployment already has a Groq key, so PydanticAI's Groq provider
    keeps model access unchanged while adding tool calling and message history.
    """
    model = GroqModel(
        settings.groq_writing_model,
        provider=GroqProvider(api_key=settings.groq_api_key),
    )
    agent = Agent(
        model,
        deps_type=LibrarianDependencies,
        output_type=ChatAgentOutput,
        instructions=INSTRUCTIONS,
        retries=1,
    )

    @agent.tool
    def search_stories(ctx: RunContext[LibrarianDependencies], request: str) -> list[ChatStoryCandidate]:
        """Search the collection for a complete, standalone child story request."""
        # This call owns all deterministic retrieval safeguards: filters,
        # dense/sparse fusion, reranking, and the maximum of five candidates.
        ctx.deps.search_was_called = True
        candidates = ctx.deps.service.candidates_for_agent(request)
        ctx.deps.allowed_story_ids = {candidate.story_id for candidate in candidates}
        return candidates

    return agent


def run_librarian_turn(
    agent: Agent[LibrarianDependencies, ChatAgentOutput],
    service: RecommendationService,
    user_input: str,
    message_history: list[Any],
) -> tuple[ChatAgentOutput, list[Any]]:
    """Run one synchronous agent turn and return only this turn's messages."""
    dependencies = LibrarianDependencies(service=service)
    result = agent.run_sync(user_input, deps=dependencies, message_history=message_history)
    output = result.output
    # Instructions alone are not a security boundary. Reject an ungrounded
    # recommendation if a model skips the required search tool.
    if output.type == "recommendations" and not dependencies.search_was_called:
        raise RuntimeError("librarian returned recommendations without searching")
    if output.type == "clarification" and output.selected_story_ids:
        raise RuntimeError("librarian attached stories to a clarification")
    if output.type == "recommendations":
        allowed_story_ids = dependencies.allowed_story_ids or set()
        if not output.selected_story_ids or not set(output.selected_story_ids).issubset(allowed_story_ids):
            raise RuntimeError("librarian selected a story outside the search result")
    return output, result.new_messages()


def client_message_history(messages: list[ChatMessage]) -> list[ModelMessage]:
    """Translate client-visible text turns into PydanticAI model history.

    Tool calls and results intentionally never cross the HTTP boundary. The
    agent can make fresh grounded tool calls for the current user request.
    """
    history: list[ModelMessage] = []
    for message in messages:
        if message.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=message.content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=message.content)]))
    return history
