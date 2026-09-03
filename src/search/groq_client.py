"""Groq calls for structured analysis and grounded recommendation wording."""

from __future__ import annotations

import json
from typing import Any

from .config import SearchSettings
from .schemas import QueryAnalysis, StoryCandidate, WritingResult


class GroqService:
    def __init__(self, settings: SearchSettings):
        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError("Install dependencies with: uv sync or uv run --with-requirements requirements.txt") from error
        self.client = Groq(api_key=settings.groq_api_key)
        self.settings = settings

    def analyse(self, query: str) -> QueryAnalysis:
        schema = QueryAnalysis.model_json_schema()
        prompt = (
            "Analyze this child story request. Return only the requested JSON. "
            "Infer age only from explicit age wording or strong direct evidence; otherwise use null and none. "
            "Infer length only from explicit short, quick, long, or chapter wording; age alone is not enough. "
            "Keep useful query terms in search_query. Request: " + query
        )
        result = self.client.chat.completions.create(
            model=self.settings.groq_analysis_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_schema", "json_schema": {"name": "query_analysis", "schema": schema}},
        )
        return QueryAnalysis.model_validate(json.loads(result.choices[0].message.content or "{}"))

    def write(self, query: str, candidates: list[StoryCandidate], limit: int) -> WritingResult:
        allowed = [item.story_id for item in candidates]
        candidate_data = [
            {
                "story_id": item.story_id,
                "title": item.title,
                "author": item.author,
                "illustrator": item.illustrator,
                "page_count": item.page_count,
                "character_count": item.character_count,
                "snippet": item.chunk_text[:1200],
            }
            for item in candidates
        ]
        prompt = (
            "You are a thoughtful, warm children's librarian helping a child find their next story. "
            "First, silently compare every supplied candidate against the child's request. Consider the requested theme, "
            "mood, characters, reading length, and age cues when those details are present. Then choose the one or two "
            "strongest fits only from the supplied candidates; do not choose a story merely because it is available. "
            "Return JSON with a one-sentence, cheerful introduction and selections. "
            "For each why_recommended, write one short, conversational sentence directly to the child. Name a concrete "
            "detail from that candidate's title or snippet—such as what happens, a character, setting, feeling, or theme—"
            "and connect that detail to what the child asked for. Explain why this particular story is a good fit, as a "
            "librarian would. Never use vague phrases such as 'it matches what you asked for', 'it has a story idea', "
            "'it may be right for you', or 'it is a good choice' without a specific supporting detail. "
            "Do not invent details that are absent from the supplied candidate data, and do not mention ranking, search, "
            "or other candidates. Ground every reason only in the supplied query, candidate metadata, and snippets. "
            f"User request: {query}\nAllowed IDs: {allowed}\nCandidates: {json.dumps(candidate_data, ensure_ascii=False)}"
        )
        result = self.client.chat.completions.create(
            model=self.settings.groq_writing_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_schema", "json_schema": {"name": "story_recommendations", "schema": WritingResult.model_json_schema()}},
        )
        return WritingResult.model_validate(json.loads(result.choices[0].message.content or "{}"))
