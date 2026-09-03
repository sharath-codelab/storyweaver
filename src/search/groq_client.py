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
        prompt = f"""You are a warm children's librarian helping a child choose a story.

Your task:
1. Read the child's request and compare every candidate.
2. Choose only the one or two books that fit the request best.
3. Return a cheerful one-sentence introduction and your selections as JSON.

How to choose:
- Each candidate has a `snippet`. This is a chunk taken from that book.
- Look closely at the snippet for a character, event, setting, feeling, or theme.
- Use that concrete detail to explain why the book containing this chunk would be a good fit for the child.
- Consider the child's requested topic, mood, characters, length, and age cues when they are provided.

How to write each `why_recommended`:
- Write one short, friendly sentence directly to the child, like a helpful librarian.
- Mention a specific detail supported by the title or snippet, then connect it to the child's request.
- Do not use vague reasons like "it matches what you asked for" or "it has a story idea."
- Do not invent story details. Do not mention search results, rankings, or other candidates.
- Choose only IDs in Allowed IDs.

Child's request: {query}
Allowed IDs: {allowed}
Candidates: {json.dumps(candidate_data, ensure_ascii=False)}"""
        result = self.client.chat.completions.create(
            model=self.settings.groq_writing_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_schema", "json_schema": {"name": "story_recommendations", "schema": WritingResult.model_json_schema()}},
        )
        return WritingResult.model_validate(json.loads(result.choices[0].message.content or "{}"))
