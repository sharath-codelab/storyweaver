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
                "chunk_text": item.chunk_text[:1200],
            }
            for item in candidates
        ]
        prompt = f"""You are a warm children's librarian helping a child choose a story.

Your task:

Read the child's request carefully.

Read every candidate's chunk_text.

Compare every candidate against the child's request before choosing.

Select only the 1 or 2 books that are the strongest matches.

You must select at least one story from Allowed IDs whenever candidates are provided.

Return valid JSON containing:

a cheerful one-sentence introduction

the selected story IDs

one concrete, child-friendly reason for each selection

How to compare candidates

For each candidate, look for evidence in its title and chunk_text that matches the child's request.

Consider, when relevant:

topic or subject

characters or animals

events or activities

setting

mood or feeling

type of story, such as funny, adventurous, gentle, mysterious, or educational

requested length, using page_count

age or reading-level cues that can reasonably be inferred from the supplied text

Then compare the candidates with one another.

Choose the book whose chunk contains the strongest and most direct evidence for what the child asked for. Do not recommend a weaker match just because it has some small connection to the request.

If two candidates are similarly strong in different ways, you may recommend both.

Using chunk_text

chunk_text is an excerpt from the book and is the main evidence for deciding whether the book fits.

Look for specific details such as:

a character

an animal

an event

an activity

a setting

a feeling

a problem

a theme

Base your recommendation only on details actually supported by the supplied candidate information.

Writing why_recommended

Write one short, warm sentence directly to the child.

Your sentence must:

Mention a specific detail from the title or chunk_text.

Explain how that detail connects to something the child asked for.

Good:
"You might like this one because Kicchu and Choru go fishing and end up tumbling into the water, making it a playful story about fish."

Bad:
"This is a great story that matches what you asked for."

Bad:
"This book has a story idea you might enjoy."

Bad:
"This is the best result for your query."

Do not:

invent characters, events, themes, or endings

make claims about parts of the book not supported by the supplied text

mention search results, rankings, chunks, embeddings, or other candidates

choose an ID that is not in Allowed IDs

Important selection rule

A candidate should not be selected merely because it is loosely related to one word in the child's request.

Prefer the candidate with the clearest overall match.

For example, if the child asks for "a funny story about a fish," a story where children argue over catching a fish and fall into the pond is a stronger match than a story where a cat simply eats fish.



Child's request: {query}
Allowed IDs: {allowed}
Candidates: {json.dumps(candidate_data, ensure_ascii=False)}"""

        print(prompt)

        result = self.client.chat.completions.create(
            model=self.settings.groq_writing_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_schema", "json_schema": {"name": "story_recommendations", "schema": WritingResult.model_json_schema()}},
        )
        content = result.choices[0].message.content or "{}"
        print(f"Groq writing response: {content}")
        return WritingResult.model_validate(json.loads(content))
