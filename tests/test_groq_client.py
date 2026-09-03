import json
import unittest
from types import SimpleNamespace

from src.search.groq_client import GroqService
from src.search.schemas import StoryCandidate


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "introduction": "What a lovely choice!",
            "selections": [{"story_id": "story-1", "why_recommended": "A tiny fox explores the moonlit garden."}],
        })))] )


class GroqWritingPromptTests(unittest.TestCase):
    def test_writing_prompt_requires_chunk_based_reasons(self):
        completions = FakeCompletions()
        service = GroqService.__new__(GroqService)
        service.settings = SimpleNamespace(groq_writing_model="test-model")
        service.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        candidate = StoryCandidate(
            story_id="story-1", title="Moon Garden", author="Writer", illustrator="Artist",
            page_count=4, character_count=600, page_start=1, page_end=2,
            chunk_text="A tiny fox explores a moonlit garden.", rrf_score=0.1,
        )

        service.write("I want a gentle story about animals", [candidate], 2)

        prompt = completions.request["messages"][0]["content"]
        self.assertIn("Each candidate has a `snippet`", prompt)
        self.assertIn("chunk taken from that book", prompt)
        self.assertIn("concrete detail", prompt)
        self.assertIn("compare every candidate", prompt)
        self.assertIn("Do not invent story details", prompt)


if __name__ == "__main__":
    unittest.main()
