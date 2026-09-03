import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.search.pinecone_client import PineconeSearch
from src.search.schemas import StoryCandidate


def candidate() -> StoryCandidate:
    return StoryCandidate(
        story_id="0001_story", title="A Story", author="Writer", illustrator="Artist",
        page_count=2, character_count=100, page_start=2, page_end=2,
        chunk_text="Only the matched block.", rrf_score=0.1,
    )


class PineconeSearchRerankTests(unittest.TestCase):
    def search_with_corpus(self, stories_dir: Path) -> PineconeSearch:
        search = PineconeSearch.__new__(PineconeSearch)
        search.settings = SimpleNamespace(ingestion=SimpleNamespace(stories_dir=stories_dir))
        search._story_text_cache = {}
        return search

    def test_rerank_document_uses_the_complete_canonical_story(self):
        with tempfile.TemporaryDirectory() as directory:
            stories_dir = Path(directory)
            (stories_dir / "0001_story.md").write_text(
                "# A Story\n\n##\nFirst page.\n\n##\nSecond page.\n\n* Text: Writer\n* Illustration: Artist\n",
                encoding="utf-8",
            )
            search = self.search_with_corpus(stories_dir)

            text = search._rerank_text(candidate(), search._full_story_text("0001_story"))

        self.assertIn("pages: 1-2 of 2", text)
        self.assertIn("First page.\n\nSecond page.", text)
        self.assertNotIn("Only the matched block.", text)

    def test_rerank_document_falls_back_to_the_matched_chunk_when_source_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            search = self.search_with_corpus(Path(directory))
            text = search._rerank_text(candidate(), search._full_story_text("0001_story"))

        self.assertIn("pages: 2-2 of 2", text)
        self.assertIn("Only the matched block.", text)


if __name__ == "__main__":
    unittest.main()
