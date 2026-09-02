import unittest

from src.search.schemas import ChunkMatch, QueryAnalysis, RecommendationRequest, WritingChoice, WritingResult
from src.search.service import RecommendationService


def match(source: str, rank: int) -> ChunkMatch:
    return ChunkMatch(record_id="story#0001", source=source, rank=rank, score=1.0, metadata={
        "story_id": "story", "display_title": "A Story", "author_credit_raw": "Writer",
        "illustrator_credit_raw": "Artist", "page_count": 2, "character_count": 200,
        "page_start": 1, "page_end": 1, "chunk_text": "A funny animal adventure.",
    })


class FakeGroq:
    def analyse(self, query):
        return QueryAnalysis(search_query=query, length_preference="short", length_confidence="high")

    def write(self, query, candidates, limit):
        return WritingResult(selections=[WritingChoice(story_id="story", why_recommended="It is funny.")])


class FakePinecone:
    def __init__(self):
        self.filters = []

    def retrieve(self, query, metadata_filter):
        self.filters.append(metadata_filter)
        return [match("dense", 1)], [match("sparse", 1)]

    def rerank(self, query, candidates):
        return {"story": 0.9}


class SearchServiceTests(unittest.TestCase):
    def test_returns_grounded_recommendation(self):
        pinecone = FakePinecone()
        service = RecommendationService(FakeGroq(), pinecone, 60, 20)
        result = service.recommend(RecommendationRequest(query="A short funny story"))
        self.assertEqual(result.recommendations[0].title, "A Story")
        self.assertIn("It is funny.", result.recommendations[0].line)
        self.assertIn("$and", pinecone.filters[0])


if __name__ == "__main__":
    unittest.main()
