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
        return WritingResult(introduction="What a fun story idea!", selections=[WritingChoice(story_id="story", why_recommended="It is funny.")])


class FakePinecone:
    def __init__(self):
        self.filters = []

    def retrieve(self, query, metadata_filter):
        self.filters.append(metadata_filter)
        return [match("dense", 1)], [match("sparse", 1)]

    def rerank(self, query, candidates, documents=None):
        return {"story": 0.9}

    def rerank_documents(self, candidates):
        return [{"id": candidate.story_id, "text": "complete story text"} for candidate in candidates]


class SearchServiceTests(unittest.TestCase):
    def test_returns_grounded_recommendation(self):
        pinecone = FakePinecone()
        service = RecommendationService(FakeGroq(), pinecone, 60, 20)
        result = service.recommend(RecommendationRequest(input="A short funny story"))
        self.assertIn("What a fun story idea!", result.response)
        self.assertIn("A Story", result.response)
        self.assertIn("It is funny.", result.response)
        self.assertIn("$and", pinecone.filters[0])

    def test_debug_response_contains_pipeline_trace(self):
        pinecone = FakePinecone()
        service = RecommendationService(FakeGroq(), pinecone, 60, 20)

        result = service.recommend_debug(RecommendationRequest(input="A short funny story"))

        self.assertIn("What a fun story idea!", result.response.message)
        self.assertFalse(result.response.analysis_fallback)
        self.assertTrue(result.response.retrieval["length_filter_applied"])
        self.assertTrue(result.response.retrieval["filter_relaxed"])
        self.assertEqual(len(result.response.retrieval["attempts"]), 2)
        self.assertEqual(result.response.fusion["rerank_candidates"][0].story_id, "story")
        self.assertTrue(result.response.reranking["applied"])
        self.assertEqual(result.response.reranking["documents"][0]["text"], "complete story text")
        self.assertEqual(result.response.finalization["top_five"][0].rerank_score, 0.9)
        self.assertFalse(result.response.finalization["writing_fallback"])


if __name__ == "__main__":
    unittest.main()
