import unittest

from src.search.fusion import fuse_chunks
from src.search.schemas import ChunkMatch


def match(record_id, source, rank, story_id):
    return ChunkMatch(record_id=record_id, source=source, rank=rank, score=1.0, metadata={
        "story_id": story_id, "display_title": story_id, "author_credit_raw": "A", "illustrator_credit_raw": "I",
        "page_count": 2, "character_count": 100, "page_start": 1, "page_end": 1, "chunk_text": "Actual content.",
    })


class FusionTests(unittest.TestCase):
    def test_fuses_both_lists_and_deduplicates_stories(self):
        output = fuse_chunks([match("a#1", "dense", 1, "a"), match("b#1", "dense", 2, "b")],
                             [match("a#1", "sparse", 1, "a"), match("a#2", "sparse", 2, "a")], 60)
        self.assertEqual([item.story_id for item in output], ["a", "b"])
        self.assertEqual(output[0].chunk_text, "Actual content.")


if __name__ == "__main__":
    unittest.main()
