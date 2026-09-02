from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ingest.chunk_stories import chunk_story
from src.ingest.parse_stories import discover_story_files, normalize_contributors, parse_and_validate, parse_story


SAMPLE = """# A Fine Story!

##
First page about a bird.

### Not a page marker

##
Second page about fruit.

##
* License: CC-BY
* Text: Ada Lovelace, Grace Hopper
* Illustration: Angie & Upesh
* Language: en
"""


class StoryParsingTests(unittest.TestCase):
    def test_parses_metadata_pages_and_credits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0001_a-fine-story.md"
            path.write_text(SAMPLE, encoding="utf-8")
            story = parse_story(path)
        self.assertEqual(story.story_id, "0001_a-fine-story")
        self.assertEqual(story.display_title, "A Fine Story!")
        self.assertEqual(story.filename_title, "a fine story")
        self.assertEqual(story.page_count, 3)
        self.assertEqual(story.character_count, len("First page about a bird.\n\n### Not a page marker\n\nSecond page about fruit."))
        self.assertEqual(story.authors, ("ada lovelace", "grace hopper"))
        self.assertEqual(story.illustrators, ("angie", "upesh"))
        self.assertEqual(len(story.pages), 2)
        self.assertNotIn("Text:", "\n".join(story.pages))

    def test_contributor_normalization(self) -> None:
        self.assertEqual(normalize_contributors("A. One and B-Two & C's Name"), ("a. one", "b-two", "c's name"))

    def test_readme_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stories_dir = Path(directory)
            (stories_dir / "README.md").write_text("# no", encoding="utf-8")
            story_path = stories_dir / "0001_a.md"
            story_path.write_text(SAMPLE, encoding="utf-8")
            self.assertEqual(discover_story_files(stories_dir), [story_path])

    def test_chunk_ids_and_page_ranges_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0001_a-fine-story.md"
            path.write_text(SAMPLE, encoding="utf-8")
            story = parse_story(path)
        chunks = chunk_story(story, target_tokens=8, max_tokens=20, overlap_tokens=2)
        self.assertTrue(chunks)
        self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))
        self.assertTrue(all(chunk.page_start <= chunk.page_end for chunk in chunks))
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(chunk.metadata["character_count"] == story.character_count for chunk in chunks))

    def test_validate_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0001_a-fine-story.md"
            path.write_text(SAMPLE, encoding="utf-8")
            stories = parse_and_validate([path])
        self.assertEqual(len(stories), 1)


if __name__ == "__main__":
    unittest.main()
