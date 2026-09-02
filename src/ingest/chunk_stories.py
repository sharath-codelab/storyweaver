"""Page-aware, deterministic story chunking."""

from __future__ import annotations

import re

from .models import Chunk, Story


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-independent estimate used for chunk boundaries."""
    return len(re.findall(r"\S+", text))


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    output: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        paragraph_size = estimate_tokens(paragraph)
        if paragraph_size > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            paragraphs_to_add = [sentence.strip() for sentence in sentences if sentence.strip()]
        else:
            paragraphs_to_add = [paragraph]
        for item in paragraphs_to_add:
            item_size = estimate_tokens(item)
            if current and current_size + item_size > max_tokens:
                output.append("\n\n".join(current))
                overlap_words = " ".join(current).split()[-overlap_tokens:]
                current = [" ".join(overlap_words)] if overlap_words else []
                current_size = estimate_tokens(" ".join(current))
            current.append(item)
            current_size += item_size
    if current:
        output.append("\n\n".join(current))
    return output


def _embedding_text(story: Story, chunk_text: str) -> str:
    return "\n".join(
        (
            f"title: {story.display_title}",
            f"authors: {story.author_credit_raw}",
            f"illustrators: {story.illustrator_credit_raw}",
            f"content: {chunk_text}",
        )
    )


def chunk_story(story: Story, target_tokens: int, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    staged: list[tuple[int, int, str]] = []
    current_pages: list[str] = []
    page_start = 1
    current_size = 0
    for page_number, page in enumerate(story.pages, start=1):
        page_size = estimate_tokens(page)
        if page_size > max_tokens:
            if current_pages:
                staged.append((page_start, page_number - 1, "\n\n".join(current_pages)))
                current_pages, current_size = [], 0
            for split in _split_long_text(page, max_tokens, overlap_tokens):
                staged.append((page_number, page_number, split))
            page_start = page_number + 1
        elif current_pages and current_size + page_size > target_tokens:
            staged.append((page_start, page_number - 1, "\n\n".join(current_pages)))
            current_pages = [page]
            page_start = page_number
            current_size = page_size
        else:
            if not current_pages:
                page_start = page_number
            current_pages.append(page)
            current_size += page_size
    if current_pages:
        staged.append((page_start, len(story.pages), "\n\n".join(current_pages)))

    chunks: list[Chunk] = []
    for number, (start, end, text) in enumerate(staged, start=1):
        clean_text = text.strip()
        if not clean_text:
            continue
        identifier = f"{story.story_id}#{number:04d}"
        metadata = {
            "story_id": story.story_id,
            "source_filename": story.source_filename,
            "display_title": story.display_title,
            "filename_title": story.filename_title,
            "page_count": story.page_count,
            "character_count": story.character_count,
            "page_start": start,
            "page_end": end,
            "chunk_number": number,
            "chunk_text": clean_text,
            "authors": list(story.authors),
            "illustrators": list(story.illustrators),
            "author_credit_raw": story.author_credit_raw,
            "illustrator_credit_raw": story.illustrator_credit_raw,
            "language": story.language or "",
            "license": story.license or "",
            "translation": story.translation or "",
            "content_sha256": story.content_sha256,
        }
        chunks.append(Chunk(identifier, story.story_id, number, start, end, clean_text, _embedding_text(story, clean_text), metadata))
    return chunks
