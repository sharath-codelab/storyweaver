"""Discovery, parsing, and validation of StoryWeaver Markdown stories."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Story

H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
PAGE_RE = re.compile(r"^##\s*$", re.MULTILINE)
CREDIT_RE = re.compile(r"^\*\s+([^:]+):\s*(.*?)\s*$")
SPLIT_CONTRIBUTORS_RE = re.compile(r"\s*(?:,|&|\band\b)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationIssue:
    filename: str
    errors: tuple[str, ...]


class CorpusValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        names = ", ".join(issue.filename for issue in issues)
        super().__init__(f"Corpus validation failed for {len(issues)} file(s): {names}")


def discover_story_files(stories_dir: Path) -> list[Path]:
    """Return sorted immediate story Markdown files, excluding the corpus README."""
    return sorted(path for path in stories_dir.glob("*.md") if path.name != "README.md")


def normalize_contributors(raw_credit: str) -> tuple[str, ...]:
    names = []
    for part in SPLIT_CONTRIBUTORS_RE.split(raw_credit):
        clean = " ".join(part.split())
        if clean:
            names.append(clean.lower())
    return tuple(names)


def _filename_title(stem: str) -> str:
    value = re.sub(r"^\d+[_-]?", "", stem)
    return " ".join(re.sub(r"[_-]+", " ", value).split())


def _split_credit_block(lines: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return content lines and the final contiguous Markdown credit block."""
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    start = end
    credits: dict[str, str] = {}
    while start:
        match = CREDIT_RE.match(lines[start - 1])
        if not match:
            break
        credits[match.group(1).strip().lower()] = match.group(2).strip()
        start -= 1
    if start == end:
        return lines, {}
    body = lines[:start]
    while body and not body[-1].strip():
        body.pop()
    return body, credits


def _pages_from_body(body: str) -> tuple[str, ...]:
    # The prefix contains the title; discard it. Empty marker-only pages are not chunks.
    parts = PAGE_RE.split(body)
    pages = [part.strip() for part in parts[1:]]
    return tuple(page for page in pages if page)


def parse_story(path: Path) -> Story:
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    title_match = H1_RE.search(source)
    lines = source.split("\n")
    body_lines, credits = _split_credit_block(lines)
    body = "\n".join(body_lines).strip()
    errors: list[str] = []
    if not title_match:
        errors.append("missing H1 title")
    if not PAGE_RE.search(source):
        errors.append("missing page marker")
    if not credits.get("text"):
        errors.append("missing Text credit")
    if not credits.get("illustration"):
        errors.append("missing Illustration credit")
    if errors:
        raise CorpusValidationError([ValidationIssue(path.name, tuple(errors))])
    pages = _pages_from_body(body)
    character_count = len("\n\n".join(pages))
    return Story(
        story_id=path.stem,
        source_filename=path.name,
        display_title=title_match.group(1).strip(),
        filename_title=_filename_title(path.stem),
        page_count=len(PAGE_RE.findall(source)),
        character_count=character_count,
        pages=pages,
        author_credit_raw=credits["text"],
        illustrator_credit_raw=credits["illustration"],
        authors=normalize_contributors(credits["text"]),
        illustrators=normalize_contributors(credits["illustration"]),
        license=credits.get("license"),
        language=credits.get("language"),
        translation=credits.get("translation"),
        content_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def parse_and_validate(paths: list[Path]) -> list[Story]:
    stories: list[Story] = []
    issues: list[ValidationIssue] = []
    for path in paths:
        try:
            stories.append(parse_story(path))
        except CorpusValidationError as error:
            issues.extend(error.issues)
    if issues:
        raise CorpusValidationError(issues)
    return stories
