# Story ingestion specification

## Purpose

Ingest all Markdown stories in `stories/` into a Pinecone hybrid index named
`swv2`. This document covers source validation, metadata extraction, chunking,
embedding, and reliable writes only.

Do not ingest `stories/README.md`. Runs must be repeatable and safe when a
source story changes.

## Verified corpus rules

The current corpus contains 395 ingestible story files and one excluded README.
Every story has an H1 title, `Text:` credit, `Illustration:` credit, and at
least one page marker. There are 3,642 page markers in total; story page counts
range from 1 to 33.

- A page marker is only a line matching `^##\s*$`; some stories contain `###`
  headings that must not be counted as pages.
- Credits may contain multiple people, separated with commas, `&`, or `and`.
- Filenames are stable IDs but do not preserve title punctuation or casing.
- The numeric filename prefix must remain in IDs because some filename slugs
  repeat under different prefixes.

## Pinecone design

Use one vector-API hybrid index, `swv2`, with both vector types on every chunk
record.

| Setting | Requirement |
| --- | --- |
| Name | `swv2` |
| Vector type | `dense` |
| Metric | `dotproduct` |
| Dimension | Exact output dimension of the configured dense model |
| Sparse data | `sparse_values` on each record |
| Namespace | `stories` |

`dotproduct` is required for this single-index dense-plus-sparse design. Cloud,
region, models, namespace, and batch size must be configuration values. Require
`PINECONE_API_KEY` from the environment and never log or commit it.

Choose and record one dense text model and one English sparse model. Generate
both vectors from the same canonical embedding text.

## Source parsing

For each `stories/*.md` file except the README:

1. Read UTF-8 text and normalize line endings to `\n`.
2. Set `story_id` to the full filename stem, such as
   `0092_the-sparrow-and-the-fruit`.
3. Set `source_filename` to the basename including `.md`.
4. Extract `display_title` from the first H1 using `^#\s+(.+?)\s*$`. This is
   the user-facing title.
5. Derive `filename_title` by removing the numeric prefix, replacing `_` and
   `-` with spaces, and collapsing whitespace. Retain it only for traceability.
6. Set `page_count` to the number of `^##\s*$` lines.
7. Remove the contiguous end-of-file attribution block made of lines like
   `* Label: value` from the story body.
8. Calculate `character_count` by removing the H1 and page-marker lines,
   joining the remaining page content with `\n\n`, and counting Unicode code
   points, including punctuation and whitespace. Store `0` for a story with no
   body content.
9. Preserve `Text:` as `author_credit_raw` and `Illustration:` as
   `illustrator_credit_raw`. Also extract `License:`, `Language:`, and
   `Translation:` when present.
9. Hash the normalized full source as `content_sha256`.

Fail validation before embedding if any story lacks an H1, `Text:`,
`Illustration:`, or page marker. Report all invalid files in one run.

## Contributor metadata

Preserve raw credit strings and derive lower-cased lists for filtering:

```json
{
  "author_credit_raw": "Mala Kumar, Manisha Chaudhry",
  "authors": ["mala kumar", "manisha chaudhry"],
  "illustrator_credit_raw": "Angie & Upesh",
  "illustrators": ["angie", "upesh"]
}
```

Split only on commas, `&`, and whole-word `and`. Trim whitespace and remove
empty values. Do not split apostrophes, initials, periods, or hyphens.

## Chunking and record schema

Ingest page-aware chunks, not whole stories:

1. Split the clean body on page markers.
2. Combine adjacent short pages to a 400–700 token target.
3. Preserve pages whole when possible.
4. If one page is too long, split on paragraphs, then sentences, with 50–100
   token overlap only between splits of that page.
5. Never create empty chunks.

Use a deterministic record ID: `<story_id>#<zero-padded chunk number>`.

Build this embedding text for each chunk and use it for both dense and sparse
embedding:

```text
title: <display title>
authors: <raw Text credit>
illustrators: <raw Illustration credit>
content: <chunk text>
```

Every record must include dense `values`, sparse `sparse_values`, and metadata:

```json
{
  "story_id": "0092_the-sparrow-and-the-fruit",
  "source_filename": "0092_the-sparrow-and-the-fruit.md",
  "display_title": "The Sparrow and The Fruit",
  "filename_title": "the sparrow and the fruit",
  "page_count": 12,
  "character_count": 4200,
  "page_start": 3,
  "page_end": 5,
  "chunk_number": 3,
  "chunk_text": "clean result excerpt",
  "authors": ["venkatramana gowda"],
  "illustrators": ["padmanabh"],
  "author_credit_raw": "Venkatramana Gowda",
  "illustrator_credit_raw": "Padmanabh",
  "language": "en",
  "license": "CC-BY",
  "translation": "Divaspathy Hegde",
  "content_sha256": "hash of normalized source"
}
```

Repeat story-level metadata on every chunk. Limit `chunk_text` to a size safe
for Pinecone metadata; use `story_id` as the stable source reference.

## Ingestion and updates

Validate sources, parse stories, create chunks, and write a local JSONL
manifest before external writes. The manifest records IDs, source hashes, model
IDs, page ranges, and statuses, but no credentials.

Create `swv2` only when it does not exist. If it exists, verify dimension and
metric before writing. Generate embeddings in bounded batches, retry transient
failures with backoff, and upsert batched records into `stories`.

For incremental changes, compare `content_sha256`. If a story changed, delete
its old records by `story_id` before upserting its new complete chunk set. For a
full rebuild, use a new namespace, validate it, then switch the application.

## Acceptance criteria

- Validation reports 395 valid stories and one intentionally excluded README.
- Every chunk repeats its story's integer `character_count` metadata.
- The vector count equals the non-empty chunk count in the manifest.
- All IDs are unique; all records have valid dense and sparse vectors.
- Sample records have expected metadata, page range, and text.
- Re-running unchanged input produces no duplicate IDs or changed hashes.
