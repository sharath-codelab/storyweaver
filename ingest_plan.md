# SWV2 ingestion implementation plan

## Goal

Implement the pipeline described in [ingest_spec.md](ingest_spec.md): ingest
395 story files into Pinecone index `swv2`. This plan covers ingestion only.

## Milestone 0 — Make the deployment choices

1. Select the dense embedding model and note its output dimension.
2. Select the English sparse embedding model.
3. Select Pinecone cloud and region.
4. Place `PINECONE_API_KEY` and model-provider credentials in local environment
   configuration; do not commit them.
5. Confirm H1 titles are the user-facing titles.

**Done when:** a configuration document or `.env.example` names every required
non-secret variable, and startup validation reports missing configuration
clearly.

## Milestone 1 — Scaffold the implementation

Create the following modules (names may vary by language):

```text
src/ingest/config.py           configuration validation
src/ingest/parse_stories.py    source discovery, parsing, validation
src/ingest/chunk_stories.py    page-aware chunking
src/ingest/embeddings.py       dense and sparse embedding calls
src/ingest/pinecone_store.py   index lifecycle and vector writes
src/ingest/ingest.py           CLI orchestration
tests/                  parser, chunking, ingestion tests
```

Add an ignored `.local/ingestion/` directory for `validation.json`,
`manifest.jsonl`, and `run-summary.json`.

**Done when:** `ingest --help` works and configuration loading does not expose
secret values.

## Milestone 2 — Discover and parse stories

1. Enumerate immediate `stories/*.md` files only.
2. Exclude `stories/README.md` exactly.
3. Sort files for deterministic runs.
4. For every file, normalize line endings and produce a parsed story object.
5. Set stable `story_id` to the complete filename stem.
6. Extract first H1 as `display_title`; separately derive `filename_title`.
7. Count only `^##\s*$` lines for `page_count`.
8. Remove the final contiguous `* Label: value` credit block from story text.
9. Calculate `character_count`: remove H1 and page-marker lines, join remaining
   page content with `\n\n`, and count Unicode code points, including whitespace
   and punctuation. Use zero for a story with no body content.
10. Extract `Text:`, `Illustration:`, `License:`, `Language:`, and `Translation:`.
11. Calculate SHA-256 of normalized source text.

Add unit cases for interior `###` headings, multi-person credits, and a final
page marker preceding credits.

**Done when:** validation reports 395 valid stories, 3,642 page markers in
total, and no missing required field.

## Milestone 3 — Normalize credits and chunk content

1. Split contributor credits on commas, `&`, and whole-word `and`.
2. Produce normalized lowercase `authors` and `illustrators` arrays while
   retaining raw credit text.
3. Split bodies by page markers and give pages one-based page numbers.
4. Combine short adjacent pages toward 550 tokens, never exceeding 700 tokens.
5. Split oversized pages by paragraphs, then sentences, with 75-token overlap.
6. Do not create empty chunks.
7. Assign IDs as `<story_id>#<zero-padded chunk number>`.
8. Create the embedding text:

   ```text
   title: <display title>
   authors: <raw Text credit>
   illustrators: <raw Illustration credit>
   content: <chunk text>
   ```

9. Emit a manifest record per chunk containing its ID, source hash, page range,
   metadata, and later model IDs.

Include the story-level integer `character_count` in every chunk's metadata so
future metadata filters can select an appropriate story length.

**Done when:** chunk IDs are unique, page ranges are valid, text is non-empty,
and a repeated run produces byte-equivalent manifest records.

## Milestone 4 — Provision Pinecone

1. Look up index `swv2`.
2. If absent, create a serverless dense index with `dotproduct`, the configured
   model dimension, and selected cloud/region.
3. If present, assert its metric and dimension match configuration.
4. Use namespace `stories`.
5. Wait for index readiness before data operations.

**Done when:** an index description confirms name, readiness, metric, and
dimension.

## Milestone 5 — Embed and upsert

1. Generate dense passage embeddings from embedding text in bounded batches.
2. Generate sparse passage embeddings from the same embedding text.
3. Validate every dense length and sparse indices/values pair.
4. Construct records with `id`, `values`, `sparse_values`, and all required
   metadata.
5. Upsert in batches into `stories`.
6. Retry transient provider failures with capped exponential backoff.
7. Persist batch outcomes and failed record IDs in the run summary.
8. Wait for Pinecone statistics to reflect manifest vector count.

For subsequent runs, compare source hashes. Skip unchanged stories; for changed
stories, delete their old vectors by `story_id` before upserting all current
chunks.

**Done when:** manifest count equals namespace count and fetched sample vectors
have expected title, raw credits, normalized contributors, character count, page
range, and text.

## Milestone 6 — Validate and operate ingestion

1. Confirm the validation report contains no invalid stories.
2. Confirm Pinecone namespace vector count equals the manifest chunk count.
3. Fetch representative records and verify their IDs, dense-vector dimensions,
   sparse values, source hashes, page ranges, credits, and metadata.
4. Re-run unchanged input and confirm no duplicate IDs or unnecessary writes.
5. Change a test story, run incrementally, and confirm stale chunks are removed
   before its new chunks are upserted.
6. Add operational alerts for validation failures, dimension mismatch, vector
   count mismatch, and repeated embedding/provider failures.

## Release checklist

- [ ] Configuration and secrets handling are complete.
- [ ] Corpus validation passes.
- [ ] Parsing and chunking tests pass.
- [ ] Index configuration matches selected models.
- [ ] Manifest and Pinecone vector counts match.
- [ ] Incremental update path removes stale chunks.
