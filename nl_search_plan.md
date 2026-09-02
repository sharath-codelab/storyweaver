# Natural-language story search: Python implementation plan

## Goal

Implement `POST /v1/story-recommendations`: accept a child-oriented query,
retrieve and rerank stories from Pinecone, then recommend up to two stories in
a librarian voice. The service uses Groq for structured query analysis and
final wording; Pinecone supplies embeddings, retrieval, and cross-encoder
reranking.

The behavioral contract is defined in [nl_search_spec.md](nl_search_spec.md).

## Phase 0 — Prepare the indexed data

### Step 1. Re-ingest `character_count`

`character_count` is now emitted by the ingestion code, but existing Pinecone
records will not have it until the full ingestion is run.

1. Confirm local validation succeeds:

   ```bash
   uv run --with-requirements requirements.txt -m src.ingest --dry-run
   ```

2. Run ingestion to upsert the current chunk schema:

   ```bash
   uv run --with-requirements requirements.txt -m src.ingest
   ```

3. Fetch a few vectors and confirm their metadata includes integer
   `character_count`, `language`, `story_id`, `chunk_text`, and contributor
   fields.

**Done when:** all active chunks have `character_count`. Do not enable length
filters before this is true.

## Phase 1 — Create the application structure

### Step 2. Add application modules

Create a separate search package so ingestion remains independent:

```text
src/search/
  __init__.py
  __main__.py             # starts uvicorn
  app.py                  # FastAPI application and routes
  config.py               # validated environment settings
  schemas.py              # Pydantic request/response/model schemas
  groq_client.py          # structured analysis and response wording
  pinecone_client.py      # query embedding, dense/sparse retrieval, rerank
  fusion.py               # reciprocal-rank fusion and story grouping
  service.py              # orchestrates the request stages
  fallback.py             # deterministic output when Groq/rerank fail
tests/
  test_fusion.py
  test_schemas.py
  test_service.py
```

Add dependencies to `requirements.txt`:

```text
fastapi
uvicorn[standard]
httpx
pydantic
groq
```

Use the existing `pinecone` and `typing_extensions` dependencies.

### Step 3. Configure environment variables

Extend `.env.example` without adding secrets:

```text
GROQ_API_KEY=
GROQ_ANALYSIS_MODEL=<configured model ID>
GROQ_WRITING_MODEL=<configured model ID>
PINECONE_RERANK_MODEL=<configured cross-encoder model ID>
SEARCH_PORT=8000
SEARCH_TIMEOUT_SECONDS=2.5
DENSE_TOP_K=50
SPARSE_TOP_K=50
RERANK_CANDIDATE_COUNT=20
RRF_K=60
```

Reuse the ingestion settings for Pinecone API key, index name, namespace,
embedding model IDs, and embedding dimension. Never return or log secrets.

**Done when:** application startup validates all required settings and reports
missing names only, never secret values.

## Phase 2 — Define and validate the API

### Step 4. Implement request and response schemas

Create Pydantic models:

```python
class RecommendationRequest(BaseModel):
    input: str = Field(min_length=1, max_length=500)
```

Trim input whitespace in a validator and reject whitespace-only input. Use `en`
as the initial fixed metadata filter.

Define a minimal public response model:

- `RecommendationResponse`: a `response` string containing one or two
  child-friendly lines. Each line names a story and explains why it fits the
  child's request.

Keep IDs, credits, counts, page ranges, scores, filter state, and fallback
diagnostics internal to the service.

### Step 5. Create the FastAPI route

Implement `POST /v1/story-recommendations` and inject one service instance
through FastAPI lifespan/dependency management. Add:

- a `/healthz` route that reports process health only;
- a request ID middleware; and
- consistent conversion of validation errors to 4xx and upstream failures to
  safe service errors or fallbacks.

**Done when:** OpenAPI documents the endpoint and invalid requests are rejected
without calling Groq or Pinecone.

## Phase 3 — Analyse queries with Groq

### Step 6. Implement structured analysis

Create `QueryAnalysis`:

```python
class QueryAnalysis(BaseModel):
    search_query: str
    age_range: AgeRange | None
    age_confidence: Literal["none", "low", "medium", "high"]
    length_preference: Literal["very_short", "short", "medium", "long"] | None
    length_confidence: Literal["none", "low", "medium", "high"]
```

Prompt Groq to return JSON only. The prompt must say:

1. use explicit age statements or strong direct signals only;
2. return no age range when unsupported;
3. infer story length only from explicit short/long/quick/chapter wording;
4. never recommend or invent titles; and
5. use the query language and preserve meaningful search terms.

Validate the provider response. If it fails, times out, or violates the schema,
fall back to `search_query=request.input` with no age or length preference.

**Done when:** mocked valid, malformed, timeout, and unsupported-age responses
all produce deterministic validated analysis data.

## Phase 4 — Build Pinecone filters and retrieve candidates

### Step 7. Build metadata filters

Create `build_filter(language, analysis, character_count_available)`.

Always filter by language when one is supplied. Apply `character_count` only if
the field has been backfilled and length confidence is medium/high.

Initial configurable ranges:

| Preference | Minimum | Maximum exclusive |
| --- | ---: | ---: |
| very_short | 0 | 1,000 |
| short | 1,000 | 5,000 |
| medium | 5,000 | 12,000 |
| long | 12,000 | 25,001 |

If a length-filtered attempt has fewer than five distinct stories, repeat
retrieval once without the count condition and mark the filter as relaxed.

### Step 8. Embed the query and run dense/sparse searches

Use Pinecone inference with the same models used by ingestion:

1. generate dense query embedding with `input_type="query"`;
2. generate sparse query embedding with `input_type="query"`;
3. run two Pinecone requests concurrently with the same metadata filter:
   - dense vector only, `top_k=50`;
   - sparse vector plus a zero-filled dense vector, `top_k=50`.

The zero vector is required because this is a dense hybrid Pinecone index; it
contributes no score, preserving sparse-only lexical ranking for that request.
4. request metadata but not vector values.

Map both responses to an internal `ChunkMatch` model containing record ID,
rank, score, story ID, required metadata, and source (`dense` or `sparse`).

**Done when:** mocks prove both searches use identical filters and independently
handle an empty response.

## Phase 5 — Fuse and group candidates

### Step 9. Implement Reciprocal Rank Fusion

Implement pure, unit-tested code:

```text
rrf_score(chunk) = sum(1 / (60 + rank_in_each_result_list))
```

For duplicate record IDs, add contributions from dense and sparse rankings. Use
dense rank, sparse rank, then ID as deterministic tiebreakers.

### Step 10. Convert chunks into story candidates

1. Group fused chunks by `story_id`.
2. Keep the highest-RRF chunk as each story's primary content passage.
3. Retain up to 20 distinct stories for reranking.
4. Preserve title, credits, page count, character count, primary chunk text,
   page range, RRF score, and first-stage ranks.

**Done when:** candidate stories are unique and preserve actual chunk content,
not merely titles.

## Phase 6 — Rerank actual story content

### Step 11. Construct cross-encoder documents

For every candidate story, create exactly one document:

```text
title: <display_title>
author: <author_credit_raw>
illustrator: <illustrator_credit_raw>
pages: <page_start>-<page_end> of <page_count>
content: <primary_chunk_text>
```

This means the cross-encoder compares the user's query with the actual
retrieved story content as well as its title and credits. It does not rank by
title alone.

Preserve title and credit fields, then truncate content at a sentence boundary
to respect the selected reranker’s documented input limit.

### Step 12. Call Pinecone reranking

Call Pinecone reranking with the normalized original query and these documents.
Map scores back to story IDs, order candidates by rerank score descending, and
use RRF score only as a deterministic tiebreaker. Keep the top five stories.

On provider error or timeout, retain RRF order and set `rerank_applied=False`.
Do not fail the full endpoint for a reranking-only failure.

**Done when:** reranker tests verify document construction, score ordering,
token truncation, and RRF fallback.

## Phase 7 — Write the final librarian response

### Step 13. Call Groq with grounded candidates only

Pass Groq only the top five candidate objects and their factual snippets. Ask
for JSON with up to `limit` selected candidate IDs and one concise,
child-friendly reason per selected ID. Each reason must connect the story
content to the child's request.

Validate every selected ID against the top-five candidates. Enrich accepted IDs
with factual metadata from Pinecone rather than trusting generated facts.

### Step 14. Provide deterministic fallback wording

If final Groq wording fails, select the top `limit` reranked candidates and
format them with a fixed template. This fallback must always return real titles,
credits, counts, and page ranges.

**Done when:** no provider failure can return an invented story or an ID outside
the reranked candidates.

## Phase 8 — Test, measure, and run

### Step 15. Test suite

Add unit tests for schemas, query analysis fallback, length filters, concurrent
retrieval, RRF, story grouping, rerank fallbacks, and final-ID validation. Add
service integration tests with mocked Groq and Pinecone clients.

### Step 16. Observability and privacy

Log request ID, stage durations, model IDs, filter state, candidate counts,
reranking status, and error class. Do not log API keys, vector values, prompts,
or full user queries by default.

Track p50/p95 end-to-end latency, Groq timeout rate, Pinecone error rate,
rerank fallback rate, and percentage of requests where the length filter was
relaxed.

### Step 17. Start the service

Target command after implementation:

```bash
uv run --with-requirements requirements.txt -m src.search
```

Run locally on port 8000 and verify:

```bash
curl -X POST http://127.0.0.1:8000/v1/story-recommendations \
  -H 'Content-Type: application/json' \
  -d '{"input":"I am seven and want a short funny animal story"}'
```

## Release checklist

- [ ] Pinecone data has been re-ingested with `character_count`.
- [ ] API schemas, health endpoint, and request IDs work.
- [ ] Groq analysis is structured and has a safe fallback.
- [ ] Dense and sparse retrieval run concurrently with the same filter.
- [ ] RRF produces unique story candidates from actual content chunks.
- [ ] Pinecone reranking sees content passages and has timeout fallback.
- [ ] Groq can select only from supplied top-five candidate IDs.
- [ ] Deterministic fallback response contains only factual candidate data.
- [ ] Tests and observability are in place before deployment.
