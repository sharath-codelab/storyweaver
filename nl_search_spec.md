# Natural-language story recommendation specification

## Goal

Build a FastAPI endpoint that accepts a child-oriented natural-language query,
finds relevant stories in Pinecone, and recommends up to two stories in a warm,
helpful librarian voice. The service uses Groq for query analysis and final
wording, Pinecone for dense/sparse retrieval and cross-encoder reranking.


## Endpoint contract

```text
POST /v1/story-recommendations
```

```json
{"input":"I am seven and want a short funny story about animals"}
```

- `input`: required, trimmed, 1–500-character user request.

The public response has exactly one field:

```json
{"output":"• The Sparrow and The Fruit — You may enjoy this lively animal adventure because you asked for a funny animal story."}
```

`output` contains one or two child-friendly recommendation lines. Each names a
real selected story and explains in simple language why it matches the child's
request. Contributor details, counts, page ranges, filter state, and reranking
diagnostics remain internal; never expose vector values, secrets, prompts, or
model reasoning.

## Request flow

```text
FastAPI validation
  → Groq structured query analysis
  → Pinecone dense and sparse retrieval (in parallel)
  → reciprocal-rank fusion and story grouping
  → Pinecone cross-encoder reranking
  → Groq librarian-style response
```

Use request IDs, stage-level timeouts, and bounded retries for transient
provider failures. A failure in reranking or response writing must still return
non-invented first-stage candidates.

## 1. Analyse the query with Groq

Request structured JSON and validate it with Pydantic:

```json
{
  "search_query":"short funny animal story",
  "age_range":{"min":6,"max":8},
  "age_confidence":"high",
  "length_preference":"short",
  "length_confidence":"high"
}
```

Rules:

1. Infer age only from explicit statements or strong direct signals; never
   present an inference as fact.
2. Return `age_range: null` and confidence `none` when it is not supported.
3. Apply a length preference only when the user explicitly asks for a short,
   quick, long, or chapter-like story. Age alone is insufficient.
4. If the call or validation fails, use the raw query and no length filter.
5. Do not persist inferred age outside request-level diagnostics.

## 2. Map explicit length to a Pinecone filter

Apply a length condition only when the analysis confidence is medium/high and
the character-count backfill is present:

| Preference | Character range |
| --- | --- |
| `very_short` | 0–999 |
| `short` | 1,000–4,999 |
| `medium` | 5,000–11,999 |
| `long` | 12,000–25,000 |
| unknown | no length filter |

Combine language and length with Pinecone metadata filtering:

```json
{"$and":[{"language":{"$eq":"en"}},{"character_count":{"$gte":1000}},{"character_count":{"$lt":5000}}]}
```

If fewer than five distinct stories survive, retry once without the length
condition but retain language filtering. Mark the filter as relaxed.

## 3. Retrieve dense and sparse candidates

Embed `search_query` using the exact dense and sparse models used by ingestion.
Run two concurrent queries against `swv2` namespace `stories`, both using the
same filter and `include_metadata=true`, `include_values=false`:

1. dense vector only, `top_k=50`;
2. sparse vector plus a zero-filled dense vector, `top_k=50`.

Pinecone's single hybrid index requires a dense vector on every query. The zero
dense vector contributes no dot-product score, so the second request remains a
pure sparse/lexical retrieval.

Returned metadata must include ID, title, chunk text, raw credits, page range,
page count, language, and character count. The two queries use the same hybrid
index; they do not require creating two Pinecone indexes.

## 4. Fuse chunks into story candidates

Apply Reciprocal Rank Fusion with `k=60`:

```text
score = Σ 1 / (k + rank_in_list)
```

Then group chunks by `story_id`, keep the highest-RRF passage as each story's
primary candidate, and retain at most 20 distinct stories. Break ties by dense
rank, sparse rank, then story ID. Keep the primary chunk's page range and text.

## 5. Rerank with Pinecone

Call Pinecone's reranking API with the original normalized query and one
document per candidate story:

```text
title: <display_title>
author: <author_credit_raw>
illustrator: <illustrator_credit_raw>
pages: <page_start>-<page_end> of <page_count>
content: <chunk_text>
```

Use a configured cross-encoder model. Preserve title and credits, then truncate
content at a sentence boundary to meet the documented query-plus-document token
limit. Sort by reranker score; use RRF only as a tiebreaker. Keep the top five
distinct stories internally. If reranking fails or times out, retain RRF order
and record `rerank_applied: false`.

## 6. Write recommendations with Groq

Give Groq only the five candidate metadata objects and snippets. Require it to
select up to `limit` IDs from that set and produce one child-friendly reason per
selection. Each reason must connect the story content to the child's request,
using only the supplied query and candidate content. Validate returned IDs
against the candidate list. On failure, return the top candidates with
deterministic template wording; never invent a story.

## Performance, resilience, and acceptance

- Run dense and sparse retrieval in parallel; target p95 end-to-end latency
  under 2.5 seconds.
- Rerank at most 20 stories; on timeout use fused ordering.
- Log request ID, model IDs, filter state, candidate counts, stage latency,
  error class, and fallback state—not full queries or secrets.
- Reject invalid requests with a clear 4xx response.
- Enable character filters only after every indexed chunk has `character_count`.
- Dense and sparse searches must use identical filters and each request 50
  chunks.
- Final results contain no duplicate story IDs, recommend at most two stories,
  and every returned story originates from the internal top five.
