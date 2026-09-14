# Handoff: embedding opennotebooklm in another project

This is a grounded question-answering engine. Given a collection of documents
you approve, it answers questions using only those documents, and says so when
they do not cover the question.

It is small: about 2,700 lines, ten modules, no ML framework. All model
inference is remote over HTTP, so the process that imports it holds no weights
and starts instantly.

## The invariant

> No factual claim reaches the user unless a passage in the approved corpus
> supports it.

Everything below exists to make that structurally true rather than merely
requested of the model. If you change one thing while integrating, do not
change that.

## What you must provide

| Dependency | Why | Notes |
|---|---|---|
| PostgreSQL 17 with pgvector | chunks, vectors, full-text, answer log | `compose.yaml` runs one; any instance works |
| A chat endpoint | reasoning and verification | OpenAI-compatible `/v1/chat/completions` **with JSON-schema support** |
| An embedding endpoint | dense retrieval | OpenAI-compatible `/v1/embeddings` |
| A rerank endpoint | ordering candidates | `/v1/rerank` returning `results[].index` and `relevance_score` |

Reference deployment is three `llama-server` processes, one per model:
Qwen3-14B for chat, BGE-M3 for embeddings, BGE-reranker-v2-m3 for reranking.
Any server speaking those shapes will do.

**The chat endpoint must support constrained decoding from a JSON schema.**
This is not a nicety. llama.cpp compiles the schema into a decoding grammar, so
structured stages cannot emit output that violates their contract. Without it,
every stage failure becomes an abstention and answer quality collapses. We
measured this: under a server that treated the schema as a hint, the model
returned bare labels, unquoted keys and `key=value` pairs.

## Two ways in

### As a library

```python
from app import db
from app.ingest import ingest
from app.pipeline import answer_question

db.init()                                   # applies schema.sql, idempotent
ingest(corpus_id, "paper.pdf", pdf_bytes)   # parse, chunk, embed, store
result = answer_question(corpus_id, "How many participants?")
```

`answer_question` returns a dict. `status` is the field to branch on:

| status | Meaning |
|---|---|
| `answered` | `answer` is prose; every claim in `claims` carries citations |
| `insufficient_evidence` | the corpus does not settle it; `answer` explains what is missing |
| `verification_failed` | a stage could not be trusted; treat exactly like insufficient |

Never render `answer` without checking `status`. The two abstention statuses
differ only in cause, never in how you should treat them.

Each entry in `claims` has `text`, `kind` (`DIRECT` or `DERIVED`), a `marker`
matching the `[n]` in the prose, and `citations` carrying `document_id`,
`title`, `page`, `section`, `chunk_id`, `quote`, and character offsets into the
document.

### As a service

`uv run uvicorn app.main:app`

| Route | Purpose |
|---|---|
| `POST /corpora` | create a corpus |
| `POST /corpora/{id}/documents` | upload a source file (multipart) |
| `GET /corpora/{id}/documents` | list sources |
| `POST /corpora/{id}/query` | ask a question |
| `GET /chunks/{chunk_id}` | resolve a citation to its passage |

`GET /` serves a single-file UI that demonstrates the citation viewer. Drop it
if you are building your own front end; nothing else depends on it.

## Integration guardrails

These are the places where a reasonable-looking change silently breaks
grounding. Each one is enforced in code and covered by a test.

**1. Never let a quote reach the user without `locate_span`.**
A span the model invents has no offsets in its chunk, so it cannot become a
citation. If you add a path that surfaces model text as a quote, route it
through `grounding.locate_span` first. This is the floor under everything else.

**2. Treat an LLM error as an abstention, never as a skipped step.**
Every stage failure returns an abstention. If you add a stage, make its failure
path return an abstention too. The one safe exception already in the code is
verbalization, which is formatting: when the writer fails, the verified claims
are emitted verbatim instead.

**3. Keep the verifier blind.**
`verify_claims` deliberately does not see the question or the other claims. A
verifier that knows which answer is wanted is a verifier that finds it. Do not
"helpfully" pass more context in.

**4. Do not trust a model's list of premises.**
`premise_stated_in_question` requires an unsupported premise to be traceable to
the question. A 14B model once returned 77 "premises" describing the entire
corpus, which abstained on every question. Keep the check.

**5. Never publish one side of a disagreement.**
`contradicts` turns a claim its own corpus disputes into a statement of the
disagreement, citing every side, and strips citations that contradict the claim
they are attached to. Conflict statements bypass the writer, which otherwise
rephrases them into two flat assertions.

**6. Scope every retrieval query by corpus in SQL.**
Corpus isolation is a security boundary, not a filter. Document ids are
corpus-scoped too, so the same file uploaded twice is two documents.

**7. Source text is data, never instructions.**
Retrieved content is delimited and the prompts say so. If you add a stage that
puts corpus text into a prompt, delimit it the same way.

**8. Embedding failure is fatal; rerank failure is not.**
A chunk with no vector is invisible to dense retrieval and would silently cost
recall, so ingestion fails loudly. Reranking is an improvement, so its failure
leaves the fused ranking intact. Preserve that asymmetry.

## Configuration

All via environment, see `.env.example`. The ones that matter when integrating:

| Variable | Default | Note |
|---|---|---|
| `DATABASE_URL` | localhost:5433 | |
| `LLAMA_URL` / `EMBED_URL` / `RERANK_URL` | 8090 / 8091 / 8092 | |
| `LLAMA_API_KEY` | none | sent as a bearer token to all three |
| `EMBED_DIM` | 1024 | **must match `vector(1024)` in `schema.sql`** |
| `RERANK_TOP_K` | 10 | passages reaching the grounding stages |
| `MIN_ANSWERABLE_CONFIDENCE` | 0.7 | below this, abstain |

Changing the embedding model means changing `EMBED_DIM`, the column width in
`schema.sql`, and re-embedding everything. The client refuses vectors of
unexpected width rather than writing them, so a mismatch fails loudly at ingest.

## What it costs

One answer is several model calls plus one per claim to verify. On a CPU-only
14B, expect 20 to 300 seconds per question. On a GPU this drops sharply. The
grounding stages are the cost; retrieval is milliseconds.

If you need it faster, the lever is fewer or cheaper model calls per question,
not concurrency. Measured on an 8-core CPU host, running four questions at once
gives about 1.3x, because a single request already occupies every thread.

## Verifying an integration

```bash
uv run pytest                                  # 63 tests; 60 need nothing, 3 skip without a database
uv run python eval/run_eval.py --workers 4     # end to end, needs both
```

The unit tests cover the guardrails above and run in under a second. Sixty of
them need nothing but Python, so they belong in your CI as they are; the three
retrieval tests skip themselves when no database is reachable.

The eval is the one that matters for grounding. It runs four classes of
question against synthetic fixtures: answerable, unanswerable-but-common-
knowledge, partially answerable, and adversarial. Ordinary accuracy is the
wrong metric here. Watch **false answer rate** on the unanswerable classes,
because that is where pretrained knowledge leaks past the corpus boundary.

Last full run, 17 of 17 passing:

| Metric | Result |
|---|---|
| False answer rate | 0% (0/9) |
| Citation precision | 100% (9/9) |
| Unsupported claim rate | 0% (0/9) |
| Retrieval recall | 100% (5/5) |

Add your own cases to `eval/cases.jsonl` for your domain, especially
unanswerable ones. A case is a question plus an expected status and optional
`must_contain` / `must_not_contain` strings.

## Known limits

- Citation precision as measured checks that a quote exists in its chunk, not
  that it supports the claim. Guardrail 5 closed the case that exposed this,
  but the metric is weaker than its name suggests.
- Conflict detection is numeric: it catches the same subject stated with
  different figures. Disagreements in wording rather than numbers pass through.
- No table extraction, so figures living only in tables are invisible.
- Chunking is structure-aware for Markdown and HTML headings, and falls back to
  line splitting on extracted PDF text, which rarely has blank lines.
- English full-text stemming only; dense retrieval is multilingual via BGE-M3.

## History

| Commit | What |
|---|---|
| `b7be8db` | the grounded pipeline |
| `28dca6f` | llama.cpp with grammar-constrained decoding, replacing Ollama |
| `0715bea` | all inference remote, torch dropped, premise and verbatim guards |
| `d2e0a12` | never assert one side of a disagreement |
| `ddac921` | concurrent eval, connection retries |

There is an approved but unimplemented plan to specialize this for a psychology
corpus: section-aware parsing, instrument alias normalization, table extraction,
study metadata, and a domain eval set.
