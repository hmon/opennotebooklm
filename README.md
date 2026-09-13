# opennotebooklm

Question answering over a collection of documents you approve, where the
documents are the only thing the system is allowed to treat as fact.

The model reads, reasons, and writes. It does not get to *know* anything. If the
sources do not establish an answer, the system says so instead of filling the
gap from what the model learned in pretraining.

```
Question
   ↓
hybrid retrieval (BM25 + dense, fused, reranked)
   ↓
answerability gate ──────────────────► abstain
   ↓
verbatim evidence spans (checked against the source text)
   ↓
atomic claims (each citing evidence, or dropped)
   ↓
per-claim entailment check ──────────► drop claim / abstain
   ↓
prose from verified claims only
   ↓
final check: no sentence outruns its claims ──► abstain
   ↓
answer with claim-level citations
```

The prompts ask the model to stay inside the corpus. The checks in
`app/grounding.py` are what make it true: a quote that is not verbatim in its
chunk cannot become a citation, a claim with no surviving evidence is dropped,
and prose that says more than its claims is never published.

## Running it

```bash
docker compose up -d          # Postgres 17 + pgvector on :5433
brew install llama.cpp
uv sync
```

Start the model server, which downloads the weights on first run (about 5.7 GB).
`--no-mmproj` skips the vision projector this repo ships, and `--reasoning-budget 0`
turns off thinking: every stage here is a small judgement, so reasoning text only
adds latency.

```bash
llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M \
  --port 8080 --ctx-size 8192 --jinja --no-mmproj --reasoning-budget 0
```

Then the app:

```bash
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000. Create a corpus, upload PDFs, Markdown, text, or
HTML, then ask. Click any `[1]` marker to see the exact passage it came from,
highlighted inside its chunk.

First run downloads the embedding and reranker models (about 1.5 GB).

## Configuration

Copy `.env.example` to `.env`. The settings that matter:

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_URL` | `http://localhost:8080` | Where llama-server is listening |
| `CHUNK_TOKENS` | `500` | Target chunk size, in words |
| `CHUNK_OVERLAP` | `0.15` | Fraction of a chunk carried into the next |
| `RERANK_TOP_K` | `10` | Passages sent to the grounding stages |
| `MIN_ANSWERABLE_CONFIDENCE` | `0.7` | Below this, abstain |

## Evaluation

Ordinary QA accuracy is the wrong metric here. The number that matters is the
false answer rate on questions the corpus cannot answer, because that is where
pretrained knowledge leaks past the boundary.

```bash
uv run python eval/run_eval.py
```

The suite runs four classes of question against synthetic fixtures:

- **A** answerable from the corpus
- **B** unanswerable, but common world knowledge (*Who wrote Hamlet?*)
- **C** partially answerable
- **D** adversarial: false premises, injected instructions, contradictory sources

It reports retrieval recall, abstention accuracy, false answer rate, citation
precision, and unsupported claim rate, and exits non-zero if any answer was
published without support.

Measured on the bundled fixtures with Qwen3.5-9B (Q4_K_M), all 17 cases passing:

| Metric | Result |
|---|---|
| False answer rate | 0% (0/9) |
| Abstention accuracy | 100% (9/9) |
| Citation precision | 100% (12/12) |
| Unsupported claim rate | 0% (0/12) |
| Retrieval recall | 100% (5/5) |
| Answerable cases answered | 100% (5/5) |

On the contradictory-year case the system reports both years with a citation to
each source and picks no winner, but it states them as two claims rather than
naming the disagreement in words. Surfacing conflicts explicitly is the clearest
place to improve next.

Expect roughly 20 to 180 seconds per question on a local 9B model. Each answer
costs several small model calls plus one more per claim to verify.

```bash
uv run pytest                 # the guards themselves; no database or model needed
```

Structured stages are schema-constrained: llama.cpp compiles each Pydantic
schema into a decoding grammar, so the model cannot emit output that violates
it. The parser still repairs near-JSON, because a stage that fails to parse
becomes an abstention and quietly costs answer quality.

## API

| Route | Purpose |
|---|---|
| `POST /corpora` | Create a corpus |
| `POST /corpora/{id}/documents` | Upload a source file |
| `POST /corpora/{id}/query` | Ask a question |
| `GET /chunks/{chunk_id}` | Resolve a citation to its source passage |

A query returns the answer, the verified claims behind it, and a citation for
each claim carrying document, page, section, and the exact quoted span.
Retrieval is always scoped to one corpus; there is no path that reads across
corpora.

`mode: "extractive"` stays close to the source wording. `mode: "synthesis"`
(the default) reads better and is verified the same way.

## What it will not do

Answer from general knowledge. Guess. Resolve a disagreement between two
sources by deciding which is right. Accept a premise the sources do not
support. Follow instructions that appear inside an uploaded document.

Those are the product, not limitations of it.

## Not yet built

A dedicated NLI entailment model alongside the LLM verifier, query
decomposition for multi-hop questions, table and figure extraction, EPUB/DOCX,
OCR, and explicit source-ranking policies for conflicts.
