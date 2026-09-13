CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS corpora (
    id          serial PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id           text PRIMARY KEY,
    corpus_id    int NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    title        text NOT NULL,
    source_type  text NOT NULL,
    file_name    text NOT NULL,
    sha256       text NOT NULL,
    version      int NOT NULL DEFAULT 1,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_corpus_idx ON documents (corpus_id);

-- Chunks are versioned: a re-ingested document bumps documents.version and
-- inserts a new generation of chunks. Old chunks are retained so historical
-- answers keep resolving to the exact text that produced them (handoff S29).
CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    document_id  text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version      int NOT NULL,
    page         int,
    section      text,
    ord          int NOT NULL,
    start_char   int NOT NULL,
    end_char     int NOT NULL,
    text         text NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding    vector(1024)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id, version);

CREATE TABLE IF NOT EXISTS answers (
    id          serial PRIMARY KEY,
    corpus_id   int NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    question    text NOT NULL,
    status      text NOT NULL,
    result      jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
