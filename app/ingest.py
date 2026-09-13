"""Parse -> chunk -> embed -> store, with provenance and versioning."""

from __future__ import annotations

import hashlib

from app import db
from app.chunking import chunk_document, guess_source_type, guess_title, parse
from app.models import embed_passages


def ingest(corpus_id: int, file_name: str, data: bytes, title: str | None = None) -> dict:
    source_type = guess_source_type(file_name)
    sha = hashlib.sha256(data).hexdigest()
    # Scoped to the corpus: the same file uploaded to two corpora is two
    # documents, so one corpus can never reach another's chunks (handoff S28).
    doc_id = f"doc_c{corpus_id}_{sha[:12]}"

    existing = db.fetchone("SELECT version FROM documents WHERE id = %s", (doc_id,))
    if existing:
        # Identical bytes already in this corpus: nothing to do (S29).
        return {
            "document_id": doc_id,
            "version": existing["version"],
            "chunks": 0,
            "status": "unchanged",
        }

    # Same file name in this corpus with different bytes => a new version of it.
    # The old document and its chunks stay, so earlier citations still resolve.
    prior = db.fetchone(
        "SELECT version FROM documents WHERE corpus_id = %s AND file_name = %s"
        " ORDER BY version DESC LIMIT 1",
        (corpus_id, file_name),
    )
    version = (prior["version"] + 1) if prior else 1

    pages = parse(data, source_type)
    chunks = chunk_document(pages)
    title = title or guess_title(pages, file_name)
    if not chunks:
        raise ValueError(f"no extractable text in {file_name}")

    vectors = embed_passages([c.text for c in chunks])

    with db.connection() as conn:
        conn.execute(
            "INSERT INTO documents (id, corpus_id, title, source_type, file_name, sha256, version)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (doc_id, corpus_id, title, source_type, file_name, sha, version),
        )
        for chunk, vector in zip(chunks, vectors, strict=True):
            page_part = f"page{chunk.page}" if chunk.page is not None else "p0"
            chunk_id = f"{doc_id}_v{version}_{page_part}_chunk{chunk.ord:03d}"
            conn.execute(
                "INSERT INTO chunks (id, document_id, version, page, section, ord,"
                " start_char, end_char, text, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (id) DO NOTHING",
                (
                    chunk_id,
                    doc_id,
                    version,
                    chunk.page,
                    chunk.section,
                    chunk.ord,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.text,
                    vector,
                ),
            )
        conn.commit()

    return {
        "document_id": doc_id,
        "version": version,
        "chunks": len(chunks),
        "status": "ingested",
    }
