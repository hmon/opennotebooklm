"""Retrieval behaviour that needs a real database. Skipped when none is up."""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip("psycopg")

from app import db  # noqa: E402
from app.chunking import Page, chunk_page  # noqa: E402


@pytest.fixture(scope="module")
def corpus() -> int:
    try:
        db.init()
    except Exception as exc:  # noqa: BLE001 - no database in this environment
        pytest.skip(f"database unavailable: {exc}")

    row = db.fetchone(
        "INSERT INTO corpora (name) VALUES ('__pytest__')"
        " ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
    )
    corpus_id = row["id"]
    with db.connection() as conn:
        conn.execute("DELETE FROM documents WHERE corpus_id = %s", (corpus_id,))
        conn.execute(
            "INSERT INTO documents (id, corpus_id, title, source_type, file_name, sha256)"
            " VALUES ('doc_test', %s, 'Test', 'text', 'test.md', 'x')",
            (corpus_id,),
        )
        conn.execute(
            "INSERT INTO chunks (id, document_id, version, page, section, ord,"
            " start_char, end_char, text)"
            " VALUES ('doc_test_v1_p0_chunk000', 'doc_test', 1, NULL, 'Methods', 0, 0, 46,"
            " 'The final sample consisted of 218 participants.')",
        )
        conn.commit()
    return corpus_id


def test_lexical_matches_a_natural_language_question(corpus):
    """Regression: an AND-ed tsquery made every multi-word question miss."""
    from app.retrieval import lexical

    hits = lexical(corpus, "how many participants were in the final sample?")
    assert [h["chunk_id"] for h in hits] == ["doc_test_v1_p0_chunk000"]


def test_lexical_returns_nothing_for_an_absent_topic(corpus):
    from app.retrieval import lexical

    assert lexical(corpus, "Who wrote Hamlet?") == []


def test_retrieval_is_scoped_to_one_corpus(corpus):
    """Corpus isolation is a security boundary, not a filter (handoff S28)."""
    from app.retrieval import lexical

    other = db.fetchone(
        "INSERT INTO corpora (name) VALUES ('__pytest_other__')"
        " ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
    )["id"]
    assert lexical(other, "how many participants") == []
