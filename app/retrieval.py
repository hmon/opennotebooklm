"""Hybrid retrieval: BM25-style lexical + dense vector, fused, then reranked.

Every query is scoped by corpus_id. Cross-corpus retrieval is a security bug
(handoff S28), so the scope lives in the SQL, not in a caller-side filter.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import db
from app.config import LEXICAL_TOP_K, RERANK_TOP_K, RRF_K, VECTOR_TOP_K
from app.models import embed_query, rerank_scores

# Only the newest version of each document participates in retrieval; older
# versions stay in the table so past citations still resolve.
_CURRENT = """
    JOIN documents d ON d.id = c.document_id AND d.version = c.version
    WHERE d.corpus_id = %(corpus_id)s
"""

_SELECT = """
    c.id AS chunk_id, c.document_id, c.version, c.page, c.section, c.ord,
    c.start_char, c.end_char, c.text, d.title
"""


@dataclass
class Passage:
    chunk_id: str
    document_id: str
    version: int
    page: int | None
    section: str | None
    ord: int
    start_char: int
    end_char: int
    text: str
    title: str
    score: float = 0.0

    def as_citation(self, quote: str, start: int, end: int) -> dict:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "version": self.version,
            "quote": quote,
            "start_char": self.start_char + start,
            "end_char": self.start_char + end,
        }


# websearch_to_tsquery ANDs every term, so a natural-language question matches
# nothing unless a chunk happens to contain all of it. Retrieval wants OR with
# rank ordering instead: stem the question through to_tsvector, then OR the
# lexemes. Going through to_tsvector also keeps the text out of tsquery syntax.
_OR_QUERY = (
    "to_tsquery('english',"
    " array_to_string(tsvector_to_array(to_tsvector('english', %(query)s)), ' | '))"
)


def lexical(corpus_id: int, query: str, top_k: int = LEXICAL_TOP_K) -> list[dict]:
    return db.fetchall(
        f"SELECT {_SELECT}, ts_rank_cd(c.tsv, {_OR_QUERY}) AS score"
        f" FROM chunks c {_CURRENT}"
        f" AND c.tsv @@ {_OR_QUERY}"
        " ORDER BY score DESC LIMIT %(k)s",
        {"corpus_id": corpus_id, "query": query, "k": top_k},
    )


def vector(corpus_id: int, query: str, top_k: int = VECTOR_TOP_K) -> list[dict]:
    return db.fetchall(
        f"SELECT {_SELECT}, 1 - (c.embedding <=> %(vec)s::vector) AS score"
        f" FROM chunks c {_CURRENT}"
        " AND c.embedding IS NOT NULL"
        " ORDER BY c.embedding <=> %(vec)s::vector LIMIT %(k)s",
        {"corpus_id": corpus_id, "vec": str(embed_query(query)), "k": top_k},
    )


def _rrf(*ranked: list[dict]) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for result_set in ranked:
        for rank, row in enumerate(result_set):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            rows.setdefault(cid, row)
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [rows[cid] | {"score": scores[cid]} for cid in order]


def hybrid_retrieve(corpus_id: int, query: str, top_k: int = RERANK_TOP_K) -> list[Passage]:
    fused = _rrf(lexical(corpus_id, query), vector(corpus_id, query))
    if not fused:
        return []
    scores = rerank_scores(query, [row["text"] for row in fused])
    ranked = sorted(zip(fused, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [
        Passage(**{k: v for k, v in row.items() if k != "score"}, score=float(score))
        for row, score in ranked[:top_k]
    ]
