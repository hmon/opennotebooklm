"""Lazily loaded local models. Import is cheap; first use pays the download."""

from __future__ import annotations

import functools

from app.config import EMBED_MODEL, RERANK_MODEL


@functools.cache
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@functools.cache
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return embedder().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    # bge asks for this prefix on the query side only.
    prefix = "Represent this sentence for searching relevant passages: "
    return embedder().encode(prefix + text, normalize_embeddings=True).tolist()


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    return [float(s) for s in reranker().predict([(question, t) for t in texts])]
