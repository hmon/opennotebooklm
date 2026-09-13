"""Embedding and reranking, served remotely by llama.cpp.

Nothing here loads weights. Each model runs as its own llama-server on the
inference host and is reached over HTTP, so this process stays small enough to
run anywhere and the models can be swapped by changing a URL.

A failure to embed is fatal to ingestion, because a chunk with no vector is
invisible to dense retrieval and would silently degrade recall. A failure to
rerank is not: retrieval still has the fused ranking to fall back on.
"""

from __future__ import annotations

import functools

import httpx

from app.config import (
    EMBED_BATCH,
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_URL,
    LLAMA_API_KEY,
    MODEL_TIMEOUT,
    RERANK_MODEL,
    RERANK_URL,
)


class EmbeddingError(RuntimeError):
    """The embedding service could not vectorize the given text."""


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {LLAMA_API_KEY}"} if LLAMA_API_KEY else {}


def _transport() -> httpx.HTTPTransport:
    """Retry on connection failures only.

    The model servers are reached through an SSH tunnel that occasionally drops
    and reconnects. A blink of the transport is not the model declining, and
    should not be reported as one. Retries here never apply to an HTTP error
    status, so a real refusal still surfaces immediately.
    """
    return httpx.HTTPTransport(retries=3)


@functools.cache
def embed_client() -> httpx.Client:
    return httpx.Client(
        base_url=EMBED_URL, timeout=MODEL_TIMEOUT, headers=_headers(), transport=_transport()
    )


@functools.cache
def rerank_client() -> httpx.Client:
    return httpx.Client(
        base_url=RERANK_URL, timeout=MODEL_TIMEOUT, headers=_headers(), transport=_transport()
    )


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, in batches the server will accept."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        try:
            response = embed_client().post(
                "/v1/embeddings", json={"model": EMBED_MODEL, "input": batch}
            )
            response.raise_for_status()
            data = response.json()["data"]
        except Exception as exc:  # noqa: BLE001 - surfaced as EmbeddingError
            raise EmbeddingError(f"embedding request failed: {exc}") from exc

        # The API does not promise ordering, and a mismatched vector is worse
        # than a missing one: it would attach a chunk to another chunk's meaning.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        if len(ordered) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} vectors, got {len(ordered)}")
        for item in ordered:
            vector = item["embedding"]
            if len(vector) != EMBED_DIM:
                raise EmbeddingError(
                    f"expected {EMBED_DIM}-dim vectors, got {len(vector)}; "
                    "EMBED_DIM and the served model disagree"
                )
            vectors.append(vector)
    return vectors


def embed_passages(texts: list[str]) -> list[list[float]]:
    return _embed(texts) if texts else []


def embed_query(text: str) -> list[float]:
    # BGE-M3 needs no instruction prefix on either side, unlike the bge-v1.5
    # family, so queries and passages are embedded the same way.
    return _embed([text])[0]


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    """Score each passage against the question.

    Returns zeros when the reranker is unreachable, which leaves the fused
    retrieval order intact rather than dropping the query.
    """
    if not texts:
        return []
    try:
        response = rerank_client().post(
            "/v1/rerank",
            json={"model": RERANK_MODEL, "query": question, "documents": texts},
        )
        response.raise_for_status()
        results = response.json()["results"]
    except Exception:  # noqa: BLE001 - reranking is an improvement, not a requirement
        return [0.0] * len(texts)

    scores = [0.0] * len(texts)
    for item in results:
        index = item.get("index")
        if index is not None and 0 <= index < len(texts):
            scores[index] = float(item["relevance_score"])
    return scores
