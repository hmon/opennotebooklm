"""Embedding and reranking over HTTP.

These guard the two failure modes that would corrupt retrieval quietly: a
vector attached to the wrong chunk, and a vector of the wrong width being
written to the database.
"""

from __future__ import annotations

import functools
import json

import httpx
import pytest

from app import models
from app.config import EMBED_DIM


def _client(monkeypatch, handler, which="embed"):
    fake = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    monkeypatch.setattr(models, f"{which}_client", functools.cache(lambda: fake))
    return fake


def _vector(seed: float) -> list[float]:
    return [seed] * EMBED_DIM


def test_embeddings_are_reordered_by_index(monkeypatch):
    """A server may answer out of order; a mismatched vector is worse than none."""

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.read())["input"]
        data = [
            {"index": i, "embedding": _vector(float(i))} for i in reversed(range(len(texts)))
        ]
        return httpx.Response(200, json={"data": data})

    _client(monkeypatch, handler)
    vectors = models.embed_passages(["a", "b", "c"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_wrong_width_is_refused(monkeypatch):
    """A dimension mismatch means the served model is not the configured one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 384}]})

    _client(monkeypatch, handler)
    with pytest.raises(models.EmbeddingError, match="dim"):
        models.embed_query("anything")


def test_a_short_reply_is_refused(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": _vector(0.1)}]})

    _client(monkeypatch, handler)
    with pytest.raises(models.EmbeddingError):
        models.embed_passages(["a", "b"])


def test_embedding_failure_is_fatal(monkeypatch):
    """Ingestion must not store chunks that dense retrieval cannot see."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no server")

    _client(monkeypatch, handler)
    with pytest.raises(models.EmbeddingError):
        models.embed_passages(["a"])


def test_embedding_batches_large_inputs(monkeypatch):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.read())["input"]
        calls.append(len(texts))
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": _vector(0.1)} for i in range(len(texts))]},
        )

    _client(monkeypatch, handler)
    monkeypatch.setattr(models, "EMBED_BATCH", 4)
    assert len(models.embed_passages(["x"] * 10)) == 10
    assert sum(calls) == 10
    assert max(calls) <= 4


def test_rerank_scores_land_on_the_right_passages(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 4.4},
                    {"index": 0, "relevance_score": -9.7},
                ]
            },
        )

    _client(monkeypatch, handler, which="rerank")
    assert models.rerank_scores("q", ["irrelevant", "relevant"]) == [-9.7, 4.4]


def test_an_unreachable_reranker_leaves_the_fused_order(monkeypatch):
    """Reranking is an improvement, not a requirement: never drop the query."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no server")

    _client(monkeypatch, handler, which="rerank")
    assert models.rerank_scores("q", ["a", "b"]) == [0.0, 0.0]
