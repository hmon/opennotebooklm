"""The wire format we send llama-server.

The grammar constraint is what keeps structured stages parseable, so the
response_format block is worth pinning: if it silently stops being sent, the
model starts free-forming JSON and stages begin failing as abstentions.
"""

from __future__ import annotations

import functools

import httpx
import pytest

from app import llm


@pytest.fixture
def captured(monkeypatch):
    """Swap the HTTP client for one that records the request and replies."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        seen["url"] = str(request.url)
        import json

        seen["json"] = json.loads(seen["body"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"sufficiency": "SUFFICIENT", "answerable": true,'
                            ' "confidence": 0.9}'
                        }
                    }
                ]
            },
        )

    fake = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    monkeypatch.setattr(llm, "client", functools.cache(lambda: fake))
    return seen


def test_schema_is_sent_as_a_grammar_constraint(captured):
    result = llm.call(llm.Answerability, "system", "user")
    assert result.sufficiency == "SUFFICIENT"

    body = captured["json"]
    assert captured["url"].endswith("/v1/chat/completions")
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]["schema"]
    # Field names, not validation aliases: the synonyms we accept are a parsing
    # concession, not what we ask the model for.
    assert "sufficiency" in schema["properties"]
    assert "evidence_id" not in schema["properties"]
    assert body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_classify_constrains_the_label_set(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["json"] = json.loads(request.read())
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"label": "ENTAILED"}'}}]}
        )

    fake = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    monkeypatch.setattr(llm, "client", functools.cache(lambda: fake))

    verdicts = ("ENTAILED", "NOT_ENTAILED", "CONTRADICTED", "AMBIGUOUS")
    assert llm.classify(verdicts, "system", "user", default="AMBIGUOUS") == "ENTAILED"
    schema = seen["json"]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["label"]["enum"] == list(verdicts)


def test_an_unreachable_server_abstains(monkeypatch):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no server")

    fake = httpx.Client(transport=httpx.MockTransport(boom), base_url="http://test")
    monkeypatch.setattr(llm, "client", functools.cache(lambda: fake))

    assert llm.classify(("YES", "NO"), "s", "u", default="NO") == "NO"
    with pytest.raises(llm.LLMError):
        llm.call(llm.Answerability, "s", "u")
