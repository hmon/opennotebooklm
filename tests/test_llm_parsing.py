"""The model's output is near-JSON often enough that parsing it is load-bearing.

Every stage that fails to parse becomes an abstention, so a syntax slip the
model makes routinely would quietly destroy answer quality.
"""

from __future__ import annotations

import pytest

from app.llm import Answerability, EvidenceSet, _coerce, _json_payload


@pytest.mark.parametrize(
    "raw",
    [
        '{"sufficiency": "SUFFICIENT", "answerable": true, "confidence": 1.0}',
        '```json\n{"sufficiency": "SUFFICIENT", "answerable": true, "confidence": 1.0}\n```',
        'Here you go:\n{"sufficiency": "SUFFICIENT", "answerable": true, "confidence": 1.0}',
        '{sufficiency: "SUFFICIENT", answerable: true, confidence: 1.0}',
        "{sufficiency=SUFFICIENT, answerable=true, confidence=1.0}",
        '{"sufficiency": "SUFFICIENT", "answerable": true, "confidence": 1.0,}',
    ],
)
def test_model_dialects_all_parse(raw):
    result = _coerce(Answerability, raw)
    assert result.sufficiency == "SUFFICIENT"
    assert result.answerable is True


def test_repair_leaves_string_contents_alone():
    """Punctuation inside a quoted span must survive: it is evidence text."""
    raw = '{"evidence": [{"evidence_id": "E1", "source_span": "Results: it ran, in 2014."}]}'
    assert _coerce(EvidenceSet, raw).evidence[0].source_span == "Results: it ran, in 2014."


def test_field_name_synonyms_are_accepted():
    raw = '{"evidence": [{"id": "E1", "span": "The final sample was 218."}]}'
    item = _coerce(EvidenceSet, raw).evidence[0]
    assert item.evidence_id == "E1"
    assert item.source_span == "The final sample was 218."


def test_json_payload_strips_fences():
    assert _json_payload('```json\n{"a": 1}\n```') == '{"a": 1}'
