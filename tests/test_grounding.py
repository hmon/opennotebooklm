"""The guards that make the corpus authoritative. All LLM calls are mocked:
these test the enforcement layer, not the model.
"""

from __future__ import annotations

import pytest

from app import grounding, llm, pipeline
from app.chunking import Page, chunk_page
from app.grounding import locate_span
from app.llm import Answerability, Claim, ClaimSet, Evidence, EvidenceSet, Prose, Sentence
from app.retrieval import Passage

TEXT = "The final sample consisted of 218 participants. Participants were randomly assigned."


def passage(text: str = TEXT, chunk_id: str = "doc_a_v1_page17_chunk002") -> Passage:
    return Passage(
        chunk_id=chunk_id,
        document_id="doc_a",
        version=1,
        page=17,
        section="Methods",
        ord=2,
        start_char=440,
        end_char=440 + len(text),
        text=text,
        title="Example Study",
    )


# --- span grounding ------------------------------------------------------

def test_locate_span_finds_verbatim_text():
    assert locate_span("218 participants", TEXT) == (30, 46)


def test_locate_span_tolerates_whitespace_only():
    assert locate_span("consisted   of\n218", TEXT) is not None


def test_locate_span_rejects_invented_text():
    assert locate_span("The trial was conducted in Germany.", TEXT) is None


def test_fabricated_span_is_dropped(monkeypatch):
    """A quote the model invents must never reach a citation."""
    monkeypatch.setattr(
        llm,
        "call",
        lambda *a, **k: EvidenceSet(
            evidence=[
                Evidence(evidence_id="E1", source_span="The final sample consisted of 218 participants."),
                Evidence(evidence_id="E1", source_span="The trial took place in Germany."),
                Evidence(evidence_id="E9", source_span="anything"),
            ]
        ),
    )
    evidence = grounding.extract_evidence("how many?", [passage()], ["E1"])
    assert [e.span for e in evidence] == ["The final sample consisted of 218 participants."]


def test_citation_offsets_point_into_the_document(monkeypatch):
    monkeypatch.setattr(
        llm,
        "call",
        lambda *a, **k: EvidenceSet(
            evidence=[Evidence(evidence_id="E1", source_span="218 participants")]
        ),
    )
    citation = grounding.extract_evidence("q", [passage()], ["E1"])[0].citation()
    assert citation["start_char"] == 440 + 30
    assert citation["quote"] == "218 participants"
    assert citation["page"] == 17


# --- the answerability gate ---------------------------------------------

@pytest.mark.parametrize(
    "result",
    [
        Answerability(sufficiency="INSUFFICIENT", answerable=False, confidence=0.98,
                      missing_information=["no sample size"]),
        Answerability(sufficiency="SUFFICIENT", answerable=True, confidence=0.4),
        Answerability(sufficiency="SUFFICIENT", answerable=True, confidence=0.99,
                      unsupported_premises=["the trial was in Germany"]),
        Answerability(sufficiency="SUFFICIENT", answerable=False, confidence=0.9),
    ],
)
def test_gate_abstains(result):
    ok, message = grounding.gate_passes(result)
    assert not ok
    assert message


def test_gate_names_the_unsupported_premise():
    result = Answerability(
        sufficiency="SUFFICIENT", answerable=True, confidence=0.99,
        unsupported_premises=["the trial was in Germany"],
    )
    _, message = grounding.gate_passes(result)
    assert "do not establish the premise" in message
    assert "Germany" in message


def test_gate_passes_when_evidence_is_sufficient():
    result = Answerability(sufficiency="SUFFICIENT", answerable=True, confidence=0.95)
    ok, message = grounding.gate_passes(result)
    assert ok and message is None


def test_gate_passes_partial_so_the_supported_part_can_be_answered():
    result = Answerability(sufficiency="PARTIAL", answerable=True, confidence=0.9)
    assert grounding.gate_passes(result)[0]


def test_gate_tolerates_a_missing_confidence():
    result = Answerability(sufficiency="SUFFICIENT", answerable=True, confidence=None)
    assert grounding.gate_passes(result)[0]


# --- claim hygiene -------------------------------------------------------

def _evidence(monkeypatch, spans: dict[str, str]):
    passages = [passage(text, f"chunk_{eid}") for eid, text in spans.items()]
    ids = list(spans)
    monkeypatch.setattr(
        llm,
        "call",
        lambda *a, **k: EvidenceSet(
            evidence=[Evidence(evidence_id=eid, source_span=text) for eid, text in spans.items()]
        ),
    )
    return grounding.extract_evidence("q", passages, ids)


def test_claims_citing_nothing_real_are_dropped(monkeypatch):
    evidence = _evidence(monkeypatch, {"E1": "The study included 218 participants."})
    monkeypatch.setattr(
        llm,
        "call",
        lambda *a, **k: ClaimSet(
            claims=[
                Claim(claim_id="C1", text="The study included 218 participants.", evidence_ids=["E1"]),
                Claim(claim_id="C2", text="The study took place in Germany.", evidence_ids=["E7"]),
                Claim(claim_id="C3", text="The study was funded publicly.", evidence_ids=[]),
            ]
        ),
    )
    claims = grounding.generate_claims("q", evidence)
    assert [c.claim_id for c in claims.claims] == ["C1"]


def test_derived_needs_two_sources(monkeypatch):
    evidence = _evidence(monkeypatch, {"E1": "The study included 218 participants."})
    monkeypatch.setattr(
        llm,
        "call",
        lambda *a, **k: ClaimSet(
            claims=[Claim(claim_id="C1", text="Both studies agree.", kind="DERIVED", evidence_ids=["E1"])]
        ),
    )
    assert grounding.generate_claims("q", evidence).claims[0].kind == "DIRECT"


def test_verifier_sees_only_the_claim_and_its_spans(monkeypatch):
    evidence = _evidence(monkeypatch, {"E1": "The trial enrolled 218 adults."})
    seen = {}

    def fake_classify(labels, system, user, default):
        seen["user"] = user
        return "ENTAILED"

    monkeypatch.setattr(llm, "classify", fake_classify)
    claims = ClaimSet(claims=[Claim(claim_id="C1", text="The trial included 218 participants.", evidence_ids=["E1"])])
    statuses = grounding.verify_claims(claims, evidence)
    assert statuses == {"C1": "ENTAILED"}
    assert "218 adults" in seen["user"]
    assert "q" not in seen["user"].split("<claim>")[0].replace("<evidence", "")


def test_unverified_claims_are_excluded(monkeypatch):
    evidence = _evidence(monkeypatch, {"E1": "The trial enrolled 218 adults."})
    monkeypatch.setattr(llm, "classify", lambda *a, **k: "NOT_ENTAILED")
    claims = ClaimSet(claims=[Claim(claim_id="C1", text="The trial included 218 men.", evidence_ids=["E1"])])
    statuses = grounding.verify_claims(claims, evidence)
    assert statuses["C1"] == "NOT_ENTAILED"
    assert [c for c in claims.claims if statuses.get(c.claim_id) == "ENTAILED"] == []


# --- the final check -----------------------------------------------------

def test_final_check_rejects_added_sentences():
    prose = Prose(
        sentences=[
            Sentence(sentence="The study included 218 participants.", claim_ids=["C1"]),
            Sentence(sentence="This is consistent with similar trials in the field.", claim_ids=[]),
        ]
    )
    ok, reason = grounding.check_final(prose, {"C1"})
    assert not ok and "unsupported sentence" in reason


def test_final_check_rejects_unknown_claim_ids():
    prose = Prose(sentences=[Sentence(sentence="The trial was in Germany.", claim_ids=["C9"])])
    ok, reason = grounding.check_final(prose, {"C1"})
    assert not ok and "unknown claim" in reason


def test_final_check_accepts_grounded_prose():
    prose = Prose(
        sentences=[
            Sentence(sentence="The study included 218 participants.", claim_ids=["C1"]),
            Sentence(sentence="However,", claim_ids=[]),
        ]
    )
    assert grounding.check_final(prose, {"C1"})[0]


# --- citation markers ----------------------------------------------------

def test_markers_are_generated_not_authored(monkeypatch):
    evidence = _evidence(monkeypatch, {"E1": "The final sample consisted of 218 participants."})
    verified = [Claim(claim_id="C1", text="The study included 218 participants.", evidence_ids=["E1"])]
    prose = Prose(sentences=[Sentence(sentence="The study included 218 participants.", claim_ids=["C1"])])
    result = pipeline._render(verified, prose, grounding.group_evidence(evidence), "synthesis")
    assert result["answer"] == "The study included 218 participants.[1]"
    assert result["claims"][0]["citations"][0]["quote"] == "The final sample consisted of 218 participants."


# --- chunking provenance -------------------------------------------------

def test_chunker_keeps_section_and_offsets():
    text = "# Paper\n\n## 3.2 Results\n\nThe effect was large.\n\nA second paragraph follows here.\n"
    chunks = chunk_page(Page(14, text), max_tokens=6)
    assert chunks
    assert chunks[0].page == 14
    assert chunks[0].section == "3.2 Results"
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char].strip() == chunk.text


def test_chunker_splits_long_pages():
    body = "\n\n".join(f"Paragraph number {i} with several words in it." for i in range(20))
    chunks = chunk_page(Page(1, body), max_tokens=20)
    assert len(chunks) > 1


# --- verbalization fallback ---------------------------------------------

def test_claims_verbatim_fallback_is_self_consistent():
    """A mangled prose stage must not discard claims that already verified."""
    verified = [
        Claim(claim_id="C1", text="The study included 218 participants.", evidence_ids=["E1"]),
        Claim(claim_id="C2", text="The study lasted 12 months.", evidence_ids=["E2"]),
    ]
    prose = pipeline._claims_verbatim(verified)
    ok, reason = grounding.check_final(prose, {"C1", "C2"})
    assert ok, reason
    assert [s.sentence for s in prose.sentences] == [c.text for c in verified]


# --- contradictory sources ----------------------------------------------

def test_conflict_claims_quote_only_verified_text(monkeypatch):
    """A conflict must be reported from grounded spans, not model-written ones."""
    from app.llm import Conflict

    evidence = _evidence(
        monkeypatch,
        {"E1": "The trial was conducted in 2014.", "E2": "The trial was conducted in 2015."},
    )
    claim_set = ClaimSet(
        claims=[],
        conflicts=[
            Conflict(
                topic="the year of the trial",
                positions=[
                    Evidence(evidence_id="E1", source_span="The trial was conducted in 2014."),
                    Evidence(evidence_id="E2", source_span="It happened in 2019, per the funder."),
                ],
            )
        ],
    )
    claims = pipeline._conflict_claims(claim_set, grounding.group_evidence(evidence))
    assert len(claims) == 1
    text = claims[0][0].text
    assert "2014" in text and "2015" in text
    assert "2019" not in text  # the invented wording never reaches the answer


def test_conflict_needs_two_surviving_sources(monkeypatch):
    from app.llm import Conflict

    evidence = _evidence(monkeypatch, {"E1": "The trial was conducted in 2014."})
    claim_set = ClaimSet(
        claims=[],
        conflicts=[
            Conflict(
                topic="the year",
                positions=[
                    Evidence(evidence_id="E1", source_span="The trial was conducted in 2014."),
                    Evidence(evidence_id="E4", source_span="invented"),
                ],
            )
        ],
    )
    assert pipeline._conflict_claims(claim_set, grounding.group_evidence(evidence)) == []


def test_chunker_splits_a_page_with_no_blank_lines():
    """Extracted PDF text is single-newline separated; a page must still split."""
    body = "\n".join(f"Line {i} of the extracted page text here." for i in range(40))
    chunks = chunk_page(Page(3, body), max_tokens=30)
    assert len(chunks) > 1
    assert all(c.page == 3 for c in chunks)
    for chunk in chunks:
        assert body[chunk.start_char : chunk.end_char].strip() == chunk.text


def test_chunker_finds_headings_in_pdf_style_text():
    body = "\n".join(
        ["Interim Report", "1. Overview", "The columns were observed for six weeks."]
        + [f"Filler sentence number {i} to push past the limit." for i in range(30)]
        + ["2. Findings", "The mean deflection was 3.4 millimetres."]
    )
    sections = {c.section for c in chunk_page(Page(1, body), max_tokens=25)}
    assert "2. Findings" in sections


def test_several_spans_from_one_source_are_all_kept(monkeypatch):
    """Two sentences from the same source must not collapse into one."""
    text = "The study included 218 participants. The study lasted 12 months."
    monkeypatch.setattr(
        llm,
        "call",
        lambda *a, **k: EvidenceSet(
            evidence=[
                Evidence(evidence_id="E1", source_span="The study included 218 participants."),
                Evidence(evidence_id="E1", source_span="The study lasted 12 months."),
            ]
        ),
    )
    evidence = grounding.extract_evidence("q", [passage(text)], ["E1"])
    grouped = grounding.group_evidence(evidence)
    assert len(grouped["E1"]) == 2

    claim = Claim(claim_id="C1", text="The study lasted 12 months.", evidence_ids=["E1"])
    prose = Prose(sentences=[Sentence(sentence=claim.text, claim_ids=["C1"])])
    result = pipeline._render([claim], prose, grouped, "synthesis")
    quotes = [c["quote"] for c in result["claims"][0]["citations"]]
    assert "The study lasted 12 months." in quotes
    assert len(quotes) == 2
