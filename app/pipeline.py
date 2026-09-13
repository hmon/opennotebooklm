"""answer_question(): the whole grounded path, with abstention at every gate."""

from __future__ import annotations

import json

from app import db, grounding, llm
from app.llm import Claim, LLMError, Prose, Sentence
from app.retrieval import hybrid_retrieve

NO_EVIDENCE = "The provided sources do not contain enough information to answer this question."
NOT_VERIFIED = (
    "The provided sources do not contain enough verified information to answer this question."
)
FAILED = "The available sources were not sufficient to produce a fully supported answer."


def _abstain(status: str, answer: str, **extra) -> dict:
    return {"status": status, "answer": answer, "claims": [], "citations": [], **extra}


def _conflict_claims(claim_set, evidence_by_id) -> list[tuple[Claim, list]]:
    """Turn detected disagreements into claims about the disagreement itself.

    We never pick a side (handoff S12). The claim asserts only that the sources
    differ, which the cited spans themselves establish.
    """
    out = []
    for i, conflict in enumerate(claim_set.conflicts):
        parts = []
        cited = []
        for position in conflict.positions:
            candidates = evidence_by_id.get(position.evidence_id) or []
            if not candidates:
                continue
            ev = candidates[0]
            # The wording quoted here must be verified text, not whatever the
            # model wrote into the conflict record. Use its span only when it is
            # genuinely present in that source; otherwise quote the span we
            # already grounded.
            quoted = ev.span
            located = grounding.locate_span(position.source_span, ev.passage.text)
            if located is not None:
                quoted = ev.passage.text[located[0] : located[1]]
            parts.append(f'"{quoted}"')
            cited.append(ev)
        if len(cited) < 2:
            continue
        claim = Claim(
            claim_id=f"X{i + 1}",
            text=(
                f"The provided sources disagree about {conflict.topic}: "
                + " / ".join(parts)
            ),
            kind="DERIVED",
            evidence_ids=list(dict.fromkeys(e.evidence_id for e in cited)),
        )
        out.append((claim, cited))
    return out


def answer_question(corpus_id: int, question: str, mode: str = "synthesis") -> dict:
    passages = hybrid_retrieve(corpus_id, question)
    if not passages:
        return _persist(corpus_id, question, _abstain("insufficient_evidence", NO_EVIDENCE))

    ids = [f"E{i + 1}" for i in range(len(passages))]

    try:
        answerability = grounding.check_answerability(question, passages, ids)
    except LLMError as exc:
        return _persist(corpus_id, question, _abstain("verification_failed", FAILED, detail=str(exc)))

    ok, message = grounding.gate_passes(answerability)
    if not ok:
        return _persist(
            corpus_id,
            question,
            _abstain(
                "insufficient_evidence",
                message,
                missing_information=answerability.missing_information,
                unsupported_premises=answerability.unsupported_premises,
            ),
        )

    try:
        evidence = grounding.extract_evidence(question, passages, ids)
    except LLMError:
        return _persist(corpus_id, question, _abstain("verification_failed", FAILED))
    if not evidence:
        return _persist(corpus_id, question, _abstain("insufficient_evidence", NO_EVIDENCE))

    evidence_by_id = grounding.group_evidence(evidence)

    try:
        claim_set = grounding.generate_claims(question, evidence)
    except LLMError:
        return _persist(corpus_id, question, _abstain("verification_failed", FAILED))
    if not claim_set.claims and not claim_set.conflicts:
        return _persist(corpus_id, question, _abstain("insufficient_evidence", NOT_VERIFIED))

    statuses = grounding.verify_claims(claim_set, evidence)
    verified = [c for c in claim_set.claims if statuses.get(c.claim_id) == "ENTAILED"]

    # Conflict claims are asserted by the spans themselves; they bypass the
    # entailment check (which compares a claim to one side at a time).
    conflicts = _conflict_claims(claim_set, evidence_by_id)
    verified += [claim for claim, _ in conflicts]

    if not verified:
        return _persist(corpus_id, question, _abstain("insufficient_evidence", NOT_VERIFIED))

    # Verbalization is formatting, not reasoning. If the model mangles it, fall
    # back to stating the verified claims themselves: that is the most
    # conservative possible rendering, already verified and already cited, so a
    # formatting slip costs fluency rather than a supported answer.
    verbalization = "model"
    try:
        prose = grounding.verbalize(question, verified, mode=mode)
    except LLMError:
        prose, verbalization = _claims_verbatim(verified), "claims"

    claim_ids = {c.claim_id for c in verified}
    ok, reason = grounding.check_final(prose, claim_ids)
    if not ok:
        if verbalization == "claims":
            return _persist(
                corpus_id, question, _abstain("verification_failed", FAILED, detail=reason)
            )
        prose, verbalization = _claims_verbatim(verified), "claims"

    result = _render(verified, prose, evidence_by_id, mode)
    result["verbalization"] = verbalization
    return _persist(corpus_id, question, result)


def _claims_verbatim(verified: list[Claim]) -> Prose:
    return Prose(
        sentences=[Sentence(sentence=c.text.strip(), claim_ids=[c.claim_id]) for c in verified]
    )


def _render(verified, prose, evidence_by_id, mode) -> dict:
    """Attach citation markers programmatically. The model never writes a [n]."""
    claims_out = []
    marker_of: dict[str, int] = {}
    for claim in verified:
        citations = [
            item.citation()
            for eid in claim.evidence_ids
            for item in evidence_by_id.get(eid, [])
        ]
        marker_of[claim.claim_id] = len(claims_out) + 1
        claims_out.append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "kind": claim.kind,
                "verification": "ENTAILED",
                "marker": marker_of[claim.claim_id],
                "citations": citations,
            }
        )

    parts = []
    for sentence in prose.sentences:
        text = sentence.sentence.strip()
        if not text:
            continue
        markers = sorted({marker_of[cid] for cid in sentence.claim_ids if cid in marker_of})
        suffix = "".join(f"[{m}]" for m in markers)
        parts.append(f"{text}{suffix}" if suffix else text)

    return {
        "status": "answered",
        "answer": " ".join(parts),
        "mode": mode,
        "claims": claims_out,
        "citations": [c for claim in claims_out for c in claim["citations"]],
    }


def _persist(corpus_id: int, question: str, result: dict) -> dict:
    try:
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO answers (corpus_id, question, status, result)"
                " VALUES (%s, %s, %s, %s)",
                (corpus_id, question, result["status"], json.dumps(result)),
            )
            conn.commit()
    except Exception:  # noqa: BLE001 - logging an answer must never fail a request
        pass
    return result
