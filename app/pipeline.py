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


CONFLICT_PREFIX = "The provided sources disagree: "


def _split_contradictions(
    verified: list[Claim], evidence_by_id: dict, all_evidence: list | None = None
) -> tuple[list[Claim], set[str]]:
    """Replace any claim its own evidence disagrees with by a statement of the
    disagreement.

    A claim can be entailed by the evidence it cites while other retrieved
    evidence gives a different figure for the same fact. Publishing it then
    picks a side, and where the disagreeing span is among its own citations it
    also cites that source as support. Neither is ours to do (handoff S12), so
    the claim becomes a report of the conflict, citing every side.

    The search covers all extracted evidence, not only what the claim cites:
    the model decides what to cite, and a claim that cites one year while the
    corpus plainly states another is exactly the case worth catching.
    """
    out: list[Claim] = []
    conflicted: set[str] = set()
    for claim in verified:
        cited = [item for eid in claim.evidence_ids for item in evidence_by_id.get(eid, [])]
        pool = all_evidence if all_evidence is not None else cited
        conflicting = []
        seen: set[tuple[str, str]] = set()
        for item in pool:
            key = (item.evidence_id, item.span)
            if key in seen or not grounding.contradicts(claim.text, item.span):
                continue
            seen.add(key)
            conflicting.append(item)
        supporting = [item for item in cited if item not in conflicting]

        if not conflicting or not supporting:
            # No disagreement, or nothing left to disagree with: keep the claim
            # but never keep a citation that contradicts it.
            keep = supporting or cited
            out.append(
                claim.model_copy(
                    update={
                        "evidence_ids": list(
                            dict.fromkeys(item.evidence_id for item in keep)
                        )
                    }
                )
            )
            continue

        sides = supporting[:1] + conflicting
        quoted = " / ".join(f'"{item.span}"' for item in sides)
        conflicted.add(claim.claim_id)
        out.append(
            claim.model_copy(
                update={
                    "text": CONFLICT_PREFIX + quoted,
                    "kind": "DERIVED",
                    "evidence_ids": list(dict.fromkeys(item.evidence_id for item in sides)),
                }
            )
        )
    return out, conflicted


def answer_question(corpus_id: int, question: str, mode: str = "synthesis") -> dict:
    passages = hybrid_retrieve(corpus_id, question)
    if not passages:
        return _persist(corpus_id, question, _abstain("insufficient_evidence", NO_EVIDENCE))

    ids = [f"E{i + 1}" for i in range(len(passages))]

    try:
        answerability = grounding.check_answerability(question, passages, ids)
    except LLMError as exc:
        return _persist(corpus_id, question, _abstain("verification_failed", FAILED, detail=str(exc)))

    ok, message = grounding.gate_passes(answerability, question)
    if not ok:
        return _persist(
            corpus_id,
            question,
            _abstain(
                "insufficient_evidence",
                message,
                missing_information=answerability.missing_information,
                unsupported_premises=[
                    p
                    for p in answerability.unsupported_premises
                    if grounding.premise_stated_in_question(p, question)
                ],
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

    # Search the retrieved text itself, not only the spans the model chose to
    # extract, so a disagreement cannot hide behind a one-sided extraction.
    conflict_pool = evidence + grounding.sentences_from_passages(passages, ids)
    verified, conflicted = _split_contradictions(verified, evidence_by_id, conflict_pool)

    # Verbalization is formatting, not reasoning. If the model mangles it, fall
    # back to stating the verified claims themselves: that is the most
    # conservative possible rendering, already verified and already cited, so a
    # formatting slip costs fluency rather than a supported answer.
    # A statement that the sources disagree must not be rephrased: the model
    # turns it back into two flat assertions, which reads as the system
    # asserting both. Those claims bypass the writer and are rendered as they
    # stand; only the undisputed ones are handed over to be made readable.
    ordinary = [c for c in verified if c.claim_id not in conflicted]
    conflicts_out = [c for c in verified if c.claim_id in conflicted]

    verbalization = "model"
    if ordinary:
        try:
            prose = grounding.verbalize(question, ordinary, mode=mode)
        except LLMError:
            prose, verbalization = _claims_verbatim(ordinary), "claims"
    else:
        prose, verbalization = Prose(sentences=[]), "claims"
    prose.sentences.extend(
        Sentence(sentence=c.text, claim_ids=[c.claim_id]) for c in conflicts_out
    )

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
