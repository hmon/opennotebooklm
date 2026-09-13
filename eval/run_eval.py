"""Grounding eval: measures what this system is actually for.

Ordinary QA accuracy is not the metric. False answer rate on unanswerable
questions is: it is the number that tells you whether pretrained knowledge is
leaking past the corpus boundary.

    uv run python eval/run_eval.py [--corpus NAME] [--keep]
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.grounding import locate_span  # noqa: E402
from app.ingest import ingest  # noqa: E402
from app.pipeline import answer_question  # noqa: E402
from app.retrieval import hybrid_retrieve  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
CASES = HERE / "cases.jsonl"


def setup_corpus(name: str) -> int:
    row = db.fetchone(
        "INSERT INTO corpora (name) VALUES (%s)"
        " ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
        (name,),
    )
    corpus_id = row["id"]
    with db.connection() as conn:
        conn.execute(
            "DELETE FROM documents WHERE corpus_id = %s", (corpus_id,)
        )
        conn.commit()
    for path in sorted(FIXTURES.glob("*.md")):
        ingest(corpus_id, path.name, path.read_bytes())
    return corpus_id


def unsupported_claims(result: dict) -> tuple[int, int]:
    """Count claims whose own quote is not present in the chunk it cites.

    Checked against the database, not against the model's say-so.
    """
    total = bad = 0
    for claim in result.get("claims", []):
        for citation in claim.get("citations", []):
            total += 1
            chunk = db.fetchone(
                "SELECT text FROM chunks WHERE id = %s", (citation["chunk_id"],)
            )
            if chunk is None or locate_span(citation["quote"], chunk["text"]) is None:
                bad += 1
    return bad, total


def run_case(corpus_id: int, case: dict) -> dict:
    started = time.time()
    retrieved = hybrid_retrieve(corpus_id, case["question"])
    recalled = None
    if case.get("must_retrieve"):
        files = {
            db.fetchone("SELECT file_name FROM documents WHERE id = %s", (p.document_id,))[
                "file_name"
            ]
            for p in retrieved
        }
        recalled = case["must_retrieve"] in files

    result = answer_question(corpus_id, case["question"])
    answer = (result.get("answer") or "").lower()

    status_ok = result["status"] == case["expect"] or (
        case["expect"] == "insufficient_evidence"
        and result["status"] in ("insufficient_evidence", "verification_failed")
    )
    contains = all(s.lower() in answer for s in case.get("must_contain", []))
    excludes = not any(s.lower() in answer for s in case.get("must_not_contain", []))
    bad_cites, total_cites = unsupported_claims(result)

    return {
        "id": case["id"],
        "class": case["class"],
        "expect": case["expect"],
        "status": result["status"],
        "status_ok": status_ok,
        "contains": contains,
        "excludes": excludes,
        "recalled": recalled,
        "bad_cites": bad_cites,
        "total_cites": total_cites,
        "pass": status_ok and contains and excludes and bad_cites == 0,
        "secs": round(time.time() - started, 1),
        "answer": result.get("answer", ""),
    }


def run(corpus_id: int, cases: list[dict], workers: int = 1) -> list[dict]:
    """Run the cases, optionally several at a time.

    The model server has four slots, and a case spends much of its wall time in
    retrieval and database work rather than generation. Concurrency is modest
    on a CPU-only host, where a single request already occupies every thread:
    expect roughly 1.3x, not 4x.
    """
    printing = threading.Lock()

    def announce(row: dict) -> None:
        with printing:
            mark = "PASS" if row["pass"] else "FAIL"
            print(f"  [{mark}] {row['id']:<3} {row['status']:<22} {row['secs']:>6}s", flush=True)

    if workers <= 1:
        rows = []
        for case in cases:
            row = run_case(corpus_id, case)
            announce(row)
            rows.append(row)
        return rows

    by_id: dict[str, dict] = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_case, corpus_id, case): case for case in cases}
        for future in futures.as_completed(pending):
            case = pending[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad case must not sink the run
                row = {
                    "id": case["id"], "class": case["class"], "expect": case["expect"],
                    "status": f"error: {exc}", "status_ok": False, "contains": False,
                    "excludes": False, "recalled": None, "bad_cites": 0, "total_cites": 0,
                    "pass": False, "secs": 0.0, "answer": "",
                }
            announce(row)
            by_id[case["id"]] = row
    # Report in the order the cases were written, not the order they finished.
    return [by_id[c["id"]] for c in cases if c["id"] in by_id]


def report(rows: list[dict]) -> int:
    by_class = lambda c: [r for r in rows if r["class"] == c]  # noqa: E731

    unanswerable = by_class("B") + [r for r in by_class("D") if r["expect"] != "answered"]
    false_answers = [r for r in unanswerable if r["status"] == "answered"]
    answerable = by_class("A")
    recall_rows = [r for r in rows if r["recalled"] is not None]
    cites = sum(r["total_cites"] for r in rows)
    bad = sum(r["bad_cites"] for r in rows)

    def pct(n, d):
        return f"{100 * n / d:5.1f}%  ({n}/{d})" if d else "    n/a"

    print("\n" + "=" * 62)
    print(f"{'Retrieval recall':<34}{pct(sum(1 for r in recall_rows if r['recalled']), len(recall_rows))}")
    print(f"{'Abstention accuracy':<34}{pct(len(unanswerable) - len(false_answers), len(unanswerable))}")
    print(f"{'FALSE ANSWER RATE':<34}{pct(len(false_answers), len(unanswerable))}")
    print(f"{'Answerable answered correctly':<34}{pct(sum(1 for r in answerable if r['pass']), len(answerable))}")
    print(f"{'Citation precision':<34}{pct(cites - bad, cites)}")
    print(f"{'Unsupported claim rate':<34}{pct(bad, cites)}")
    print(f"{'Overall pass':<34}{pct(sum(1 for r in rows if r['pass']), len(rows))}")
    print("=" * 62)

    if false_answers:
        print("\nFalse answers (answered when the corpus does not support it):")
        for r in false_answers:
            print(f"  {r['id']}: {r['answer'][:150]}")
    failures = [r for r in rows if not r["pass"] and r not in false_answers]
    if failures:
        print("\nOther failures:")
        for r in failures:
            why = []
            if not r["status_ok"]:
                why.append(f"status {r['status']} != {r['expect']}")
            if not r["contains"]:
                why.append("missing required content")
            if not r["excludes"]:
                why.append("contains forbidden content")
            if r["bad_cites"]:
                why.append(f"{r['bad_cites']} unsupported citation(s)")
            print(f"  {r['id']}: {', '.join(why)}")
            if r["answer"]:
                print(f"      {r['answer'][:150]}")

    return 1 if (false_answers or bad) else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="__eval__")
    parser.add_argument("--only", help="comma-separated case ids or classes")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("EVAL_WORKERS", "4")),
        help="cases to run at once (1 to disable)",
    )
    args = parser.parse_args()

    db.init()
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted or c["class"] in wanted]

    print(f"Ingesting fixtures into corpus '{args.corpus}'...")
    corpus_id = setup_corpus(args.corpus)
    print(f"Running {len(cases)} cases, {args.workers} at a time...\n")
    started = time.time()
    rows = run(corpus_id, cases, workers=args.workers)
    print(f"\nWall time: {time.time() - started:.0f}s")
    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main())
