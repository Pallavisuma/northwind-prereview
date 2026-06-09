"""Evaluation harness. Drop in a JSON of expected outcomes and get back the
numbers that matter for this system:

  * Verdict accuracy + violation-detection precision/recall/F1 (the core: does it
    catch real violations without over-flagging clean items?)
  * Reimbursable-amount accuracy (within a per-item tolerance)
  * Citation faithfulness (are quoted clauses real?) and citation recall (did it
    cite the clause the violation actually rests on?)
  * Retrieval recall@k (was the deciding clause even retrieved?)
  * Q&A refusal: correct-refusal rate on out-of-scope vs false-refusal on in-scope

Why these: the brief rewards a system that demonstrably works AND is honest about
uncertainty. Violation recall catches misses; faithfulness/citation-recall catch
unsupported reasoning; refusal metrics catch fabrication. Run:

    cd backend
    python -m scripts.evaluate ../eval/expected_sample.json [--out results.json]
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from app.config import SUBMISSIONS_DIR
from app.context import load_employee, Employee
from app.extraction import extract_path, SUPPORTED_EXTS
from app.ingest.embed import build_index
from app import qa
from app.review import review_line_item, LineItemReview

VIOLATION = {"flagged", "rejected"}


def parse_cite(s: str) -> tuple[str, str]:
    m = re.match(r"\s*([A-Z]+-\d+)\s*§?\s*([\d.]*)", s)
    return (m.group(1), m.group(2)) if m else (s.strip(), "")


def cites_of(review: LineItemReview) -> set[tuple[str, str]]:
    return {(c.citation.doc_id, c.citation.section) for c in review.citation_checks}


def retrieved_clauses(review: LineItemReview) -> set[tuple[str, str]]:
    return {(h.doc_id, h.section) for h in review.retrieved}


@dataclass
class Acc:
    n: int = 0
    hits: int = 0
    def add(self, ok: bool):
        self.n += 1; self.hits += int(ok)
    def rate(self) -> float:
        return self.hits / self.n if self.n else float("nan")


@dataclass
class Counts:
    tp: int = 0; fp: int = 0; fn: int = 0; tn: int = 0
    def prf(self):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")
        f = 2 * p * r / (p + r) if p and r and (p + r) else float("nan")
        return p, r, f


def _employee_for(sub_spec: dict) -> tuple[Employee, list]:
    if "name" in sub_spec:
        d = SUBMISSIONS_DIR / sub_spec["name"]
        emp = load_employee(d)
        files = sorted(p for p in (d / "receipts").iterdir()
                       if p.suffix.lower() in SUPPORTED_EXTS)
        return emp, files
    emp = Employee.model_validate(sub_spec["employee"])      # inline (held-out)
    from pathlib import Path
    files = [Path(p) for p in sub_spec.get("receipts", [])]
    return emp, files


def evaluate(expected: dict) -> dict:
    build_index(verbose=False)
    verdict_exact = Acc(); violation = Counts()
    reimb = Acc(); cite_recall = Acc(); faithful = Acc(); retr_recall = Acc()
    per_item = []

    for spec in expected.get("submissions", []):
        emp, files = _employee_for(spec)
        default_v = spec.get("default_verdict", "compliant")
        li_exp = spec.get("line_items", {})
        for f in files:
            r = extract_path(f, cache=True)
            rev = review_line_item(r, emp, filename=f.name)
            pred = rev.verdict.verdict.value
            exp = li_exp.get(f.name, {})
            exp_v = exp.get("verdict", default_v)

            verdict_exact.add(pred == exp_v)
            ev, pv = exp_v in VIOLATION, pred in VIOLATION
            if ev and pv: violation.tp += 1
            elif pv and not ev: violation.fp += 1
            elif ev and not pv: violation.fn += 1
            else: violation.tn += 1

            if "reimbursable_amount" in exp:
                tol = exp.get("reimbursable_tolerance", 1.0)
                got = rev.verdict.reimbursable_amount
                reimb.add(got is not None and abs(got - exp["reimbursable_amount"]) <= tol)
            if rev.citation_checks:
                faithful.add(rev.citations_faithful)
            if exp.get("citations"):
                want = {parse_cite(c) for c in exp["citations"]}
                cite_recall.add(bool(want & cites_of(rev)))
                retr_recall.add(bool(want & retrieved_clauses(rev)))

            per_item.append({
                "submission": spec.get("name", "?"), "file": f.name,
                "expected": exp_v, "predicted": pred,
                "match": pred == exp_v, "reimbursable": rev.verdict.reimbursable_amount,
                "citations_faithful": rev.citations_faithful,
            })

    # --- Q&A ---
    qa_out = Acc(); qa_in = Acc(); qa_cite = Acc(); qa_rows = []
    for item in expected.get("qa", []):
        res = qa.ask(item["question"])
        refused = res["refused"]
        if item.get("refused"):
            qa_out.add(refused)                       # correctly refused?
        else:
            qa_in.add(not refused)                    # correctly answered?
            if item.get("citations") and not refused:
                want = {parse_cite(c) for c in item["citations"]}
                got = {(c["doc_id"], c["section"]) for c in res["citations"]}
                qa_cite.add(bool(want & got))
        qa_rows.append({"q": item["question"], "expected_refused": item.get("refused", False),
                        "refused": refused})

    p, r, f1 = violation.prf()
    return {
        "verdict_exact_accuracy": round(verdict_exact.rate(), 3),
        "violation_detection": {"precision": round(p, 3), "recall": round(r, 3),
                                "f1": round(f1, 3), **violation.__dict__},
        "reimbursable_within_tolerance": round(reimb.rate(), 3) if reimb.n else None,
        "citation_faithfulness_rate": round(faithful.rate(), 3) if faithful.n else None,
        "citation_recall": round(cite_recall.rate(), 3) if cite_recall.n else None,
        "retrieval_recall_at_k": round(retr_recall.rate(), 3) if retr_recall.n else None,
        "qa_correct_refusal_rate": round(qa_out.rate(), 3) if qa_out.n else None,
        "qa_false_refusal_rate": round(1 - qa_in.rate(), 3) if qa_in.n else None,
        "qa_answer_citation_recall": round(qa_cite.rate(), 3) if qa_cite.n else None,
        "n_line_items": verdict_exact.n, "n_qa": len(qa_rows),
        "per_item": per_item, "qa_rows": qa_rows,
    }


def _print(m: dict) -> None:
    print("\n" + "=" * 64 + "\nEVALUATION RESULTS\n" + "=" * 64)
    print(f"Line items: {m['n_line_items']}   Q&A: {m['n_qa']}\n")
    print(f"Verdict exact accuracy        : {m['verdict_exact_accuracy']}")
    vd = m["violation_detection"]
    print(f"Violation detection           : P={vd['precision']} R={vd['recall']} "
          f"F1={vd['f1']}  (tp={vd['tp']} fp={vd['fp']} fn={vd['fn']} tn={vd['tn']})")
    print(f"Reimbursable within tolerance : {m['reimbursable_within_tolerance']}")
    print(f"Citation faithfulness rate    : {m['citation_faithfulness_rate']}")
    print(f"Citation recall (right clause): {m['citation_recall']}")
    print(f"Retrieval recall@k            : {m['retrieval_recall_at_k']}")
    print(f"Q&A correct-refusal rate      : {m['qa_correct_refusal_rate']}")
    print(f"Q&A false-refusal rate        : {m['qa_false_refusal_rate']}")
    print(f"Q&A answer citation recall    : {m['qa_answer_citation_recall']}")
    print("\nPer line item:")
    for it in m["per_item"]:
        mark = "✓" if it["match"] else "✗"
        print(f"  {mark} {it['submission']}/{it['file']}: "
              f"expected={it['expected']} predicted={it['predicted']}")
    print("\nQ&A:")
    for q in m["qa_rows"]:
        mark = "✓" if q["refused"] == q["expected_refused"] else "✗"
        print(f"  {mark} refused={q['refused']} (expected {q['expected_refused']}): {q['q'][:60]}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.evaluate <expected.json> [--out results.json]")
        return
    expected = json.loads(open(sys.argv[1]).read())
    metrics = evaluate(expected)
    _print(metrics)
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        json.dump(metrics, open(out, "w"), indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
