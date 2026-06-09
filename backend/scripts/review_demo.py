"""Phase 3 acceptance demo: full pipeline on one submission. For each receipt:
extract -> retrieve -> verdict -> verify citations, then print a reviewer-style
card. This is the end-to-end heart of the system, minus the UI.

Run locally (Gemini reachable). ~2 calls per receipt, throttled to ~9/min:

    cd backend
    python -m scripts.review_demo                      # 04_alcohol_solo_travel
    python -m scripts.review_demo 03_dinner_over_cap
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import SUBMISSIONS_DIR
from app.context import load_employee
from app.extraction import extract_path, SUPPORTED_EXTS
from app.ingest.embed import build_index
from app.review import review_line_item
from app.schemas import VerdictLabel

ICON = {VerdictLabel.compliant: "✅ COMPLIANT", VerdictLabel.flagged: "⚠️  FLAGGED",
        VerdictLabel.rejected: "⛔ REJECTED", VerdictLabel.needs_info: "❓ NEEDS INFO"}


def _process(path, emp):
    r = extract_path(path, cache=True)          # cached: re-runs are instant
    return path.name, r, review_line_item(r, emp, filename=path.name)


def _render(name, r, rev) -> str:
    v = rev.verdict
    money = lambda x: f"${x:,.2f}" if x is not None else "—"
    out = [f"\n{name}  —  {ICON[v.verdict]}   (conf {v.confidence:.2f}, "
           f"retrieval {rev.top_score:.2f})",
           f"  {r.vendor} [{r.category.value}]  total {money(r.total)}  "
           f"→ reimbursable {money(v.reimbursable_amount)}"]
    if v.issues:
        out.append(f"  issues: {', '.join(v.issues)}")
    out.append(f"  reasoning: {v.reasoning}")
    for c in rev.citation_checks:
        mark = "✓" if c.verified else "✗ UNVERIFIED"
        out.append(f"    [{mark}] {c.citation.doc_id} §{c.citation.section}: "
                   f"\"{c.citation.quote[:90]}{'…' if len(c.citation.quote) > 90 else ''}\"")
    return "\n".join(out)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "04_alcohol_solo_travel"
    sub = SUBMISSIONS_DIR / name
    if not sub.exists():
        print(f"No such submission: {name}")
        return

    build_index(verbose=True)  # no-op if already embedded
    emp = load_employee(sub)
    files = sorted(p for p in (sub / "receipts").iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    print(f"\nSubmission: {name}\n{emp.context_brief()}")
    print(f"Processing {len(files)} receipts concurrently (free-tier paced; "
          f"first run does API calls, re-runs hit the cache)...\n" + "=" * 80)

    from app.gemini import QuotaExceeded, client
    client()  # warm the shared Gemini client once before spawning workers
    t0 = time.time()
    results: dict[str, str] = {}
    reimb: dict[str, float] = {}
    try:
        with ThreadPoolExecutor(max_workers=min(6, len(files))) as ex:
            futs = {ex.submit(_process, p, emp): p.name for p in files}
            for fut in as_completed(futs):
                nm, r, rev = fut.result()
                results[nm] = _render(nm, r, rev)
                reimb[nm] = rev.verdict.reimbursable_amount or 0.0
                print(f"  …done {nm}")  # live heartbeat as each finishes
    except QuotaExceeded as e:
        print(f"\n⛔ {e}\n(Extractions already done are cached, so the next run "
              f"resumes without repeating them.)")
        return

    for nm in sorted(results):  # stable, readable order
        print(results[nm])
    print("\n" + "=" * 80)
    print(f"Total reimbursable: ${sum(reimb.values()):,.2f}   "
          f"| {len(files)} receipts in {time.time() - t0:.0f}s")
    print("Check: are verdicts right, citations faithful (✓), and uncertainty honest?")


if __name__ == "__main__":
    main()
