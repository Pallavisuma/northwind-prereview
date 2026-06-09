"""Phase 2 acceptance demo. Extracts every receipt in one sample submission and
prints the structured result, so we can confirm the model reads messy receipts,
isolates alcohol, and captures decisive on-receipt notes.

Run locally (Gemini reachable). Default submission exercises alcohol detection:

    cd backend
    python -m scripts.extract_demo                       # 04_alcohol_solo_travel
    python -m scripts.extract_demo 03_dinner_over_cap    # any submission folder
"""
from __future__ import annotations

import sys

from app.config import SUBMISSIONS_DIR
from app.extraction import extract_path, SUPPORTED_EXTS


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "04_alcohol_solo_travel"
    folder = SUBMISSIONS_DIR / name / "receipts"
    if not folder.exists():
        print(f"No such submission: {name}\nAvailable: "
              f"{[p.name for p in SUBMISSIONS_DIR.iterdir() if p.is_dir()]}")
        return

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    print(f"Extracting {len(files)} receipts from {name} "
          f"(throttled to free-tier ~9/min)...\n" + "=" * 78)
    for p in files:
        r = extract_path(p)
        money = lambda x: f"${x:,.2f}" if x is not None else "—"
        print(f"\n{p.name}")
        print(f"  vendor   : {r.vendor}  [{r.category.value}"
              f"{'/' + r.meal_type.value if r.meal_type else ''}]")
        print(f"  where    : {r.city or '—'}, {r.state_or_country or '—'}   date: {r.date or '—'}")
        print(f"  totals   : subtotal {money(r.subtotal)}  tax {money(r.tax)}  "
              f"tip {money(r.tip)}  TOTAL {money(r.total)}  ({r.currency})")
        if r.category.value == "meal":
            print(f"  alcohol  : {money(r.alcohol_total)}   itemized: {r.is_itemized}   "
                  f"diners: {r.diner_count or '—'}  external: {r.has_external_attendees}")
            for li in r.line_items:
                tag = "  🍺ALC" if li.is_alcohol else ""
                print(f"      - {li.description:<32} {money(li.amount)}{tag}")
        if r.flight_class:
            print(f"  flight   : class={r.flight_class.value}")
        if r.lodging_nightly_rate:
            print(f"  lodging  : {money(r.lodging_nightly_rate)}/night x {r.lodging_nights}")
        if r.notes_on_receipt:
            print(f"  NOTE     : “{r.notes_on_receipt}”")
        print(f"  confidence: {r.extraction_confidence:.2f}"
              + (f"   warnings: {r.extraction_warnings}" if r.extraction_warnings else ""))
    print("\n" + "=" * 78)
    print("Check: are totals right, alcohol lines flagged, and notes captured?")


if __name__ == "__main__":
    main()
