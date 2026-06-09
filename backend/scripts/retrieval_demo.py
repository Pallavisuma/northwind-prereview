"""Phase 1 acceptance demo. Builds the policy index (embeds on first run, then
cached) and runs a set of probe queries — including an out-of-scope one — so we
can eyeball whether retrieval pulls the right clause and ranks noise low.

Run locally (where the Gemini API is reachable):

    cd backend
    python -m scripts.retrieval_demo
"""
from __future__ import annotations

from app.ingest.embed import build_index
from app.retrieval import retriever

# (query, what we hope to see at/near the top)
PROBES = [
    ("Can I expense alcohol when traveling alone on business?", "TEP-003 §3.1 (solo travel)"),
    ("What is the per-person dinner spending limit?", "TEP-002 §2 (meal caps)"),
    ("When am I allowed to fly business class?", "TEP-005 §2.3 (intl >=10h)"),
    ("nightly hotel rate cap for a tier 1 city", "TEP-004 §3 (lodging tiers)"),
    ("Do I need a receipt for a $12 coffee?", "TEP-007 §4.1 (under-$25)"),
    ("trips that qualify for per diem", "TEP-008 §2.1 (>=3 nights)"),
    # Out-of-scope: should surface only weakly-related/noise chunks at low scores.
    ("How many vacation days do new employees accrue?", "(out of scope — expect low scores/noise)"),
]


def main() -> None:
    n = build_index(verbose=True)
    print(f"\nIndex ready: {n} chunks.\n" + "=" * 72)
    for q, expect in PROBES:
        hits = retriever.search(q, k=4)
        print(f"\nQ: {q}\n   want ~ {expect}")
        for h in hits:
            snippet = h.text.replace("\n", " ")[:70]
            print(f"   {h.score:.3f}  {h.citation():13s} | {snippet}")
    print("\n" + "=" * 72)
    print("Eyeball check: does the top hit for each in-scope query match 'want'?")
    print("Does the out-of-scope query stay below ~0.6 and surface no real rule?")


if __name__ == "__main__":
    main()
