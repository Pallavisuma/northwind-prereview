"""The verdict engine — the core of the system.

For each extracted line item we:
  1. Retrieve the most relevant policy clauses with a few category-aware
     sub-queries (better recall than one generic query).
  2. Ask the model for a schema-constrained Verdict, grounded ONLY in the
     provided clauses + the employee's trip context, with verbatim citations.
  3. Verify every citation against the actual corpus text. A quote that doesn't
     match its clause is caught here and downgrades confidence — this is how we
     guarantee the "citation faithfulness" the brief spot-checks.

We pass pre-computed numbers (food vs alcohol split, tip %, trip nights) into
the prompt so the model reasons about policy rather than doing error-prone
arithmetic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app import gemini
from app.config import RETRIEVAL_TOP_K
from app.context import Employee
from app.retrieval import Hit, retriever
from app.schemas import Category, Citation, ExtractedReceipt, Verdict, VerdictLabel

# Category-aware sub-queries. Union of hits gives the model the clauses it needs
# without hard-coding which document is authoritative.
CATEGORY_QUERIES: dict[Category, list[str]] = {
    Category.meal: [
        "per-person meal cap breakfast lunch dinner reimbursement limit",
        "alcohol reimbursement solo travel team meal client entertainment",
        "tip gratuity reimbursable percentage limit",
        "high-cost tier 1 city meal cap increase",
        "client entertainment meal per person cap external attendees approval",
    ],
    Category.airfare: [
        "air travel class of service economy premium business first eligibility",
        "checked baggage seat selection wifi reimbursable airfare",
    ],
    Category.lodging: [
        "lodging nightly rate cap city tier hotel",
        "resort fee incidentals mini-bar lodging non-reimbursable",
    ],
    Category.ground_transport: [
        "ground transportation rideshare taxi reimbursable premium category",
        "parking tolls mileage ground transport reimbursement",
    ],
    Category.conference: [
        "conference registration fee reimbursable meals included",
        "conference attendance approval threshold",
    ],
    Category.other: [
        "business expense reimbursement general principles business purpose",
        "non-reimbursable personal expenses",
    ],
}
COMMON_QUERIES = ["expense approval authority by amount and grade",
                  "receipt requirements itemization amount must match"]

SYSTEM = (
    "You are an expense-compliance pre-reviewer for Northwind Logistics. A human "
    "makes the final call; your job is to do the heavy lifting and explain it so "
    "they can trust or override you.\n\n"
    "Decide a verdict for ONE line item using ONLY: (a) the POLICY CLAUSES "
    "provided, (b) the employee's TRIP CONTEXT, and (c) the RECEIPT FACTS. Do not "
    "rely on outside knowledge of dollar amounts — read every threshold from the "
    "clauses given. If the clauses needed to judge this item are not present, say "
    "so via verdict=needs_info.\n\n"
    "VERDICTS:\n"
    "- compliant: clearly within policy. reimbursable_amount = the total.\n"
    "- flagged: a likely or partial violation that needs a human, OR only part is "
    "reimbursable (e.g. alcohol inside an otherwise-fine meal, or spend over a "
    "cap). Set reimbursable_amount to the allowed portion.\n"
    "- rejected: a clear violation with nothing reimbursable. reimbursable_amount = 0.\n"
    "- needs_info: genuinely ambiguous, or policy support is weak. Lower your "
    "confidence and explain what's missing. Prefer this over guessing.\n\n"
    "INTERPRETING RECEIPT FACTS: These are a structured extraction, not the raw "
    "receipt. A field that is null/None means it was NOT captured — that is not "
    "evidence it is absent from the receipt. Do NOT treat a null field as a policy "
    "violation. Only doubt a receipt's completeness when extraction_warnings or a "
    "low extraction_confidence say it was illegible or incomplete.\n\n"
    "SUBSTANCE vs DOCUMENTATION: Base flagged/rejected mainly on substantive "
    "policy — spend over a cap, disallowed alcohol, wrong cabin class, an amount "
    "that doesn't match, or a personal/non-business charge. If a required "
    "documentation element merely appears missing (e.g. no payment method "
    "captured), the correct response is verdict = needs_info: a human should "
    "confirm it on the original receipt. NEVER set reimbursable_amount to 0 over "
    "a documentation gap on an expense that is otherwise within policy — set "
    "reimbursable_amount to the substantively-allowed amount (the full total if "
    "otherwise compliant) and explain what the reviewer should verify. Reserve "
    "reimbursable_amount = 0 strictly for charges with no reimbursable substance "
    "(e.g. an entirely personal or disallowed item).\n\n"
    "KEY REASONING:\n"
    "- Alcohol is reimbursable ONLY as sanctioned client entertainment (an external "
    "client physically present AND VP pre-approval). On solo or employee-only meals, "
    "alcohol is not reimbursable — flag the item and reimburse only the food portion. "
    "Use the receipt's attendee evidence; a trip purpose that merely mentions meeting "
    "partners does NOT by itself make a solo dinner client entertainment.\n"
    "- Meal caps are per person and inclusive of tax and tip. Standard caps differ "
    "from client-entertainment caps. Tier-1 cities raise caps by a stated percentage.\n"
    "- Compare the relevant per-person amount to the cap; if over, flag and set "
    "reimbursable_amount to the cap.\n\n"
    "CITATIONS: quote verbatim from the provided clauses and give the exact doc_id "
    "and section. Every non-trivial claim must be backed by a citation. Keep "
    "reasoning to 2-4 sentences, written for a busy reviewer."
)


@dataclass
class CitationCheck:
    citation: Citation
    verified: bool
    matched_chunk_id: str | None
    match_ratio: float


@dataclass
class LineItemReview:
    filename: str
    receipt: ExtractedReceipt
    verdict: Verdict
    citation_checks: list[CitationCheck] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    top_score: float = 0.0

    @property
    def citations_faithful(self) -> bool:
        return all(c.verified for c in self.citation_checks)


def _dedupe_hits(hits: list[Hit], k: int) -> list[Hit]:
    best: dict[str, Hit] = {}
    for h in hits:
        if h.chunk_id not in best or h.score > best[h.chunk_id].score:
            best[h.chunk_id] = h
    return sorted(best.values(), key=lambda h: -h.score)[:k]


def retrieve_for_line(receipt: ExtractedReceipt, k: int = RETRIEVAL_TOP_K) -> list[Hit]:
    queries = list(CATEGORY_QUERIES.get(receipt.category, CATEGORY_QUERIES[Category.other]))
    queries += COMMON_QUERIES
    # Anchor one query in the concrete facts to sharpen recall.
    facts = f"{receipt.category.value} {receipt.vendor} {receipt.city or ''} " \
            f"{'with alcohol' if receipt.alcohol_total else ''} total {receipt.total}"
    queries.append(facts)
    all_hits: list[Hit] = []
    for hits in retriever.search_many(queries, k=4):  # one batched embed call
        all_hits.extend(hits)
    return _dedupe_hits(all_hits, k)


def _tokens(s: str) -> list[str]:
    """Word/number tokens, punctuation stripped — robust to layout/whitespace."""
    return re.findall(r"[a-z0-9$%.]+", s.lower())


VERIFY_THRESHOLD = 0.80


def _support_ratio(quote: str, text: str) -> float:
    """Fraction of the quote's tokens that appear in the clause, allowing the
    quote to be assembled from several real spans of the SAME clause (e.g. a cap
    line plus a city from its list). This rewards faithful stitching while a
    fabricated quote — whose tokens aren't in the clause — still scores low."""
    q = _tokens(quote)
    if not q:
        return 0.0
    t = _tokens(text)
    if " ".join(q) in " ".join(t):
        return 1.0
    matched = sum(b.size for b in SequenceMatcher(None, q, t).get_matching_blocks())
    return matched / len(q)


def verify_citation(c: Citation) -> CitationCheck:
    """Check the quote is genuinely supported by the cited clause (or any clause
    if the section is wrong). Guards against fabricated or altered quotes."""
    if not _tokens(c.quote):
        return CitationCheck(c, False, None, 0.0)
    clause = retriever.get_clause(c.doc_id, c.section)
    if clause is not None:
        r = _support_ratio(c.quote, clause)
        cid = f"{c.doc_id}§{c.section}" if c.section else c.doc_id
        return CitationCheck(c, r >= VERIFY_THRESHOLD, cid, r)
    # Cited section not found — scan the (tiny) corpus to catch a mis-cited §.
    best_id, best_r = None, 0.0
    for row in retriever._meta:
        r = _support_ratio(c.quote, row.text)
        if r > best_r:
            best_id, best_r = row.chunk_id, r
    return CitationCheck(c, best_r >= VERIFY_THRESHOLD, best_id, best_r)


def _facts_block(r: ExtractedReceipt) -> str:
    food = None
    if r.subtotal is not None:
        food = round(r.subtotal - (r.alcohol_total or 0), 2)
    tip_pct = None
    if r.tip and r.subtotal:
        tip_pct = round(100 * r.tip / r.subtotal, 1)
    lines = "\n".join(
        f"    - {li.description}: {li.amount}" + (" [ALCOHOL]" if li.is_alcohol else "")
        for li in r.line_items)
    return (
        f"vendor: {r.vendor}\ncategory: {r.category.value}"
        + (f" / {r.meal_type.value}" if r.meal_type else "")
        + f"\nlocation: {r.city or '?'}, {r.state_or_country or '?'}\ndate: {r.date or '?'}\n"
        f"currency: {r.currency}\nsubtotal: {r.subtotal}\ntax: {r.tax}\ntip: {r.tip}"
        + (f" ({tip_pct}% of pre-tax)" if tip_pct is not None else "")
        + f"\ntotal: {r.total}\npayment_method: {r.payment_method}\n"
        f"alcohol_total: {r.alcohol_total}\nfood_subtotal: {food}\n"
        f"diner_count: {r.diner_count}\nhas_external_attendees: {r.has_external_attendees}\n"
        f"flight_class: {r.flight_class.value if r.flight_class else None}\n"
        f"lodging: {r.lodging_nightly_rate}/night x {r.lodging_nights}\n"
        f"is_itemized: {r.is_itemized}\nnotes_on_receipt: {r.notes_on_receipt}\n"
        f"extraction_confidence: {r.extraction_confidence}\n"
        f"extraction_warnings: {r.extraction_warnings}"
        + (f"\nline_items:\n{lines}" if lines else ""))


def _clauses_block(hits: list[Hit]) -> str:
    out = []
    for h in hits:
        loc = f"{h.doc_id} §{h.section}" if h.section else h.doc_id
        out.append(f"[{loc}] (relevance {h.score:.2f}) — \"{h.text}\"")
    return "\n\n".join(out)


def review_line_item(receipt: ExtractedReceipt, employee: Employee,
                     filename: str = "") -> LineItemReview:
    hits = retrieve_for_line(receipt)
    top = hits[0].score if hits else 0.0
    prompt = (
        f"TRIP CONTEXT:\n{employee.context_brief()}\n\n"
        f"RECEIPT FACTS:\n{_facts_block(receipt)}\n\n"
        f"POLICY CLAUSES (verbatim; cite by doc_id and section, quote exactly):\n"
        f"{_clauses_block(hits)}\n\n"
        f"Note: the highest clause relevance score is {top:.2f}. If that is low "
        f"(weak support) or the facts are ambiguous, prefer verdict=needs_info."
    )
    verdict = Verdict.model_validate(
        gemini.generate_json(prompt, schema=Verdict, system=SYSTEM))

    # Normalize section labels (models sometimes return "§2" or " 2.3 ").
    for c in verdict.citations:
        c.section = c.section.lstrip("§ ").strip()

    checks = [verify_citation(c) for c in verdict.citations]
    # Faithfulness guardrail: if any quote can't be verified, never let the item
    # read as confidently compliant — downgrade and tell the reviewer.
    if checks and not all(c.verified for c in checks):
        verdict.confidence = min(verdict.confidence, 0.5)
        verdict.issues = list(dict.fromkeys(verdict.issues + ["unverified_citation"]))
        if verdict.verdict == VerdictLabel.compliant:
            verdict.verdict = VerdictLabel.needs_info

    return LineItemReview(filename=filename, receipt=receipt, verdict=verdict,
                          citation_checks=checks, retrieved=hits, top_score=top)
