"""Ad-hoc policy Q&A with grounded answers and honest refusal.

Two layers of "don't fabricate":
  1. A retrieval gate — if the best clause is below a similarity threshold, the
     question is out of scope and we refuse without even calling the LLM.
  2. The model is instructed to answer ONLY from the provided clauses and to
     refuse otherwise, and its citations are verified against the corpus.

The threshold is calibrated from observed scores: in-scope policy questions
retrieve at ~0.72-0.80, clearly out-of-scope ones peak around ~0.64.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app import gemini
from app.review import _clauses_block, verify_citation
from app.retrieval import retriever
from app.schemas import Citation

REFUSE_THRESHOLD = 0.66

SYSTEM = (
    "You answer questions about Northwind Logistics' expense & travel policy. "
    "Answer ONLY from the policy clauses provided. If they do not contain the "
    "answer, set refused=true and say it's outside the available policy library "
    "— never use outside knowledge or guess. Quote clauses verbatim in citations "
    "with exact doc_id and section. Keep answers concise."
)


class PolicyAnswer(BaseModel):
    answer: str = Field(description="Grounded answer, or a brief refusal if out of scope.")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    refused: bool = Field(default=False, description="True if the question is out of policy scope.")


def ask(question: str, k: int = 6) -> dict:
    hits = retriever.search(question, k=k)
    top = hits[0].score if hits else 0.0

    if not hits or top < REFUSE_THRESHOLD:
        return {
            "answer": "I can't answer that from Northwind's policy library — it "
                      "appears to be outside the documents I have. Try rephrasing "
                      "around a travel/expense policy topic.",
            "citations": [], "confidence": 0.0, "refused": True,
            "retrieval_top_score": top,
        }

    prompt = (f"QUESTION: {question}\n\nPOLICY CLAUSES (verbatim; cite exactly):\n"
              f"{_clauses_block(hits)}")
    ans = PolicyAnswer.model_validate(
        gemini.generate_json(prompt, schema=PolicyAnswer, system=SYSTEM))
    for c in ans.citations:
        c.section = c.section.lstrip("§ ").strip()
    checks = [verify_citation(c) for c in ans.citations]
    if checks and not all(c.verified for c in checks):
        ans.confidence = min(ans.confidence, 0.5)
    return {
        "answer": ans.answer,
        "citations": [{
            "doc_id": c.citation.doc_id, "section": c.citation.section,
            "quote": c.citation.quote, "verified": c.verified,
            "match_ratio": round(c.match_ratio, 3),
        } for c in checks],
        "confidence": ans.confidence, "refused": ans.refused,
        "retrieval_top_score": top,
    }
