"""Retrieval over the policy index. The corpus is tiny (~700 chunks), so we load
all vectors into one numpy matrix and do exact cosine similarity in-process — a
vector DB would be pure operational overhead here, and we defend that in the
README. The matrix is cached and refreshed when the row count changes."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from app import gemini
from app.config import EMBED_DIM, RETRIEVAL_TOP_K
from app.db import SessionLocal
from app.models import PolicyChunk


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    heading: str
    text: str
    page_start: int
    source_pdf: str
    score: float

    def citation(self) -> str:
        return f"{self.doc_id} §{self.section}" if self.section else self.doc_id


class PolicyRetriever:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._meta: list[PolicyChunk] = []
        self._mat: np.ndarray | None = None
        self._count = -1
        self._lock = threading.Lock()

    def _load(self) -> None:
        db = SessionLocal()
        try:
            rows = db.query(PolicyChunk).all()
            self._meta = rows
            self._ids = [r.chunk_id for r in rows]
            if rows:
                mat = np.vstack([
                    np.frombuffer(r.embedding, dtype=np.float32) for r in rows
                ])
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                self._mat = mat / np.clip(norms, 1e-8, None)
            else:
                self._mat = None
            self._count = len(rows)
        finally:
            db.close()

    def _ensure_fresh(self) -> None:
        db = SessionLocal()
        try:
            n = db.query(PolicyChunk).count()
        finally:
            db.close()
        if n != self._count:
            with self._lock:           # one reload at a time across worker threads
                if n != self._count:
                    self._load()

    def search(self, query: str, k: int = RETRIEVAL_TOP_K,
               doc_ids: list[str] | None = None) -> list[Hit]:
        self._ensure_fresh()
        if self._mat is None:
            return []
        qv = np.asarray(
            gemini.embed([query], task_type="retrieval_query")[0], dtype=np.float32)
        qv = qv / max(np.linalg.norm(qv), 1e-8)
        scores = self._mat @ qv
        idx = np.argsort(-scores)
        hits: list[Hit] = []
        for i in idx:
            r = self._meta[i]
            if doc_ids and r.doc_id not in doc_ids:
                continue
            hits.append(Hit(
                chunk_id=r.chunk_id, doc_id=r.doc_id, doc_title=r.doc_title,
                section=r.section, heading=r.heading, text=r.text,
                page_start=r.page_start, source_pdf=r.source_pdf,
                score=float(scores[i]),
            ))
            if len(hits) >= k:
                break
        return hits


    def search_many(self, queries: list[str], k: int = 4) -> list[list[Hit]]:
        """Embed all queries in ONE batched call, then score each — far fewer
        round-trips than calling search() per query."""
        self._ensure_fresh()
        if self._mat is None or not queries:
            return [[] for _ in queries]
        qvs = gemini.embed(queries, task_type="RETRIEVAL_QUERY")
        results: list[list[Hit]] = []
        for qv in qvs:
            v = np.asarray(qv, dtype=np.float32)
            v = v / max(np.linalg.norm(v), 1e-8)
            scores = self._mat @ v
            idx = np.argsort(-scores)[:k]
            results.append([self._hit(i, float(scores[i])) for i in idx])
        return results

    def _hit(self, i: int, score: float) -> Hit:
        r = self._meta[i]
        return Hit(chunk_id=r.chunk_id, doc_id=r.doc_id, doc_title=r.doc_title,
                   section=r.section, heading=r.heading, text=r.text,
                   page_start=r.page_start, source_pdf=r.source_pdf, score=score)

    def get_clause(self, doc_id: str, section: str) -> str | None:
        """Return the verbatim text of a clause by id+section, for citation
        verification (independent of what was retrieved)."""
        self._ensure_fresh()
        sec = (section or "").lstrip("§ ").strip()
        for r in self._meta:
            if r.doc_id == doc_id and (r.section or "") == sec:
                return r.text
        return None


# Module-level singleton (cheap; lazy-loads on first search).
retriever = PolicyRetriever()
