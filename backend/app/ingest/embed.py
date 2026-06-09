"""Build the policy index: chunk -> embed -> persist in SQLite.

Idempotent: re-running only re-embeds if the corpus changed (hash check), so a
container restart doesn't burn free-tier quota.
"""
from __future__ import annotations

import hashlib

import numpy as np

from app import gemini
from app.config import POLICIES_DIR, EMBED_DIM
from app.db import SessionLocal, init_db
from app.ingest.chunker import chunk_all
from app.models import PolicyChunk, Meta


def _corpus_hash() -> str:
    h = hashlib.sha256()
    for pdf in sorted(POLICIES_DIR.glob("*.pdf")):
        h.update(pdf.name.encode())
        h.update(str(pdf.stat().st_size).encode())
    return h.hexdigest()


def index_built(db) -> bool:
    row = db.get(Meta, "corpus_hash")
    n = db.query(PolicyChunk).count()
    return bool(row and row.value == _corpus_hash() and n > 0)


def build_index(force: bool = False, verbose: bool = True) -> int:
    init_db()
    db = SessionLocal()
    try:
        if not force and index_built(db):
            n = db.query(PolicyChunk).count()
            if verbose:
                print(f"Index up to date ({n} chunks). Use force=True to rebuild.")
            return n

        db.query(PolicyChunk).delete()
        chunks = chunk_all(POLICIES_DIR)
        if verbose:
            print(f"Embedding {len(chunks)} chunks "
                  f"(free tier ~{gemini.config.EMBED_RPM}/min, so a few minutes)...")
        vecs = gemini.embed([c.embed_text() for c in chunks],
                            task_type="RETRIEVAL_DOCUMENT", progress=verbose)
        for c, v in zip(chunks, vecs):
            arr = np.asarray(v, dtype=np.float32)
            assert arr.shape[0] == EMBED_DIM, f"dim {arr.shape[0]} != {EMBED_DIM}"
            db.add(PolicyChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title,
                family=c.family, section=c.section, heading=c.heading,
                text=c.text, source_pdf=c.source_pdf, page_start=c.page_start,
                embedding=arr.tobytes(),
            ))
        meta = db.get(Meta, "corpus_hash") or Meta(key="corpus_hash")
        meta.value = _corpus_hash()
        db.merge(meta)
        db.commit()
        if verbose:
            print(f"Indexed {len(chunks)} chunks.")
        return len(chunks)
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    build_index(force="--force" in sys.argv)
