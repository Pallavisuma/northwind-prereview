"""FastAPI app: the reviewer-facing API plus startup wiring. Serves the built
frontend (Phase 5) when present, so the whole thing deploys as one service."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import qa, service, store
from app.config import REPO_ROOT
from app.db import SessionLocal, init_db
from app.gemini import QuotaExceeded
from app.models import Employee, LineItem, Submission


# --- startup --------------------------------------------------------------

def _load_seed_index_if_needed() -> None:
    """On first boot, if the DB has no policy chunks but a shipped seed index
    exists, port its rows in — so the app works immediately with no embedding
    calls. Backend-agnostic (reads the SQLite seed, writes via SQLAlchemy), so
    it works the same for SQLite and Postgres."""
    import sqlite3
    from app.config import SEED_INDEX
    from app.models import PolicyChunk, Meta
    db = SessionLocal()
    try:
        if db.query(PolicyChunk).count() > 0 or not SEED_INDEX.exists():
            return
        con = sqlite3.connect(str(SEED_INDEX))
        cols = ["chunk_id", "doc_id", "doc_title", "family", "section",
                "heading", "text", "source_pdf", "page_start", "embedding"]
        rows = con.execute(f"SELECT {','.join(cols)} FROM policy_chunks").fetchall()
        db.bulk_save_objects([PolicyChunk(**dict(zip(cols, r))) for r in rows])
        for k, v in con.execute("SELECT key, value FROM meta").fetchall():
            db.merge(Meta(key=k, value=v))
        con.close()
        db.commit()
        print(f"[startup] loaded {len(rows)} policy chunks from seed index")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _load_seed_index_if_needed()
    db = SessionLocal()
    try:
        store.seed_employees(db)
    finally:
        db.close()
    # Build the index only if still missing (no seed shipped and not yet built).
    try:
        from app.ingest.embed import build_index, index_built
        with SessionLocal() as s:
            ready = index_built(s)
        if not ready:
            build_index(verbose=False)
    except Exception as e:  # don't block startup if quota/key missing
        print(f"[startup] policy index not ready: {e}")
    yield


app = FastAPI(title="Northwind Expense Pre-Review", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- serialization --------------------------------------------------------

def employee_dict(e: Employee) -> dict:
    return {"id": e.id, "employee_id": e.employee_id, "name": e.name, "grade": e.grade,
            "title": e.title, "department": e.department, "manager_id": e.manager_id,
            "home_base": e.home_base, "trip_purpose": e.trip_purpose,
            "trip_dates": e.trip_dates, "is_seed": e.is_seed}


def lineitem_dict(li: LineItem) -> dict:
    return {
        "id": li.id, "filename": li.filename, "category": li.category,
        "receipt": li.receipt,
        "system_verdict": li.verdict, "effective_verdict": li.effective_verdict,
        "reasoning": li.reasoning, "confidence": li.confidence,
        "reimbursable_amount": li.reimbursable_amount, "issues": li.issues or [],
        "citations": li.citations or [], "retrieval_top_score": li.retrieval_top_score,
        "citations_faithful": li.citations_faithful,
        "overrides": [{"id": o.id, "reviewer": o.reviewer, "new_verdict": o.new_verdict,
                       "comment": o.comment, "created_at": o.created_at.isoformat()}
                      for o in li.overrides],
    }


def submission_dict(s: Submission, *, detail: bool = False) -> dict:
    out = {"id": s.id, "status": s.status, "label": s.label,
           "created_at": s.created_at.isoformat(), "updated_at": s.updated_at.isoformat(),
           "employee": employee_dict(s.employee) if s.employee else None,
           "line_item_count": len(s.line_items)}
    flagged = sum(1 for li in s.line_items
                  if li.effective_verdict in ("flagged", "rejected"))
    out["flagged_count"] = flagged
    out["total_reimbursable"] = round(
        sum(li.reimbursable_amount or 0 for li in s.line_items), 2)
    if detail:
        out["line_items"] = [lineitem_dict(li) for li in s.line_items]
    if getattr(s, "errors", None):
        out["errors"] = [{"file": f, "error": msg} for f, msg in s.errors]
    return out


# --- request bodies -------------------------------------------------------

class EmployeeIn(BaseModel):
    name: str
    grade: int
    title: str = ""
    department: str = ""
    manager_id: str = ""
    home_base: str = ""
    trip_purpose: str = ""
    trip_dates: str = ""


class OverrideIn(BaseModel):
    new_verdict: str
    comment: str = ""
    reviewer: str = "reviewer"


class AskIn(BaseModel):
    question: str


# --- endpoints ------------------------------------------------------------

@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    from app.models import PolicyChunk
    return {"ok": True, "employees": db.query(Employee).count(),
            "policy_chunks": db.query(PolicyChunk).count()}


@app.get("/api/employees")
def get_employees(db: Session = Depends(get_db)):
    return [employee_dict(e) for e in store.list_employees(db)]


@app.post("/api/employees")
def post_employee(body: EmployeeIn, db: Session = Depends(get_db)):
    emp = store.create_employee(db, **body.model_dump())
    return employee_dict(emp)


@app.get("/api/submissions")
def get_submissions(employee_pk: int | None = None, status: str | None = None,
                    db: Session = Depends(get_db)):
    subs = store.list_submissions(db, employee_pk=employee_pk, status=status)
    return [submission_dict(s) for s in subs]


@app.get("/api/submissions/{submission_id}")
def get_submission(submission_id: int, db: Session = Depends(get_db)):
    s = store.get_submission(db, submission_id)
    if not s:
        raise HTTPException(404, "submission not found")
    return submission_dict(s, detail=True)


@app.post("/api/submissions")
async def create_submission(employee_pk: int = Form(...), label: str = Form(""),
                            files: list[UploadFile] = File(...),
                            db: Session = Depends(get_db)):
    emp = store.get_employee(db, employee_pk)
    if not emp:
        raise HTTPException(404, "employee not found")
    payloads = [(f.filename, await f.read()) for f in files]
    bad = [fn for fn, _ in payloads if not service.valid_receipt(fn)]
    if bad:
        raise HTTPException(400, f"unsupported file types: {bad}")

    sub = store.create_submission(db, employee_pk=employee_pk,
                                  label=label or (emp.trip_purpose or ""))
    try:
        service.process_submission(db, sub, payloads, store.employee_ctx(emp))
    except QuotaExceeded as e:
        raise HTTPException(429, str(e))
    db.refresh(sub)
    return submission_dict(sub, detail=True)


@app.post("/api/line-items/{line_item_id}/override")
def post_override(line_item_id: int, body: OverrideIn, db: Session = Depends(get_db)):
    if body.new_verdict not in ("compliant", "flagged", "rejected", "needs_info"):
        raise HTTPException(400, "invalid verdict")
    ev = store.add_override(db, line_item_id, body.new_verdict, body.comment, body.reviewer)
    if ev is None:
        raise HTTPException(404, "line item not found")
    li = db.get(LineItem, line_item_id)
    return lineitem_dict(li)


@app.post("/api/policy/ask")
def policy_ask(body: AskIn):
    if not body.question.strip():
        raise HTTPException(400, "empty question")
    try:
        return qa.ask(body.question)
    except QuotaExceeded as e:
        raise HTTPException(429, str(e))


# --- serve the single-file frontend (no build step) ----------------------

_FRONTEND = REPO_ROOT / "frontend"
if (_FRONTEND / "index.html").exists():
    @app.get("/")
    def _index():
        return FileResponse(_FRONTEND / "index.html")

    # Any other static assets (favicon, etc.) if added later.
    if (_FRONTEND / "static").exists():
        app.mount("/static", StaticFiles(directory=_FRONTEND / "static"), name="static")
