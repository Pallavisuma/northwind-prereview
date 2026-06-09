"""Persistence service: all reads/writes of operational state go through here,
so the API and the processing pipeline never touch the ORM directly. Everything
lands in SQLite on a persistent volume — submissions processed yesterday are
visible today after a restart."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import SUBMISSIONS_DIR
from app.context import Employee as EmployeeCtx
from app.models import Employee, LineItem, OverrideEvent, Submission
from app.review import LineItemReview


# --- Employees ------------------------------------------------------------

def seed_employees(db: Session) -> int:
    """Load the five sample employees on startup (idempotent: upsert by id)."""
    n = 0
    for sub in sorted(p for p in SUBMISSIONS_DIR.iterdir() if p.is_dir()):
        info = sub / "employee_info.json"
        if not info.exists():
            continue
        ctx = EmployeeCtx.model_validate_json(info.read_text())
        row = db.scalar(select(Employee).where(Employee.employee_id == ctx.employee_id))
        if row is None:
            row = Employee(employee_id=ctx.employee_id, is_seed=True)
            db.add(row)
        row.name, row.grade, row.title = ctx.name, ctx.grade, ctx.title
        row.department, row.manager_id = ctx.department, ctx.manager_id or ""
        row.home_base, row.trip_purpose = ctx.home_base or "", ctx.trip_purpose or ""
        row.trip_dates, row.is_seed = ctx.trip_dates or "", True
        n += 1
    db.commit()
    return n


def list_employees(db: Session) -> list[Employee]:
    return list(db.scalars(select(Employee).order_by(Employee.employee_id)))


def get_employee(db: Session, pk: int) -> Optional[Employee]:
    return db.get(Employee, pk)


def create_employee(db: Session, **fields) -> Employee:
    emp = Employee(is_seed=False, **fields)
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return emp


def employee_ctx(emp: Employee) -> EmployeeCtx:
    """ORM Employee -> the context object the verdict engine expects."""
    return EmployeeCtx(
        employee_id=emp.employee_id, name=emp.name, grade=emp.grade, title=emp.title,
        department=emp.department, manager_id=emp.manager_id, home_base=emp.home_base,
        trip_purpose=emp.trip_purpose, trip_dates=emp.trip_dates)


# --- Submissions & line items --------------------------------------------

def create_submission(db: Session, employee_pk: int, label: str = "") -> Submission:
    sub = Submission(employee_pk=employee_pk, label=label, status="pending")
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub


def set_status(db: Session, sub: Submission, status: str) -> None:
    sub.status = status
    db.commit()


def save_review(db: Session, submission_id: int, review: LineItemReview,
                receipt_path: str = "") -> LineItem:
    """Persist one extracted+reviewed receipt as a LineItem."""
    v = review.verdict
    citations = [{
        "doc_id": c.citation.doc_id, "section": c.citation.section,
        "quote": c.citation.quote, "verified": c.verified,
        "match_ratio": round(c.match_ratio, 3), "matched_chunk_id": c.matched_chunk_id,
    } for c in review.citation_checks]
    li = LineItem(
        submission_id=submission_id, filename=review.filename, receipt_path=receipt_path,
        receipt=review.receipt.model_dump(mode="json"), category=review.receipt.category.value,
        verdict=v.verdict.value, reasoning=v.reasoning, confidence=v.confidence,
        reimbursable_amount=v.reimbursable_amount, issues=list(v.issues),
        citations=citations, retrieval_top_score=review.top_score,
        citations_faithful=review.citations_faithful,
    )
    db.add(li)
    db.commit()
    db.refresh(li)
    return li


def list_submissions(db: Session, *, employee_pk: Optional[int] = None,
                     status: Optional[str] = None,
                     since: Optional[datetime] = None) -> list[Submission]:
    q = select(Submission).order_by(Submission.created_at.desc())
    if employee_pk is not None:
        q = q.where(Submission.employee_pk == employee_pk)
    if status:
        q = q.where(Submission.status == status)
    if since:
        q = q.where(Submission.created_at >= since)
    return list(db.scalars(q))


def get_submission(db: Session, submission_id: int) -> Optional[Submission]:
    return db.get(Submission, submission_id)


# --- Overrides (append-only audit) ---------------------------------------

def add_override(db: Session, line_item_id: int, new_verdict: str, comment: str,
                 reviewer: str = "reviewer") -> Optional[OverrideEvent]:
    li = db.get(LineItem, line_item_id)
    if li is None:
        return None
    ev = OverrideEvent(line_item_id=line_item_id, new_verdict=new_verdict,
                       comment=comment, reviewer=reviewer)
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev
