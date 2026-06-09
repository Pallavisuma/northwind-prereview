"""ORM models. The policy index (Phase 1) plus the operational tables —
employees, submissions, line items with their verdicts, and an append-only
override log. Rich LLM outputs (extraction, citations) are stored as JSON for
fidelity; key fields are also flattened into columns so history can be filtered
by employee / status / date without parsing JSON."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (Column, Integer, String, Text, LargeBinary, Float,
                        DateTime, Boolean, ForeignKey, JSON)
from sqlalchemy.orm import relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PolicyChunk(Base):
    """One clause-level chunk of a policy document, plus its embedding."""
    __tablename__ = "policy_chunks"

    id = Column(Integer, primary_key=True)
    chunk_id = Column(String, unique=True, index=True)   # "TEP-002§2.3"
    doc_id = Column(String, index=True)                  # "TEP-002"
    doc_title = Column(String)
    family = Column(String, index=True)                  # "TEP" | "SEC" | ...
    section = Column(String)                             # "2.3"
    heading = Column(String)
    text = Column(Text)                                  # verbatim clause
    source_pdf = Column(String)
    page_start = Column(Integer)
    embedding = Column(LargeBinary)                      # float32[EMBED_DIM]


class Meta(Base):
    """Tiny key/value table for ingestion bookkeeping (corpus hash, etc.)."""
    __tablename__ = "meta"
    key = Column(String, primary_key=True)
    value = Column(Text)


class Employee(Base):
    """An employee plus their trip context. Five are seeded on startup; more can
    be created from the UI."""
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    employee_id = Column(String, unique=True, index=True)   # "NW-04821"
    name = Column(String, nullable=False)
    grade = Column(Integer, nullable=False)
    title = Column(String, default="")
    department = Column(String, default="")
    manager_id = Column(String, default="")
    home_base = Column(String, default="")
    trip_purpose = Column(Text, default="")
    trip_dates = Column(String, default="")
    is_seed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)

    submissions = relationship("Submission", back_populates="employee")


class Submission(Base):
    """One expense submission (a set of receipts) for an employee."""
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    employee_pk = Column(Integer, ForeignKey("employees.id"), index=True)
    label = Column(String, default="")            # e.g. trip purpose snapshot
    status = Column(String, default="pending", index=True)  # pending|processing|reviewed|error
    created_at = Column(DateTime, default=_utcnow, index=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    employee = relationship("Employee", back_populates="submissions")
    line_items = relationship("LineItem", back_populates="submission",
                              cascade="all, delete-orphan")


class LineItem(Base):
    """One receipt = one line item, with its extraction and system verdict.
    Verdict fields are the SYSTEM's; reviewer overrides live in OverrideEvent."""
    __tablename__ = "line_items"

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), index=True)
    filename = Column(String, default="")
    receipt_path = Column(String, default="")     # stored upload, for re-view
    receipt = Column(JSON)                         # full ExtractedReceipt
    category = Column(String, index=True)

    # System verdict (immutable record of what the model said)
    verdict = Column(String, index=True)           # compliant|flagged|rejected|needs_info
    reasoning = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    reimbursable_amount = Column(Float)
    issues = Column(JSON, default=list)
    citations = Column(JSON, default=list)         # [{doc_id, section, quote, verified, ratio}]
    retrieval_top_score = Column(Float, default=0.0)
    citations_faithful = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)

    submission = relationship("Submission", back_populates="line_items")
    overrides = relationship("OverrideEvent", back_populates="line_item",
                             cascade="all, delete-orphan",
                             order_by="OverrideEvent.created_at")

    @property
    def effective_verdict(self) -> str:
        return self.overrides[-1].new_verdict if self.overrides else self.verdict


class OverrideEvent(Base):
    """Append-only audit log of reviewer overrides. We never mutate a verdict in
    place — each override is a new row, so the full history is preserved."""
    __tablename__ = "override_events"

    id = Column(Integer, primary_key=True)
    line_item_id = Column(Integer, ForeignKey("line_items.id"), index=True)
    reviewer = Column(String, default="reviewer")
    new_verdict = Column(String, nullable=False)
    comment = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow)

    line_item = relationship("LineItem", back_populates="overrides")
