"""Orchestration: turn uploaded receipts into a fully reviewed, persisted
submission. Receipts are processed concurrently (each does extract -> retrieve
-> verdict); persistence happens on the request's session afterward so we keep a
single writer. Uploaded files are saved to the volume so a submission can be
re-opened or re-reviewed later."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy.orm import Session

from app import store
from app.config import UPLOADS_DIR
from app.context import Employee as EmployeeCtx
from app.extraction import extract_receipt_cached, guess_mime, SUPPORTED_EXTS
from app.gemini import QuotaExceeded
from app.models import Submission
from app.review import review_line_item


def _save_upload(submission_id: int, filename: str, data: bytes) -> str:
    d = UPLOADS_DIR / str(submission_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    path.write_bytes(data)
    return str(path)


def process_submission(db: Session, submission: Submission,
                       files: list[tuple[str, bytes]], emp: EmployeeCtx,
                       max_workers: int = 6) -> Submission:
    """Extract + review every receipt, then persist. Unsupported files are
    skipped with a clear error line item rather than crashing the whole batch."""
    store.set_status(db, submission, "processing")

    def work(item: tuple[str, bytes]):
        filename, data = item
        guess_mime(filename)  # raises ValueError on unsupported format
        path = _save_upload(submission.id, filename, data)
        receipt = extract_receipt_cached(data, filename)
        review = review_line_item(receipt, emp, filename=filename)
        return filename, review, path

    results, errors = [], []
    try:
        with ThreadPoolExecutor(max_workers=min(max_workers, max(len(files), 1))) as ex:
            futs = {ex.submit(work, it): it[0] for it in files}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except QuotaExceeded:
                    # Systemic outage, not a per-file problem — abort the whole
                    # batch so the API returns a clear 429 to the reviewer.
                    store.set_status(db, submission, "error")
                    raise
                except Exception as e:  # noqa: BLE001 — genuine per-file issue
                    errors.append((futs[fut], str(e)))
    except QuotaExceeded:
        raise
    except Exception:
        store.set_status(db, submission, "error")
        raise

    for _, review, path in results:
        store.save_review(db, submission.id, review, receipt_path=path)
    store.set_status(db, submission, "reviewed" if not errors else "reviewed_with_errors")
    submission.errors = errors  # transient, surfaced in the API response
    return submission


def valid_receipt(filename: str) -> bool:
    from pathlib import Path
    return Path(filename).suffix.lower() in SUPPORTED_EXTS
