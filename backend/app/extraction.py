"""Receipt extraction. Each receipt (PDF / PNG / JPG / TXT) becomes one typed
ExtractedReceipt via a single schema-constrained Gemini call.

Design choices (defended in the README):
  * Vision-first. We hand the raw file to a multimodal model instead of running
    OCR + regex. Real receipts vary wildly in layout; a vision model reads a
    crumpled photo or an odd PDF far more robustly than brittle parsing, and it
    can also read on-receipt notes ("Solo diner. No external attendees") that
    turn out to decide verdicts.
  * Extract facts, not judgments. This stage never decides compliance — it only
    reports what the receipt says, including a self-reported confidence and
    explicit warnings when something is illegible. Verdicts come later, so the
    reasoning is auditable and each stage is independently testable.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from app import gemini
from app.config import STATE_DIR
from app.schemas import ExtractedReceipt

_CACHE_DIR = STATE_DIR / "extract_cache"

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".txt": "text/plain",
    ".text": "text/plain",
}

SUPPORTED_EXTS = set(MIME_BY_EXT)

EXTRACTION_SYSTEM = (
    "You are a meticulous expense-receipt extractor for a corporate finance "
    "system. Read the receipt and return ONLY the structured fields requested. "
    "Rules:\n"
    "- Transcribe what is actually printed. Do NOT invent or infer values that "
    "are not visible. If a field is absent or unreadable, leave it null.\n"
    "- For meals, itemize every line and mark each as alcohol or not. Beer, "
    "wine, spirits, cocktails, hard seltzer = alcohol. Coffee, soda, juice, "
    "sparkling/still water, mocktails = NOT alcohol.\n"
    "- Set alcohol_total to the sum of alcoholic line amounts (null if none).\n"
    "- Infer meal_type from the time of day and items (breakfast/lunch/dinner).\n"
    "- For flights, set flight_class from the fare/cabin shown (economy, "
    "premium_economy, business, first); use 'unknown' if not shown.\n"
    "- Capture the method of payment if shown (e.g. 'Visa ****8829', 'Amex "
    "****1001', or 'Cash'); set payment_method to null only if truly absent.\n"
    "- Capture any handwritten or printed annotation verbatim in "
    "notes_on_receipt. Only set has_external_attendees / diner_count when the "
    "receipt explicitly states it.\n"
    "- Report extraction_confidence honestly and list extraction_warnings for "
    "anything ambiguous, blurry, or partially obscured."
)


def guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in MIME_BY_EXT:
        raise ValueError(
            f"Unsupported receipt format '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")
    return MIME_BY_EXT[ext]


def _reconcile(r: ExtractedReceipt) -> ExtractedReceipt:
    """Light, deterministic cleanup of model output (no judgments)."""
    # Derive alcohol_total from line items when the model left it null.
    if r.alcohol_total is None and r.line_items:
        alc = sum(li.amount for li in r.line_items if li.is_alcohol)
        r.alcohol_total = round(alc, 2) if alc > 0 else None
    # A negative tip is nonsensical — it means subtotal+tax exceeded the total,
    # i.e. an unparsed discount/comp. Drop the bogus value and flag the gap so a
    # reviewer (and the verdict engine) see a clean number with a caveat.
    if r.tip is not None and r.tip < 0:
        r.extraction_warnings.append(
            f"Total is below subtotal+tax (implied tip {r.tip:.2f}); likely a "
            "discount/comp not itemized. Tip set to null.")
        r.tip = None
    # Derive total from parts when missing.
    if r.total is None:
        parts = [r.subtotal or 0, r.tax or 0, r.tip or 0]
        if any(p for p in parts):
            r.total = round(sum(parts), 2)
    return r


def _pdf_text(data: bytes) -> str:
    """Best-effort text layer from a PDF (empty if scanned/unparseable)."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages).strip()
    except Exception:
        return ""


def extract_receipt(data: bytes, filename: str) -> ExtractedReceipt:
    """Extract one receipt from raw bytes. `filename` drives format detection.

    Hybrid strategy: we always give the model the original file (so it can use
    layout/vision), and for PDFs we ALSO attach the extracted text layer. Clear
    lines like 'Payment: Visa ****3214' or exact totals are then available as
    text and far less likely to be missed than from vision alone — while images
    and scanned PDFs still fall back to pure vision."""
    mime = guess_mime(filename)
    instruction = f"Extract this receipt (source file: {filename})."

    if mime == "text/plain":
        text = data.decode("utf-8", errors="replace")
        parts: list = [f"{instruction}\n\nReceipt text:\n{text}"]
    elif mime == "application/pdf":
        text = _pdf_text(data)
        lead = instruction + (
            f"\n\nExtracted text layer (use together with the document image; "
            f"prefer it for exact amounts and the payment line):\n{text}"
            if text else "")
        parts = [lead, gemini.part_from_bytes(data, mime)]
    else:  # images
        parts = [instruction, gemini.part_from_bytes(data, mime)]

    raw = gemini.generate_multimodal_json(
        parts, schema=ExtractedReceipt, system=EXTRACTION_SYSTEM)
    return _reconcile(ExtractedReceipt.model_validate(raw))


def extract_receipt_cached(data: bytes, filename: str) -> ExtractedReceipt:
    """Same as extract_receipt, but memoized on (file bytes + extractor version)
    so re-processing an identical receipt is free. Content-addressed, so it stays
    correct if a file changes. Disable by deleting STATE_DIR/extract_cache."""
    key = hashlib.sha256(data + filename.encode() + EXTRACTION_SYSTEM.encode()).hexdigest()
    cache_file = _CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return ExtractedReceipt.model_validate_json(cache_file.read_text())
        except Exception:
            pass  # corrupt cache entry -> re-extract
    result = extract_receipt(data, filename)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result.model_dump_json())
    return result


def extract_path(path: str | Path, *, cache: bool = False) -> ExtractedReceipt:
    p = Path(path)
    fn = extract_receipt_cached if cache else extract_receipt
    return fn(p.read_bytes(), p.name)
