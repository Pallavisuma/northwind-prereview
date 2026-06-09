"""Pydantic schemas that double as Gemini `response_schema`s. Using schema-
constrained generation (not free-text parsing) is the single biggest lever for
reliability here: the model must return exactly these fields and types, so
downstream code never parses prose."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    airfare = "airfare"
    lodging = "lodging"
    ground_transport = "ground_transport"
    meal = "meal"
    conference = "conference"
    other = "other"


class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    other = "other"


class FlightClass(str, Enum):
    economy = "economy"
    premium_economy = "premium_economy"
    business = "business"
    first = "first"
    unknown = "unknown"


class ReceiptLine(BaseModel):
    """One ordered item on the receipt (esp. for meals, to isolate alcohol)."""
    description: str
    amount: float
    is_alcohol: bool = Field(
        description="True only if this line is an alcoholic beverage "
                    "(beer, wine, spirits, cocktail). Non-alcoholic drinks are False."
    )


class ExtractedReceipt(BaseModel):
    """Everything we pull from a single receipt. Optional fields are null when
    the receipt doesn't show them — the model must NOT guess."""
    vendor: str = Field(description="Merchant / business name.")
    category: Category
    meal_type: Optional[MealType] = Field(
        default=None, description="Only for category=meal; infer from time/items.")

    city: Optional[str] = None
    state_or_country: Optional[str] = Field(
        default=None, description="State (US) or country, as printed.")
    date: Optional[str] = Field(default=None, description="Transaction date, ISO 8601 (YYYY-MM-DD).")

    currency: str = Field(default="USD", description="ISO currency code.")
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    total: Optional[float] = Field(default=None, description="Grand total actually charged.")
    payment_method: Optional[str] = Field(
        default=None,
        description="Method of payment as shown, e.g. 'Visa ****8829' or 'Cash'. "
                    "Null only if the receipt truly does not show one.")

    line_items: list[ReceiptLine] = Field(
        default_factory=list,
        description="Itemized lines if the receipt shows them; empty if not itemized.")
    alcohol_total: Optional[float] = Field(
        default=None, description="Sum of alcoholic line amounts; null if none/unknown.")

    # Meal/entertainment context that can appear ON the receipt itself.
    diner_count: Optional[int] = Field(
        default=None, description="Number of guests/covers if printed on the receipt.")
    has_external_attendees: Optional[bool] = Field(
        default=None,
        description="Set ONLY from explicit receipt text (e.g. a note naming external "
                    "clients, or 'no external attendees'). Otherwise null.")
    notes_on_receipt: Optional[str] = Field(
        default=None, description="Any handwritten or printed annotation, verbatim.")

    # Category-specific extras (null when not applicable).
    flight_class: Optional[FlightClass] = None
    lodging_nights: Optional[int] = None
    lodging_nightly_rate: Optional[float] = None

    is_itemized: bool = Field(
        description="Whether the receipt shows an itemized breakdown of charges.")
    extraction_confidence: float = Field(
        ge=0.0, le=1.0,
        description="0-1 confidence the fields above are correct and legible.")
    extraction_warnings: list[str] = Field(
        default_factory=list,
        description="Legibility/ambiguity problems, e.g. 'total partially obscured'.")


# --- Verdict (Phase 3) ----------------------------------------------------

class VerdictLabel(str, Enum):
    compliant = "compliant"    # clearly within policy
    flagged = "flagged"        # likely/partial violation — needs a human, may be partly reimbursable
    rejected = "rejected"      # clear violation, nothing reimbursable
    needs_info = "needs_info"  # genuinely ambiguous or weak policy support — honest "I don't know"


class Citation(BaseModel):
    """A policy clause the verdict relies on. `quote` MUST be copied verbatim
    from the provided clause text — it is verified against the corpus after
    generation, so fabricated or altered quotes are caught."""
    doc_id: str = Field(description="e.g. 'TEP-003'")
    section: str = Field(description="e.g. '3.1' (empty string if none)")
    quote: str = Field(description="Verbatim sentence(s) from the clause that support the verdict.")


class Verdict(BaseModel):
    verdict: VerdictLabel
    reasoning: str = Field(
        description="Concise, reviewer-facing explanation grounded in the cited clauses.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Clauses relied on. Required unless verdict is needs_info with no relevant policy.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0-1 confidence in this verdict given policy support and receipt clarity.")
    reimbursable_amount: Optional[float] = Field(
        default=None,
        description="Amount that should be reimbursed: full total if compliant, the allowed "
                    "portion if partially flagged, 0 if rejected, null if undetermined.")
    issues: list[str] = Field(
        default_factory=list,
        description="Short machine-readable tags, e.g. 'alcohol_on_solo_travel', 'over_dinner_cap'.")
