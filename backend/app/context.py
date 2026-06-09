"""Employee / trip context. Loaded from employee_info.json and threaded into the
verdict engine — several sample cases can only be judged correctly with trip
context, not the receipt alone (e.g. alcohol hinges on whether a meal was solo
or sanctioned client entertainment)."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Employee(BaseModel):
    employee_id: str
    name: str
    grade: int
    title: str
    department: str
    manager_id: Optional[str] = None
    home_base: Optional[str] = None
    trip_purpose: Optional[str] = None
    trip_dates: Optional[str] = None  # "2025-04-14 to 2025-04-16"

    def _parse_dates(self) -> tuple[Optional[date], Optional[date]]:
        if not self.trip_dates:
            return None, None
        parts = [p.strip() for p in self.trip_dates.replace("—", "-").split(" to ")]
        out: list[Optional[date]] = []
        for p in parts[:2]:
            try:
                out.append(datetime.strptime(p, "%Y-%m-%d").date())
            except ValueError:
                out.append(None)
        while len(out) < 2:
            out.append(out[0] if out else None)
        return out[0], out[1]

    @property
    def trip_start(self) -> Optional[date]:
        return self._parse_dates()[0]

    @property
    def trip_end(self) -> Optional[date]:
        return self._parse_dates()[1]

    @property
    def trip_nights(self) -> Optional[int]:
        s, e = self._parse_dates()
        if s and e:
            return max((e - s).days, 0)
        return None

    def context_brief(self) -> str:
        """Compact, factual context block for the verdict prompt."""
        nights = self.trip_nights
        return (
            f"Employee: {self.name} (ID {self.employee_id}), Grade {self.grade}, "
            f"{self.title}, {self.department}.\n"
            f"Home base: {self.home_base or 'unknown'}.\n"
            f"Trip purpose: {self.trip_purpose or 'unknown'}.\n"
            f"Trip dates: {self.trip_dates or 'unknown'}"
            + (f" ({nights} night{'s' if nights != 1 else ''})." if nights is not None else ".")
        )


def load_employee(submission_dir: str | Path) -> Employee:
    p = Path(submission_dir) / "employee_info.json"
    return Employee.model_validate_json(p.read_text())
