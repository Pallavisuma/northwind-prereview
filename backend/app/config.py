"""Central configuration. All paths are absolute and derived from this file's
location so the app behaves the same locally and in a container."""
from __future__ import annotations

import os
from pathlib import Path

# backend/app/config.py -> backend/ -> repo root
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

# Load backend/.env if present (local dev convenience; no-op in prod).
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

# Seed data shipped with the repo
DATA_DIR = REPO_ROOT / "data"
POLICIES_DIR = DATA_DIR / "policies"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

# Runtime state. Lives on a persistent volume in production.
# Override with NW_STATE_DIR (e.g. a mounted disk) in deployment.
STATE_DIR = Path(os.environ.get("NW_STATE_DIR", BACKEND_DIR / "data"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STATE_DIR / "northwind.db"
UPLOADS_DIR = STATE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DB_URL = f"sqlite:///{DB_PATH}"

# Pre-built policy index shipped in the repo. On first boot, if the runtime DB
# doesn't exist yet, we restore from this so the deployed app is usable
# immediately without re-embedding (saves quota + avoids a slow cold start).
SEED_INDEX = DATA_DIR / "policy_index.db"

# --- Gemini ---------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Model tiers. The free tier caps gemini-2.5-flash at ~20 requests/DAY, so we
# default BOTH stages to Flash-Lite (far higher daily allowance, 30 RPM) to keep
# the system usable end-to-end on a free key. For higher-quality verdicts on a
# key with quota/billing, set NW_MODEL_REASONING=gemini-2.5-flash.
MODEL_REASONING = os.environ.get("NW_MODEL_REASONING", "gemini-2.5-flash-lite")
MODEL_EXTRACT = os.environ.get("NW_MODEL_EXTRACT", "gemini-2.5-flash-lite")
EMBED_MODEL = os.environ.get("NW_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.environ.get("NW_EMBED_DIM", "768"))
# Free-tier embedding quota counts each content against a per-minute limit.
# Stay just under 100 to be safe; raise this on a paid tier for fast ingestion.
EMBED_RPM = int(os.environ.get("NW_EMBED_RPM", "95"))
EMBED_WINDOW_SECS = float(os.environ.get("NW_EMBED_WINDOW_SECS", "60"))
# Generation free-tier limits, per model and per minute. Flash-Lite has a
# higher cap; we stay just under each documented limit to be safe.
GEN_RPM = int(os.environ.get("NW_GEN_RPM", "13"))            # gemini-2.5-flash (15 RPM)
GEN_RPM_LITE = int(os.environ.get("NW_GEN_RPM_LITE", "28"))  # gemini-2.5-flash-lite (30 RPM)
GEN_WINDOW_SECS = float(os.environ.get("NW_GEN_WINDOW_SECS", "60"))


def gen_rpm_for(model: str) -> int:
    return GEN_RPM_LITE if "lite" in (model or "").lower() else GEN_RPM

# Retrieval defaults
RETRIEVAL_TOP_K = int(os.environ.get("NW_RETRIEVAL_TOP_K", "8"))
