"""Thin Gemini client wrapper over the current `google-genai` SDK: embeddings +
schema-constrained generation, with retry/backoff for free-tier rate limits
(10 RPM). One module so model choice and failure handling live in one place."""
from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any

from google import genai
from google.genai import types

from app import config

_client: genai.Client | None = None
_client_lock = threading.Lock()

# Thread-safe rolling-window limiters for the free-tier quotas. Embedding counts
# each *content*; generation counts each request, with a SEPARATE budget per
# model (flash vs flash-lite have independent quotas, so extraction and verdicts
# don't compete). Reserving a slot atomically then sleeping outside the lock lets
# concurrent workers fill each budget without bursting past it.
_lock = threading.Lock()
_embed_times: deque[float] = deque()
_gen_times: dict[str, deque[float]] = defaultdict(deque)


def _reserve(times: deque[float], n: int, limit: int, window: float) -> None:
    while True:
        with _lock:
            now = time.time()
            while times and now - times[0] >= window:
                times.popleft()
            if len(times) + n <= limit:
                times.extend([now] * n)
                return
            wait = window - (now - times[0]) + 0.05
        time.sleep(max(wait, 0.05))


def client() -> genai.Client:
    global _client
    if _client is None:
        # Double-checked locking: without this, concurrent workers can each build
        # a client and the discarded duplicate's HTTP session gets closed mid-use.
        with _client_lock:
            if _client is None:
                if not config.GEMINI_API_KEY:
                    raise RuntimeError(
                        "GEMINI_API_KEY is not set. Export it or put it in backend/.env")
                _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


class QuotaExceeded(RuntimeError):
    """Raised when a per-DAY free-tier quota is exhausted (retrying won't help
    until reset). Carries a human-actionable message."""


def _retry_delay_secs(msg: str) -> float | None:
    """Pull the server-suggested wait out of a 429 message, if present."""
    m = re.search(r"retry in ([0-9.]+)s", msg) or re.search(r"retrydelay['\":\s]+([0-9.]+)s", msg)
    return float(m.group(1)) if m else None


def _is_per_day(msg: str) -> bool:
    return "perday" in msg or "per day" in msg or "requests per day" in msg


def _retry(fn, *, tries: int = 6, base: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except QuotaExceeded:
            raise
        except Exception as e:  # includes 429 RESOURCE_EXHAUSTED, 5xx
            last = e
            msg = str(e).lower()
            # A per-DAY cap won't clear by waiting seconds — fail fast & clearly.
            if "429" in msg and _is_per_day(msg):
                m = re.search(r"model:\s*([\w.\-]+)", msg)
                model = m.group(1) if m else "this model"
                raise QuotaExceeded(
                    f"Daily free-tier request quota is exhausted for {model}. "
                    f"The free tier allows very few requests/day for this model. "
                    f"Options: wait for the daily reset (~midnight Pacific), point "
                    f"NW_MODEL_REASONING/NW_MODEL_EXTRACT at a model that still has "
                    f"quota, or enable billing on the API key."
                ) from e
            if any(k in msg for k in ("429", "rate", "quota", "exhaust",
                                      "resource", "503", "500", "unavailable")):
                wait = _retry_delay_secs(msg)
                time.sleep((wait + 1.0) if wait else min(base ** i, 30))
                continue
            raise
    raise last


def _throttle_embed(n: int) -> None:
    """Pace embedding to stay within the per-minute content quota."""
    _reserve(_embed_times, n, config.EMBED_RPM, config.EMBED_WINDOW_SECS)


def _throttle_gen(model: str) -> None:
    """Pace generation per-model (independent free-tier quotas)."""
    _reserve(_gen_times[model], 1, config.gen_rpm_for(model), config.GEN_WINDOW_SECS)


def embed(texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT",
          batch_size: int | None = None, progress: bool = False) -> list[list[float]]:
    """Embed texts. task_type is RETRIEVAL_DOCUMENT for the corpus and
    RETRIEVAL_QUERY for user queries (asymmetric embeddings improve recall).

    The SDK sends a list as a single batchEmbedContents call, but the free tier
    counts each *content* against a per-minute quota — so we pace via a rolling
    window and honor the server's retry hints. A one-time corpus build of ~700
    chunks takes a few minutes, then it's cached and never re-embedded."""
    cfg = types.EmbedContentConfig(
        task_type=task_type.upper(), output_dimensionality=config.EMBED_DIM)
    bs = batch_size or config.EMBED_RPM  # never exceed one window per batch

    def _one_call(contents):
        _throttle_embed(len(contents))
        r = _retry(lambda: client().models.embed_content(
            model=config.EMBED_MODEL, contents=contents, config=cfg))
        _embed_times.extend([time.time()] * len(contents))
        return [e.values for e in r.embeddings]

    out: list[list[float]] = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        out.extend(_one_call(batch))
        if progress and len(texts) > bs:
            print(f"  embedded {min(i + bs, len(texts))}/{len(texts)}")
    return out


def generate_json(prompt: str, schema: Any, *, model: str | None = None,
                  system: str | None = None, temperature: float = 0.0) -> Any:
    """Schema-constrained text generation. Returns parsed JSON."""
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        response_mime_type="application/json", response_schema=schema)
    m = model or config.MODEL_REASONING
    _throttle_gen(m)
    r = _retry(lambda: client().models.generate_content(
        model=m, contents=prompt, config=cfg))
    return json.loads(r.text)


def generate_multimodal_json(parts: list[Any], schema: Any, *,
                             model: str | None = None, system: str | None = None,
                             temperature: float = 0.0) -> Any:
    """Schema-constrained generation over mixed parts (text + receipt bytes)."""
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        response_mime_type="application/json", response_schema=schema)
    m = model or config.MODEL_EXTRACT
    _throttle_gen(m)
    r = _retry(lambda: client().models.generate_content(
        model=m, contents=parts, config=cfg))
    return json.loads(r.text)


def part_from_bytes(data: bytes, mime_type: str):
    """Build a multimodal Part from raw file bytes (PDF/PNG/JPG)."""
    return types.Part.from_bytes(data=data, mime_type=mime_type)
