# Quickstart

See **[README.md](./README.md)** for full setup, architecture, and deploy steps.

TL;DR — run locally:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY
python -m app.ingest.embed    # one-time: build the policy index
uvicorn app.main:app --port 8000   # open http://localhost:8000
```

Phase demo scripts (each runs locally against Gemini):

```bash
python -m scripts.retrieval_demo                 # Phase 1: retrieval quality
python -m scripts.extract_demo 04_alcohol_solo_travel   # Phase 2: receipt extraction
python -m scripts.review_demo  04_alcohol_solo_travel   # Phase 3: full verdicts
python -m scripts.evaluate ../eval/expected_sample.json # Phase 6: metrics
```
