# Single image: FastAPI backend + the static reviewer UI it serves.
FROM python:3.12-slim

WORKDIR /app

# System deps for pdfplumber/pillow are pure-python wheels here; none needed.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code, seed data (policies + sample submissions + prebuilt index), UI.
COPY backend /app/backend
COPY data /app/data
COPY frontend /app/frontend

ENV PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1 \
    NW_STATE_DIR=/data

# Runtime state (SQLite + uploads) lives on a mounted volume at /data.
VOLUME ["/data"]
EXPOSE 8080

CMD ["sh", "-c", "cd /app/backend && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
