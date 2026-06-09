"""SQLAlchemy engine/session. SQLite on a persistent volume — state survives
restarts (no in-memory anything)."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DB_URL, IS_SQLITE

# SQLite needs check_same_thread off for our worker threads; Postgres wants
# pre-ping to survive Neon/Render idle connection drops.
engine = create_engine(
    DB_URL, future=True,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
    pool_pre_ping=not IS_SQLITE,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
Base = declarative_base()


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    import app.models  # noqa: F401  (register tables)
    Base.metadata.create_all(engine)
