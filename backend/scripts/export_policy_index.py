"""Export the embedded policy index (and only that) into data/policy_index.db,
a small seed file committed to the repo. The deployed container restores it on
first boot, so production needs ZERO embedding calls to come online.

No API/quota used — this just copies the policy_chunks + meta tables from your
already-built runtime DB. Run once locally after the index is built:

    cd backend
    python -m scripts.embed_then_export   # or: python -m app.ingest.embed && python -m scripts.export_policy_index
"""
from __future__ import annotations

import sqlite3

from app.config import DB_PATH, SEED_INDEX


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"No runtime DB at {DB_PATH}. Build the index first: "
                         f"python -m app.ingest.embed")
    SEED_INDEX.parent.mkdir(parents=True, exist_ok=True)
    if SEED_INDEX.exists():
        SEED_INDEX.unlink()

    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(SEED_INDEX)
    # Copy schema + data for just the policy tables.
    for tbl in ("policy_chunks", "meta"):
        row = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
        ).fetchone()
        if not row:
            continue
        dst.execute(row[0])
        cols = [r[1] for r in src.execute(f"PRAGMA table_info({tbl})")]
        placeholders = ",".join("?" * len(cols))
        rows = src.execute(f"SELECT {','.join(cols)} FROM {tbl}").fetchall()
        dst.executemany(f"INSERT INTO {tbl} VALUES ({placeholders})", rows)
        print(f"  copied {len(rows)} rows from {tbl}")
    dst.commit()
    n = dst.execute("SELECT COUNT(*) FROM policy_chunks").fetchone()[0]
    src.close(); dst.close()
    size_mb = SEED_INDEX.stat().st_size / 1e6
    print(f"Wrote {SEED_INDEX} ({n} chunks, {size_mb:.1f} MB). Commit this file.")


if __name__ == "__main__":
    main()
