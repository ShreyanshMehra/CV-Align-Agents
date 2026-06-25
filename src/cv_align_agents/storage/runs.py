"""SQLite persistence for screening runs (explainability / audit trail).

Uses the standard-library ``sqlite3`` (no extra dependency). Each screening run
is stored as a row with its full JSON payload, so a run can be retrieved later
via ``GET /runs/{id}`` and recent runs can be listed. A fresh connection is
opened per operation, which keeps the store safe to call from FastAPI's worker
threads.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cv_align_agents.state import ScreeningResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    mode        TEXT NOT NULL,
    job_title   TEXT,
    n_candidates INTEGER NOT NULL,
    result_json TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """A small SQLite-backed store for screening runs."""

    def __init__(self, db_path: str | Path = "runs.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        # ":memory:" databases do not persist across connections, so creating a
        # fresh one per operation would lose data; guard against that footgun.
        if self.db_path == ":memory:":
            raise ValueError(
                "RunStore needs a file path; ':memory:' is not supported "
                "because each operation opens a new connection."
            )
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def save(self, result: ScreeningResult) -> str:
        """Persist a screening result and return its run id.

        Mutates ``result`` in place to set ``run_id`` and ``created_at`` if they
        are not already populated.
        """
        run_id = result.run_id or uuid4().hex
        created_at = result.created_at or _now_iso()
        result.run_id = run_id
        result.created_at = created_at

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(id, created_at, mode, job_title, n_candidates, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    created_at,
                    result.mode,
                    result.job_title,
                    len(result.candidates),
                    result.model_dump_json(),
                ),
            )
        return run_id

    def get(self, run_id: str) -> ScreeningResult | None:
        """Return a stored run by id, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return ScreeningResult.model_validate_json(row["result_json"])

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return summaries of recent runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, mode, job_title, n_candidates "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
