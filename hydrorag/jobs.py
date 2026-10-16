from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = ("pending", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IndexJobStore:
    """Small persistent job ledger for the single-process ingest worker."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'succeeded', 'failed')),
                    document_id TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_index_jobs_status
                    ON index_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_index_jobs_source
                    ON index_jobs(source_path, status);
                """
            )

    def create(self, source_path: str, file_name: str) -> str:
        job_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO index_jobs(job_id, source_path, file_name, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (job_id, source_path, file_name, _now()),
            )
        return job_id

    def active_for_path(self, source_path: str) -> bool:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT 1 FROM index_jobs WHERE source_path=? AND status IN ({placeholders}) LIMIT 1",
                (source_path, *ACTIVE_STATUSES),
            ).fetchone()
        return row is not None

    def mark_running(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE index_jobs SET status='running', started_at=? WHERE job_id=?",
                (_now(), job_id),
            )

    def complete(self, job_id: str, document_id: str, chunk_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_jobs
                SET status='succeeded', document_id=?, chunk_count=?, finished_at=?
                WHERE job_id=?
                """,
                (document_id, int(chunk_count), _now(), job_id),
            )

    def fail(self, job_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_jobs SET status='failed', error=?, finished_at=? WHERE job_id=?
                """,
                (str(error)[:2000], _now(), job_id),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, file_name, status, document_id, chunk_count, error,
                       created_at, started_at, finished_at
                FROM index_jobs WHERE job_id=?
                """,
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def recover_after_restart(self) -> list[dict[str, str]]:
        with self._connect() as connection:
            connection.execute(
                "UPDATE index_jobs SET status='pending', error='', started_at='' WHERE status='running'",
            )
            rows = connection.execute(
                "SELECT job_id, source_path FROM index_jobs WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM index_jobs GROUP BY status"
            ).fetchall()
        result = {status: 0 for status in ("pending", "running", "succeeded", "failed")}
        result.update({str(row["status"]): int(row["count"]) for row in rows})
        return result
