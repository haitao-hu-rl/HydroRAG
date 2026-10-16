from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable


class SQLiteMemoryStore:
    """Persist conversation context separately from the document knowledge base."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_messages_scope
                    ON messages(user_id, thread_id, id);
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, thread_id)
                );
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'preference',
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'explicit_user_request',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, memory_type, content)
                );
                CREATE INDEX IF NOT EXISTS idx_long_term_user
                    ON long_term_memories(user_id, id);
                """
            )

    def recent_history(
        self,
        user_id: str,
        thread_id: str,
        turns: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        limit = max(1, int(turns)) * 2
        with self._connect() as connection:
            summary_row = connection.execute(
                "SELECT summary FROM conversation_summaries WHERE user_id=? AND thread_id=?",
                (user_id, thread_id),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT role, content, citations_json
                FROM messages
                WHERE user_id=? AND thread_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, thread_id, limit),
            ).fetchall()
        history = [
            {
                "role": row["role"],
                "content": row["content"],
                "citations": json.loads(row["citations_json"] or "[]"),
            }
            for row in reversed(rows)
        ]
        return (summary_row["summary"] if summary_row else ""), history

    def append_turn(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        output: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO messages(user_id, thread_id, role, content, citations_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (user_id, thread_id, "user", query, "[]"),
                    (
                        user_id,
                        thread_id,
                        "assistant",
                        output,
                        json.dumps(citations or [], ensure_ascii=False),
                    ),
                ],
            )

    def compact_history(
        self,
        user_id: str,
        thread_id: str,
        keep_turns: int,
        summarizer: Callable[[str, str], str] | None = None,
    ) -> None:
        keep_messages = max(1, int(keep_turns)) * 2
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content FROM messages
                WHERE user_id=? AND thread_id=? ORDER BY id
                """,
                (user_id, thread_id),
            ).fetchall()
            if len(rows) <= keep_messages:
                return
            older = rows[:-keep_messages]
            previous = connection.execute(
                "SELECT summary FROM conversation_summaries WHERE user_id=? AND thread_id=?",
                (user_id, thread_id),
            ).fetchone()
            previous_summary = previous["summary"] if previous else ""
            transcript = "\n".join(f"{row['role']}: {row['content']}" for row in older)
            if summarizer:
                summary = summarizer(previous_summary, transcript).strip()
            else:
                combined = f"{previous_summary}\n{transcript}".strip()
                summary = combined[-4000:]
            connection.execute(
                """
                INSERT INTO conversation_summaries(user_id, thread_id, summary, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, thread_id) DO UPDATE SET
                    summary=excluded.summary, updated_at=CURRENT_TIMESTAMP
                """,
                (user_id, thread_id, summary),
            )
            connection.execute(
                f"DELETE FROM messages WHERE id IN ({','.join('?' for _ in older)})",
                tuple(row["id"] for row in older),
            )

    def clear_conversation(self, user_id: str, thread_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE user_id=? AND thread_id=?",
                (user_id, thread_id),
            )
            connection.execute(
                "DELETE FROM conversation_summaries WHERE user_id=? AND thread_id=?",
                (user_id, thread_id),
            )

    def add_long_term(
        self,
        user_id: str,
        content: str,
        memory_type: str = "preference",
    ) -> int:
        content = content.strip()
        if not content:
            raise ValueError("Memory content must not be empty.")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO long_term_memories(user_id, memory_type, content)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, memory_type, content) DO UPDATE SET
                    updated_at=CURRENT_TIMESTAMP
                """,
                (user_id, memory_type, content),
            )
            row = connection.execute(
                """
                SELECT id FROM long_term_memories
                WHERE user_id=? AND memory_type=? AND content=?
                """,
                (user_id, memory_type, content),
            ).fetchone()
        return int(row["id"])

    def list_long_term(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, memory_type, content, source, created_at, updated_at
                FROM long_term_memories WHERE user_id=? ORDER BY id
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_long_term(self, user_id: str, memory_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM long_term_memories WHERE user_id=? AND id=?",
                (user_id, int(memory_id)),
            )
        return cursor.rowcount > 0

    def clear_long_term(self, user_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM long_term_memories WHERE user_id=?",
                (user_id,),
            )
        return max(cursor.rowcount, 0)

    def forget_matching(self, user_id: str, text: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM long_term_memories WHERE user_id=? AND content LIKE ?",
                (user_id, f"%{text.strip()}%"),
            )
        return max(cursor.rowcount, 0)
