"""SQLite relational + vector memory storage."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Sequence

import numpy as np

from config import DB_PATH, EMBEDDING_DIMENSION
from core.schema import Fact, FactStatus, utc_now

logger = logging.getLogger(__name__)


def _serialize_embedding(vector: Sequence[float]) -> bytes:
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class DatabaseError(Exception):
    """Raised when a database operation fails."""


class MemoryDatabase:
    """Hybrid SQLite store for conversation turns and vector-backed memories."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError(f"SQLite operation failed: {exc}") from exc
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON conversation_turns(session_id, timestamp);

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    subject TEXT DEFAULT 'user',
                    predicate TEXT DEFAULT '',
                    object TEXT DEFAULT '',
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    superseded_by TEXT,
                    metadata_json TEXT DEFAULT '{}',
                    embedding BLOB,
                    embedding_dim INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_memories_status
                    ON memories(status);
                CREATE INDEX IF NOT EXISTS idx_memories_category
                    ON memories(category);
                CREATE INDEX IF NOT EXISTS idx_memories_subject
                    ON memories(subject);
                """
            )

    @staticmethod
    def _dt_to_str(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @staticmethod
    def _str_to_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            text=row["text"],
            subject=row["subject"] or "user",
            predicate=row["predicate"] or "",
            object=row["object"] or "",
            category=row["category"],
            confidence=float(row["confidence"]),
            status=FactStatus(row["status"]),
            created_at=self._str_to_dt(row["created_at"]) or utc_now(),
            updated_at=self._str_to_dt(row["updated_at"]) or utc_now(),
            last_accessed_at=self._str_to_dt(row["last_accessed_at"]),
            superseded_by=row["superseded_by"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        ts = self._dt_to_str(timestamp or utc_now())
        meta_json = json.dumps(metadata or {})
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_turns (session_id, role, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, ts, meta_json),
            )
            return int(cursor.lastrowid)

    def get_turns(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp, metadata_json
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def clear_session_turns(self, session_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_turns WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount

    def forget_fact(self, fact_id: str) -> bool:
        """Remove a memory from active retrieval while retaining its audit row."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status = ?, superseded_by = NULL, updated_at = ? WHERE id = ? AND status = ?",
                (
                    FactStatus.SUPERSEDED.value,
                    self._dt_to_str(utc_now()),
                    fact_id,
                    FactStatus.ACTIVE.value,
                ),
            )
            return cursor.rowcount > 0

    def supersede_active_locations(self, new_fact_id: str, subject: str = "user") -> list[str]:
        """Retire every active location atomically, returning retired fact IDs."""
        try:
            with self._connect() as db_conn:
                rows = db_conn.execute(
                    "SELECT id FROM memories WHERE subject = ? AND predicate = ? AND status = ? AND id != ?",
                    (subject, "lives_in", FactStatus.ACTIVE.value, new_fact_id),
                ).fetchall()
                old_ids = [str(row["id"]) for row in rows]
                if old_ids:
                    now = self._dt_to_str(utc_now())
                    db_conn.executemany(
                        "UPDATE memories SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ? AND status = ?",
                        [
                            (FactStatus.SUPERSEDED.value, new_fact_id, now, fact_id, FactStatus.ACTIVE.value)
                            for fact_id in old_ids
                        ],
                    )
                return old_ids
        except Exception as exc:
            try:
                db_conn.rollback()
            except (NameError, sqlite3.Error):
                pass
            raise DatabaseError(f"Location supersession failed: {exc}") from exc

    def upsert_fact(self, fact: Fact, embedding: Sequence[float] | None = None) -> Fact:
        now = utc_now()
        fact.updated_at = now
        embedding_blob = None
        embedding_dim = 0
        if embedding is not None:
            embedding_blob = _serialize_embedding(embedding)
            embedding_dim = len(embedding)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, text, subject, predicate, object, category,
                    confidence, status, created_at, updated_at, last_accessed_at,
                    superseded_by, metadata_json, embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    subject=excluded.subject,
                    predicate=excluded.predicate,
                    object=excluded.object,
                    category=excluded.category,
                    confidence=excluded.confidence,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    last_accessed_at=excluded.last_accessed_at,
                    superseded_by=excluded.superseded_by,
                    metadata_json=excluded.metadata_json,
                    embedding=COALESCE(excluded.embedding, memories.embedding),
                    embedding_dim=CASE
                        WHEN excluded.embedding_dim > 0 THEN excluded.embedding_dim
                        ELSE memories.embedding_dim
                    END
                """,
                (
                    fact.id,
                    fact.text,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.category.value if hasattr(fact.category, "value") else str(fact.category),
                    fact.confidence,
                    fact.status.value,
                    self._dt_to_str(fact.created_at),
                    self._dt_to_str(fact.updated_at),
                    self._dt_to_str(fact.last_accessed_at) if fact.last_accessed_at else None,
                    fact.superseded_by,
                    json.dumps(fact.metadata),
                    embedding_blob,
                    embedding_dim,
                ),
            )
        return fact

    def get_fact(self, fact_id: str) -> Fact | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (fact_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_fact(row)

    def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> None:
        now = self._dt_to_str(utc_now())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET status = ?, superseded_by = ?, updated_at = ?
                WHERE id = ?
                """,
                (FactStatus.SUPERSEDED.value, new_fact_id, now, old_fact_id),
            )

    def update_fact_confidence(self, fact_id: str, confidence: float, touch_access: bool = True) -> None:
        now = utc_now()
        with self._connect() as conn:
            if touch_access:
                conn.execute(
                    """
                    UPDATE memories
                    SET confidence = ?, updated_at = ?, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (confidence, self._dt_to_str(now), self._dt_to_str(now), fact_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE memories
                    SET confidence = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (confidence, self._dt_to_str(now), fact_id),
                )

    def touch_fact_access(self, fact_id: str) -> None:
        now = self._dt_to_str(utc_now())
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
                (now, fact_id),
            )

    def list_active_facts(self, limit: int = 100) -> list[Fact]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (FactStatus.ACTIVE.value, limit),
            ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_active_candidates(
        self,
        subject: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        clauses = ["status = ?"]
        params: list[Any] = [FactStatus.ACTIVE.value]
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if category:
            clauses.append("category = ?")
            params.append(category)
        params.append(limit)
        query = f"""
            SELECT * FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def search_memories(
        self,
        query_embedding: Sequence[float],
        top_k: int = 8,
        status: FactStatus = FactStatus.ACTIVE,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[Fact, float]]:
        """Vector similarity search over stored embeddings."""
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm

        clauses = ["status = ?", "embedding IS NOT NULL", "confidence >= ?"]
        params: list[Any] = [status.value, min_confidence]
        if category:
            clauses.append("category = ?")
            params.append(category)

        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: list[tuple[Fact, float]] = []
        for row in rows:
            blob = row["embedding"]
            if not blob:
                continue
            vec = _deserialize_embedding(blob)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            vec = vec / vec_norm
            similarity = float(np.dot(query_vec, vec))
            scored.append((self._row_to_fact(row), similarity))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def keyword_search_memories(
        self,
        keywords: list[str],
        top_k: int = 8,
        status: FactStatus = FactStatus.ACTIVE,
    ) -> list[Fact]:
        if not keywords:
            return []
        clauses = []
        params: list[Any] = [status.value]
        for kw in keywords:
            clauses.append("(text LIKE ? OR predicate LIKE ? OR object LIKE ?)")
            pattern = f"%{kw}%"
            params.extend([pattern, pattern, pattern])
        sql = f"""
            SELECT * FROM memories
            WHERE status = ? AND ({' OR '.join(clauses)})
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(top_k)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def count_facts(self, status: FactStatus | None = None) -> int:
        with self._connect() as conn:
            if status is None:
                row = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories WHERE status = ?",
                    (status.value,),
                ).fetchone()
        return int(row["c"]) if row else 0

    def reset_all(self) -> None:
        """Dangerous: wipe all data."""
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_turns")
            conn.execute("DELETE FROM memories")

    @staticmethod
    def validate_embedding_dimension(vector: Sequence[float]) -> None:
        if len(vector) != EMBEDDING_DIMENSION:
            logger.warning(
                "Embedding dimension %d differs from configured %d",
                len(vector),
                EMBEDDING_DIMENSION,
            )
