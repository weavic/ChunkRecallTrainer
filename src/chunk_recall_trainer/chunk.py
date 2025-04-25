# chunk.py – data model + SM‑2 scheduler for *Chunk Recall Trainer*

"""A minimal, self‑contained data layer that
• stores chunks in SQLite
• fetches ≤5 overdue chunks for each session
• updates scheduling fields with a lightweight SM‑2 algorithm

Usage example (inside Streamlit):
>>> repo = ChunkRepo()
>>> overdue = repo.get_overdue()
>>> st.write(overdue[0].jp_prompt)
>>> updated = sm2_update(overdue[0], quality=4)
>>> repo.update(updated)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
import sqlite3
from typing import List, Optional, Union

DB_PATH = "chunks.db"


@dataclass
class Chunk:
    id: Optional[int]
    jp_prompt: str
    en_answer: str
    ef: float = 2.5  # Easiness Factor
    interval: int = 0  # days until next review
    next_due_date: date = date.today()
    review_count: int = 0
    created_at: datetime = datetime.now(timezone.utc)
    updated_at: datetime = datetime.now(timezone.utc)

    # ──────────────────────────────── DB helpers ────────────────────────────────
    @staticmethod
    def create_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                jp_prompt     TEXT    NOT NULL,
                en_answer     TEXT    NOT NULL,
                ef            REAL    NOT NULL DEFAULT 2.5,
                interval      INTEGER NOT NULL DEFAULT 0,
                next_due_date DATE    NOT NULL,
                review_count  INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT    NOT NULL,
                updated_at    TEXT    NOT NULL
            );
            """
        )
        conn.commit()

    def save(self, conn: sqlite3.Connection) -> "Chunk":
        payload = {
            **asdict(self),
            "next_due_date": self.next_due_date.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.id is None:
            cur = conn.execute(
                """
                INSERT INTO chunks (jp_prompt, en_answer, ef, interval, next_due_date,
                                    review_count, created_at, updated_at)
                VALUES (:jp_prompt, :en_answer, :ef, :interval, :next_due_date,
                        :review_count, :created_at, :updated_at);
                """,
                payload,
            )
            self.id = cur.lastrowid
        else:
            self.updated_at = datetime.now(timezone.utc)
            payload.update({"id": self.id, "updated_at": self.updated_at.isoformat()})
            conn.execute(
                """
                UPDATE chunks SET
                    jp_prompt=:jp_prompt,
                    en_answer=:en_answer,
                    ef=:ef,
                    interval=:interval,
                    next_due_date=:next_due_date,
                    review_count=:review_count,
                    updated_at=:updated_at
                WHERE id=:id;
                """,
                payload,
            )
        conn.commit()
        return self

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Chunk":
        def _parse_date(val: Union[str, date, datetime]) -> date:
            if isinstance(val, date) and not isinstance(val, datetime):
                return val
            if isinstance(val, datetime):
                return val.date()
            # assume str
            return datetime.fromisoformat(val).date()

        def _parse_dt(val: Union[str, datetime]) -> datetime:
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return Chunk(
            id=row["id"],
            jp_prompt=row["jp_prompt"],
            en_answer=row["en_answer"],
            ef=row["ef"],
            interval=row["interval"],
            next_due_date=_parse_date(row["next_due_date"]),
            review_count=row["review_count"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )


# ────────────────────────────── Repository layer ──────────────────────────────
class ChunkRepo:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        Chunk.create_table(self.conn)

    def add(self, chunk: Chunk) -> Chunk:
        return chunk.save(self.conn)

    def update(self, chunk: Chunk) -> None:
        chunk.save(self.conn)

    def get_overdue(self, limit: int = 5) -> List[Chunk]:
        rows = self.conn.execute(
            """
            SELECT * FROM chunks
            WHERE DATE(next_due_date) <= DATE('now')
            ORDER BY next_due_date ASC, review_count ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [Chunk.from_row(r) for r in rows]


# ────────────────────────────── SM‑2 Scheduling ───────────────────────────────
def sm2_update(chunk: Chunk, quality: int) -> Chunk:
    """Update a chunk with a recall *quality* score 0–5.
    Ref: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2
    """
    assert 0 <= quality <= 5, "quality must be 0‥5"

    if quality < 3:
        chunk.interval = 1
    else:
        if chunk.review_count == 0:
            chunk.interval = 1
        elif chunk.review_count == 1:
            chunk.interval = 6
        else:
            chunk.interval = int(round(chunk.interval * chunk.ef))
        # Easiness Factor update
        chunk.ef = max(
            1.3, chunk.ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        )

    chunk.review_count += 1
    chunk.next_due_date = date.today() + timedelta(days=chunk.interval)
    return chunk
