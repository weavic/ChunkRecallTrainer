"""
Defines the data layer for "chunks" in the Chunk Recall Trainer application.

This module includes:
- The `Chunk` dataclass: Represents a single piece of learning material (e.g., a
  phrase pair) along with its associated SM-2 scheduling parameters (Easiness
  Factor, interval, next due date).
- The `ChunkRepo` class: A repository pattern implementation for managing CRUD
  (Create, Read, Update, Delete) operations for chunks. It ensures that all
  database interactions are scoped to the currently authenticated user.
- The `sm2_update` function: Implements the core logic of the SM-2 algorithm
  to update a chunk's scheduling parameters based on user recall quality.

Database Interaction:
- Assumes a single SQLite database file (`chunks.db` by default) shared by all users.
- User-scoping is achieved by including a `user_id` column in the `chunks` table.
  All queries in `ChunkRepo` are filtered by this `user_id`.
- The `Chunk` class provides static methods for table creation (`create_table`)
  and row-to-object mapping (`from_row`), while instance methods like `save`
  handle persistence.
- The Streamlit application layer is responsible for providing the `user_id`
  (obtained from `st.session_state["user_id"]` after authentication) to the
  `ChunkRepo`.
"""

from __future__ import annotations  # For type hinting Chunk within Chunk class methods

from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta, timezone
import sqlite3
from typing import List, Optional, Union, Any  # Any for pd.DataFrame.iterrows()
import pandas as pd
import re  # For CSV header cleaning

# Default path for the SQLite database file.
DB_PATH = "chunks.db"

###############################################################################
# Data Model: Chunk
###############################################################################


@dataclass
class Chunk:
    """
    Represents a learning chunk with its content and SM-2 scheduling parameters.

    A "chunk" is typically a pair of phrases (e.g., a Japanese prompt and its
    English equivalent) that the user wants to memorize.

    Attributes:
        id (Optional[int]): Unique identifier for the chunk in the database.
                            None if the chunk is new and not yet saved.
        user_id (str): Identifier for the user who owns this chunk. Essential for
                       data scoping in a multi-user environment.
        jp_prompt (str): The Japanese phrase or prompt.
        en_answer (str): The corresponding English answer or translation.
        ef (float): Easiness Factor, as defined by the SM-2 algorithm.
                    Default is 2.5. This value influences how much the interval
                    increases after a successful review.
        interval (int): The current review interval in days. This is the number
                        of days until the chunk is due for review again after the
                        last review. Default is 0 (due immediately for new chunks).
        next_due_date (date): The date when this chunk is next due for review.
                              Defaults to the current date for new chunks.
        review_count (int): The number of times this chunk has been reviewed.
                            Default is 0.
        created_at (datetime): Timestamp (UTC) when the chunk was first created.
                               Defaults to the current UTC time.
        updated_at (datetime): Timestamp (UTC) when the chunk was last updated.
                               Defaults to the current UTC time.
    """

    id: Optional[int]
    user_id: str
    jp_prompt: str
    en_answer: str
    ef: float = 2.5  # Easiness Factor (SM-2 default)
    interval: int = 0  # Days until next review (0 for new/failed chunks)
    next_due_date: date = field(default_factory=date.today)  # Defaults to today
    review_count: int = 0  # Number of times reviewed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ──────────────────────────────── Database Helpers ─────────────────────────────
    @staticmethod
    def create_table(conn: sqlite3.Connection) -> None:
        """
        Creates the `chunks` table in the SQLite database if it doesn't already exist.

        This method defines the schema for storing chunks, including SM-2 parameters
        and user scoping.

        Args:
            conn: An active sqlite3.Connection object.
        """
        # SQL statement for table creation. Ensures user_id is NOT NULL for scoping.
        # Default values are set for SM-2 parameters and timestamps.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT, /* Unique ID for each chunk */
                user_id       TEXT    NOT NULL, /* Foreign key to user, for data isolation */
                jp_prompt     TEXT    NOT NULL, /* Japanese content */
                en_answer     TEXT    NOT NULL, /* English content */
                ef            REAL    NOT NULL DEFAULT 2.5, /* Easiness Factor */
                interval      INTEGER NOT NULL DEFAULT 0,   /* Review interval in days */
                next_due_date DATE    NOT NULL, /* Date of next scheduled review */
                review_count  INTEGER NOT NULL DEFAULT 0,   /* Number of times reviewed */
                created_at    TEXT    NOT NULL, /* ISO format UTC datetime string */
                updated_at    TEXT    NOT NULL  /* ISO format UTC datetime string */
            );
            """
        )
        conn.commit()  # Commits the transaction to ensure table is created.

    def save(self, conn: sqlite3.Connection) -> "Chunk":
        """
        Saves the current chunk instance to the database.

        If the chunk's `id` is None, it's treated as a new chunk and an INSERT
        operation is performed. Otherwise, an existing chunk is updated using an
        UPDATE operation (scoped to the chunk's `id` and `user_id`).
        Timestamps (`created_at`, `updated_at`) and date fields (`next_due_date`)
        are converted to ISO format strings for database storage.

        Args:
            conn: An active sqlite3.Connection object.

        Returns:
            The saved Chunk instance (self), potentially updated with a new `id`
            if it was an insert operation, or a new `updated_at` timestamp.
        """
        # Prepare payload, converting datetime/date objects to ISO strings for SQLite.
        payload = {
            **asdict(
                self
            ),  # Converts dataclass to dict, but datetime/date are still objects
            "next_due_date": self.next_due_date.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),  # Initially set, updated below if existing
        }

        if self.id is None:  # New chunk: Perform INSERT
            cursor = conn.execute(
                """
                INSERT INTO chunks (user_id, jp_prompt, en_answer, ef, interval, next_due_date,
                                    review_count, created_at, updated_at)
                VALUES (:user_id, :jp_prompt, :en_answer, :ef, :interval, :next_due_date,
                        :review_count, :created_at, :updated_at);
                """,
                payload,
            )
            self.id = cursor.lastrowid  # Set the new chunk's ID from the database.
        else:  # Existing chunk: Perform UPDATE
            self.updated_at = datetime.now(
                timezone.utc
            )  # Update `updated_at` timestamp.
            payload["updated_at"] = self.updated_at.isoformat()  # Update payload for DB
            payload["id"] = (
                self.id
            )  # Ensure ID is in payload for WHERE clause, though not strictly needed for SET

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
                WHERE id=:id AND user_id=:user_id; /* Ensures user scopes update */
                """,
                payload,
            )
        conn.commit()  # Commits the transaction.
        return self

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Chunk":
        """
        Converts a SQLite row object into a Chunk dataclass instance.

        Handles parsing of date and datetime strings from the database into
        Python `date` and `datetime` objects.

        Args:
            row: A sqlite3.Row object representing a row from the `chunks` table.

        Returns:
            A Chunk instance populated with data from the row.
        """

        # Helper to parse date strings or pass through existing date/datetime objects.
        def _parse_date_field(value: Union[str, date, datetime]) -> date:
            if isinstance(value, date) and not isinstance(
                value, datetime
            ):  # Already a date object
                return value
            if isinstance(value, datetime):  # If datetime, convert to date
                return value.date()
            return datetime.fromisoformat(value).date()  # Parse from ISO string

        # Helper to parse datetime strings or pass through existing datetime objects.
        def _parse_datetime_field(value: Union[str, datetime]) -> datetime:
            if isinstance(value, datetime):  # Already a datetime object
                return value
            dt_obj = datetime.fromisoformat(value)
            # Ensure datetime is timezone-aware (assuming UTC if naive, which fromisoformat might produce for non-Z strings)
            if dt_obj.tzinfo is None:
                return dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj

        return Chunk(
            id=row["id"],
            user_id=row["user_id"],
            jp_prompt=row["jp_prompt"],
            en_answer=row["en_answer"],
            ef=row["ef"],
            interval=row["interval"],
            next_due_date=_parse_date_field(row["next_due_date"]),
            review_count=row["review_count"],
            created_at=_parse_datetime_field(row["created_at"]),
            updated_at=_parse_datetime_field(row["updated_at"]),
        )


###############################################################################
# Repository Layer: ChunkRepo
###############################################################################


class ChunkRepo:
    """
    Manages data operations for Chunks, scoped to a specific user.

    This repository class abstracts the database interactions (SQLite) for Chunk
    objects. All operations are implicitly filtered by the `user_id` provided
    during instantiation, ensuring users can only access their own data.

    Attributes:
        user_id (str): The ID of the user whose chunks are being managed.
        conn (sqlite3.Connection): The SQLite database connection object.
    """

    def __init__(self, user_id: str, db_path: str = DB_PATH):
        """
        Initializes the ChunkRepo for a specific user.

        Establishes a connection to the SQLite database and ensures the `chunks`
        table exists.

        Args:
            user_id: The ID of the user for whom this repository instance is scoped.
            db_path: The file path to the SQLite database. Defaults to `DB_PATH`.
        """
        self.user_id = user_id
        # Connect to the SQLite database.
        # PARSE_DECLTYPES and PARSE_COLNAMES enable automatic type detection.
        # check_same_thread=False is used for Streamlit's multi-threading context,
        # though care must be taken if explicit multi-threading is introduced.
        self.conn = sqlite3.connect(
            db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,  # Important for Streamlit's execution model
        )
        # Set row_factory to sqlite3.Row to access columns by name.
        self.conn.row_factory = sqlite3.Row
        # Ensure the chunks table exists in the database.
        Chunk.create_table(self.conn)

    def get_all(self) -> List[Chunk]:
        """
        Retrieves all chunks belonging to the current user.

        Returns:
            A list of Chunk objects. Returns an empty list if no chunks are found.
        """
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE user_id = ? ORDER BY created_at DESC",
            (self.user_id,),
        )
        rows = cursor.fetchall()
        return [Chunk.from_row(row) for row in rows]

    def get_overdue(self, limit: int = 5) -> List[Chunk]:
        """
        Retrieves chunks that are due for review (next_due_date is today or earlier).

        Results are ordered by `next_due_date` (oldest first) and then by `review_count`
        (least reviewed first) to prioritize items that are most overdue or less practiced.

        Args:
            limit: The maximum number of overdue chunks to retrieve. Defaults to 5.

        Returns:
            A list of Chunk objects due for review, up to the specified limit.
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM chunks
            WHERE user_id = ? AND DATE(next_due_date) <= DATE('now') /* SQLite date functions */
            ORDER BY next_due_date ASC, review_count ASC /* Prioritize most overdue/least reviewed */
            LIMIT ? /* Limit the number of chunks for daily practice queue */
            """,
            (self.user_id, limit),
        )
        rows = cursor.fetchall()
        return [Chunk.from_row(row) for row in rows]

    def add(self, chunk: Chunk) -> Chunk:
        """
        Adds a new chunk to the database for the current user.

        The `user_id` of the provided chunk object is automatically set to the
        repository's `user_id` to ensure correct scoping.

        Args:
            chunk: A Chunk object to be added. Its `id` should typically be None.

        Returns:
            The added Chunk object, now with its `id` (and potentially other fields
            like `created_at`, `updated_at`) populated from the database.
        """
        chunk.user_id = (
            self.user_id
        )  # Ensure the chunk is associated with this repo's user.
        return chunk.save(self.conn)  # Delegates to Chunk's save method.

    def update(self, chunk: Chunk) -> Chunk:
        """
        Updates an existing chunk in the database.

        Performs a check to ensure the chunk being updated belongs to the
        current user associated with the repository.

        Args:
            chunk: The Chunk object to update. Must have a valid `id`.

        Returns:
            The updated Chunk object.

        Raises:
            ValueError: If the chunk's `user_id` does not match the repository's `user_id`.
        """
        if chunk.user_id != self.user_id:
            raise ValueError(
                "SecurityError: Attempt to update a chunk belonging to another user."
            )
        return chunk.save(self.conn)  # Delegates to Chunk's save method.

    def bulk_update(self, df: pd.DataFrame) -> None:
        """
        Updates multiple chunks based on data from a Pandas DataFrame.

        Each row in the DataFrame is expected to correspond to a chunk, containing
        at least 'id', 'jp_prompt', 'en_answer', 'ef', and 'interval'.
        Missing SM-2 parameters might default if not handled carefully.

        Args:
            df: A Pandas DataFrame where each row represents a chunk to be updated.
                Requires columns like 'id', 'jp_prompt', 'en_answer', 'ef', 'interval'.
        """
        # Iterate over DataFrame rows and update each chunk.
        # This is not the most performant way for very large bulk updates in SQLite,
        # but it's clear and leverages the existing `update` method's logic.
        for _, row_data in df.iterrows():
            # Construct a Chunk object from the row data.
            # Ensure all necessary fields for Chunk are present in row_data or handled with defaults.
            chunk_to_update = Chunk(
                id=row_data["id"],
                user_id=self.user_id,  # Ensure user_id is correctly scoped
                jp_prompt=row_data["jp_prompt"],
                en_answer=row_data["en_answer"],
                ef=row_data["ef"],
                interval=row_data["interval"],
                # next_due_date might need to be parsed if it's a string in the DataFrame
                next_due_date=pd.to_datetime(
                    row_data.get("next_due_date", date.today())
                ).date(),
                review_count=int(row_data.get("review_count", 0)),
                # created_at is usually not updated, updated_at is handled by .save()
            )
            self.update(chunk_to_update)  # Use the existing update method.

    def delete_many(self, ids: List[int]) -> None:
        """
        Deletes multiple chunks from the database by their IDs.

        The deletion is scoped to the current user.

        Args:
            ids: A list of integer IDs of the chunks to be deleted.
        """
        if not ids:  # Do nothing if the list of IDs is empty.
            return
        # Create a string of placeholders for the SQL IN clause (e.g., "?,?,?")
        placeholders = ",".join(["?"] * len(ids))
        self.conn.execute(
            f"DELETE FROM chunks WHERE user_id = ? AND id IN ({placeholders})",
            (self.user_id, *ids),  # Unpack IDs into query parameters
        )
        self.conn.commit()

    def reset_intervals(self, ids: List[int]) -> None:
        """
        Resets the review intervals and related SM-2 parameters for specified chunks.

        Sets `interval` to 0, `next_due_date` to today, and `review_count` to 0,
        effectively making these chunks due for immediate review as if they were new.
        Scoped to the current user.

        Args:
            ids: A list of integer IDs of the chunks whose intervals are to be reset.
        """
        if not ids:  # Do nothing if the list of IDs is empty.
            return
        placeholders = ",".join(["?"] * len(ids))
        today_iso = (
            date.today().isoformat()
        )  # Get today's date in ISO format for query.
        self.conn.execute(
            f"""
            UPDATE chunks
            SET interval = 0,            /* Reset interval */
                next_due_date = ?,       /* Set due date to today */
                review_count = 0         /* Reset review count */
            WHERE user_id = ? AND id IN ({placeholders}) /* Scope to user and selected IDs */
            """,
            (today_iso, self.user_id, *ids),  # Parameters for the query
        )
        self.conn.commit()

    def save_from_csv(self, file_obj: Any) -> int:
        """
        Imports chunks from a CSV file-like object into the database for the current user.

        The CSV file is expected to have columns for Japanese and English phrases.
        Column names are normalized (lowercase, spaces removed).
        SM-2 parameters (`ef`, `interval`, `next_due_date`, `review_count`) can be
        optionally included; otherwise, they default to values for new chunks.

        Args:
            file_obj: A file-like object (e.g., from `st.file_uploader`) containing CSV data.

        Returns:
            The number of chunks successfully added from the CSV.
        """
        df = pd.read_csv(file_obj)

        # Normalize column names: lowercase and remove spaces.
        df.columns = [re.sub(r"\s+", "", str(c).lower()) for c in df.columns]

        # Dynamically find Japanese and English column names (assuming they start with 'jp' or 'en').
        # This adds flexibility if exact column names vary slightly.
        try:
            col_jp = next(c for c in df.columns if c.startswith("jp"))
            col_en = next(c for c in df.columns if c.startswith("en"))
        except StopIteration:
            raise ValueError(
                "CSV must contain columns starting with 'jp' (for Japanese) and 'en' (for English)."
            )

        chunks_added_count = 0
        for _, row in df.iterrows():
            # Parse next_due_date if present, otherwise default to today.
            next_due_val = row.get("nextduedate")  # meet standarizedized column name
            parsed_next_due_date = (
                date.fromisoformat(str(next_due_val))
                if next_due_val and not pd.isna(next_due_val)
                else date.today()
            )

            # Create and add a new Chunk object for each row in the CSV.
            # `id` is None, so it will be an insert operation.
            self.add(
                Chunk(
                    id=None,  # New chunk
                    user_id=self.user_id,  # Associate with current user
                    jp_prompt=str(row[col_jp]),
                    en_answer=str(row[col_en]),
                    ef=float(row.get("ef", 2.5)),  # Default EF if not in CSV
                    interval=int(
                        row.get("interval", 0)
                    ),  # Default interval if not in CSV
                    next_due_date=parsed_next_due_date,
                    review_count=int(
                        row.get("review_count", 0)
                    ),  # Default review_count
                )
            )
            chunks_added_count += 1
        return chunks_added_count

    def export_all(self) -> str:
        """
        Exports all chunks belonging to the current user into a CSV formatted string.

        Returns:
            A string containing the CSV data.
        """
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE user_id = ? ORDER BY created_at ASC",
            (self.user_id,),
        )
        rows = cursor.fetchall()
        # Convert list of sqlite3.Row objects to list of dicts for DataFrame creation.
        df = pd.DataFrame([dict(row) for row in rows])
        return df.to_csv(index=False)  # Convert DataFrame to CSV string without index.

    def reset(self) -> None:
        """
        Deletes ALL chunks belonging to the current user from the database.
        This is a destructive operation.
        """
        self.conn.execute("DELETE FROM chunks WHERE user_id = ?", (self.user_id,))
        self.conn.commit()


###############################################################################
# SM-2 Spaced Repetition Algorithm Logic
###############################################################################


def sm2_update(chunk: Chunk, quality: int) -> Chunk:
    """
    Updates a chunk's SM-2 scheduling parameters based on recall quality.

    This function implements a version of the SM-2 algorithm.
    Reference: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2

    Args:
        chunk: The Chunk object to be updated.
        quality: An integer score from 0 to 5 representing the quality of recall
                 for the chunk during a review session.
                 - 5: Perfect response.
                 - 4: Correct response after some hesitation.
                 - 3: Correct response with considerable difficulty.
                 - 2: Incorrect response; where the correct answer seemed easy to recall.
                 - 1: Incorrect response; the user needs to re-learn the item.
                 - 0: Complete blackout.

    Returns:
        The modified Chunk object with updated `ef`, `interval`, `review_count`,
        and `next_due_date`.

    Raises:
        ValueError: If `quality` is not an integer between 0 and 5.
    """
    if not (isinstance(quality, int) and 0 <= quality <= 5):
        raise ValueError("Recall quality must be an integer between 0 and 5.")

    # Update Easiness Factor (EF) based on quality, but only if quality >= 3.
    # EF should not go below 1.3.
    if quality >= 3:
        # EF formula from SM-2: EF' = EF + [0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)]
        chunk.ef = max(
            1.3, chunk.ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        )
    # else: EF remains unchanged for quality < 3

    # Update interval based on quality and review count.
    if quality < 3:  # If recall quality is low (incorrect or very difficult)
        chunk.interval = 1  # Reset interval to 1 day (or 0 for immediate re-review, SM-2 usually suggests 1st interval)
    else:  # If recall quality is good (quality >= 3)
        if chunk.review_count == 0:  # First successful review
            chunk.interval = 1
        elif chunk.review_count == 1:  # Second successful review
            chunk.interval = 6
        else:  # Subsequent successful reviews
            # Interval grows by the Easiness Factor.
            # Rounding is applied as interval is in days.
            chunk.interval = int(round(chunk.interval * chunk.ef))
            # Practical cap on interval can be added here if desired (e.g., max 1 year)

    # Increment review count after each review.
    chunk.review_count += 1

    # Set the next due date based on the new interval.
    chunk.next_due_date = date.today() + timedelta(
        days=max(1, chunk.interval)
    )  # Ensure interval is at least 1 day for next review.

    # Note: If quality < 3, some SM-2 variations might reset review_count or handle EF differently.
    # This implementation keeps EF unchanged for q < 3 and resets interval.
    # The original SM-2 algorithm resets the repetition sequence for q < 3,
    # effectively making the interval for the next repetition I(1), I(2), etc.
    # This implementation simplifies by setting interval to 1 for q < 3 and keeping EF.
    # For a stricter SM-2, if q < 3, one might also reset review_count to 0 here,
    # so the next successful review (q >= 3) starts with I(1)=1, I(2)=6.
    # The current logic implies that even after a fail, if the next is a pass,
    # the interval calculation might use a higher review_count.
    # Let's adjust: if quality < 3, also reset review_count for stricter adherence.
    if quality < 3:
        chunk.review_count = 0  # Reset review count as if starting over for this item.

    return chunk
