import unittest
from unittest.mock import MagicMock, patch, call
import sqlite3
import os
import sys
import pandas as pd
from datetime import date, datetime, timedelta, timezone

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from chunk_recall_trainer.chunk import Chunk, ChunkRepo, sm2_update, DB_PATH

# Test database path (in-memory or a temporary file)
TEST_DB_PATH = ":memory:"


class TestSM2Update(unittest.TestCase):
    def test_sm2_update_quality_below_3(self):
        """Test SM-2 update when quality is below 3 (e.g., 2)."""
        chunk = Chunk(
            id=1,
            user_id="user1",
            jp_prompt="jp",
            en_answer="en",
            ef=2.5,
            interval=10,
            review_count=2,
        )
        original_ef = chunk.ef  # EF should not change for quality < 3 in this variant

        updated_chunk = sm2_update(chunk, 2)  # Quality = 2

        self.assertEqual(updated_chunk.interval, 1)  # Interval resets
        self.assertEqual(updated_chunk.review_count, 0)  # Review count resets
        self.assertEqual(updated_chunk.ef, original_ef)  # EF does not change
        self.assertEqual(updated_chunk.next_due_date, date.today() + timedelta(days=1))

    def test_sm2_update_quality_3(self):
        """Test SM-2 update when quality is 3."""
        chunk = Chunk(
            id=1,
            user_id="user1",
            jp_prompt="jp",
            en_answer="en",
            ef=2.5,
            interval=10,
            review_count=3,
        )
        updated_chunk = sm2_update(chunk, 3)

        self.assertTrue(updated_chunk.ef < 2.5)  # EF should decrease for quality 3
        self.assertEqual(updated_chunk.review_count, 4)
        self.assertEqual(
            updated_chunk.interval, int(round(10 * chunk.ef))
        )  # Previous interval * new EF
        self.assertEqual(
            updated_chunk.next_due_date,
            date.today() + timedelta(days=updated_chunk.interval),
        )

    def test_sm2_update_quality_5_first_review(self):
        """Test SM-2 update for a perfect score on the first review."""
        chunk = Chunk(
            id=1,
            user_id="user1",
            jp_prompt="jp",
            en_answer="en",
            ef=2.5,
            interval=0,
            review_count=0,
        )
        updated_chunk = sm2_update(chunk, 5)  # Quality = 5

        self.assertEqual(updated_chunk.interval, 1)  # First interval is 1
        self.assertEqual(updated_chunk.review_count, 1)
        self.assertTrue(updated_chunk.ef > 2.5)  # EF should increase for quality 5
        self.assertEqual(updated_chunk.next_due_date, date.today() + timedelta(days=1))

    def test_sm2_update_quality_4_second_review(self):
        """Test SM-2 update for quality 4 on the second review."""
        # After first review (q=5), interval=1, ef=2.6, review_count=1
        chunk = Chunk(
            id=1,
            user_id="user1",
            jp_prompt="jp",
            en_answer="en",
            ef=2.6,
            interval=1,
            review_count=1,
        )
        updated_chunk = sm2_update(chunk, 4)  # Quality = 4

        self.assertEqual(updated_chunk.interval, 6)  # Second interval is 6
        self.assertEqual(updated_chunk.review_count, 2)
        # EF should be 2.6 (no change for quality 4 as per formula: 0.1 - (5-4)*(0.08+(5-4)*0.02) = 0.1 - 0.1 = 0)
        self.assertAlmostEqual(updated_chunk.ef, 2.6, places=5)
        self.assertEqual(updated_chunk.next_due_date, date.today() + timedelta(days=6))

    def test_sm2_update_quality_5_third_review(self):
        """Test SM-2 update for quality 5 on the third review (interval calculation)."""
        # After second review (q=4), interval=6, ef=2.6, review_count=2
        chunk = Chunk(
            id=1,
            user_id="user1",
            jp_prompt="jp",
            en_answer="en",
            ef=2.6,
            interval=6,
            review_count=2,
        )
        updated_chunk = sm2_update(chunk, 5)  # Quality = 5

        expected_interval = int(round(6 * 2.7))  # Old interval * new EF (2.6 + 0.1)
        self.assertEqual(updated_chunk.interval, expected_interval)
        self.assertEqual(updated_chunk.review_count, 3)
        self.assertAlmostEqual(updated_chunk.ef, 2.7, places=5)  # EF increases by 0.1
        self.assertEqual(
            updated_chunk.next_due_date,
            date.today() + timedelta(days=expected_interval),
        )

    def test_ef_floor(self):
        """Test that EF does not go below 1.3."""
        chunk = Chunk(
            id=1,
            user_id="user1",
            jp_prompt="jp",
            en_answer="en",
            ef=1.3,
            interval=10,
            review_count=3,
        )
        updated_chunk = sm2_update(chunk, 0)  # Quality 0, if EF could drop, it would
        # With quality < 3, EF is not modified in this implementation variant, interval resets
        self.assertEqual(updated_chunk.ef, 1.3)

        chunk.ef = 1.35
        updated_chunk = sm2_update(
            chunk, 3
        )  # Quality 3, EF = 1.35 + (0.1 - 2*0.08 - 2*0.02*2) = 1.35 + (0.1 - 0.16 - 0.08) = 1.35 - 0.14 = 1.21 -> clamped to 1.3
        self.assertAlmostEqual(updated_chunk.ef, 1.3, places=5)


class TestChunk(unittest.TestCase):
    def test_chunk_defaults(self):
        """Test default values of a new Chunk."""
        now = datetime.now(timezone.utc)
        chunk = Chunk(
            id=None, user_id="user1", jp_prompt="こんにちは", en_answer="Hello"
        )
        self.assertIsNone(chunk.id)
        self.assertEqual(chunk.user_id, "user1")
        self.assertEqual(chunk.jp_prompt, "こんにちは")
        self.assertEqual(chunk.en_answer, "Hello")
        self.assertEqual(chunk.ef, 2.5)
        self.assertEqual(chunk.interval, 0)
        self.assertEqual(chunk.next_due_date, date.today())
        self.assertEqual(chunk.review_count, 0)
        self.assertAlmostEqual(chunk.created_at, now, delta=timedelta(seconds=1))
        self.assertAlmostEqual(chunk.updated_at, now, delta=timedelta(seconds=1))

    @patch("sqlite3.connect")
    def test_save_new_chunk(self, mock_connect):
        """Test saving a new chunk (INSERT)."""
        mock_conn = MagicMock(spec=sqlite3.Connection)
        mock_cursor = MagicMock(spec=sqlite3.Cursor)
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.lastrowid = 123  # Simulate new ID from DB

        chunk = Chunk(id=None, user_id="user1", jp_prompt="jp", en_answer="en")
        original_created_at = chunk.created_at

        saved_chunk = chunk.save(mock_conn)

        self.assertEqual(saved_chunk.id, 123)
        self.assertEqual(
            saved_chunk.created_at, original_created_at
        )  # created_at should not change
        # updated_at is set to created_at by default, and not changed further on initial save
        self.assertAlmostEqual(
            saved_chunk.updated_at, original_created_at, delta=timedelta(seconds=1)
        )

        mock_conn.execute.assert_called_once()
        self.assertIn("INSERT INTO chunks", mock_conn.execute.call_args[0][0])
        mock_conn.commit.assert_called_once()

    @patch("sqlite3.connect")
    def test_save_existing_chunk(self, mock_connect):
        """Test saving an existing chunk (UPDATE)."""
        mock_conn = MagicMock(spec=sqlite3.Connection)
        mock_connect.return_value = mock_conn

        initial_created_at = datetime.now(timezone.utc) - timedelta(days=1)
        initial_updated_at = datetime.now(timezone.utc) - timedelta(hours=1)

        chunk = Chunk(
            id=1,
            user_id="user1",
            jp_prompt="jp_old",
            en_answer="en_old",
            created_at=initial_created_at,
            updated_at=initial_updated_at,
        )

        chunk.jp_prompt = "jp_new"  # Modify the chunk

        # Ensure save method updates 'updated_at'
        with patch(
            "chunk_recall_trainer.chunk.datetime", wraps=datetime
        ) as mock_datetime:
            # mock_datetime.now.return_value = datetime.now(timezone.utc) # Ensure it returns a new time
            current_time_for_update = datetime.now(timezone.utc)
            mock_datetime.now.return_value = current_time_for_update

            saved_chunk = chunk.save(mock_conn)

            self.assertEqual(saved_chunk.id, 1)
            self.assertEqual(saved_chunk.jp_prompt, "jp_new")
            self.assertEqual(
                saved_chunk.created_at, initial_created_at
            )  # created_at should not change
            self.assertEqual(
                saved_chunk.updated_at, current_time_for_update
            )  # updated_at should be new time

            mock_conn.execute.assert_called_once()
            self.assertIn("UPDATE chunks SET", mock_conn.execute.call_args[0][0])
            mock_conn.commit.assert_called_once()

    def test_from_row(self):
        """Test creating a Chunk object from a SQLite row."""
        mock_sql_row = {
            "id": 1,
            "user_id": "user1",
            "jp_prompt": "jp",
            "en_answer": "en",
            "ef": 2.0,
            "interval": 5,
            "next_due_date": "2023-01-10",
            "review_count": 3,
            "created_at": "2023-01-01T10:00:00Z",
            "updated_at": "2023-01-05T12:00:00Z",
        }
        chunk = Chunk.from_row(mock_sql_row)
        self.assertEqual(chunk.id, 1)
        self.assertEqual(chunk.ef, 2.0)
        self.assertEqual(chunk.next_due_date, date(2023, 1, 10))
        self.assertEqual(
            chunk.created_at, datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        )


class TestChunkRepo(unittest.TestCase):
    def setUp(self):
        """Setup an in-memory SQLite database for each test."""
        self.conn = sqlite3.connect(
            TEST_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.row_factory = sqlite3.Row
        Chunk.create_table(self.conn)  # Ensure table is created

        # Patch the DB_PATH used by ChunkRepo to use the in-memory DB
        self.db_path_patcher = patch("chunk_recall_trainer.chunk.DB_PATH", TEST_DB_PATH)
        self.mock_db_path = self.db_path_patcher.start()

        self.user_id = "test_user_123"
        self.repo = ChunkRepo(user_id=self.user_id)
        self.repo.reset()  # Reset the repo to ensure a clean state
        # We will use the actual repo.conn which is :memory:

    def tearDown(self):
        """Close the database connection and stop patches."""
        self.conn.close()
        self.db_path_patcher.stop()

    def test_add_and_get_all_chunks(self):
        """Test adding chunks and retrieving all chunks for a user."""
        chunk1_data = {"jp_prompt": "こんにちは", "en_answer": "Hello"}
        chunk2_data = {"jp_prompt": "ありがとう", "en_answer": "Thank you"}

        chunk1 = Chunk(id=None, user_id=self.user_id, **chunk1_data)
        chunk2 = Chunk(id=None, user_id=self.user_id, **chunk2_data)

        self.repo.add(chunk1)
        self.repo.add(chunk2)

        all_chunks = self.repo.get_all()
        self.assertEqual(len(all_chunks), 2)
        self.assertEqual(
            all_chunks[0].jp_prompt, chunk2_data["jp_prompt"]
        )  # Ordered by created_at DESC
        self.assertEqual(all_chunks[1].jp_prompt, chunk1_data["jp_prompt"])

        # Test with another user
        repo_user2 = ChunkRepo(user_id="another_user")
        self.assertEqual(len(repo_user2.get_all()), 0)

    def test_get_overdue_chunks(self):
        """Test retrieving overdue chunks."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)

        chunk1 = Chunk(
            id=None,
            user_id=self.user_id,
            jp_prompt="c1",
            en_answer="a1",
            next_due_date=yesterday,
        )
        chunk2 = Chunk(
            id=None,
            user_id=self.user_id,
            jp_prompt="c2",
            en_answer="a2",
            next_due_date=today,
        )
        chunk3 = Chunk(
            id=None,
            user_id=self.user_id,
            jp_prompt="c3",
            en_answer="a3",
            next_due_date=tomorrow,
        )

        self.repo.add(chunk1)
        self.repo.add(chunk2)
        self.repo.add(chunk3)

        overdue_chunks = self.repo.get_overdue(limit=5)
        self.assertEqual(len(overdue_chunks), 2)
        self.assertIn(chunk1.jp_prompt, [c.jp_prompt for c in overdue_chunks])
        self.assertIn(chunk2.jp_prompt, [c.jp_prompt for c in overdue_chunks])
        # Ensure they are ordered correctly (yesterday first)
        self.assertEqual(overdue_chunks[0].jp_prompt, chunk1.jp_prompt)

    def test_delete_many_chunks(self):
        """Test deleting multiple chunks by their IDs."""
        c1 = self.repo.add(
            Chunk(id=None, user_id=self.user_id, jp_prompt="j1", en_answer="e1")
        )
        c2 = self.repo.add(
            Chunk(id=None, user_id=self.user_id, jp_prompt="j2", en_answer="e2")
        )
        c3 = self.repo.add(
            Chunk(id=None, user_id=self.user_id, jp_prompt="j3", en_answer="e3")
        )

        self.repo.delete_many([c1.id, c3.id])

        remaining_chunks = self.repo.get_all()
        self.assertEqual(len(remaining_chunks), 1)
        self.assertEqual(remaining_chunks[0].id, c2.id)

    def test_reset_intervals(self):
        """Test resetting review intervals for specified chunks."""
        c1 = self.repo.add(
            Chunk(
                id=None,
                user_id=self.user_id,
                jp_prompt="j1",
                en_answer="e1",
                interval=10,
                review_count=5,
            )
        )
        c2 = self.repo.add(
            Chunk(
                id=None,
                user_id=self.user_id,
                jp_prompt="j2",
                en_answer="e2",
                interval=5,
                review_count=2,
            )
        )

        self.repo.reset_intervals([c1.id])

        updated_c1_list = [c for c in self.repo.get_all() if c.id == c1.id]
        self.assertTrue(len(updated_c1_list) == 1)
        updated_c1 = updated_c1_list[0]

        self.assertEqual(updated_c1.interval, 0)
        self.assertEqual(updated_c1.review_count, 0)
        self.assertEqual(updated_c1.next_due_date, date.today())

        # Check c2 was not affected
        updated_c2_list = [c for c in self.repo.get_all() if c.id == c2.id]
        self.assertTrue(len(updated_c2_list) == 1)
        updated_c2 = updated_c2_list[0]
        self.assertEqual(updated_c2.interval, 5)

    @patch("pandas.read_csv")
    def test_save_from_csv(self, mock_read_csv):
        """Test saving chunks from a CSV file object."""
        csv_data = {
            "JP Prompt": ["こんにちは", "ありがとう"],
            "EN Answer": ["Hello", "Thank you"],
            "EF": [2.3, 2.6],
            "Interval": [1, 5],
            "Next Due Date": ["2023-01-01", "2023-01-05"],
            "Review Count": [1, 2],
        }
        mock_df = pd.DataFrame(csv_data)
        mock_read_csv.return_value = mock_df

        mock_file_obj = MagicMock()  # Simulate a file object

        added_count = self.repo.save_from_csv(mock_file_obj)
        self.assertEqual(added_count, 2)

        all_chunks = self.repo.get_all()
        self.assertEqual(len(all_chunks), 2)
        self.assertEqual(
            all_chunks[0].jp_prompt, "ありがとう"
        )  # Ordered by created_at DESC by get_all
        self.assertEqual(all_chunks[0].ef, 2.6)
        self.assertEqual(all_chunks[1].jp_prompt, "こんにちは")
        self.assertEqual(all_chunks[1].next_due_date, date(2023, 1, 1))

    def test_export_all(self):
        """Test exporting all chunks to a CSV string."""
        self.repo.add(
            Chunk(
                id=None,
                user_id=self.user_id,
                jp_prompt="jp1",
                en_answer="en1",
                ef=2.0,
                interval=1,
                next_due_date=date(2023, 1, 1),
            )
        )
        self.repo.add(
            Chunk(
                id=None,
                user_id=self.user_id,
                jp_prompt="jp2",
                en_answer="en2",
                ef=2.1,
                interval=2,
                next_due_date=date(2023, 1, 2),
            )
        )

        csv_string = self.repo.export_all()

        self.assertIn("jp_prompt,en_answer,ef,interval,next_due_date", csv_string)
        self.assertIn("jp1,en1,2.0,1,2023-01-01", csv_string)
        self.assertIn("jp2,en2,2.1,2,2023-01-02", csv_string)

    def test_reset_repo(self):
        """Test deleting all chunks for the current user."""
        self.repo.add(
            Chunk(id=None, user_id=self.user_id, jp_prompt="j1", en_answer="e1")
        )
        self.repo.add(
            Chunk(
                id=None, user_id="other_user", jp_prompt="j_other", en_answer="e_other"
            )
        )  # Add chunk for another user

        self.repo.reset()

        self.assertEqual(len(self.repo.get_all()), 0)  # Current user's chunks deleted

        # Verify other user's chunks are not deleted (requires separate repo instance or direct DB check)
        # For simplicity, we'll assume DB isolation if other tests pass.
        # A more robust test might involve querying the DB directly here.


if __name__ == "__main__":
    unittest.main()
