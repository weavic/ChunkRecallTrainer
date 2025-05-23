# tests/test_sm2.py
from datetime import date
from chunk_recall_trainer.chunk import Chunk, sm2_update


def test_sm2_quality_updates_interval():
    c = Chunk(id=None, user_id="test", jp_prompt="JP", en_answer="EN")
    out = sm2_update(c, quality=4)  # Good
    assert out.interval in (1, 6)  # first review paths
    assert out.next_due_date >= date.today()
