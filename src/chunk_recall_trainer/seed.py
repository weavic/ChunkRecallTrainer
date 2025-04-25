# seed.py  — data seeding script for *Chunk Recall Trainer*
from chunk import ChunkRepo, Chunk

examples = [
    ("おはようございます。調子はどうですか？", "Good morning. How are you?"),
    ("こんばんは。調子はどうですか？", "Good evening. How are you?"),
    ("えーと、なんて言えばいいかな…", "Let me think for a sec."),
    (
        "日々のトレーニングの積み重ねが大事だよね",
        "It's the daily training that matters.",
    ),
    ("情報をキャッチアップは頻繁に", "I've got a lot to catch up on information-wise."),
]

repo = ChunkRepo()
for jp, en in examples:
    repo.add(Chunk(id=None, jp_prompt=jp, en_answer=en))
    print(
        f"seeded: {jp} -> {en}, id={repo.conn.execute('SELECT last_insert_rowid()').fetchone()[0]}"
    )
