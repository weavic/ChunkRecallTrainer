from datetime import date
import streamlit as st
from chunk import ChunkRepo, sm2_update

st.set_page_config(page_title="Chunk Recall Trainer", page_icon="📚", layout="centered")

repo = ChunkRepo()

# ──────────────────────────────────────────────────────────────────────────────
# 1. Session‑scoped queue (max 5 overdue chunks per day)
# ──────────────────────────────────────────────────────────────────────────────
if "queue_date" not in st.session_state or st.session_state.queue_date != date.today():
    st.session_state.queue = repo.get_overdue(limit=5)
    st.session_state.queue_date = date.today()
    st.session_state.queue_total = len(st.session_state.queue)

queue = st.session_state.queue
remaining = len(queue)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Exit early when queue is empty
# ──────────────────────────────────────────────────────────────────────────────
if remaining == 0:
    st.success("🎉 All caught up for today! See you tomorrow.")
    st.stop()

# Always work on the first chunk in the queue
chunk = queue[0]

# ──────────────────────────────────────────────────────────────────────────────
# 3. UI – prompt & reveal logic
# ──────────────────────────────────────────────────────────────────────────────
st.title("Chunk Recall Trainer")
total = st.session_state.queue_total
current = total - remaining + 1
st.subheader(f"Progress: {current} / {total}")

st.markdown(f"**JP Prompt:** {chunk.jp_prompt}")

if "revealed" not in st.session_state:
    st.session_state.revealed = False

if not st.session_state.revealed:
    if st.button("Reveal", key=f"reveal_{chunk.id}"):
        st.session_state.revealed = True
        st.rerun()
    st.stop()

st.markdown(f"✅ **Answer:** {chunk.en_answer}")

cols = st.columns(3)

# Mapping: label -> SM‑2 quality score
for col, (label, score) in zip(cols, {"Hard": 2, "Good": 4, "Easy": 5}.items()):
    key = f"{chunk.id}_{label}"

    if col.button(label, key=key):
        # Update scheduling in DB
        repo.update(sm2_update(chunk, score))
        # Remove this chunk from today's queue
        queue.pop(0)
        # Reset reveal state and rerun
        st.session_state.revealed = False
        st.rerun()
