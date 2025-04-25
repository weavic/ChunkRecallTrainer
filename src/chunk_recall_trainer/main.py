from datetime import date
import streamlit as st
from chunk import ChunkRepo, sm2_update, Chunk
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Chunk Recall Trainer", page_icon="📚", layout="centered")
st.sidebar.header("📂 Data Management")
if st.session_state.get("just_added"):
    st.toast("✅ Chunk saved!")
    st.session_state.just_added = False
if st.session_state.get("just_reset"):
    st.toast("🗑️ Database cleared!", icon="🔥")
    st.session_state.just_reset = False

repo = ChunkRepo()

# ──────────────────────────────────────────────────────────────────────────────
# CSV Import
# ──────────────────────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)

    csv_string = df.to_csv(index=False)
    csv_buffer = StringIO(csv_string)

    count = repo.save_from_csv(csv_buffer)
    st.sidebar.success(f"✅ Imported {count} chunks!")

# ──────────────────────────────────────────────────────────────────────────────
# CSV Export
# ──────────────────────────────────────────────────────────────────────────────
if st.sidebar.button("Download all Chunks"):
    csv = repo.export_all()
    st.sidebar.download_button(
        label="Save CSV",
        data=csv,
        file_name="chunks.csv",
        mime="text/csv",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Single Chunk Add Form
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar.form("add_chunk"):
    st.markdown("### ➕ Add a single chunk")
    jp = st.text_area("JP Prompt", height=80)
    en = st.text_area("EN Answer", height=80)
    submitted = st.form_submit_button("Add")
    if submitted and jp and en:
        repo.add(Chunk(id=None, jp_prompt=jp, en_answer=en))
        st.session_state.just_added = True  # Flag to indicate a new chunk was added
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Dangerous chunk deletion
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚠️ Dangerous chunk deletion")
with st.sidebar.form("reset_db"):
    confirm = st.checkbox("Yes, delete **all** chunks")
    reset = st.form_submit_button("Reset database", type="primary")
    if reset and confirm:
        repo.reset()
        st.session_state.just_reset = True

        # clear the session state
        st.session_state.queue = []
        st.session_state.queue_date = date.today()
        st.session_state.queue_total = 0
        st.rerun()

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
