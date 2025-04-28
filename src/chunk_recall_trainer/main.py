from datetime import date
import streamlit as st
from chunk import ChunkRepo, sm2_update, Chunk
import pandas as pd
from io import StringIO
import uuid
from exercises import ExerciseGenerator

st.set_page_config(page_title="Chunk Recall Trainer", page_icon="📚", layout="centered")
st.sidebar.header("📂 Data Management")
if st.session_state.get("just_added"):
    st.toast("✅ Chunk saved!")
    st.session_state.just_added = False
if st.session_state.get("just_reset"):
    st.toast("🗑️ Database cleared!", icon="🔥")
    st.session_state.just_reset = False

# ──────────────────────────────────────────────────────────────────────────────
# UUID Generation as a User ID
# ──────────────────────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    query_params = st.query_params
    uid = query_params.get("uid")
    if "uid" in query_params:
        st.session_state.user_id = uid if isinstance(uid, str) else uid[0]
    else:
        st.session_state.user_id = str(uuid.uuid4())
        st.query_params["uid"] = st.session_state.user_id

    user_id = st.session_state.user_id
    st.sidebar.markdown(f"**User ID:** {user_id}")


repo = ChunkRepo(user_id=st.session_state.user_id)

# ──────────────────────────────────────────────────────────────────────────────
# API Key Import
# ──────────────────────────────────────────────────────────────────────────────
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    st.session_state["api_key"] = api_key

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
csv = repo.export_all()
st.sidebar.download_button(
    label="💾 Export CSV",
    data=csv,
    file_name="chunks_export.csv",
    mime="text/csv",
    help="Download your current chunk deck as CSV",
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

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
    st.stop()

gen = ExerciseGenerator(api_key=st.session_state["api_key"])

for ch in repo.get_overdue():
    with st.expander(ch.jp_prompt):
        if st.button("📝 Practice", key=f"q_{ch.id}"):
            ex = gen.create_exercise(ch.jp_prompt, ch.en_answer)
            st.session_state[f"ex_{ch.id}"] = ex

        ex = st.session_state.get(f"ex_{ch.id}")
        if ex:
            st.markdown(f"**Prompt:** {ex.question}")
            ans = st.text_area("Your answer", key=f"ans_{ch.id}")
            if st.button("✅ Check", key=f"chk_{ch.id}"):
                fb = gen.review_answer(ans, ex, ch.en_answer)
                st.markdown(fb.comment_md)
