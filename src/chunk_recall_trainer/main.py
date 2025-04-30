from datetime import date
import streamlit as st
import os
from chunk import ChunkRepo, sm2_update, Chunk
import pandas as pd
from io import StringIO
import uuid
from exercises import ExerciseGenerator

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit app, page config, and repo
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chunk Recall Trainer", page_icon="📚", layout="centered")

API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
# Set the API key in session state if it exists
if API_KEY and "api_key" not in st.session_state:
    st.session_state["api_key"] = API_KEY

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

if st.session_state.get("just_added"):
    st.toast("✅ Chunk saved!")
    st.session_state.just_added = False
if st.session_state.get("just_reset"):
    st.toast("🗑️ Database cleared!", icon="🔥")
    st.session_state.just_reset = False

# ──────────────────────────────────────────────────────────────────────────────
# UI. Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data")
    # ──────────────────────────────────────────────────────────────────────────────
    # CSV Import
    # ──────────────────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        csv_string = df.to_csv(index=False)
        csv_buffer = StringIO(csv_string)

        count = repo.save_from_csv(csv_buffer)
        st.sidebar.success(f"✅ Imported {count} chunks!")
    st.markdown("---")
    # ──────────────────────────────────────────────────────────────────────────────
    # CSV Export
    # ──────────────────────────────────────────────────────────────────────────────
    csv = repo.export_all()
    st.download_button(
        label="💾 Export CSV",
        data=csv,
        file_name="chunks_export.csv",
        mime="text/csv",
        help="Download your current chunk deck as CSV",
    )
    # ──────────────────────────────────────────────────────────────────────────────
    # Single Chunk Add Form
    # ──────────────────────────────────────────────────────────────────────────────
    with st.form("add_chunk"):
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
    st.markdown("### ⚠️ Dangerous chunk deletion")
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
    # Settings
    # ──────────────────────────────────────────────────────────────────────────────
    st.header("🛠 Settings")
    # ──────────────────────────────────────────────────────────────────────────────
    # API Key Input: if the API key is set in the environment, disable the input
    # ──────────────────────────────────────────────────────────────────────────────
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Enter your OpenAI API key",
        disabled=bool(API_KEY),
    )
    if api_key and api_key != st.session_state.get("api_key"):
        st.session_state["api_key"] = api_key

# ──────────────────────────────────────────────────────────────────────────────
# UI – prompt & reveal logic
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Session‑scoped queue (max 5 overdue chunks per day)
# ──────────────────────────────────────────────────────────────────────────────
if "queue_date" not in st.session_state or st.session_state.queue_date != date.today():
    st.session_state.queue = repo.get_overdue(limit=5)
    st.session_state.queue_date = date.today()
    st.session_state.queue_total = len(st.session_state.queue)

queue = st.session_state.queue
remaining = len(queue)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("Chunk Recall Trainer")
with col2:
    total = st.session_state.queue_total
    current = total - remaining + 1
    st.metric("Progress", f"{current}/{total}")

# Tabbed interface for practice and management
tab_practice, tab_manage = st.tabs(["💬 Practice", "📋 Manage Chunks"])

with tab_practice:
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # ──────────────────────────────────────────────────────────────────────────────
    # Exit early when queue is empty
    # ──────────────────────────────────────────────────────────────────────────────
    if remaining == 0:
        st.success("🎉 All caught up for today! See you tomorrow.")
        st.stop()

    # Always work on the first chunk in the queue
    chunk = queue[0]

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

with tab_manage:
    st.subheader("📋 Chunk List")

    df = pd.DataFrame([c.__dict__ for c in repo.get_all()])
    edited = st.data_editor(
        df[["id", "jp_prompt", "en_answer", "ef", "interval"]],
        num_rows="dynamic",
        use_container_width=True,
    )

    save, col_del, col_reset = st.columns([1, 1, 1])
    with save:
        if st.button("💾 Save edits"):
            repo.bulk_update(edited)
            st.success("Saved!")

    with col_del:
        sel_ids = st.multiselect("Select rows", edited["id"])
        if st.button("🗑 Delete", disabled=not sel_ids):
            repo.delete_many(sel_ids)
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset intervals", disabled=not sel_ids):
            repo.reset_intervals(sel_ids)
            st.rerun()

    df["next_due_date"] = pd.to_datetime(df["next_due_date"]).dt.date
    due_today = (df["next_due_date"] <= date.today()).sum()

    st.divider()
    st.subheader("📊 Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(df))
    col2.metric("Avg EF", round(df["ef"].mean(), 2))
    col3.metric("Due today", due_today)
