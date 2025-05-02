from datetime import date
import streamlit as st
import os
from chunk import ChunkRepo, Chunk
import pandas as pd
from io import StringIO
import uuid
from openai import OpenAI
from graph import app as graph_app
from streamlit_firebase_auth import FirebaseAuth

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit app, page config, and repo
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chunk Recall Trainer", page_icon="📚", layout="centered")

API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
# Set the API key in session state if it exists
if API_KEY and "api_key" not in st.session_state:
    st.session_state["api_key"] = API_KEY

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS for styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp [data-testid="stDataFrame"]            {border-radius:12px;}
    .stApp div[data-testid="stExpander"] > div    {border-radius:8px; box-shadow:0 2px 4px #0003;}
    .stApp button[kind="secondary"]               {border-radius:8px;}
    .stApp button[kind="secondary"]:hover         {background:#6C63FF22;}
    .stApp [data-baseweb="tab-list"] button[data-selected="true"]   {border-color:#6C63FF;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# login with Firebase
# ──────────────────────────────────────────────────────────────────────────────
allowed_emails = os.getenv("ALLOWED_EMAILS", "")
allowed_list = (
    [e.strip() for e in allowed_emails.split(",") if e.strip()]
    if allowed_emails
    else None  # None indicates no restriction
)

auth = FirebaseAuth(
    {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
    }
)
user = auth.check_session()
if user:
    if allowed_list is not None and user["email"] not in allowed_list:
        st.sidebar.error("You are not authorized to use this app.")
        st.stop()
    else:
        st.sidebar.success(f"Signed in as {user['email']}")
        st.session_state["user_id"] = user["uid"]

else:
    st.sidebar.markdown("### 🔑 Login")
    user = auth.login_form()
    if user:
        st.session_state["user_id"] = user["uid"]
        st.sidebar.success(f"Signed in as {user['email']}")
    else:
        st.sidebar.error("Login failed")
        st.session_state["user_id"] = None
        st.sidebar.info("Please login to use the app")
        st.stop()

if "user_id" not in st.session_state:
    st.error("You must be logged in to use this app.")
    st.stop()

# Hide or disable the logout button until the functionality is stable
logout_stable = False  # Set to True when logout functionality is implemented

if logout_stable and st.sidebar.button("Logout", use_container_width=True):
    rtn = auth.logout_form()
    st.write("Logging out...", rtn)
    st.sidebar.success("Logged out")
    st.session_state["user_id"] = None
    st.stop()
elif not logout_stable:
    st.sidebar.info("Logout functionality is currently unavailable.")

# ──────────────────────────────────────────────────────────────────────────────
# Initialize the session state
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
            repo.add(
                Chunk(
                    id=None,
                    user_id=st.session_state.user_id,
                    jp_prompt=jp,
                    en_answer=en,
                )
            )
            st.session_state.just_added = True  # Flag to indicate a new chunk was added
            st.rerun()

    # ──────────────────────────────────────────────────────────────────────────────
    # Dangerous chunk deletion
    # ──────────────────────────────────────────────────────────────────────────────
    st.markdown("### ⚠️ Dangerous chunk deletion")
    with st.form("reset_db"):
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
    if not st.session_state.get("api_key"):
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

    for ch in repo.get_overdue():
        with st.expander(ch.jp_prompt):
            if st.button("📝 Practice", key=f"graph_{ch.id}"):
                result = graph_app.invoke(
                    {
                        "jp_prompt": ch.jp_prompt,
                        "en_answer": ch.en_answer,
                        "user_input": st.session_state.get(f"ans_{ch.id}", ""),
                    },
                    config={"run-name": f"chunk-{ch.id}"},
                )
                if result and isinstance(result, dict) and "question" in result:
                    st.session_state[f"ex_{ch.id}"] = result.get("question")
                else:
                    st.warning(f"⚠️ Unable to retrieve 'question' for chunk ID {ch.id}.")

            question = st.session_state.get(f"ex_{ch.id}")
            if question:
                st.markdown(f"**Prompt:** {question}")

                # tentative function to generate audio files : this will be replaced by a audio recorder
                st.markdown("### 🎙️ Upload and Transcribe")
                audio_file = st.file_uploader(
                    "Upload an audio file", type=["wav", "mp3"], key=f"audio_{ch.id}"
                )
                if audio_file and "api_key" in st.session_state:
                    # Transcribe the audio file using OpenAI's Whisper API
                    with st.spinner("Transcribing..."):
                        client = OpenAI(api_key=st.session_state["api_key"])
                        try:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                response_format="text",
                            )
                            if transcript:
                                st.session_state[f"ans_{ch.id}"] = transcript
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")

                ans = st.text_area("Your answer", key=f"ans_{ch.id}")

                if st.button("✅ Check", key=f"chk_{ch.id}", disabled=not ans.strip()):
                    result = graph_app.invoke(
                        {
                            "jp_prompt": ch.jp_prompt,
                            "en_answer": ch.en_answer,
                            "user_input": ans,  # user's answer. this will be used to generate feedback
                        },
                        config={"run-name": f"chunk-{ch.id}"},
                    )
                    st.markdown(f"**Feedback**:\n\n{result.get('feedback')}")

with tab_manage:
    st.subheader("📋 Chunk List")

    df = pd.DataFrame([c.__dict__ for c in repo.get_all()])
    edited = st.data_editor(
        df[["id", "jp_prompt", "en_answer", "ef", "interval"]],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True),
            "jp_prompt": st.column_config.TextColumn("JP Prompt"),
            "en_answer": st.column_config.TextColumn("EN Answer"),
            "ef": st.column_config.NumberColumn(
                "EF", min_value=1.0, max_value=5.0, step=0.1
            ),
            "interval": st.column_config.NumberColumn("Interval", min_value=0),
        },
    )

    left, mid, right = st.columns([1, 2, 1])
    with right:
        if st.button("💾 Save edits", use_container_width=True):
            repo.bulk_update(edited)
            st.success("Saved!")

    with st.container(border=True):
        col_sel, col_del, col_reset = st.columns([2, 1, 1])

        with col_sel:
            sel_id = st.selectbox(
                "Target row ID",
                edited["id"],
                label_visibility="visible",
                index=None,
            )

        with col_del:
            st.markdown("**Danger**", help="Delete Selected Chunk")
            if st.button("🗑 Delete", disabled=sel_id is None, use_container_width=True):
                repo.delete_many([sel_id])
                st.rerun()

        with col_reset:
            st.markdown("**Reset review**", help="Reset review interval/EF")
            if st.button(
                "🔄 Reset intv", disabled=sel_id is None, use_container_width=True
            ):
                repo.reset_intervals([sel_id])
                st.rerun()

    df["next_due_date"] = pd.to_datetime(df["next_due_date"]).dt.date
    due_today = (df["next_due_date"] <= date.today()).sum()

    st.divider()
    st.subheader("📊 Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(df))
    col2.metric("Avg EF", round(df["ef"].mean(), 2))
    col3.metric("Due today", due_today)
