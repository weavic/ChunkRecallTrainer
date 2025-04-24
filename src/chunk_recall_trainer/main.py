# app.py (minimal demo)
import streamlit as st
from chunk import ChunkRepo, sm2_update, Chunk  # auto-import via devcontainer
from datetime import date

repo = ChunkRepo()
overdue = repo.get_overdue()

st.title("Streamlit App (Python) - debugging")

if not overdue:
    st.success("🎉 All caught up! Come back tomorrow.")
    st.stop()

# Track position in session
idx = st.session_state.get("idx", 0)
chunk = overdue[idx]

st.subheader(f"Chunk {idx + 1} / {len(overdue)}")
st.markdown(f"**JP Prompt:** {chunk.jp_prompt}")

if st.button("Reveal"):
    st.markdown(f"✅ **Answer:** {chunk.en_answer}")

    col1, col2, col3 = st.columns(3)
    if col1.button("Hard"):
        repo.update(sm2_update(chunk, 2))
        st.session_state.idx = idx + 1
        st.experimental_rerun()
    if col2.button("Good"):
        repo.update(sm2_update(chunk, 4))
        st.session_state.idx = idx + 1
        st.experimental_rerun()
    if col3.button("Easy"):
        repo.update(sm2_update(chunk, 5))
        st.session_state.idx = idx + 1
        st.experimental_rerun()
