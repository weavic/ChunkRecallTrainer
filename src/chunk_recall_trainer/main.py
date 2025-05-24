"""
Main application file for the Chunk Recall Trainer.

This Streamlit application provides a user interface for practicing language chunks
(e.g., Japanese phrases and their English equivalents) using a spaced repetition
algorithm (SM2). It integrates Firebase for user authentication, OpenAI for
exercise generation and feedback (via a LangGraph chain), and a local SQLite
database (via the `chunk` module) for storing user-specific chunk data.

The UI is structured as follows:
- A sidebar for user authentication (login/logout), data management operations
  (CSV import/export, adding individual chunks, database reset), and application
  settings (OpenAI API key input).
- A main content area with two tabs:
    - "Practice" tab: Displays chunks due for review, allows users to generate
      practice exercises, input their answers (textually or via audio upload
      which is then transcribed), and receive AI-generated feedback on their
      performance. Exercise generation and feedback are handled by the LangGraph app.
    - "Manage Chunks" tab: Shows a list of all chunks for the logged-in user,
      allowing for editing of chunk content, EF/interval values, and deletion
      of selected chunks. Also displays overall statistics about the chunk collection.

Session state is used extensively to manage the practice queue, user inputs,
generated questions/feedback, and API keys.
"""
from datetime import date
import streamlit as st
# import os # Not directly used; API keys and configs are managed via AppConfig.
from chunk import ChunkRepo, Chunk # Data models and repository for chunks.
import pandas as pd # For CSV import/export and data display.
from io import StringIO # For handling CSV data in memory.
import uuid # For generating unique keys for Streamlit widgets.
from openai import OpenAI # For audio transcription using Whisper API.
from graph import app as graph_app # LangGraph application for exercise/feedback.
# FirebaseAuth is handled within the auth module.
from .config import app_config # Centralized application configuration.
from .auth import initialize_auth, authenticate_user, render_logout_button # Auth functions.
from .logger import logger # Import the logger

# ──────────────────────────────────────────────────────────────────────────────
# Application Setup: Page Configuration and Global Settings
# ──────────────────────────────────────────────────────────────────────────────
# Configure the Streamlit page. This must be the first Streamlit command executed.
logger.info("Chunk Recall Trainer application started.")
st.set_page_config(page_title="Chunk Recall Trainer", page_icon="📚", layout="centered")

# Initialize OpenAI API key in session state from AppConfig if not already set by the user.
# This key is crucial for:
# 1. The LangGraph application (`graph_app`) which makes LLM calls for exercises/feedback.
# 2. Direct OpenAI calls, such as using the Whisper API for audio transcription.
if app_config.openai_api_key and "api_key" not in st.session_state:
    st.session_state["api_key"] = app_config.openai_api_key
    logger.info("OpenAI API key found in session state/config.")

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS Styling
# ──────────────────────────────────────────────────────────────────────────────
# Apply custom CSS to enhance the visual appearance of Streamlit elements.
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
# ──────────────────────────────────────────────────────────────────────────────
# User Authentication and Initialization
# ──────────────────────────────────────────────────────────────────────────────
# Initialize the Firebase authentication service using configurations from `app_config`.
auth_handler = initialize_auth(app_config)
# Perform user authentication. This function handles session checks, displays a login
# form in the sidebar (if needed), and validates user email against an allowed list.
# It calls `st.stop()` internally if authentication fails or the user is not authorized,
# preventing the rest of the script from running.
current_user = authenticate_user(auth_handler, app_config)

# If authentication is successful, `current_user` contains user details.
# Store the unique user ID in session state for use throughout the app.
if current_user:
    st.session_state["user_id"] = current_user["uid"]
    logger.info(f"User {st.session_state.user_id} authenticated successfully.")
else:
    # This is a fallback; `authenticate_user` should have already stopped execution.
    # If `current_user` is None, it means auth failed or user is not yet logged in.
    if "user_id" not in st.session_state or not st.session_state.get("user_id"):
         logger.warning("User failed authentication or is not logged in.")
         st.sidebar.error("Critical: Authentication process did not complete. Please try again.")
         st.stop() # Ensure app stops if control reaches here without a user_id.

# Initialize the ChunkRepo for the authenticated user. This provides access to
# user-specific chunk data stored in the database.
# This step must occur after `st.session_state.user_id` is confirmed.
if "user_id" in st.session_state and st.session_state.user_id:
    try:
        # Create a repository instance scoped to the current user.
        repo = ChunkRepo(user_id=st.session_state.user_id)
        logger.info(f"ChunkRepo initialized for user {st.session_state.user_id}")
    except Exception as e: # Catch potential errors during database initialization.
        logger.error(f"Fatal Error: Could not initialize user data repository for user {st.session_state.user_id}. Details: {e}")
        st.error(f"Fatal Error: Could not initialize user data repository. Details: {e}")
        st.stop() # Stop the app if the repo cannot be initialized.
else:
    # This should be an unreachable state if authentication flow is correct.
    logger.error("Fatal Error: User ID is missing after authentication. Cannot initialize data.")
    st.error("Fatal Error: User ID is missing after authentication. Cannot initialize data.")
    st.stop()

# Render the logout button in the sidebar.
# The `logout_stable` parameter can be used to enable/disable the button (e.g., during development).
render_logout_button(auth_handler, logout_stable=False) 

# ──────────────────────────────────────────────────────────────────────────────
# Session State Flags & Toast Notifications
# ──────────────────────────────────────────────────────────────────────────────
# These flags are used to trigger one-time toast notifications after certain actions.
# They are set within form submission logic and reset here after display.
if st.session_state.get("just_added"):
    st.toast("✅ Chunk successfully saved!")
    st.session_state.just_added = False # Reset the flag
if st.session_state.get("just_reset"):
    st.toast("🗑️ All chunks have been cleared from the database!", icon="🔥")
    st.session_state.just_reset = False # Reset the flag

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar UI: Data Management and Settings
# ──────────────────────────────────────────────────────────────────────────────
# The Streamlit sidebar is populated by authentication functions (login/logout messages)
# and then by the data management and settings sections defined below.
with st.sidebar:
    # Data management features are available only if a user is logged in and the
    # data repository (`repo`) has been successfully initialized.
    if "user_id" in st.session_state and st.session_state.user_id and 'repo' in locals():
        st.header("📂 Data Management")
        
        # CSV Import section
        uploaded_csv_file = st.file_uploader("Import Chunks from CSV", type="csv", key="csv_uploader")
        if uploaded_csv_file is not None:
            logger.info(f"Attempting to import CSV file: {uploaded_csv_file.name}")
            try:
                df_import = pd.read_csv(uploaded_csv_file)
                # Convert DataFrame to a CSV string in memory for the repo method
                csv_in_memory = StringIO()
                df_import.to_csv(csv_in_memory, index=False)
                csv_in_memory.seek(0) # Rewind buffer to the beginning for reading
                
                imported_count = repo.save_from_csv(csv_in_memory)
                logger.info(f"Successfully imported {imported_count} chunks from CSV.")
                st.sidebar.success(f"✅ Successfully imported {imported_count} chunks!")
                # Consider st.rerun() or other UI update mechanism if changes need immediate reflection
            except Exception as e:
                logger.error(f"CSV import failed: {e}")
                st.sidebar.error(f"CSV import failed: {e}")
        st.markdown("---") # Visual separator
        
        # CSV Export section
        logger.info("Exporting all chunks to CSV.")
        all_chunks_as_csv_str = repo.export_all()
        st.download_button(
            label="💾 Export All Chunks to CSV",
            data=all_chunks_as_csv_str, # Must be string or bytes
            file_name="chunks_export.csv",
            mime="text/csv",
            help="Download all your chunks in a CSV file.",
            key="export_csv_button" # Unique key for the widget
        )
        
        # Form for adding a single new chunk
        with st.form("add_chunk_form", clear_on_submit=True):
            st.markdown("### ➕ Add a New Chunk")
            new_jp_prompt_input = st.text_area("Japanese Prompt (JP)", height=80, key="new_jp_prompt_text_area")
            new_en_answer_input = st.text_area("English Answer/Chunk (EN)", height=80, key="new_en_answer_text_area")
            add_chunk_submit_button = st.form_submit_button("Add Chunk")
            
            if add_chunk_submit_button:
                if new_jp_prompt_input and new_en_answer_input: # Basic validation
                    logger.info(f"Adding new chunk: JP='{new_jp_prompt_input}', EN='{new_en_answer_input}'")
                    try:
                        # The Chunk class might handle ID generation if `id=None` is passed.
                        # user_id must be from the active session.
                        repo.add(
                            Chunk( 
                                id=None, # Database will typically auto-generate ID
                                user_id=st.session_state.user_id,
                                jp_prompt=new_jp_prompt_input,
                                en_answer=new_en_answer_input,
                                # EF, interval, etc., will be set to defaults by Pydantic model or DB
                            )
                        )
                        logger.info("Chunk added successfully.")
                        st.session_state.just_added = True # Trigger toast notification
                        st.rerun() # Refresh UI to reflect the new chunk and update stats/queue
                    except Exception as e:
                        logger.error(f"Failed to add chunk: {e}")
                        st.sidebar.error(f"Failed to add chunk: {e}")
                else:
                    st.sidebar.warning("Both Japanese and English fields are required to add a new chunk.")
        
        # "Danger Zone" for operations like resetting the database
        st.markdown("### ⚠️ Danger Zone")
        with st.form("reset_database_form"):
            st.markdown("This will delete **all** your chunks. This action cannot be undone.")
            confirm_reset_checkbox = st.checkbox("Yes, I understand and wish to delete all my chunks.", key="confirm_reset_db_checkbox")
            reset_db_submit_button = st.form_submit_button(
                "Reset Entire Database for This User", 
                type="primary", # Emphasizes caution
                disabled=not confirm_reset_checkbox # Button disabled until checkbox is ticked
            )
            
            if reset_db_submit_button and confirm_reset_checkbox:
                logger.info("User initiated database reset.")
                try:
                    repo.reset() # Method to delete all chunks for the current user
                    logger.info("Database reset successful.")
                    st.session_state.just_reset = True # Trigger toast notification
                    # Clear any session state variables that might hold outdated data
                    st.session_state.queue = []
                    st.session_state.queue_date = date.today() # Reset queue date
                    st.session_state.queue_total = 0
                    # Potentially clear other session states like q_*, ak_*, fb_*, ans_val_* if they exist
                    st.rerun() # Refresh UI
                except Exception as e:
                    logger.error(f"Database reset operation failed: {e}")
                    st.sidebar.error(f"Database reset operation failed: {e}")

    # Settings section for API Key configuration.
    # This section is available regardless of user login status, as API key might be
    # needed by parts of the app or for initial configuration.
    st.header("🛠 Application Settings")
    # Allow users to input their OpenAI API key if it's not pre-configured via environment variables or Streamlit secrets.
    # `app_config.openai_api_key` provides the value from env/secrets (if any).
    # `st.session_state.api_key` stores the key active for the current session, which could be
    # the one from `app_config` or one entered by the user.
    user_provided_api_key = st.text_input(
        "OpenAI API Key",
        type="password", # Masks the input
        value=st.session_state.get("api_key", app_config.openai_api_key or ""), # Display current session key or configured key
        help="Enter your OpenAI API key if not already configured. This will override any system-set key for your current session.",
        disabled=bool(app_config.openai_api_key), # Disable input if key is pre-loaded from app_config (env/secrets)
    )
    # If user enters a new key and it's different from the current session key, update it.
    if user_provided_api_key and user_provided_api_key != st.session_state.get("api_key"):
        st.session_state["api_key"] = user_provided_api_key # Update the API key in session state
        logger.info("OpenAI API key updated by user for the current session.")
        st.success("API key has been updated for the current session.")
    elif not st.session_state.get("api_key"):
        logger.warning("OpenAI API key is not set.")


# ──────────────────────────────────────────────────────────────────────────────
# Main UI: Title, Practice Queue, and Tabs
# ──────────────────────────────────────────────────────────────────────────────
st.title("📚 Chunk Recall Trainer") # Main title of the application

# Initialize or update the daily practice queue for the logged-in user.
# The queue contains chunks that are "overdue" for review based on SM2 algorithm.
# This logic runs only if a user is authenticated and the `repo` is available.
if "user_id" in st.session_state and st.session_state.user_id and 'repo' in locals():
    # Check if the queue needs refreshing (e.g., new day or first load).
    if "queue_date" not in st.session_state or st.session_state.queue_date != date.today():
        try:
            # Fetch up to 5 overdue chunks to form the daily practice queue.
            st.session_state.queue = repo.get_overdue(limit=5)
            st.session_state.queue_date = date.today() # Mark queue as updated for today
            st.session_state.queue_total = len(st.session_state.queue) # Total items for today's session
            logger.info(f"Practice queue updated for user {st.session_state.user_id}. Found {len(st.session_state.queue)} items.")
        except Exception as e:
            logger.error(f"Failed to load practice queue: {e}")
            st.error(f"Failed to load practice queue: {e}")
            st.session_state.queue = [] # Default to empty queue on error
            st.session_state.queue_total = 0
else:
    # Fallback if user/repo not properly initialized (should be rare).
    logger.warning("User or repo not initialized, practice queue cannot be loaded.")
    st.session_state.queue = []
    st.session_state.queue_total = 0
    
# Display progress for the current practice session.
practice_chunks_queue = st.session_state.get("queue", [])
num_remaining_in_practice_queue = len(practice_chunks_queue)
total_chunks_for_today_session = st.session_state.get("queue_total", 0)
# Calculate how many chunks have been (or are being) processed from today's queue.
# If queue is not empty, count includes the current one.
current_progress_display_count = total_chunks_for_today_session - num_remaining_in_practice_queue + \
                                 (1 if num_remaining_in_practice_queue > 0 and total_chunks_for_today_session > 0 else 0)


# Layout for title (already set) and progress metric.
main_title_column, progress_metric_column = st.columns([3, 1])
with main_title_column:
    pass # Main title `st.title("📚 Chunk Recall Trainer")` is already rendered above.
with progress_metric_column:
    if total_chunks_for_today_session > 0:
        st.metric("Today's Review Progress", f"{current_progress_display_count}/{total_chunks_for_today_session}")
    else:
        # Show N/A if there are no chunks in the queue for today.
        st.metric("Today's Review Progress", "N/A")


# Tabbed interface for "Practice" and "Manage Chunks" sections.
tab_practice_section, tab_manage_chunks_section = st.tabs(["💬 Practice Exercises", "📋 Manage Your Chunks"])

# ──────────────────────────────────────────────────────────────────────────────
# "Practice" Tab Logic: Handles exercise display, interaction, and feedback.
# ──────────────────────────────────────────────────────────────────────────────
with tab_practice_section:
    # An OpenAI API key is essential for the Practice tab as it involves LLM calls
    # via the LangGraph app for generating questions and providing feedback.
    if not st.session_state.get("api_key"):
        logger.warning("OpenAI API key is not set. Practice features will be limited.")
        st.warning("⚠️ OpenAI API key is not set. Please configure it in the sidebar under Settings to use practice features.")
        st.stop() # Stop rendering this tab if API key is missing.

    # If the practice queue is empty for today, display a success message.
    if not practice_chunks_queue: # Checks the daily limited queue
        st.success("🎉 All chunks for today's session are reviewed! Check back tomorrow or add new chunks.")
        st.stop() # Stop rendering further if nothing to practice.

    # The UI iterates through ALL overdue chunks from `repo.get_overdue()`.
    # This allows users to practice any overdue chunk, not just those in the limited daily queue.
    # Session state (e.g., `q_{id}`, `ak_{id}`) is used to manage the state of each chunk's exercise individually.
    # This means a user can generate a question for one chunk, answer another, etc.
    
    # Display each overdue chunk within an expander for individual practice.
    for chunk_item_to_practice in repo.get_overdue(): # Iterates all overdue, not just the daily `practice_chunks_queue`
        chunk_id_as_str = str(chunk_item_to_practice.id) # Use string ID for session state keys for robustness.
        
        # Expander title shows Japanese prompt and next due date for context.
        with st.expander(f"🇯🇵 JP: {chunk_item_to_practice.jp_prompt} (Next due: {chunk_item_to_practice.next_due_date.strftime('%Y-%m-%d')})"):
            st.markdown(f"🇬🇧 EN: {chunk_item_to_practice.en_answer}") # Display the English answer/chunk.

            # "Generate Practice Question" button:
            # Invokes the LangGraph app (`graph_app`) to generate a question and its model answer key.
            if st.button("📝 Generate Practice Question", key=f"practice_btn_{chunk_id_as_str}"):
                logger.info(f"Generating practice question for chunk ID: {chunk_id_as_str}")
                # Prepare input for the graph: only `jp_prompt` and `en_answer` are needed for question generation.
                # Other fields are set to None as per the graph's expected input state.
                graph_input_for_question = {
                    "jp_prompt": chunk_item_to_practice.jp_prompt,
                    "en_answer": chunk_item_to_practice.en_answer,
                    "user_input": None, # No user input at this stage.
                    "question": None,   # Question will be generated by the graph.
                    "answer_key": None, # Answer key will be generated by the graph.
                    "feedback": None,   # No feedback yet.
                }
                try:
                    # Invoke the LangGraph app. `config` adds a run name for LangSmith tracing (if enabled).
                    question_generation_result = graph_app.invoke(
                        graph_input_for_question,
                        config={"run-name": f"generate_exercise_chunk-{chunk_id_as_str}"},
                    )
                    # Store the generated question and its answer key in session state, keyed by chunk ID.
                    # This allows multiple exercises to be active simultaneously.
                    st.session_state[f"q_{chunk_id_as_str}"] = question_generation_result.get("question")
                    st.session_state[f"ak_{chunk_id_as_str}"] = question_generation_result.get("answer_key")
                    # Clear any previous feedback or answer for this chunk when a new question is generated.
                    st.session_state[f"fb_{chunk_id_as_str}"] = None
                    st.session_state[f"ans_val_{chunk_id_as_str}"] = ""
                except Exception as e:
                    logger.error(f"Error generating question for chunk ID {chunk_id_as_str}: {e}")
                    st.error(f"An error occurred while generating the question: {e}")
            
            # Display the generated question if it exists in the session state for this chunk.
            current_practice_question = st.session_state.get(f"q_{chunk_id_as_str}")
            if current_practice_question:
                st.markdown(f"**🤔 Your Question:** {current_practice_question}")

                # Audio input section (experimental): Allows user to upload a spoken answer.
                st.markdown("##### 🎙️ Optionally, answer by voice (Upload audio file)")
                audio_file_upload = st.file_uploader(
                    "Upload your spoken answer (WAV or MP3 format)", type=["wav", "mp3"], key=f"audio_uploader_{chunk_id_as_str}"
                )
                if audio_file_upload and st.session_state.get("api_key"): # Check for API key again before OpenAI call
                    logger.info(f"Transcribing audio for chunk ID: {chunk_id_as_str}")
                    with st.spinner("Transcribing your audio answer... Please wait."):
                        try:
                            # Initialize OpenAI client specifically for this Whisper API call.
                            # Consider creating a shared client if making frequent calls to optimize.
                            openai_whisper_client = OpenAI(api_key=st.session_state["api_key"])
                            transcription_response = openai_whisper_client.audio.transcriptions.create(
                                model="whisper-1",      # Specify Whisper model
                                file=audio_file_upload, # The uploaded file object
                                response_format="text", # Request plain text output
                            )
                            if transcription_response:
                                # Populate the text area below with the transcribed text.
                                st.session_state[f"ans_val_{chunk_id_as_str}"] = str(transcription_response).strip()
                                logger.info(f"Audio transcribed successfully for chunk ID: {chunk_id_as_str}")
                                st.success("Audio transcribed successfully! Your answer has been populated below.")
                        except Exception as e:
                            logger.error(f"Audio transcription failed for chunk ID {chunk_id_as_str}: {e}")
                            st.error(f"Audio transcription failed: {e}")

                # Text area for the user to type or see their transcribed answer.
                # Its value is bound to `st.session_state[f"ans_val_{chunk_id_as_str}"]`
                # to persist input across interactions or reflect transcribed audio.
                user_typed_answer = st.text_area(
                    "Your Answer:", 
                    key=f"ans_text_area_{chunk_id_as_str}", # Unique key for the text area
                    value=st.session_state.get(f"ans_val_{chunk_id_as_str}", "") # Default to empty or previous value
                )
                # Ensure session state is updated if user types directly into the text area.
                st.session_state[f"ans_val_{chunk_id_as_str}"] = user_typed_answer

                # "Check My Answer" button: Disabled if no answer is provided.
                # This sends the user's answer to the LangGraph app for feedback.
                if st.button("✅ Check My Answer", key=f"check_btn_{chunk_id_as_str}", disabled=not user_typed_answer.strip()):
                    logger.info(f"Checking answer for chunk ID: {chunk_id_as_str}")
                    # Retrieve the question and its model answer key from session state to provide context for review.
                    question_context_for_review = st.session_state.get(f"q_{chunk_id_as_str}")
                    answer_key_context_for_review = st.session_state.get(f"ak_{chunk_id_as_str}")

                    if not question_context_for_review or not answer_key_context_for_review:
                        st.error("⚠️ Cannot check answer: The practice question or its model answer key was not found. Please generate a question first.")
                    else:
                        # Prepare input for the graph's review path.
                        # This includes the original chunk, user's answer, the generated question, and its model answer.
                        graph_input_for_feedback = {
                            "jp_prompt": chunk_item_to_practice.jp_prompt,    # Original JP part of the chunk
                            "en_answer": chunk_item_to_practice.en_answer,    # Original EN part of the chunk
                            "user_input": user_typed_answer,                  # User's submitted answer
                            "question": question_context_for_review,          # The question user was answering
                            "answer_key": answer_key_context_for_review,      # The model answer to that question
                            "feedback": None,                                 # Feedback will be generated by the graph.
                        }
                        try:
                            feedback_result = graph_app.invoke(
                                graph_input_for_feedback,
                                config={"run-name": f"review_answer_chunk-{chunk_id_as_str}"}, # LangSmith run name
                            )
                            # Store the AI-generated feedback in session state.
                            st.session_state[f"fb_{chunk_id_as_str}"] = feedback_result.get("feedback")
                        except Exception as e:
                            logger.error(f"Error getting feedback for chunk ID {chunk_id_as_str}: {e}")
                            st.error(f"An error occurred while getting feedback: {e}")
                
                # Display the feedback if it's available in session state for this chunk.
                current_feedback_markdown = st.session_state.get(f"fb_{chunk_id_as_str}")
                if current_feedback_markdown:
                    st.markdown(f"**💡 Feedback:**\n\n{current_feedback_markdown}")
            st.markdown("---") # Visual separator between chunks in the practice list.

# ──────────────────────────────────────────────────────────────────────────────
# "Manage Chunks" Tab Logic: Viewing, editing, and managing the chunk collection.
# ──────────────────────────────────────────────────────────────────────────────
with tab_manage_chunks_section:
    st.subheader("📋 Your Chunk Collection")
    # Ensure user is logged in and repository is initialized before showing management tools.
    if not ('repo' in locals() and st.session_state.get("user_id")):
        st.warning("Please log in to manage your chunks.")
        st.stop() # Stop rendering this tab if prerequisites are not met.

    all_user_chunks_list = repo.get_all() # Fetch all chunks for the current user.
    if not all_user_chunks_list:
        st.info("You haven't added any chunks yet. Use the 'Add a New Chunk' form in the sidebar or import a CSV file!")
    else:
        # Convert list of Chunk Pydantic models to a Pandas DataFrame for display in st.data_editor.
        # Using model_dump() for proper serialization if Chunks are Pydantic models.
        # If Chunks are simple dicts or dataclasses, `c.__dict__` might be okay but model_dump is safer.
        try:
            chunks_df = pd.DataFrame([c.model_dump() for c in all_user_chunks_list])
        except AttributeError: # Fallback if .model_dump() is not available (e.g. not Pydantic)
            logger.warning("Using __dict__ for DataFrame conversion as model_dump is not available.")
            chunks_df = pd.DataFrame([c.__dict__ for c in all_user_chunks_list])
            
    edited = st.data_editor(
        chunks_df[["id", "jp_prompt", "en_answer", "ef", "interval"]],
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
            try:
                repo.bulk_update(edited)
                logger.info(f"Bulk update successful for user {st.session_state.user_id}.")
                st.success("Saved!")
            except Exception as e:
                logger.error(f"Bulk update failed for user {st.session_state.user_id}: {e}")
                st.error(f"Failed to save edits: {e}")


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
                try:
                    repo.delete_many([sel_id])
                    logger.info(f"Deleted chunk with ID {sel_id} for user {st.session_state.user_id}.")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to delete chunk with ID {sel_id} for user {st.session_state.user_id}: {e}")
                    st.error(f"Failed to delete chunk: {e}")


        with col_reset:
            st.markdown("**Reset review**", help="Reset review interval/EF")
            if st.button(
                "🔄 Reset intv", disabled=sel_id is None, use_container_width=True
            ):
                try:
                    repo.reset_intervals([sel_id])
                    logger.info(f"Reset interval for chunk ID {sel_id} for user {st.session_state.user_id}.")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to reset interval for chunk ID {sel_id} for user {st.session_state.user_id}: {e}")
                    st.error(f"Failed to reset interval: {e}")


    chunks_df["next_due_date"] = pd.to_datetime(chunks_df["next_due_date"]).dt.date
    due_today = (chunks_df["next_due_date"] <= date.today()).sum()

    st.divider()
    st.subheader("📊 Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(chunks_df))
    col2.metric("Avg EF", round(chunks_df["ef"].mean(), 2) if not chunks_df.empty else 0)
    col3.metric("Due today", due_today)
