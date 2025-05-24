"""
Authentication module for the Chunk Recall Trainer application.

This module handles user authentication using Firebase via the streamlit-firebase-auth
library. It provides functions to initialize the authentication service,
manage the login/session check process, and render a logout button.
User access can be restricted based on an allowed list of email addresses
defined in the application configuration.
"""
import streamlit as st
from streamlit_firebase_auth import FirebaseAuth # type: ignore # library type hints may be missing
from .config import AppConfig
from typing import Optional, Dict, Any
from .logger import logger # Import the logger

# Type alias for user object for clarity, actual structure depends on FirebaseAuth library
User = Dict[str, Any]

def initialize_auth(app_config: AppConfig) -> FirebaseAuth:
    """
    Initializes and returns a FirebaseAuth object using Firebase configurations.

    Args:
        app_config: The application configuration object containing Firebase settings.

    Returns:
        An initialized FirebaseAuth instance.
    """
    logger.info("Initializing Firebase Auth.")
    try:
        auth_instance = FirebaseAuth(app_config.firebase_config)
        logger.info("Firebase Auth initialized successfully.")
        return auth_instance
    except Exception as e:
        logger.error(f"Firebase Admin SDK initialization failed: {e}")
        st.error(f"Firebase initialization failed: {e}") # Also show to user
        st.stop() # Stop the app if Firebase cannot be initialized


def authenticate_user(auth: FirebaseAuth, app_config: AppConfig) -> Optional[User]:
    """
    Manages user authentication, including session checking and login form display.

    This function checks for an active session. If found, it validates the user's
    email against an allowed list (if configured). If no active session, it
    displays a login form in the Streamlit sidebar. 
    
    Side Effects:
        - Renders login form, success/error messages, and authorization status in `st.sidebar`.
        - Calls `st.stop()` to halt app execution if:
            - User is not authorized (email not in allowed list).
            - Login fails or the login form is displayed awaiting input.

    Args:
        auth: The initialized FirebaseAuth instance.
        app_config: The application configuration object.

    Returns:
        The user object (a dictionary) if authentication is successful and the user
        is authorized. Returns `None` if authentication fails or the user is not
        authorized, in which case `st.stop()` is also called.
    """
    logger.info("Attempting to authenticate user.")
    user: Optional[User] = auth.check_session() # Check for existing authenticated session

    if user:
        # User is already signed in (active session)
        email = user.get('email', 'unknown_user')
        logger.info(f"User {email} session found.")
        st.sidebar.info(f"Welcome back, {email}!") # Welcome message
        if app_config.allowed_emails_list and email not in app_config.allowed_emails_list:
            logger.warning(f"User {email} is not authorized for this application.")
            st.sidebar.error("Access Denied: Your email is not authorized for this application.")
            st.stop()  # Halt execution if email is not in the allowed list
        else:
            # User is authorized
            logger.info(f"User {email} is authorized and signed in.")
            st.sidebar.success(f"Signed in as {email}")
            return user
    else:
        # No active session, display login form
        logger.info("No active session found. Displaying login form.")
        st.sidebar.markdown("### 🔑 Please Login")
        # Note: streamlit-firebase-auth's login_form() can involve user creation/verification internally
        # based on its own logic, which we don't have direct hooks into for detailed logging here
        # without modifying that library. We log the outcome.
        try:
            user = auth.login_form() # This call renders the form and returns user info on successful login
        except Exception as e: # Catching potential errors during the login_form process itself
            logger.error(f"Error during login_form display or interaction: {e}")
            st.sidebar.error("An unexpected error occurred during login. Please try again.")
            st.stop()
            return None # Should be unreachable due to st.stop()

        if user:
            email = user.get('email', 'unknown_user')
            # User has just logged in successfully
            logger.info(f"User {email} logged in successfully via form.")
            if app_config.allowed_emails_list and email not in app_config.allowed_emails_list:
                logger.warning(f"User {email} logged in but is not authorized.")
                st.sidebar.error("Access Denied: Your email is not authorized for this application.")
                # Consider clearing session_state or explicitly logging out user here if library doesn't handle it
                st.stop() # Halt execution
            else:
                logger.info(f"User {email} successfully logged in and is authorized.")
                st.sidebar.success(f"Successfully logged in as {email}")
                return user
        else:
            # Login form is displayed, or login attempt failed
            # streamlit-firebase-auth handles some messages, but we can add more context.
            logger.info("Login form displayed or login attempt failed. Halting execution until login.")
            st.sidebar.info("Enter your credentials to access the application.")
            st.stop() # Halt execution until user logs in

    # This path should ideally not be reached if st.stop() is effective or user is returned.
    logger.warning("authenticate_user function reached an unexpected end state.")
    return None

def render_logout_button(auth: FirebaseAuth, logout_stable: bool = False):
    """
    Renders a logout button in the Streamlit sidebar and handles the logout process.

    Args:
        auth: The initialized FirebaseAuth instance.
        logout_stable: A boolean flag to enable/disable the logout functionality.
                       (Currently defaults to False, meaning logout is shown as unavailable).
    
    Side Effects:
        - Renders a button in `st.sidebar`.
        - If logout is successful:
            - Displays a success message in `st.sidebar`.
            - Clears "user_id" from `st.session_state`.
            - Calls `st.rerun()` to refresh the app state.
        - If `logout_stable` is False, displays an info message about unavailability.
    """
    if logout_stable:
        if st.sidebar.button("Logout", use_container_width=True, key="logout_button"):
            logged_out_user_email = st.session_state.get('email', 'unknown_user') # Get email before logout
            logger.info(f"User {logged_out_user_email} initiated logout.")
            try:
                # The logout_form method in streamlit_firebase_auth typically handles
                # clearing Firebase session cookies and local state.
                auth.logout_form()
                logger.info(f"User {logged_out_user_email} logged out successfully.")
                st.sidebar.success("You have been successfully logged out.")
                # Clear any app-specific user identifiers from session state
                if "user_id" in st.session_state:
                    del st.session_state["user_id"]
                if "email" in st.session_state: # Also clear email if stored
                    del st.session_state["email"]
                # Optionally clear other session state variables tied to the user
                # e.g., st.session_state.queue = []
                st.rerun() # Rerun the app to reflect the logged-out state
            except Exception as e:
                logger.error(f"Error during logout for user {logged_out_user_email}: {e}")
                st.sidebar.error("An error occurred during logout. Please try again.")
    else:
        st.sidebar.info("Logout functionality is currently marked as unavailable.")
        logger.debug("Logout button not rendered as logout_stable is False.")
