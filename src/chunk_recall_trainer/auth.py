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
    return FirebaseAuth(app_config.firebase_config)

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
    user: Optional[User] = auth.check_session() # Check for existing authenticated session

    if user:
        # User is already signed in (active session)
        st.sidebar.info(f"Welcome back, {user.get('email', 'user')}!") # Welcome message
        if app_config.allowed_emails_list and user.get("email") not in app_config.allowed_emails_list:
            st.sidebar.error("Access Denied: Your email is not authorized for this application.")
            st.stop()  # Halt execution if email is not in the allowed list
        else:
            # User is authorized
            st.sidebar.success(f"Signed in as {user.get('email', 'Unknown User')}")
            return user
    else:
        # No active session, display login form
        st.sidebar.markdown("### 🔑 Please Login")
        user = auth.login_form() # This call renders the form and returns user info on successful login
        
        if user:
            # User has just logged in successfully
            if app_config.allowed_emails_list and user.get("email") not in app_config.allowed_emails_list:
                st.sidebar.error("Access Denied: Your email is not authorized for this application.")
                # Consider clearing session_state or explicitly logging out user here if library doesn't handle it
                st.stop() # Halt execution
            else:
                st.sidebar.success(f"Successfully logged in as {user.get('email', 'Unknown User')}")
                return user
        else:
            # Login form is displayed, or login attempt failed
            # streamlit-firebase-auth handles some messages, but we can add more context.
            st.sidebar.info("Enter your credentials to access the application.")
            st.stop() # Halt execution until user logs in

    # This path should ideally not be reached if st.stop() is effective or user is returned.
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
            # The logout_form method in streamlit_firebase_auth typically handles
            # clearing Firebase session cookies and local state.
            auth.logout_form() 
            st.sidebar.success("You have been successfully logged out.")
            # Clear any app-specific user identifiers from session state
            if "user_id" in st.session_state:
                del st.session_state["user_id"]
            # Optionally clear other session state variables tied to the user
            # e.g., st.session_state.queue = [] 
            st.rerun() # Rerun the app to reflect the logged-out state
    else:
        st.sidebar.info("Logout functionality is currently marked as unavailable.")
