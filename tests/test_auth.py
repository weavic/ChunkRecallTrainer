import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import sys

# Add src directory to sys.path to allow importing AppConfig and auth functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from chunk_recall_trainer.auth import (
    initialize_auth,
    authenticate_user,
    render_logout_button,
)
from chunk_recall_trainer.config import AppConfig

# Mock streamlit_firebase_auth.FirebaseAuth before it's imported by auth.py
# However, since it's passed as an argument, we can often mock the instance.
# For initialize_auth, we need to mock the class itself.


class TestAuth(unittest.TestCase):

    def setUp(self):
        """Setup common test resources."""
        # Create a mock AppConfig instance
        self.mock_app_config = MagicMock(spec=AppConfig)
        self.mock_app_config.firebase_config = {
            "apiKey": "test_key",
            "authDomain": "test_domain",
        }
        self.mock_app_config.allowed_emails_list = [
            "test@example.com",
            "another@example.com",
        ]

    @patch("chunk_recall_trainer.auth.FirebaseAuth")
    def test_initialize_auth(self, MockFirebaseAuth):
        """Test that initialize_auth correctly initializes FirebaseAuth."""
        auth_instance = initialize_auth(self.mock_app_config)
        MockFirebaseAuth.assert_called_once_with(self.mock_app_config.firebase_config)
        self.assertIs(auth_instance, MockFirebaseAuth.return_value)

    @patch("chunk_recall_trainer.auth.st")
    def test_authenticate_user_session_exists_allowed(self, mock_st):
        """Test authenticate_user when user session exists and email is allowed."""
        mock_auth = MagicMock()
        mock_user = {"email": "test@example.com", "uid": "123"}
        mock_auth.check_session.return_value = mock_user

        user = authenticate_user(mock_auth, self.mock_app_config)

        self.assertEqual(user, mock_user)
        mock_auth.check_session.assert_called_once()
        mock_st.sidebar.info.assert_called_once_with(
            f"Welcome back, {mock_user['email']}!"
        )
        mock_st.sidebar.success.assert_called_once_with(
            f"Signed in as {mock_user['email']}"
        )
        mock_st.stop.assert_not_called()
        mock_auth.login_form.assert_not_called()

    @patch("chunk_recall_trainer.auth.st")
    def test_authenticate_user_session_exists_not_allowed(self, mock_st):
        """Test authenticate_user when user session exists but email is not allowed."""
        mock_auth = MagicMock()
        mock_user = {"email": "forbidden@example.com", "uid": "456"}
        mock_auth.check_session.return_value = mock_user

        authenticate_user(mock_auth, self.mock_app_config)  # Will call st.stop()

        mock_auth.check_session.assert_called_once()
        mock_st.sidebar.info.assert_called_once_with(
            f"Welcome back, {mock_user['email']}!"
        )
        mock_st.sidebar.error.assert_called_once_with(
            "Access Denied: Your email is not authorized for this application."
        )
        mock_st.stop.assert_called_once()
        mock_auth.login_form.assert_not_called()

    @patch("chunk_recall_trainer.auth.st")
    def test_authenticate_user_login_form_allowed(self, mock_st):
        """Test authenticate_user with login form, successful login, email allowed."""
        mock_auth = MagicMock()
        mock_auth.check_session.return_value = None  # No active session
        mock_login_user = {"email": "another@example.com", "uid": "789"}
        mock_auth.login_form.return_value = mock_login_user

        user = authenticate_user(mock_auth, self.mock_app_config)

        self.assertEqual(user, mock_login_user)
        mock_auth.check_session.assert_called_once()
        mock_st.sidebar.markdown.assert_called_once_with("### 🔑 Please Login")
        mock_auth.login_form.assert_called_once()
        mock_st.sidebar.success.assert_called_once_with(
            f"Successfully logged in as {mock_login_user['email']}"
        )
        mock_st.stop.assert_not_called()

    @patch("chunk_recall_trainer.auth.st")
    def test_authenticate_user_login_form_not_allowed(self, mock_st):
        """Test authenticate_user with login form, successful login, but email not allowed."""
        mock_auth = MagicMock()
        mock_auth.check_session.return_value = None  # No active session
        mock_login_user = {"email": "unlisted@example.com", "uid": "101"}
        mock_auth.login_form.return_value = mock_login_user

        authenticate_user(mock_auth, self.mock_app_config)  # Will call st.stop()

        mock_auth.check_session.assert_called_once()
        mock_auth.login_form.assert_called_once()
        mock_st.sidebar.error.assert_called_once_with(
            "Access Denied: Your email is not authorized for this application."
        )
        mock_st.stop.assert_called_once()

    @patch("chunk_recall_trainer.auth.st")
    def test_authenticate_user_login_form_fails(self, mock_st):
        """Test authenticate_user with login form, login fails."""
        mock_auth = MagicMock()
        mock_auth.check_session.return_value = None  # No active session
        mock_auth.login_form.return_value = None  # Login attempt fails

        authenticate_user(mock_auth, self.mock_app_config)  # Will call st.stop()

        mock_auth.check_session.assert_called_once()
        mock_auth.login_form.assert_called_once()
        mock_st.sidebar.info.assert_called_once_with(
            "Enter your credentials to access the application."
        )
        mock_st.stop.assert_called_once()

    @patch("chunk_recall_trainer.auth.st")
    def test_render_logout_button_stable_and_clicked(self, mock_st):
        """Test render_logout_button when stable and button is clicked."""
        mock_auth = MagicMock()
        mock_st.sidebar.button.return_value = True  # Simulate button click

        # Mock session_state as a dictionary
        mock_st.session_state = {"user_id": "123"}

        render_logout_button(mock_auth, logout_stable=True)

        mock_st.sidebar.button.assert_called_once_with(
            "Logout", use_container_width=True, key="logout_button"
        )
        mock_auth.logout_form.assert_called_once()
        mock_st.sidebar.success.assert_called_once_with(
            "You have been successfully logged out."
        )
        self.assertNotIn(
            "user_id", mock_st.session_state
        )  # Check if user_id is deleted
        mock_st.rerun.assert_called_once()

    @patch("chunk_recall_trainer.auth.st")
    def test_render_logout_button_stable_not_clicked(self, mock_st):
        """Test render_logout_button when stable and button is not clicked."""
        mock_auth = MagicMock()
        mock_st.sidebar.button.return_value = False  # Simulate button not clicked

        render_logout_button(mock_auth, logout_stable=True)

        mock_st.sidebar.button.assert_called_once_with(
            "Logout", use_container_width=True, key="logout_button"
        )
        mock_auth.logout_form.assert_not_called()
        mock_st.rerun.assert_not_called()

    @patch("chunk_recall_trainer.auth.st")
    def test_render_logout_button_not_stable(self, mock_st):
        """Test render_logout_button when not stable."""
        mock_auth = MagicMock()  # Not strictly needed here as it won't be called

        render_logout_button(mock_auth, logout_stable=False)

        mock_st.sidebar.info.assert_called_once_with(
            "Logout functionality is currently marked as unavailable."
        )
        mock_st.sidebar.button.assert_not_called()


if __name__ == "__main__":
    unittest.main()
