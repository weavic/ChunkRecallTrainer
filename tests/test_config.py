import unittest
from unittest.mock import patch
import os
import sys

# Add src directory to sys.path to allow importing AppConfig
# This is a common pattern for tests outside the main package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chunk_recall_trainer.config import AppConfig

# Mock streamlit before it's imported by config module if AppConfig uses st.secrets at import time
# For AppConfig, st.secrets is called within methods/init, so direct patching is fine.
# If st was used at module level in config.py, this would be more complex.

class TestAppConfig(unittest.TestCase):

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env_openai_key"}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_openai_api_key_from_environ(self, mock_st):
        """Test OpenAI API key is retrieved from environment variable (priority)."""
        mock_st.secrets = {} # Ensure st.secrets is empty or does not have the key
        config = AppConfig()
        self.assertEqual(config.openai_api_key, "env_openai_key")

    @patch.dict(os.environ, {}, clear=True) # Clear os.environ for this test
    @patch("chunk_recall_trainer.config.st")
    def test_openai_api_key_from_st_secrets(self, mock_st):
        """Test OpenAI API key is retrieved from st.secrets if not in os.environ."""
        mock_st.secrets = {"OPENAI_API_KEY": "secrets_openai_key"}
        config = AppConfig()
        self.assertEqual(config.openai_api_key, "secrets_openai_key")

    @patch.dict(os.environ, {}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_openai_api_key_missing(self, mock_st):
        """Test OpenAI API key is None if not in os.environ or st.secrets."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertIsNone(config.openai_api_key)

    @patch.dict(os.environ, {
        "FIREBASE_API_KEY": "fb_api_key",
        "FIREBASE_AUTH_DOMAIN": "fb_auth_domain",
        "FIREBASE_MEASUREMENT_ID": "fb_measurement_id"
    }, clear=True)
    @patch("chunk_recall_trainer.config.st") # Mock st even if not directly used by these attributes
    def test_firebase_configs_from_environ(self, mock_st):
        """Test Firebase configurations are correctly retrieved from os.environ."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertEqual(config.firebase_api_key, "fb_api_key")
        self.assertEqual(config.firebase_auth_domain, "fb_auth_domain")
        self.assertEqual(config.firebase_measurement_id, "fb_measurement_id")

    @patch.dict(os.environ, {"ALLOWED_EMAILS": ""}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_allowed_emails_empty(self, mock_st):
        """Test allowed_emails_list with an empty string."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertIsNone(config.allowed_emails_list)

    @patch.dict(os.environ, {"ALLOWED_EMAILS": "user1@example.com"}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_allowed_emails_single(self, mock_st):
        """Test allowed_emails_list with a single email."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertEqual(config.allowed_emails_list, ["user1@example.com"])

    @patch.dict(os.environ, {"ALLOWED_EMAILS": "user1@example.com,user2@example.com"}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_allowed_emails_multiple(self, mock_st):
        """Test allowed_emails_list with multiple emails."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertEqual(config.allowed_emails_list, ["user1@example.com", "user2@example.com"])

    @patch.dict(os.environ, {"ALLOWED_EMAILS": " user1@example.com , user2@example.com  "}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_allowed_emails_with_spaces(self, mock_st):
        """Test allowed_emails_list with emails having leading/trailing spaces."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertEqual(config.allowed_emails_list, ["user1@example.com", "user2@example.com"])
        
    @patch.dict(os.environ, {"ALLOWED_EMAILS": "user1@example.com,,user2@example.com"}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_allowed_emails_with_empty_strings_between_commas(self, mock_st):
        """Test allowed_emails_list with empty strings between commas."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertEqual(config.allowed_emails_list, ["user1@example.com", "user2@example.com"])

    @patch.dict(os.environ, {
        "FIREBASE_API_KEY": "key123",
        "FIREBASE_AUTH_DOMAIN": "domain.com",
        "FIREBASE_MEASUREMENT_ID": "id789"
    }, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_firebase_config_property(self, mock_st):
        """Test the firebase_config property returns the correct dictionary."""
        mock_st.secrets = {}
        config = AppConfig()
        expected_config = {
            "apiKey": "key123",
            "authDomain": "domain.com",
            "measurementId": "id789",
        }
        self.assertEqual(config.firebase_config, expected_config)

    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "ls_api_key"}, clear=True)
    @patch("chunk_recall_trainer.config.st")
    def test_langsmith_api_key(self, mock_st):
        """Test LangSmith API key is retrieved from environment variable."""
        mock_st.secrets = {}
        config = AppConfig()
        self.assertEqual(config.langsmith_api_key, "ls_api_key")

if __name__ == "__main__":
    unittest.main()
