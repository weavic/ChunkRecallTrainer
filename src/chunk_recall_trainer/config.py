"""
Configuration module for the Chunk Recall Trainer application.

This module defines the AppConfig class, which centralizes the application's
configuration settings, primarily loaded from environment variables or Streamlit
secrets. It provides a single point of access for configurations like API keys
and Firebase settings.

Attributes:
    app_config (AppConfig): A singleton instance of the AppConfig class.
"""
import os
import streamlit as st
from typing import List, Optional, Dict

class AppConfig:
    """
    Manages application configuration settings.

    This class loads various API keys and settings from environment variables
    or Streamlit secrets. It provides convenient properties to access these
    configurations, such as Firebase settings and a list of allowed email
    addresses for application access.

    Attributes:
        openai_api_key (Optional[str]): OpenAI API key.
        firebase_api_key (Optional[str]): Firebase API key for web apps.
        firebase_auth_domain (Optional[str]): Firebase authentication domain.
        firebase_measurement_id (Optional[str]): Firebase measurement ID for Analytics.
        allowed_emails (str): Comma-separated string of allowed email addresses.
        langsmith_api_key (Optional[str]): LangSmith API key (if used).
    """
    def __init__(self):
        """
        Initializes AppConfig by loading settings from environment variables
        or Streamlit secrets.
        """
        # Attempt to load OpenAI API key from environment variables first, then Streamlit secrets
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        
        # Firebase configuration details
        self.firebase_api_key: Optional[str] = os.getenv("FIREBASE_API_KEY")
        self.firebase_auth_domain: Optional[str] = os.getenv("FIREBASE_AUTH_DOMAIN")
        self.firebase_measurement_id: Optional[str] = os.getenv("FIREBASE_MEASUREMENT_ID")
        
        # Allowed emails for application access control
        self.allowed_emails: str = os.getenv("ALLOWED_EMAILS", "") # Defaults to empty string if not set
        
        # LangSmith API key (retained for potential future use, though currently not active in graph)
        self.langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")

    @property
    def firebase_config(self) -> Dict[str, Optional[str]]:
        """
        Returns the Firebase configuration as a dictionary.

        This format is typically required by Firebase SDKs.
        """
        return {
            "apiKey": self.firebase_api_key,
            "authDomain": self.firebase_auth_domain,
            "measurementId": self.firebase_measurement_id,
            # Note: Other Firebase config fields like 'projectId', 'storageBucket', 
            # 'messagingSenderId', 'appId' might be needed depending on Firebase services used.
            # These are not currently configured here.
        }

    @property
    def allowed_emails_list(self) -> Optional[List[str]]:
        """
        Returns a list of allowed email addresses.

        Parses the comma-separated `allowed_emails` string into a list.
        Returns None if `allowed_emails` is empty or not set.
        """
        if self.allowed_emails:
            return [e.strip() for e in self.allowed_emails.split(",") if e.strip()]
        return None

# Initialize a single config instance for global application use.
# This makes it easy to access configuration settings from anywhere in the app
# by simply importing `app_config` from this module.
app_config = AppConfig()
