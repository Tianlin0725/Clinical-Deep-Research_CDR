"""
Tests for the config module and environment variable validation.
"""

import os
import pytest
from unittest.mock import patch

from cdr.core.config import (
    EnvValidationError,
    validate_environment,
    get_env_or_raise,
    get_settings,
    reset_settings,
)


class TestEnvValidationError:
    """Test the EnvValidationError exception."""

    def test_error_message_format(self):
        """Test that error message is well-formatted."""
        missing = ["API_KEY", "SECRET_TOKEN"]
        error = EnvValidationError(missing, ".env.example")
        
        assert "ENVIRONMENT CONFIGURATION ERROR" in str(error)
        assert "API_KEY" in str(error)
        assert "SECRET_TOKEN" in str(error)
        assert "cp .env.example .env" in str(error)
        assert error.missing_vars == missing

    def test_single_missing_var(self):
        """Test with a single missing variable."""
        error = EnvValidationError(["API_KEY"], ".env.example")
        assert "API_KEY" in str(error)


class TestValidateEnvironment:
    """Test the validate_environment function."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    def test_valid_gemini_environment(self):
        """Test validation passes with valid Gemini env."""
        # Should not raise
        validate_environment("gemini")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_gemini_api_key(self):
        """Test validation fails when GEMINI_API_KEY is missing."""
        with pytest.raises(EnvValidationError) as exc_info:
            validate_environment("gemini")
        
        assert "GEMINI_API_KEY" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_valid_openai_environment(self):
        """Test validation passes with valid OpenAI env."""
        validate_environment("openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_openai_api_key(self):
        """Test validation fails when OPENAI_API_KEY is missing."""
        with pytest.raises(EnvValidationError) as exc_info:
            validate_environment("openai")
        
        assert "OPENAI_API_KEY" in str(exc_info.value)

    @patch.dict(os.environ, {"LLM_DEFAULT_PROVIDER": "openai", "OPENAI_API_KEY": "test"}, clear=True)
    def test_uses_default_provider_from_env(self):
        """Test that it uses LLM_DEFAULT_PROVIDER from env when provider is None."""
        # Should not raise
        validate_environment()

    @patch.dict(os.environ, {"CLOUDFLARE_API_KEY": "key", "CLOUDFLARE_ACCOUNT_ID": "id"}, clear=True)
    def test_cloudflare_requires_two_vars(self):
        """Test Cloudflare requires both API key and Account ID."""
        validate_environment("cloudflare")

    @patch.dict(os.environ, {"CLOUDFLARE_API_KEY": "key"}, clear=True)
    def test_cloudflare_missing_account_id(self):
        """Test Cloudflare fails when ACCOUNT_ID is missing."""
        with pytest.raises(EnvValidationError) as exc_info:
            validate_environment("cloudflare")
        
        assert "CLOUDFLARE_ACCOUNT_ID" in str(exc_info.value)


class TestGetEnvOrRaise:
    """Test the get_env_or_raise function."""

    @patch.dict(os.environ, {"MY_VAR": "my-value"}, clear=True)
    def test_existing_variable(self):
        """Test getting an existing variable."""
        value = get_env_or_raise("MY_VAR")
        assert value == "my-value"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_variable(self):
        """Test raising when variable is missing."""
        with pytest.raises(EnvValidationError) as exc_info:
            get_env_or_raise("MISSING_VAR")
        
        assert "MISSING_VAR" in str(exc_info.value)

    @patch.dict(os.environ, {"EMPTY_VAR": ""}, clear=True)
    def test_empty_variable(self):
        """Test raising when variable is empty."""
        with pytest.raises(EnvValidationError) as exc_info:
            get_env_or_raise("EMPTY_VAR")
        
        assert "EMPTY_VAR" in str(exc_info.value)

    @patch.dict(os.environ, {"WHITESPACE_VAR": "   "}, clear=True)
    def test_whitespace_only_variable(self):
        """Test raising when variable is whitespace only."""
        with pytest.raises(EnvValidationError) as exc_info:
            get_env_or_raise("WHITESPACE_VAR")
        
        assert "WHITESPACE_VAR" in str(exc_info.value)


class TestSettings:
    """Test the Settings class."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    def test_get_settings_singleton(self):
        """Test that get_settings returns a singleton."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False)
    def test_log_level_from_env(self):
        """Test that log level is read from env."""
        reset_settings()
        settings = get_settings()
        assert settings.logging.log_level == "DEBUG"

    @patch.dict(os.environ, {"CDR_DEBUG": "true"}, clear=False)
    def test_debug_flag_from_env(self):
        """Test that debug flag is read from env."""
        reset_settings()
        settings = get_settings()
        assert settings.features.debug == True
