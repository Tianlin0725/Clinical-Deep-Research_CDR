"""
Tests for CDR environment variable validation.

These tests verify that the environment validator provides clear error messages
when required environment variables are missing.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestEnvValidationError:
    """Tests for EnvValidationError class."""

    def test_error_message_contains_missing_vars(self):
        """Test that error message lists missing variables."""
        from cdr.core.config import EnvValidationError

        error = EnvValidationError(missing_vars=["GEMINI_API_KEY", "OPENAI_API_KEY"])

        assert "GEMINI_API_KEY" in str(error)
        assert "OPENAI_API_KEY" in str(error)
        assert "missing" in str(error).lower()

    def test_error_message_references_env_example(self):
        """Test that error message references .env.example."""
        from cdr.core.config import EnvValidationError

        error = EnvValidationError(
            missing_vars=["GEMINI_API_KEY"],
            env_example_path=".env.example"
        )

        assert ".env.example" in str(error)
        assert "cp .env.example .env" in str(error)

    def test_error_has_missing_vars_attribute(self):
        """Test that error has missing_vars attribute."""
        from cdr.core.config import EnvValidationError

        missing = ["TEST_VAR"]
        error = EnvValidationError(missing_vars=missing)

        assert error.missing_vars == missing
        assert error.env_example_path == ".env.example"


class TestValidateEnvironment:
    """Tests for validate_environment function."""

    def test_no_error_when_provider_configured(self):
        """Test no error when required provider is configured."""
        from cdr.core.config import validate_environment

        # With GEMINI_API_KEY set, should not raise
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            # Should not raise
            validate_environment(provider="gemini")

    def test_raises_error_when_missing_provider_key(self):
        """Test that error is raised when provider key is missing."""
        from cdr.core.config import validate_environment, EnvValidationError

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvValidationError) as exc_info:
                validate_environment(provider="gemini")

            assert "GEMINI_API_KEY" in exc_info.value.missing_vars

    def test_uses_default_provider_when_none_specified(self):
        """Test that default provider is used when none specified."""
        from cdr.core.config import validate_environment

        # Without GEMINI_API_KEY, should raise
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):
                validate_environment()

    def test_validates_openai_provider(self):
        """Test validation for OpenAI provider."""
        from cdr.core.config import validate_environment, EnvValidationError

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvValidationError) as exc_info:
                validate_environment(provider="openai")

            assert "OPENAI_API_KEY" in exc_info.value.missing_vars

    def test_validates_huggingface_provider(self):
        """Test validation for HuggingFace provider."""
        from cdr.core.config import validate_environment, EnvValidationError

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvValidationError) as exc_info:
                validate_environment(provider="huggingface")

            assert "HF_TOKEN" in exc_info.value.missing_vars


class TestGetEnvOrRaise:
    """Tests for get_env_or_raise function."""

    def test_returns_value_when_set(self):
        """Test that value is returned when environment variable is set."""
        from cdr.core.config import get_env_or_raise

        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            result = get_env_or_raise("TEST_KEY")
            assert result == "test_value"

    def test_raises_error_when_missing(self):
        """Test that error is raised when environment variable is missing."""
        from cdr.core.config import get_env_or_raise, EnvValidationError

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvValidationError) as exc_info:
                get_env_or_raise("MISSING_KEY")

            assert "MISSING_KEY" in exc_info.value.missing_vars

    def test_raises_error_when_empty(self):
        """Test that error is raised when environment variable is empty."""
        from cdr.core.config import get_env_or_raise, EnvValidationError

        with patch.dict(os.environ, {"EMPTY_KEY": ""}):
            with pytest.raises(EnvValidationError):
                get_env_or_raise("EMPTY_KEY")


class TestRequiredEnvVars:
    """Tests for REQUIRED_ENV_VARS configuration."""

    def test_all_providers_have_required_vars(self):
        """Test that all providers have required variables defined."""
        from cdr.core.config import REQUIRED_ENV_VARS

        assert "gemini" in REQUIRED_ENV_VARS
        assert "openai" in REQUIRED_ENV_VARS
        assert "anthropic" in REQUIRED_ENV_VARS
        assert "huggingface" in REQUIRED_ENV_VARS

    def test_gemini_requires_api_key(self):
        """Test that Gemini requires GEMINI_API_KEY."""
        from cdr.core.config import REQUIRED_ENV_VARS

        assert "GEMINI_API_KEY" in REQUIRED_ENV_VARS["gemini"]

    def test_openai_requires_api_key(self):
        """Test that OpenAI requires OPENAI_API_KEY."""
        from cdr.core.config import REQUIRED_ENV_VARS

        assert "OPENAI_API_KEY" in REQUIRED_ENV_VARS["openai"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
