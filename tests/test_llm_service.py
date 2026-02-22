"""
Tests for get_litellm_params: config validation and ValueError paths.
"""
import pytest
from django.test import override_settings

from agentcore_metering.adapters.django.services import get_litellm_params


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmServiceOpenAI:
    @override_settings(LLM_PROVIDER="openai", OPENAI_CONFIG={})
    def test_missing_api_key_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "OpenAI" in str(exc_info.value)
        err = str(exc_info.value).lower()
        assert "incomplete" in err or "api" in err


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmServiceAzureOpenAI:
    @override_settings(LLM_PROVIDER="azure_openai", AZURE_OPENAI_CONFIG={})
    def test_empty_config_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Azure" in str(exc_info.value)

    @override_settings(
        LLM_PROVIDER="azure_openai",
        AZURE_OPENAI_CONFIG={"api_key": "key", "api_base": ""},
    )
    def test_missing_api_base_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Azure" in str(exc_info.value)

    @override_settings(
        LLM_PROVIDER="azure_openai",
        AZURE_OPENAI_CONFIG={"api_key": "", "api_base": "https://example.com"},
    )
    def test_missing_api_key_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Azure" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmServiceGemini:
    @override_settings(LLM_PROVIDER="gemini", GEMINI_CONFIG={})
    def test_missing_api_key_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Gemini" in str(exc_info.value)
        assert "incomplete" in str(exc_info.value).lower()
