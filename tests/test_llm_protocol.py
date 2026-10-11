"""
AGIOne per-model wire-protocol routing.

AGIOne exposes three protocols and rejects a model on the wrong one, so the
LiteLLM model string (and provider) has to be chosen from the model id's
first segment: openai/* -> Responses, anthropic/* -> Messages, everything
else -> Chat Completions.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services.litellm_params import (
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_MESSAGES,
    PROTOCOL_RESPONSES,
    build_litellm_params_from_config,
    resolve_llm_protocol,
)
from agentcore_metering.adapters.django.services.runtime_config import (
    validate_llm_config,
)

AGIONE_BASE = "https://agione.pro/hyperone/xapi/api"


def _agione_config(model):
    return {"api_key": "key", "api_base": AGIONE_BASE, "model": model}


def _mock_completion_response(content="ok", model="openai/gpt-4o-mini"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ),
        model=model,
    )


@pytest.mark.unit
class TestResolveLlmProtocol:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("openai/gpt-5.4-nano/d688d", PROTOCOL_RESPONSES),
            ("openai/gpt-5.3-codex/b687c", PROTOCOL_RESPONSES),
            ("anthropic/claude-sonnet-4.6/e6887", PROTOCOL_MESSAGES),
            ("anthropic/claude-opus-4.6/a6b6f", PROTOCOL_MESSAGES),
            ("deepseek/deepseek-v4-flash/02acd", PROTOCOL_CHAT_COMPLETIONS),
            ("qwen/qwen3.6-flash/82a89", PROTOCOL_CHAT_COMPLETIONS),
            ("gemini/gemini-3.5-flash/a17fd", PROTOCOL_CHAT_COMPLETIONS),
            ("z-ai/glm-5/e577e", PROTOCOL_CHAT_COMPLETIONS),
            ("", PROTOCOL_CHAT_COMPLETIONS),
        ],
    )
    def test_agione_protocol_by_family(self, model, expected):
        assert resolve_llm_protocol("agione", model) == expected

    def test_other_providers_always_chat_completions(self):
        # A custom gateway is not AGIOne and must not inherit AGIOne's rules.
        assert (
            resolve_llm_protocol("openai_compatible", "openai/gpt-5.4-nano/d688d")
            == PROTOCOL_CHAT_COMPLETIONS
        )
        assert (
            resolve_llm_protocol("anthropic", "claude-sonnet-4.6")
            == PROTOCOL_CHAT_COMPLETIONS
        )

    def test_family_match_is_case_insensitive(self):
        assert (
            resolve_llm_protocol("agione", "OpenAI/GPT-5.4-nano/d688d")
            == PROTOCOL_RESPONSES
        )


@pytest.mark.unit
class TestAgioneModelString:
    def test_openai_family_uses_responses_bridge_marker(self):
        params = build_litellm_params_from_config(
            "agione", _agione_config("openai/gpt-5.4-nano/d688d")
        )
        # One "openai/" is the provider prefix LiteLLM strips, "responses/"
        # selects the bridge, and the extra family segment survives to the
        # gateway, which needs "openai/gpt-5.4-nano/d688d" verbatim.
        assert params["model"] == (
            "openai/responses/openai/openai/gpt-5.4-nano/d688d"
        )
        assert params["custom_llm_provider"] == "openai"

    def test_anthropic_family_uses_messages_provider(self):
        params = build_litellm_params_from_config(
            "agione", _agione_config("anthropic/claude-sonnet-4.6/e6887")
        )
        assert params["model"] == (
            "anthropic/anthropic/claude-sonnet-4.6/e6887"
        )
        assert params["custom_llm_provider"] == "anthropic"

    def test_other_families_keep_chat_completions(self):
        params = build_litellm_params_from_config(
            "agione", _agione_config("deepseek/deepseek-v4-flash/02acd")
        )
        assert params["model"] == (
            "openai/deepseek/deepseek-v4-flash/02acd"
        )
        assert params["custom_llm_provider"] == "openai"


@pytest.mark.unit
class TestOpenAiCompatibleUnchanged:
    def test_openai_compatible_still_openai_chat_completions(self):
        params = build_litellm_params_from_config(
            "openai_compatible",
            {
                "api_key": "key",
                "api_base": "https://gateway.example/v1",
                "model": "openai/gpt-5.4-nano/d688d",
            },
        )
        assert params["model"] == "openai/openai/gpt-5.4-nano/d688d"
        assert params["custom_llm_provider"] == "openai"


@pytest.mark.unit
@pytest.mark.django_db
class TestValidateUsesProtocolModelString:
    @pytest.mark.parametrize(
        "model,expected_model,expected_provider",
        [
            (
                "openai/gpt-5.4-nano/d688d",
                "openai/responses/openai/openai/gpt-5.4-nano/d688d",
                "openai",
            ),
            (
                "anthropic/claude-sonnet-4.6/e6887",
                "anthropic/anthropic/claude-sonnet-4.6/e6887",
                "anthropic",
            ),
        ],
    )
    def test_validate_llm_config_routes_by_protocol(
        self, model, expected_model, expected_provider
    ):
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_completion_response()
            ok, message = validate_llm_config(
                "agione", _agione_config(model), user=None
            )

        assert ok is True
        assert message == ""
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == expected_model
        assert kwargs["custom_llm_provider"] == expected_provider

    def test_get_litellm_params_routes_from_db_config(self):
        from agentcore_metering.adapters.django.services import (
            get_litellm_params,
        )

        LLMConfig.objects.create(
            scope=LLMConfig.Scope.GLOBAL,
            user=None,
            model_type=LLMConfig.MODEL_TYPE_LLM,
            provider="agione",
            config=_agione_config("openai/gpt-5.4-nano/d688d"),
            is_active=True,
        )

        params = get_litellm_params()

        assert params["model"] == (
            "openai/responses/openai/openai/gpt-5.4-nano/d688d"
        )
        assert params["custom_llm_provider"] == "openai"
