"""
Tests for get_litellm_params: config validation and ValueError paths.

Config is read from DB only; tests create a global LLMConfig so that
validation (e.g. missing api_key) is exercised.
"""
import pytest

from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services import get_litellm_params
from agentcore_metering.adapters.django.services.litellm_params import (
    apply_reasoning_effort,
)


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmServiceOpenAI:
    def test_missing_api_key_raises_value_error(self):
        LLMConfig.objects.create(
            scope=LLMConfig.Scope.GLOBAL,
            user=None,
            model_type=LLMConfig.MODEL_TYPE_LLM,
            provider="openai",
            config={},
            is_active=True,
        )
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "OpenAI" in str(exc_info.value)
        err = str(exc_info.value).lower()
        assert "incomplete" in err or "api" in err


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmServiceAzureOpenAI:
    def test_empty_config_raises_value_error(self):
        LLMConfig.objects.create(
            scope=LLMConfig.Scope.GLOBAL,
            user=None,
            model_type=LLMConfig.MODEL_TYPE_LLM,
            provider="azure_openai",
            config={},
            is_active=True,
        )
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Azure" in str(exc_info.value)

    def test_missing_api_base_raises_value_error(self):
        LLMConfig.objects.create(
            scope=LLMConfig.Scope.GLOBAL,
            user=None,
            model_type=LLMConfig.MODEL_TYPE_LLM,
            provider="azure_openai",
            config={"api_key": "key", "api_base": ""},
            is_active=True,
        )
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Azure" in str(exc_info.value)

    def test_missing_api_key_raises_value_error(self):
        LLMConfig.objects.create(
            scope=LLMConfig.Scope.GLOBAL,
            user=None,
            model_type=LLMConfig.MODEL_TYPE_LLM,
            provider="azure_openai",
            config={"api_key": "", "api_base": "https://example.com"},
            is_active=True,
        )
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Azure" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmServiceGemini:
    def test_missing_api_key_raises_value_error(self):
        LLMConfig.objects.create(
            scope=LLMConfig.Scope.GLOBAL,
            user=None,
            model_type=LLMConfig.MODEL_TYPE_LLM,
            provider="gemini",
            config={},
            is_active=True,
        )
        with pytest.raises(ValueError) as exc_info:
            get_litellm_params()
        assert "Gemini" in str(exc_info.value)
        assert "incomplete" in str(exc_info.value).lower()


@pytest.mark.unit
class TestApplyReasoningEffort:
    """
    LiteLLM's DeepSeek mapping can turn thinking on but never off --
    reasoning_effort="none" is dropped, leaving DeepSeek's default (on) in
    place. These pin that "none" now actually disables it, and that nothing
    else changes shape.
    """

    def test_none_effort_disables_thinking_for_deepseek(self):
        params = {"model": "deepseek/deepseek-chat"}
        apply_reasoning_effort(params, "none")
        assert params["reasoning_effort"] == "none"
        assert params["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_other_efforts_do_not_add_a_body(self):
        params = {"model": "deepseek/deepseek-chat"}
        apply_reasoning_effort(params, "high")
        assert params["reasoning_effort"] == "high"
        assert "extra_body" not in params

    def test_absent_effort_is_a_no_op(self):
        params = {"model": "deepseek/deepseek-chat"}
        apply_reasoning_effort(params, None)
        assert params == {"model": "deepseek/deepseek-chat"}

    def test_gateway_served_deepseek_also_gets_the_body(self):
        """Reversed deliberately.

        This used to assert the gateway route was left alone, on the theory
        that forwarding a vendor field is the gateway's business. The
        consequence was that a caller's explicit "none" became a silent
        no-op there -- no way to express the request at all. Probing a live
        gateway settled it: the field is accepted, not rejected, and the
        same one-line prompt drops from completion=40 to completion=7.
        """
        params = {"model": "openai/deepseek/DeepSeek-V4-Flash/8f94e"}
        apply_reasoning_effort(params, "none")
        assert params["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_a_stricter_gateway_can_opt_back_out(self, settings):
        """The escape hatch, for a deployment whose gateway rejects unknown
        body fields -- config rather than a fork."""
        settings.AGENTCORE_DEEPSEEK_THINKING_MODEL_PREFIXES = ("deepseek/",)
        params = {"model": "openai/deepseek/DeepSeek-V4-Flash/8f94e"}
        apply_reasoning_effort(params, "none")
        assert "extra_body" not in params

        # The narrower setting must still cover the native route.
        native = {"model": "deepseek/deepseek-chat"}
        apply_reasoning_effort(native, "none")
        assert native["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_the_body_is_only_sent_when_none_was_asked_for(self):
        """Bounds the blast radius: a caller that never asked for "none"
        sends nothing new to any gateway, whatever the prefixes are."""
        for effort in ("high", "low", "medium"):
            params = {"model": "openai/deepseek/DeepSeek-V4-Flash/8f94e"}
            apply_reasoning_effort(params, effort)
            assert "extra_body" not in params, effort

    def test_other_providers_are_left_alone(self):
        params = {"model": "gpt-4o-mini"}
        apply_reasoning_effort(params, "none")
        assert "extra_body" not in params

    def test_existing_extra_body_is_preserved_and_not_overridden(self):
        params = {
            "model": "deepseek/deepseek-chat",
            "extra_body": {"thinking": {"type": "enabled"}, "foo": 1},
        }
        apply_reasoning_effort(params, "none")
        assert params["extra_body"] == {
            "thinking": {"type": "enabled"},
            "foo": 1,
        }


@pytest.mark.unit
class TestDeepseekPrefixOverride:
    """The override is deployment config, so its failure modes are config
    mistakes rather than code paths."""

    def test_a_bare_string_is_one_prefix_not_a_bag_of_characters(
        self, settings
    ):
        """tuple("deepseek/") is ("d", "e", "e", ...), and startswith on
        single characters matches nearly every model string -- the body
        would go to providers that never asked for it."""
        settings.AGENTCORE_DEEPSEEK_THINKING_MODEL_PREFIXES = "deepseek/"

        params = {"model": "gpt-4o-mini"}
        apply_reasoning_effort(params, "none")
        assert "extra_body" not in params, (
            "a non-DeepSeek model matched a character of the prefix string"
        )

        deepseek = {"model": "deepseek/deepseek-chat"}
        apply_reasoning_effort(deepseek, "none")
        assert deepseek["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_an_empty_override_disables_the_body_everywhere(self, settings):
        settings.AGENTCORE_DEEPSEEK_THINKING_MODEL_PREFIXES = ()
        params = {"model": "deepseek/deepseek-chat"}
        apply_reasoning_effort(params, "none")
        assert "extra_body" not in params
        assert params["reasoning_effort"] == "none"
