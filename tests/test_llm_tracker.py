"""
Tests for trackers.llm.LLMTracker (LiteLLM): call_and_track exception paths.
"""
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from agentcore_metering.adapters.django import LLMTracker


@pytest.mark.unit
class TestCallAndTrackValidation:
    """
    call_and_track raises ValueError for invalid input before calling LLM.
    """

    def test_empty_messages_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            LLMTracker.call_and_track(messages=[])
        err = str(exc_info.value).lower()
        assert "cannot be empty" in err or "empty" in err

    def test_none_messages_raises_value_error(self):
        with pytest.raises(ValueError):
            LLMTracker.call_and_track(messages=[])


@pytest.mark.unit
class TestCallAndTrackServiceReturnsNone:
    """
    When litellm.completion returns None, call_and_track raises ValueError.
    """

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_completion_returns_none_raises_value_error(
        self, mock_params, mock_litellm
    ):
        mock_params.return_value = {
            "model": "gpt-4", "api_key": "sk-x", "messages": []
        }
        mock_litellm.completion.return_value = None

        with pytest.raises(ValueError) as exc_info:
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}]
            )
        assert "None" in str(exc_info.value)


@pytest.mark.unit
class TestCallAndTrackEmptyResponse:
    """
    When LiteLLM returns empty response, call_and_track raises ValueError.
    """

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_empty_response_content_raises_value_error(
        self, mock_params, mock_litellm
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        msg = MagicMock()
        msg.content = ""
        choice = MagicMock()
        choice.message = msg
        mock_litellm.completion.return_value = MagicMock(
            choices=[choice],
            usage=MagicMock(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            model="gpt-4",
        )

        with pytest.raises(ValueError) as exc_info:
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}]
            )
        assert "empty" in str(exc_info.value).lower()


@pytest.mark.unit
class TestCallAndTrackUsageExtraction:
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_extracts_cached_and_reasoning_tokens_from_nested_usage_details(
        self, mock_params, mock_litellm, mock_save_usage
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=0,
            reasoning_tokens=0,
            prompt_tokens_details={"cached_tokens": 3},
            completion_tokens_details={"reasoning_tokens": 2},
        )
        message = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=message)
        mock_litellm.completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model="gpt-4",
            _hidden_params={},
        )

        content, usage_dict = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}]
        )

        assert content == "ok"
        assert usage_dict["cached_tokens"] == 3
        assert usage_dict["reasoning_tokens"] == 2
        save_kwargs = mock_save_usage.call_args.kwargs
        assert save_kwargs["cached_tokens"] == 3
        assert save_kwargs["reasoning_tokens"] == 2

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_invalid_hidden_response_cost_does_not_break_call(
        self, mock_params, mock_litellm, mock_save_usage
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        message = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=message)
        mock_litellm.completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model="gpt-4",
            _hidden_params={"response_cost": "not-a-number"},
        )

        content, usage_dict = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}]
        )

        assert content == "ok"
        assert usage_dict["cost"] is None
        save_kwargs = mock_save_usage.call_args.kwargs
        assert save_kwargs["cost"] is None

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_whitespace_only_response_raises_value_error(
        self, mock_params, mock_litellm
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        msg = MagicMock()
        msg.content = "   \n  "
        choice = MagicMock()
        choice.message = msg
        mock_litellm.completion.return_value = MagicMock(
            choices=[choice],
            usage=MagicMock(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            model="gpt-4",
        )

        with pytest.raises(ValueError) as exc_info:
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}]
            )
        assert "empty" in str(exc_info.value).lower()
