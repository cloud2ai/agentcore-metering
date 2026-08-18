"""
Tests for trackers.llm.LLMTracker (LiteLLM): call_and_track exception paths.
"""
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentcore_metering.adapters.django import LLMTracker
from agentcore_metering.adapters.django.trackers.llm import (
    _assistant_message_payload,
)


def test_assistant_message_payload_preserves_reasoning_content():
    message = SimpleNamespace(
        content="answer",
        reasoning_content="complete reasoning",
        tool_calls=[],
    )

    payload = _assistant_message_payload(message, "stop")

    assert payload["reasoning_content"] == "complete reasoning"


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

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_empty_response_persists_raw_reasoning_for_diagnosis(
        self, mock_params, mock_litellm, mock_save_usage
    ):
        """The exception message only carries reasoning_len (a count) —
        the raw text has to survive into the persisted usage row, or a
        future failure is exactly as undiagnosable as this one was."""
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        msg = MagicMock()
        msg.content = ""
        msg.reasoning_content = "the model thought about it and stopped"
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        mock_litellm.completion.return_value = MagicMock(
            choices=[choice],
            usage=MagicMock(
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
            ),
            model="gpt-4",
        )

        with pytest.raises(ValueError):
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}]
            )

        saved_state = mock_save_usage.call_args.kwargs["state"]
        diagnostics = saved_state["metadata"]["empty_response_diagnostics"]
        assert diagnostics["finish_reason"] == "stop"
        assert diagnostics["reasoning_content"] == (
            "the model thought about it and stopped"
        )
        assert diagnostics["content_repr"] == repr("")

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_empty_response_reasoning_capture_is_truncated(
        self, mock_params, mock_litellm, mock_save_usage
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        msg = MagicMock()
        msg.content = ""
        msg.reasoning_content = "x" * 10_000
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "length"
        mock_litellm.completion.return_value = MagicMock(
            choices=[choice],
            usage=MagicMock(
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
            ),
            model="gpt-4",
        )

        with pytest.raises(ValueError):
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}]
            )

        saved_state = mock_save_usage.call_args.kwargs["state"]
        diagnostics = saved_state["metadata"]["empty_response_diagnostics"]
        assert len(diagnostics["reasoning_content"]) == 4000


@pytest.mark.unit
class TestCallAndTrackReasoningContentRecovery:
    """
    #223: deepseek-v4-flash sometimes writes its full JSON answer into
    reasoning_content and leaves content empty under forced json_object
    mode, reporting finish_reason='stop' as if nothing were wrong. When
    that reasoning text is itself valid JSON, recover it instead of
    failing the call — but only under json_mode, where a JSON object is
    actually expected.
    """

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_json_mode_recovers_valid_json_from_reasoning_content(
        self, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {"model": "deepseek/deepseek-v4-flash"}
        answer = '{"verdict": "rejected", "confidence": 0.95}'
        message = SimpleNamespace(content="", reasoning_content=answer)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        mock_completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=0, total_tokens=10,
            ),
            model="deepseek/deepseek-v4-flash",
            _hidden_params={},
        )

        content, usage = LLMTracker.call_and_track(
            messages=[{"role": "system", "content": "review this"}],
            json_mode=True,
        )

        assert json.loads(content) == json.loads(answer)
        saved_kwargs = mock_save_usage.call_args.kwargs
        assert saved_kwargs["success"] is True
        assert saved_kwargs["state"]["metadata"][
            "recovered_from_reasoning_content"
        ] is True

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_json_mode_recovery_reaches_return_message_payload(
        self, mock_params, mock_completion, mock_save_usage
    ):
        """return_message must reflect the recovered content, not the raw
        (empty) message.content — it builds its payload straight from the
        response message, so recovery has to be threaded through
        explicitly or this path silently loses it."""
        mock_params.return_value = {"model": "deepseek/deepseek-v4-flash"}
        answer = '{"verdict": "passed", "confidence": 0.9}'
        message = SimpleNamespace(content="", reasoning_content=answer)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        mock_completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=0, total_tokens=10,
            ),
            model="deepseek/deepseek-v4-flash",
            _hidden_params={},
        )

        payload, _usage = LLMTracker.call_and_track(
            messages=[{"role": "system", "content": "review this"}],
            json_mode=True,
            json_repair=False,
            return_message=True,
        )

        assert json.loads(payload["content"]) == json.loads(answer)

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_non_json_mode_does_not_recover_from_reasoning_content(
        self, mock_params, mock_completion, mock_save_usage
    ):
        """Outside json_mode, reasoning_content is genuine chain-of-thought
        prose, not a misplaced answer — even if it happens to parse as
        JSON, treating it as the response would smuggle unvetted text past
        a caller that never asked for structured output."""
        mock_params.return_value = {"model": "deepseek/deepseek-v4-flash"}
        message = SimpleNamespace(
            content="", reasoning_content='{"looks": "like json"}'
        )
        choice = SimpleNamespace(message=message, finish_reason="stop")
        mock_completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=0, total_tokens=10,
            ),
            model="deepseek/deepseek-v4-flash",
            _hidden_params={},
        )

        with pytest.raises(ValueError, match="empty"):
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}],
                json_mode=False,
            )

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_json_mode_does_not_recover_on_length_truncation(
        self, mock_params, mock_completion, mock_save_usage
    ):
        """finish_reason='length' (#177 — output budget exhausted) must
        never go through recovery, even when the truncated reasoning text
        happens to repair into syntactically valid JSON: json_repair closes
        brackets around whatever fields landed before the cutoff, which can
        fabricate a complete-looking {"verdict": "passed", "confidence":
        0.95} out of a review that never finished. Confirmed by hand:
        repair_json('{"verdict": "passed", "confidence": 0.95, "reason":
        "cle') returns exactly that, dropping nothing that looks wrong.
        Only finish_reason='stop' (model claims it finished on its own) is
        trusted enough to recover from."""
        mock_params.return_value = {"model": "deepseek/deepseek-v4-flash"}
        truncated_but_parseable = (
            '{"verdict": "passed", "confidence": 0.95, "reason": "cle'
        )
        message = SimpleNamespace(
            content="", reasoning_content=truncated_but_parseable
        )
        choice = SimpleNamespace(message=message, finish_reason="length")
        mock_completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=0, total_tokens=10,
            ),
            model="deepseek/deepseek-v4-flash",
            _hidden_params={},
        )

        with pytest.raises(ValueError, match="empty"):
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}],
                json_mode=True,
            )

        saved_kwargs = mock_save_usage.call_args.kwargs
        assert saved_kwargs["success"] is False

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_json_mode_does_not_recover_non_json_reasoning_content(
        self, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {"model": "deepseek/deepseek-v4-flash"}
        message = SimpleNamespace(
            content="",
            reasoning_content="the model thought about it and stopped",
        )
        choice = SimpleNamespace(message=message, finish_reason="stop")
        mock_completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=0, total_tokens=10,
            ),
            model="deepseek/deepseek-v4-flash",
            _hidden_params={},
        )

        with pytest.raises(ValueError, match="empty"):
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}],
                json_mode=True,
            )

        saved_kwargs = mock_save_usage.call_args.kwargs
        assert saved_kwargs["success"] is False


@pytest.mark.unit
class TestCallAndTrackToolCalling:
    """
    call_and_track forwards tools/tool_choice and returns parsed tool calls.
    """

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_tools_forwarded_and_tool_calls_returned(
        self, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        tool_call = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"city": "SF"}',
            ),
        )
        # Empty content with a tool call must not raise "empty response".
        message = SimpleNamespace(content="", tool_calls=[tool_call])
        choice = SimpleNamespace(
            message=message,
            finish_reason="tool_calls",
        )
        mock_completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model="gpt-4",
            _hidden_params={},
        )

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        message_payload, usage_dict = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools,
            tool_choice="auto",
            return_message=True,
        )

        # tools/tool_choice are forwarded to litellm.completion.
        completion_kwargs = mock_completion.call_args.kwargs
        assert completion_kwargs["tools"] == tools
        assert completion_kwargs["tool_choice"] == "auto"

        # return_message yields an assistant payload with parsed tool calls.
        assert message_payload["role"] == "assistant"
        assert len(message_payload["tool_calls"]) == 1
        parsed = message_payload["tool_calls"][0]
        assert parsed["id"] == "call_1"
        assert parsed["function"]["name"] == "get_weather"
        assert parsed["function"]["arguments"] == '{"city": "SF"}'
        assert message_payload["finish_reason"] == "tool_calls"
        assert usage_dict["total_tokens"] == 15


@pytest.mark.unit
class TestCallAndTrackMetadata:
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_non_stream_forwards_litellm_metadata(
        self, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {
            "model": "gpt-4",
            "api_key": "sk-x",
            "metadata": {"existing": "value"},
            "proxy_server_request": {
                "headers": {"x-request-id": "request-1"},
            },
        }
        mock_completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
            model="gpt-4",
            _hidden_params={},
        )

        LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            state={
                "litellm_metadata": {
                    "session_id": "session-1",
                    "trace_metadata": {"run_uuid": "run-1"},
                },
                "otel_traceparent": (
                    "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-"
                    "bbbbbbbbbbbbbbbb-01"
                ),
            },
        )

        assert mock_completion.call_args.kwargs["metadata"] == {
            "existing": "value",
            "session_id": "session-1",
            "trace_metadata": {"run_uuid": "run-1"},
        }
        assert mock_completion.call_args.kwargs[
            "proxy_server_request"
        ] == {
            "headers": {
                "x-request-id": "request-1",
                "traceparent": (
                    "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-"
                    "bbbbbbbbbbbbbbbb-01"
                ),
            }
        }


@pytest.mark.unit
class TestCallAndTrackUsageExtraction:
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
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
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
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


@pytest.mark.unit
class TestCallAndTrackTokenFallback:
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm_usage.token_counter"
    )
    def test_sync_fallback_uses_token_counter_when_usage_is_zero(
        self, mock_token_counter, mock_params, mock_litellm, mock_save_usage
    ):
        def _side_effect(
            *,
            model,
            custom_tokenizer=None,
            text=None,
            messages=None,
            **_kwargs,
        ):
            if messages is not None:
                return 5
            if text is not None:
                return 7
            return 0

        mock_token_counter.side_effect = _side_effect
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        message = SimpleNamespace(content="hello")
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

        assert content == "hello"
        assert usage_dict["prompt_tokens"] == 5
        assert usage_dict["completion_tokens"] == 7
        assert usage_dict["total_tokens"] == 12
        save_kwargs = mock_save_usage.call_args.kwargs
        assert save_kwargs["prompt_tokens"] == 5
        assert save_kwargs["completion_tokens"] == 7
        assert save_kwargs["total_tokens"] == 12

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm_usage.token_counter"
    )
    def test_sync_fallback_guards_min_completion_token_when_token_counter_fails(
        self,
        mock_token_counter,
        mock_params,
        mock_litellm,
        mock_save_usage,
    ):
        def _side_effect(
            *,
            model,
            custom_tokenizer=None,
            text=None,
            messages=None,
            **_kwargs,
        ):
            raise RuntimeError("boom")

        mock_token_counter.side_effect = _side_effect
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        message = SimpleNamespace(content="hello")
        choice = SimpleNamespace(message=message)
        mock_litellm.completion.return_value = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model="gpt-4",
            _hidden_params={},
        )

        _content, usage_dict = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}]
        )

        assert usage_dict["completion_tokens"] == 1
        assert usage_dict["total_tokens"] == 1


@pytest.mark.unit
class TestCallAndTrackJsonRepairRetry:
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.time.sleep"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.timezone.now"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_json_mode_retries_and_succeeds(
        self,
        mock_params,
        mock_litellm,
        mock_save_usage,
        mock_now,
        mock_sleep,
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        mock_now.side_effect = [
            datetime(2026, 3, 17, 10, 0, 0),
            datetime(2026, 3, 17, 10, 0, 1),
        ]

        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        choice_bad = SimpleNamespace(message=SimpleNamespace(content=":"))
        choice_ok = SimpleNamespace(
            message=SimpleNamespace(content='{"cleaned_content":"ok"}')
        )
        response_bad = SimpleNamespace(
            choices=[choice_bad],
            usage=usage,
            model="gpt-4",
            _hidden_params={},
        )
        response_ok = SimpleNamespace(
            choices=[choice_ok],
            usage=usage,
            model="gpt-4",
            _hidden_params={},
        )
        mock_litellm.completion.side_effect = [response_bad, response_ok]

        content, _usage = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )

        assert '"cleaned_content"' in content
        assert mock_litellm.completion.call_count == 2
        assert mock_save_usage.call_count == 2
        mock_sleep.assert_called_once_with(0.5)
        first_started_at = mock_save_usage.call_args_list[0].kwargs[
            "started_at"
        ]
        second_started_at = mock_save_usage.call_args_list[1].kwargs[
            "started_at"
        ]
        assert first_started_at != second_started_at

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.time.sleep"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_json_mode_raises_after_retry_exhausted(
        self,
        mock_params,
        mock_litellm,
        mock_save_usage,
        mock_sleep,
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        choice_bad = SimpleNamespace(message=SimpleNamespace(content=":"))
        response_bad = SimpleNamespace(
            choices=[choice_bad],
            usage=usage,
            model="gpt-4",
            _hidden_params={},
        )
        mock_litellm.completion.side_effect = [
            response_bad,
            response_bad,
            response_bad,
        ]

        with pytest.raises(ValueError) as exc_info:
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}],
                json_mode=True,
            )
        assert "Invalid JSON response after 3 attempts" in str(exc_info.value)
        assert mock_litellm.completion.call_count == 3
        assert mock_save_usage.call_count == 3
        assert mock_sleep.call_count == 2


@pytest.mark.unit
class TestCallAndTrackStreaming:
    """
    call_and_track(stream=True) yields chunks and calls _save_usage_to_db with
    is_streaming=True and first_chunk_at when first non-empty content arrives.
    """

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_stream_true_yields_chunks_and_saves_with_is_streaming(
        self, mock_params, mock_litellm, mock_save_usage
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        delta1 = SimpleNamespace(content="Hello")
        delta2 = SimpleNamespace(content=" world")
        choice1 = SimpleNamespace(delta=delta1)
        choice2 = SimpleNamespace(delta=delta2)
        usage_ns = SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )
        chunk1 = SimpleNamespace(choices=[choice1], usage=None, model="gpt-4")
        chunk2 = SimpleNamespace(
            choices=[choice2], usage=usage_ns, model="gpt-4"
        )
        mock_litellm.completion.return_value = iter([chunk1, chunk2])

        gen = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        chunks = []
        try:
            while True:
                chunks.append(next(gen))
        except StopIteration as e:
            usage_return = e.value
        # The leading space in " world" must be preserved verbatim;
        # stripping it collapses inter-word spacing in the assembled answer.
        assert chunks == [("content", "Hello"), ("content", " world")]
        assert usage_return is not None
        assert usage_return.get("model") == "gpt-4"
        assert mock_save_usage.called
        save_kwargs = mock_save_usage.call_args.kwargs
        assert save_kwargs["is_streaming"] is True
        assert save_kwargs.get("first_chunk_at") is not None

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_stream_forwards_litellm_metadata(
        self, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
            model="gpt-4",
        )
        mock_completion.return_value = iter([chunk])

        generator = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            state={
                "litellm_metadata": {
                    "session_id": "session-1",
                },
                "otel_traceparent": (
                    "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-"
                    "bbbbbbbbbbbbbbbb-01"
                ),
            },
            stream=True,
        )
        list(generator)

        assert mock_completion.call_args.kwargs["metadata"] == {
            "session_id": "session-1",
        }
        assert mock_completion.call_args.kwargs["stream"] is True
        assert mock_completion.call_args.kwargs[
            "proxy_server_request"
        ]["headers"]["traceparent"] == (
            "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-"
            "bbbbbbbbbbbbbbbb-01"
        )

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_stream_returns_provider_finish_reason(
        self, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        usage = SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="partial"),
                    finish_reason="length",
                )
            ],
            usage=usage,
            model="gpt-4",
        )
        mock_completion.return_value = iter([chunk])

        gen = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            result = exc.value

        assert result["_finish_reason"] == "length"

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm_usage.token_counter"
    )
    def test_stream_fallback_counts_tokens_from_streamed_content(
        self, mock_token_counter, mock_params, mock_litellm, mock_save_usage
    ):
        def _side_effect(
            *,
            model,
            custom_tokenizer=None,
            text=None,
            messages=None,
            **_kwargs,
        ):
            if messages is not None:
                return 3
            if text is not None:
                return 8
            return 0

        mock_token_counter.side_effect = _side_effect
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        delta1 = SimpleNamespace(content="Hello")
        delta2 = SimpleNamespace(content=" world")
        choice1 = SimpleNamespace(delta=delta1)
        choice2 = SimpleNamespace(delta=delta2)
        chunk1 = SimpleNamespace(choices=[choice1], usage=None, model="gpt-4")
        chunk2 = SimpleNamespace(choices=[choice2], usage=None, model="gpt-4")
        mock_litellm.completion.return_value = iter([chunk1, chunk2])

        gen = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        try:
            while True:
                next(gen)
        except StopIteration as e:
            usage_return = e.value

        assert usage_return["prompt_tokens"] == 3
        assert usage_return["completion_tokens"] == 8
        assert usage_return["total_tokens"] == 11
        save_kwargs = mock_save_usage.call_args.kwargs
        assert save_kwargs["prompt_tokens"] == 3
        assert save_kwargs["completion_tokens"] == 8
        assert save_kwargs["total_tokens"] == 11

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.litellm"
    )
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_stream_preserves_whitespace_for_markdown(
        self, mock_params, mock_litellm, mock_save_usage
    ):
        """Whitespace-only and leading-space deltas survive verbatim.

        LLMs stream Markdown as token deltas where inter-word spaces
        arrive as leading-space chunks (" complete") and paragraph
        breaks arrive as whitespace-only chunks ("\\n\\n"). Stripping
        per chunk welds words together and drops the blank lines that
        headings/lists depend on, so the answer renders as a wall of
        unformatted text. The assembled content must match byte-for-byte.
        """
        mock_params.return_value = {"model": "gpt-4", "api_key": "sk-x"}
        deltas = [
            "Here's", " the", " complete", " summary", ":",
            "\n\n", "---", "\n\n", "## ", "1", ". Metering",
        ]
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=d))],
                usage=None,
                model="gpt-4",
            )
            for d in deltas
        ]
        mock_litellm.completion.return_value = iter(chunks)

        gen = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        streamed = "".join(
            text for kind, text in gen if kind == "content"
        )

        assert streamed == (
            "Here's the complete summary:\n\n---\n\n## 1. Metering"
        )
        # Defining symptoms of the old per-chunk strip must not reappear:
        assert " the complete summary" in streamed   # spaces kept
        assert "\n\n---\n\n## 1" in streamed          # blank lines kept


@pytest.mark.unit
class TestCallAndTrackProviderRetries:
    class TransientProviderError(Exception):
        status_code = 503

    @staticmethod
    def _response(content="ok"):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
            model="gpt-4",
            _hidden_params={},
        )

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    @patch(
        "agentcore_metering.adapters.django.services.litellm_retry.time.sleep"
    )
    def test_non_stream_retry_saves_successful_usage_once(
        self, mock_sleep, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {
            "model": "gpt-4",
            "api_key": "sk-x",
            "timeout": 30,
            "num_retries": 2,
        }
        mock_completion.side_effect = [
            self.TransientProviderError("busy"),
            self._response(),
        ]
        state = {}

        content, usage = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            state=state,
        )

        assert content == "ok"
        assert usage["total_tokens"] == 2
        assert mock_completion.call_count == 2
        # One row for the failed-but-retried first attempt (it may have
        # already been billed by the provider even though this process
        # never saw a response) plus one for the eventual success — every
        # provider-side request gets a row so billing reconciliation can
        # COUNT(*) and match the provider's bill, not just the final
        # outcome. See completion_with_retry's on_attempt_error.
        assert mock_save_usage.call_count == 2
        assert len(state["llm_calls"]) == 2
        assert state["llm_calls"][0]["success"] is False
        assert state["llm_calls"][1]["success"] is True

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    @patch(
        "agentcore_metering.adapters.django.services.litellm_retry.time.sleep"
    )
    def test_non_stream_retry_marks_attempt_row_without_leaking_metadata(
        self, mock_sleep, mock_params, mock_completion, mock_save_usage,
    ):
        """The retried-attempt row is tagged metadata["retry_attempt"]=True
        so it stays distinguishable from a genuine terminal failure (e.g.
        via .exclude(metadata__retry_attempt=True) for callers that want
        "logical call" counts instead of "provider request" counts) — and
        that tag must not leak into the final (successful) record, since
        both share the same effective_state object."""
        mock_params.return_value = {
            "model": "gpt-4",
            "api_key": "sk-x",
            "timeout": 30,
            "num_retries": 2,
        }
        mock_completion.side_effect = [
            self.TransientProviderError("busy"),
            self._response(),
        ]

        content, _usage = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
        )

        assert content == "ok"
        assert mock_save_usage.call_count == 2
        retry_call, final_call = mock_save_usage.call_args_list
        assert retry_call.kwargs["success"] is False
        assert (
            retry_call.kwargs["state"]["metadata"]["retry_attempt"] is True
        )
        assert final_call.kwargs["success"] is True
        assert "retry_attempt" not in (
            final_call.kwargs["state"].get("metadata") or {}
        )

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    @patch(
        "agentcore_metering.adapters.django.services.litellm_retry.time.sleep"
    )
    def test_retry_is_independent_of_previous_failure_in_same_process(
        self, mock_sleep, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {
            "model": "gpt-4",
            "api_key": "sk-x",
            "timeout": 30,
            "num_retries": 1,
        }
        mock_completion.side_effect = [
            self.TransientProviderError("first attempt"),
            self.TransientProviderError("first call exhausted"),
            self.TransientProviderError("second call first attempt"),
            self._response("second call succeeded"),
        ]

        with pytest.raises(self.TransientProviderError):
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "first"}]
            )

        content, _usage = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "second"}]
        )

        assert content == "second call succeeded"
        assert mock_completion.call_count == 4
        # First call_and_track(): 1 retried-attempt row + 1 terminal-
        # failure row. Second: 1 retried-attempt row + 1 success row.
        # See on_attempt_error in completion_with_retry.
        assert mock_save_usage.call_count == 4

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    @patch(
        "agentcore_metering.adapters.django.services.litellm_retry.time.sleep"
    )
    def test_stream_retries_failure_before_first_content(
        self, mock_sleep, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {
            "model": "gpt-4",
            "api_key": "sk-x",
            "timeout": 30,
            "num_retries": 2,
        }

        def failing_stream():
            raise self.TransientProviderError("busy")
            yield

        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="recovered"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
            model="gpt-4",
        )
        mock_completion.side_effect = [failing_stream(), iter([chunk])]

        chunks = list(
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )

        assert chunks == [("content", "recovered")]
        assert mock_completion.call_count == 2
        # One row for the failed-before-first-chunk retried attempt, one
        # for the eventual success. See on_attempt_error in
        # iter_completion_with_retry.
        assert mock_save_usage.call_count == 2
        assert mock_save_usage.call_args_list[0].kwargs["success"] is False
        assert mock_save_usage.call_args.kwargs["success"] is True

    @patch(
        "agentcore_metering.adapters.django.trackers.llm.LLMTracker"
        "._save_usage_to_db"
    )
    @patch("litellm.completion")
    @patch(
        "agentcore_metering.adapters.django.trackers.llm.get_litellm_params"
    )
    def test_stream_does_not_retry_failure_after_first_content(
        self, mock_params, mock_completion, mock_save_usage
    ):
        mock_params.return_value = {
            "model": "gpt-4",
            "api_key": "sk-x",
            "timeout": 30,
            "num_retries": 2,
        }

        def partially_failing_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="partial"),
                        finish_reason=None,
                    )
                ],
                usage=None,
                model="gpt-4",
            )
            raise self.TransientProviderError("busy after content")

        mock_completion.return_value = partially_failing_stream()
        stream = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

        assert next(stream) == ("content", "partial")
        with pytest.raises(self.TransientProviderError):
            next(stream)

        assert mock_completion.call_count == 1
        assert mock_save_usage.call_count == 1
        assert mock_save_usage.call_args.kwargs["success"] is False


def test_raise_friendly_noop_when_disabled():
    # Default (friendly_errors=False): no-op, original exception left to caller.
    from agentcore_metering.adapters.django.trackers.llm import _raise_friendly

    _raise_friendly(Exception("Insufficient Balance"), False)  # must not raise


def test_raise_friendly_raises_when_enabled():
    import pytest as _pytest

    from agentcore_metering.adapters.django.services.runtime_config import (
        LLMProviderError,
    )
    from agentcore_metering.adapters.django.trackers.llm import _raise_friendly

    with _pytest.raises(LLMProviderError):
        _raise_friendly(Exception("Insufficient Balance"), True)
