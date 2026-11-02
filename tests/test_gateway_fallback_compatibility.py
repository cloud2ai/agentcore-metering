"""Keep legacy error formats and stream replay protections intact."""

import json
from unittest.mock import patch

import httpx
import pytest

from agentcore_metering.adapters.django import LLMTracker
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage
from agentcore_metering.adapters.django.services.fallback_config import (
    save_policy,
)
from agentcore_metering.adapters.django.services.model_fallback import (
    FallbackTarget,
    execute_fallbacks,
    fallback_reason,
)
from tests.test_fallback_execution import stream_chunk
from tests.test_gateway_error_fallback import gateway_body, sdk_error


@pytest.mark.parametrize("body", [
    {"detail": "quota exhausted"},
    [{"detail": "insufficient_quota"}],
    "insufficient balance",
])
def test_legacy_raw_response_formats_keep_their_quota_signal(body):
    request = httpx.Request("POST", "https://fixture.invalid/completions")
    response = httpx.Response(502, request=request, json=body)
    error = httpx.HTTPStatusError(
        "Unavailable", request=request, response=response,
    )
    assert fallback_reason(error) == "quota_exhausted"


@pytest.mark.parametrize("body", [
    json.dumps("insufficient balance"),
    json.dumps([{"detail": "quota exhausted"}]),
    {"message": {"detail": "insufficient_quota"}},
])
def test_legacy_sdk_body_formats_keep_their_quota_signal(body):
    assert fallback_reason(sdk_error(body)) == "quota_exhausted"


@pytest.mark.parametrize("on_response", [False, True])
def test_legacy_raw_policy_signal_still_wins_over_quota(on_response):
    body = {
        "error": {"code": "insufficient_quota"},
        "detail": "content_policy_violation",
    }
    if on_response:
        request = httpx.Request("POST", "https://fixture.invalid/completions")
        response = httpx.Response(502, request=request, json=body)
        error = httpx.HTTPStatusError(
            "Unavailable", request=request, response=response,
        )
    else:
        error = sdk_error(json.dumps(body))
    assert fallback_reason(error) is None


@pytest.mark.parametrize("encoded", [False, True])
def test_parameter_named_insufficient_balance_is_not_a_quota_error(encoded):
    body = {"error": {
        "code": "invalid_request_error",
        "message": "Invalid question name: insufficient_balance",
    }}
    assert fallback_reason(sdk_error(
        json.dumps(body) if encoded else body, 400,
    )) is None


def test_nested_quota_only_switches_when_the_condition_is_enabled():
    calls = []
    error = sdk_error(gateway_body())

    def call(target, timeout, attempt):
        calls.append(target.config_uuid)
        raise error

    with pytest.raises(type(error)) as caught:
        execute_fallbacks(
            [FallbackTarget("primary", 30), FallbackTarget("backup", 30)],
            call, timeout_seconds=60, conditions=["timeout"],
        )
    assert caught.value is error
    assert calls == ["primary"]


@pytest.mark.django_db
def test_nested_quota_without_a_backup_policy_keeps_legacy_behavior():
    primary = LLMConfig.objects.create(
        scope="global", provider="openai_compatible", config={
            "model": "legacy-primary", "api_key": "fixture-key",
            "api_base": "https://fixture.invalid/v1", "num_retries": 0,
        },
    )
    error = sdk_error(gateway_body())
    with patch("litellm.completion", side_effect=error) as request:
        with pytest.raises(type(error)) as caught:
            LLMTracker.call_and_track(
                messages=[{"role": "user", "content": "Test"}],
                model_uuid=str(primary.uuid),
            )
    assert caught.value is error
    assert request.call_count == 1
    assert not getattr(error, "agentcore_no_retry", False)
    row = LLMUsage.objects.get()
    assert not row.success and row.cost is None
    assert "fallback_call_id" not in row.metadata


@pytest.mark.django_db
@pytest.mark.parametrize("kind", ["content", "reasoning"])
def test_nested_quota_after_stream_output_never_replays(kind):
    primary, backup = [
        LLMConfig.objects.create(
            scope="global", provider="openai_compatible", config={
                "model": name, "api_key": "fixture-key",
                "api_base": "https://fixture.invalid/v1",
            },
        )
        for name in ("stream-primary", "stream-backup")
    ]
    primary = save_policy(primary, {
        "model_uuids": [str(backup.uuid)],
        "conditions": ["quota_exhausted"],
    })
    error = sdk_error(gateway_body())

    def interrupted():
        chunk = stream_chunk("partial", None, usage=False)
        if kind == "reasoning":
            chunk.choices[0].delta.content = None
            chunk.choices[0].delta.reasoning_content = "partial"
        yield chunk
        raise error

    with patch("litellm.completion", return_value=interrupted()) as request:
        stream = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "Test"}],
            model_uuid=str(primary.uuid), stream=True,
        )
        assert next(stream) == (kind, "partial")
        with pytest.raises(type(error)) as caught:
            next(stream)
    assert caught.value is error
    assert request.call_count == 1
    row = LLMUsage.objects.get()
    assert not row.success
    assert row.metadata["usage_may_be_incomplete"] is True
