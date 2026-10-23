"""Transport-independent failover boundaries and budget guarantees."""

from unittest.mock import Mock

import httpx
import pytest

from agentcore_metering.adapters.django.services import model_fallback
from agentcore_metering.adapters.django.services.model_fallback import (
    FallbackTarget,
    execute_fallbacks,
    fallback_reason,
)


def http_error(status, message="Upstream unavailable"):
    request = httpx.Request("POST", "https://example.test/v1/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(message, request=request, response=response)


@pytest.mark.parametrize(
    "error, reason",
    [
        (http_error(402), "quota_exhausted"),
        (http_error(429), "rate_limit"),
        (http_error(500), "upstream_error"),
        (http_error(503), "upstream_error"),
        (http_error(504), "timeout"),
        (http_error(400, "insufficient_quota"), "quota_exhausted"),
        (http_error(502, "租户额度不足"), "quota_exhausted"),
        (httpx.ReadTimeout("Timed out"), "timeout"),
        (TimeoutError("Timed out"), "timeout"),
        (httpx.ConnectError("Connection failed"), "connection_error"),
        (httpx.RemoteProtocolError("Remote closed"), "connection_error"),
        (http_error(401), None),
        (http_error(403), None),
        (http_error(404), None),
        (http_error(400), None),
        (http_error(500, "content_policy_violation"), None),
        (http_error(500, "model_refusal: insufficient_quota"), None),
        (ValueError("Invalid article category"), None),
        (TypeError("Invalid question schema"), None),
    ],
)
def test_only_provider_availability_errors_can_fail_over(error, reason):
    assert fallback_reason(error) == reason


def test_explicit_quota_code_is_detected_without_copying_provider_message():
    error = http_error(500)
    error.body = {"error": {"code": "insufficient_quota"}}
    assert fallback_reason(error) == "quota_exhausted"


def test_policy_error_in_cause_is_not_hidden_by_gateway_500():
    error = http_error(500)
    error.__cause__ = type("ContentPolicyViolationError", (Exception,), {})()
    assert fallback_reason(error) is None


def test_wrapped_network_error_and_cyclic_cause_terminate():
    error = RuntimeError("Provider failure")
    error.__cause__ = httpx.ConnectError("Disconnected")
    assert fallback_reason(error) == "connection_error"
    error.__cause__ = error
    assert fallback_reason(error) is None


@pytest.fixture
def targets():
    return [FallbackTarget("primary", 120), FallbackTarget("backup", 120)]


def test_success_never_contacts_backup(targets):
    call = Mock(return_value=("answer", {"total_tokens": 12}))
    assert execute_fallbacks(targets, call, timeout_seconds=120) == (
        "answer",
        {"total_tokens": 12},
    )
    call.assert_called_once()
    assert call.call_args.args[0] == targets[0]


def test_refusal_payload_is_returned_without_trying_to_bypass_it(targets):
    result = [{"name": "review", "type": "refusal"}]
    call = Mock(return_value=result)
    assert execute_fallbacks(targets, call, timeout_seconds=120) is result
    call.assert_called_once()


def test_switch_once_and_keep_actual_usage_and_safe_provenance(targets):
    calls = []
    usage = {"model": "backup-model", "total_tokens": 17, "cost": None}

    def call(target, timeout, attempt):
        calls.append((target, timeout, attempt.metadata()))
        if target == targets[0]:
            raise http_error(502, "insufficient_quota secret-api-key")
        return "answer", usage

    assert execute_fallbacks(targets, call, timeout_seconds=120) == (
        "answer",
        usage,
    )
    assert [row[0] for row in calls] == targets
    assert calls[0][2]["fallback_call_id"] == (calls[1][2]["fallback_call_id"])
    assert calls[1][2]["requested_config_uuid"] == "primary"
    assert calls[1][2]["actual_config_uuid"] == "backup"
    assert calls[1][2]["fallback_attempt"] == 2
    assert calls[1][2]["fallback_reason"] == "quota_exhausted"
    assert "secret-api-key" not in str(calls)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Invalid response"),
        http_error(400),
        http_error(500, "model_refusal"),
    ],
)
def test_non_availability_error_never_calls_backup(targets, error):
    call = Mock(side_effect=error)
    with pytest.raises(type(error)) as caught:
        execute_fallbacks(targets, call, timeout_seconds=120)
    assert caught.value is error
    call.assert_called_once()


def test_exhausted_chain_raises_last_failure_without_restarting(targets):
    errors = [http_error(429), http_error(503)]
    call = Mock(side_effect=errors)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        execute_fallbacks(targets, call, timeout_seconds=120)
    assert caught.value is errors[-1]
    assert call.call_count == 2


def test_primary_timeout_reserves_time_for_backup(targets, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(model_fallback.time, "monotonic", lambda: clock[0])
    observed = []

    def call(target, timeout, attempt):
        observed.append(timeout)
        if target == targets[0]:
            clock[0] += timeout
            raise httpx.ReadTimeout("Timed out")
        return "answer"

    assert execute_fallbacks(targets, call, timeout_seconds=60) == "answer"
    assert observed == [30, 30]


def test_different_model_timeouts_get_proportional_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(model_fallback.time, "monotonic", lambda: clock[0])
    targets = [FallbackTarget("primary", 120), FallbackTarget("backup", 30)]
    observed = []

    def call(target, timeout, attempt):
        observed.append(timeout)
        if target == targets[0]:
            clock[0] += timeout
            raise httpx.ReadTimeout("Timed out")
        return "answer"

    assert execute_fallbacks(targets, call, timeout_seconds=150) == "answer"
    assert observed == [120, 30]


def test_total_deadline_never_starts_a_late_backup(targets, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(model_fallback.time, "monotonic", lambda: clock[0])

    def call(target, timeout, attempt):
        clock[0] += 61
        raise httpx.ReadTimeout("Provider exceeded timeout")

    call = Mock(side_effect=call)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        execute_fallbacks(targets, call, timeout_seconds=60)
    call.assert_called_once()


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), 3601])
def test_invalid_budget_rejected_before_request(targets, timeout):
    call = Mock()
    with pytest.raises(ValueError):
        execute_fallbacks(targets, call, timeout_seconds=timeout)
    call.assert_not_called()


@pytest.mark.parametrize(
    "targets",
    [
        [],
        [FallbackTarget("same", 120)] * 2,
        [FallbackTarget(str(index), 120) for index in range(5)],
        [FallbackTarget("primary", float("inf"))],
    ],
)
def test_invalid_chain_rejected_before_request(targets):
    call = Mock()
    with pytest.raises(ValueError):
        execute_fallbacks(targets, call, timeout_seconds=120)
    call.assert_not_called()


def test_optional_retry_is_bounded_and_quota_skips_it(targets):
    quota = RuntimeError("quota exhausted")
    call = Mock(side_effect=[TimeoutError(), quota, "backup ok"])
    assert (
        execute_fallbacks(
            targets,
            call,
            timeout_seconds=120,
            retries=1,
        )
        == "backup ok"
    )
    attempts = [item.args[2] for item in call.call_args_list]
    assert [item.actual_config_uuid for item in attempts] == [
        "primary",
        "primary",
        "backup",
    ]
    assert [item.model_attempt for item in attempts] == [1, 2, 1]


def test_unselected_condition_stops_without_a_backup(targets):
    call = Mock(side_effect=TimeoutError())
    with pytest.raises(TimeoutError):
        execute_fallbacks(
            targets, call, timeout_seconds=60, conditions=["rate_limit"]
        )
    assert call.call_count == 1


@pytest.mark.parametrize("status", [400, 500])
def test_httpx_gateway_body_distinguishes_quota_from_status(status):
    request = httpx.Request("POST", "https://gateway.invalid/v1/decisions")
    response = httpx.Response(
        status,
        request=request,
        json={
            "error": {
                "code": "insufficient_quota",
                "message": "Balance empty",
            },
        },
    )
    error = httpx.HTTPStatusError(
        "Request failed", request=request, response=response
    )
    assert fallback_reason(error) == "quota_exhausted"
