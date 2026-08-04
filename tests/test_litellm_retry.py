from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentcore_metering.adapters.django.services.litellm_retry import (
    calculate_retry_delay,
    completion_with_retry,
    iter_completion_with_retry,
)


class TransientProviderError(Exception):
    def __init__(self, status_code=503, retry_after=None):
        super().__init__(f"provider returned {status_code}")
        self.status_code = status_code
        headers = {}
        if retry_after is not None:
            headers["retry-after"] = retry_after
        self.response = SimpleNamespace(headers=headers)


class AuthenticationError(Exception):
    status_code = 401


class BadRequestError(Exception):
    status_code = 400


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [429, 502, 503])
def test_non_stream_retries_transient_error_and_disables_nested_retries(
    status_code,
):
    completion = MagicMock(side_effect=[TransientProviderError(status_code), "success"])

    with patch(
        "agentcore_metering.adapters.django.services.litellm_retry.time.sleep"
    ) as sleep:
        result = completion_with_retry(
            completion,
            {"model": "test", "timeout": 30, "num_retries": 2},
        )

    assert result == "success"
    assert completion.call_count == 2
    assert sleep.call_count == 1
    for call in completion.call_args_list:
        assert call.kwargs["num_retries"] == 0
        assert call.kwargs["max_retries"] == 0
        assert set(call.kwargs["retry_policy"].values()) == {0}


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [AuthenticationError("bad key"), BadRequestError("invalid request")],
)
def test_non_stream_does_not_retry_permanent_errors(error):
    completion = MagicMock(side_effect=error)

    with pytest.raises(type(error), match=str(error)):
        completion_with_retry(
            completion,
            {"model": "test", "timeout": 30, "num_retries": 2},
        )

    assert completion.call_count == 1


@pytest.mark.unit
def test_non_stream_retries_timeout_before_succeeding():
    completion = MagicMock(side_effect=[TimeoutError("slow"), "success"])

    with patch("agentcore_metering.adapters.django.services.litellm_retry.time.sleep"):
        result = completion_with_retry(
            completion,
            {"model": "test", "timeout": 30, "num_retries": 1},
        )

    assert result == "success"
    assert completion.call_count == 2


@pytest.mark.unit
def test_zero_retries_disables_retrying():
    completion = MagicMock(side_effect=TransientProviderError(503))

    with pytest.raises(TransientProviderError):
        completion_with_retry(
            completion,
            {"model": "test", "timeout": 30, "num_retries": 0},
        )

    assert completion.call_count == 1


@pytest.mark.unit
def test_retry_after_is_honored_and_bounded_by_remaining_timeout():
    error = TransientProviderError(429, retry_after="5")

    with patch(
        "agentcore_metering.adapters.django.services.litellm_retry.random.uniform",
        return_value=0,
    ):
        assert calculate_retry_delay(error, 0, remaining_timeout=10) == 5
        assert calculate_retry_delay(error, 0, remaining_timeout=0.25) == 0.25


@pytest.mark.unit
def test_stream_retries_iterator_failure_before_first_emitted_chunk():
    def failing_stream():
        raise TransientProviderError(502)
        yield

    completion = MagicMock(side_effect=[failing_stream(), iter(["first", "second"])])

    with patch("agentcore_metering.adapters.django.services.litellm_retry.time.sleep"):
        chunks = list(
            iter_completion_with_retry(
                completion,
                {
                    "model": "test",
                    "timeout": 30,
                    "num_retries": 2,
                    "stream": True,
                },
                has_emitted=lambda: False,
            )
        )

    assert chunks == ["first", "second"]
    assert completion.call_count == 2


@pytest.mark.unit
def test_stream_does_not_replay_after_first_emitted_chunk():
    emitted = False

    def partially_failing_stream():
        yield "first"
        raise TransientProviderError(503)

    completion = MagicMock(return_value=partially_failing_stream())
    stream = iter_completion_with_retry(
        completion,
        {
            "model": "test",
            "timeout": 30,
            "num_retries": 2,
            "stream": True,
        },
        has_emitted=lambda: emitted,
    )

    assert next(stream) == "first"
    emitted = True
    with pytest.raises(TransientProviderError):
        next(stream)

    assert completion.call_count == 1
