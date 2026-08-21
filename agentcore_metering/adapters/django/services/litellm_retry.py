"""Deterministic, per-call retry handling around LiteLLM completions."""

import email.utils
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, Optional

from agentcore_metering.constants import LITELLM_NUM_RETRIES

logger = logging.getLogger(__name__)

RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_MAX_DELAY_SECONDS = 8.0
RETRY_JITTER_SECONDS = 0.25
_DISABLED_LITELLM_RETRY_POLICY = {
    "AuthenticationErrorRetries": 0,
    "BadRequestErrorRetries": 0,
    "ContentPolicyViolationErrorRetries": 0,
    "InternalServerErrorRetries": 0,
    "RateLimitErrorRetries": 0,
    "TimeoutErrorRetries": 0,
}

_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        "TimeoutError",
    }
)
_PERMANENT_EXCEPTION_NAMES = frozenset(
    {
        "AuthenticationError",
        "BadRequestError",
        "ContentPolicyViolationError",
        "NotFoundError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
        "ValidationError",
    }
)


def _status_code(exc: Exception) -> Optional[int]:
    value = getattr(exc, "status_code", None)
    if value is None:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable_exception(exc: Exception) -> bool:
    """Return whether a provider failure is safe to retry."""
    type_name = type(exc).__name__
    if type_name in _PERMANENT_EXCEPTION_NAMES:
        return False

    status_code = _status_code(exc)
    if status_code is not None:
        return status_code == 408 or status_code == 429 or status_code >= 500

    if type_name in _TRANSIENT_EXCEPTION_NAMES:
        return True

    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, Exception) and cause is not exc:
        return is_retryable_exception(cause)
    return False


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None

    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None

    try:
        seconds = float(value)
        return max(0.0, seconds)
    except (TypeError, ValueError):
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(str(value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (retry_at - datetime.now(timezone.utc)).total_seconds(),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def calculate_retry_delay(
    exc: Exception,
    retry_number: int,
    *,
    remaining_timeout: Optional[float] = None,
) -> float:
    """Return a jittered, bounded delay for the next retry attempt."""
    retry_after = _retry_after_seconds(exc)
    if retry_after is None:
        delay = RETRY_BASE_DELAY_SECONDS * (2 ** max(0, retry_number))
    else:
        delay = retry_after
    delay = min(delay, RETRY_MAX_DELAY_SECONDS)
    delay += random.uniform(0.0, RETRY_JITTER_SECONDS)
    delay = min(delay, RETRY_MAX_DELAY_SECONDS)
    if remaining_timeout is not None:
        delay = min(delay, max(0.0, remaining_timeout))
    return max(0.0, delay)


def _configured_retries(params: Dict[str, Any]) -> int:
    value = params.get("num_retries", LITELLM_NUM_RETRIES)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return LITELLM_NUM_RETRIES


def _deadline(params: Dict[str, Any]) -> Optional[float]:
    timeout = params.get("timeout")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return None
    if timeout <= 0:
        return None
    return time.monotonic() + float(timeout)


def _remaining_timeout(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _attempt_params(
    params: Dict[str, Any], remaining_timeout: Optional[float]
) -> Dict[str, Any]:
    attempt = dict(params)
    # Agentcore owns the retry loop. Explicitly disable both LiteLLM's wrapper
    # retries and provider-SDK retries so process-global state cannot add hidden
    # attempts or replay a stream after content has escaped.
    attempt["num_retries"] = 0
    attempt["max_retries"] = 0
    attempt["retry_policy"] = dict(_DISABLED_LITELLM_RETRY_POLICY)
    if remaining_timeout is not None:
        attempt["timeout"] = max(0.001, remaining_timeout)
    return attempt


def _wait_for_retry(
    exc: Exception,
    retry_number: int,
    deadline: Optional[float],
    params: Dict[str, Any],
) -> bool:
    remaining = _remaining_timeout(deadline)
    if remaining is not None and remaining <= 0:
        return False
    delay = calculate_retry_delay(
        exc,
        retry_number,
        remaining_timeout=remaining,
    )
    if remaining is not None and delay >= remaining:
        return False
    logger.warning(
        "Retrying LiteLLM call model=%s attempt=%s delay=%.3fs error=%s",
        params.get("model", "unknown"),
        retry_number + 2,
        delay,
        type(exc).__name__,
    )
    if delay > 0:
        time.sleep(delay)
    return True


def completion_with_retry(
    completion: Callable[..., Any],
    params: Dict[str, Any],
    *,
    on_attempt_error: Optional[Callable[[Exception, int], None]] = None,
) -> Any:
    """Call a non-streaming LiteLLM completion within one timeout budget.

    on_attempt_error, if given, fires once per attempt that fails but is
    about to be retried (never for the final attempt that raises out —
    callers already observe that via the exception itself). A failed
    attempt may still have reached the provider and been billed even
    though this process never saw a response for it; without this hook,
    a call that ultimately succeeds on a later attempt leaves zero trace
    of the earlier billed-but-lost attempt anywhere (confirmed live:
    10 DeepSeek requests billed on 2026-08-20 with no matching row —
    success or failure — in llm_tracker_usage). Exceptions raised by the
    callback itself are logged and swallowed, never allowed to break an
    otherwise-successful retry.
    """
    retries = _configured_retries(params)
    deadline = _deadline(params)
    for attempt_number in range(retries + 1):
        remaining = _remaining_timeout(deadline)
        try:
            return completion(**_attempt_params(params, remaining))
        except Exception as exc:
            will_retry = (
                attempt_number < retries
                and is_retryable_exception(exc)
                and _wait_for_retry(
                    exc,
                    attempt_number,
                    deadline,
                    params,
                )
            )
            if not will_retry:
                raise
            if on_attempt_error is not None:
                try:
                    on_attempt_error(exc, attempt_number)
                except Exception:
                    logger.exception(
                        "on_attempt_error callback failed; continuing retry"
                    )
    raise RuntimeError("unreachable retry state")


def iter_completion_with_retry(
    completion: Callable[..., Any],
    params: Dict[str, Any],
    *,
    has_emitted: Callable[[], bool],
    on_retry: Optional[Callable[[], None]] = None,
    on_attempt_error: Optional[Callable[[Exception, int], None]] = None,
) -> Generator[Any, None, None]:
    """Yield stream chunks, replaying only before caller-visible output.

    on_attempt_error: see completion_with_retry — same billed-but-lost-
    attempt audit trail hook, fired once per retried attempt. Distinct
    from on_retry (which resets caller-side stream-accumulation state);
    both fire together when an attempt is retried.
    """
    retries = _configured_retries(params)
    deadline = _deadline(params)
    for attempt_number in range(retries + 1):
        remaining = _remaining_timeout(deadline)
        try:
            response = completion(**_attempt_params(params, remaining))
            yield from response
            return
        except Exception as exc:
            will_retry = (
                not has_emitted()
                and attempt_number < retries
                and is_retryable_exception(exc)
                and _wait_for_retry(
                    exc,
                    attempt_number,
                    deadline,
                    params,
                )
            )
            if not will_retry:
                raise
            if on_attempt_error is not None:
                try:
                    on_attempt_error(exc, attempt_number)
                except Exception:
                    logger.exception(
                        "on_attempt_error callback failed; continuing retry"
                    )
            if on_retry is not None:
                on_retry()
