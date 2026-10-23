"""Bounded model failover policy, independent of transport and metering.

Callers own parameter adaptation and one usage record per provider attempt.
This module does not persist usage or replay emitted responses.
"""

import math
import time
from dataclasses import dataclass
from uuid import uuid4

import httpx

MAX_FALLBACK_MODELS = 3
MAX_TOTAL_TIMEOUT_SECONDS = 3600


class EmptyResponseError(RuntimeError):
    """A transport completed without an answer, refusal or truncation."""


class FallbackUnavailableError(RuntimeError):
    """No model can be attempted under the current cooldown policy."""


_POLICY_ERRORS = {
    "ContentPolicyViolationError",
    "ModelRefusalError",
    "PermissionDeniedError",
    "AuthenticationError",
}
_VALIDATION_ERRORS = {
    "BadRequestError",
    "ValidationError",
    "UnprocessableEntityError",
    "NotFoundError",
}
_QUOTA_MARKERS = (
    "insufficient_quota",
    "quota_exceeded",
    "quota exhausted",
    "insufficient balance",
    "insufficient credits",
    "credit balance",
    "billing_hard_limit_reached",
    "余额不足",
    "额度不足",
    "配额耗尽",
)
_POLICY_MARKERS = (
    "content_policy_violation",
    "content_filter",
    "model_refusal",
)


def _error_text(exc):
    """Inspect bounded error fields locally; never copy them to metadata."""
    parts = [str(exc)[:4096]]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            parts.extend(
                str(error.get(key, ""))[:1024]
                for key in (
                    "code",
                    "type",
                    "message",
                )
            )
    elif isinstance(body, str):
        parts.append(body[:4096])
    # HTTPX status errors keep gateway error details on the response,
    # not on exc.body. A 400/5xx wrapper must not hide exhausted quota.
    response = getattr(exc, "response", None)
    try:
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            parts.append(content[:4096].decode("utf-8", errors="replace"))
    except (httpx.ResponseNotRead, RuntimeError):
        pass
    for key in ("code", "type"):
        value = getattr(exc, key, None)
        if isinstance(value, str):
            parts.append(value[:1024])
    return " ".join(parts).lower()


def fallback_reason(exc):
    """Return a safe reason code, or None for a non-failover error.

    Refusal and validation failures must not be worked around by selecting
    a different model. A provider's explicit quota error can use a 400 or
    5xx status; it should switch once rather than retry that provider.
    """
    seen = set()
    chain = []
    while isinstance(exc, Exception) and id(exc) not in seen:
        seen.add(id(exc))
        chain.append((exc, _error_text(exc)))
        exc = getattr(exc, "__cause__", None)
    # A wrapper's HTTP status must not hide a policy failure in its cause.
    if any(
        type(error).__name__ in _POLICY_ERRORS
        or any(marker in text for marker in _POLICY_MARKERS)
        for error, text in chain
    ):
        return None
    for exc, text in chain:
        kind = type(exc).__name__
        if isinstance(exc, EmptyResponseError):
            return "empty_response"
        if isinstance(exc, (ValueError, TypeError)):
            return None
        if any(marker in text for marker in _QUOTA_MARKERS):
            return "quota_exhausted"
        if kind in _VALIDATION_ERRORS:
            return None
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):
            status = None
        if status == 402:
            return "quota_exhausted"
        if status == 429:
            return "rate_limit"
        if (
            status in (408, 504)
            or isinstance(
                exc,
                (
                    TimeoutError,
                    httpx.TimeoutException,
                ),
            )
            or kind in {"Timeout", "APITimeoutError"}
        ):
            return "timeout"
        if (
            isinstance(
                exc,
                (
                    ConnectionError,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ),
            )
            or kind == "APIConnectionError"
        ):
            return "connection_error"
        if status is not None:
            return "upstream_error" if 500 <= status < 600 else None
        if kind in {"InternalServerError", "ServiceUnavailableError"}:
            return "upstream_error"
        if kind == "RateLimitError":
            return "rate_limit"
    return None


def _positive_timeout(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= MAX_TOTAL_TIMEOUT_SECONDS
    ):
        raise ValueError("Fallback timeout must be in (0, 3600] seconds.")
    return float(value)


@dataclass(frozen=True)
class FallbackTarget:
    """A resolved configuration identity and its maximum request duration."""

    config_uuid: str
    timeout_seconds: float


@dataclass(frozen=True)
class FallbackAttempt:
    """Safe provenance passed to the existing per-attempt usage recorder."""

    call_id: str
    requested_config_uuid: str
    actual_config_uuid: str
    attempt_number: int
    reason: str | None
    model_attempt: int = 1

    def metadata(self):
        return {
            "fallback_call_id": self.call_id,
            "requested_config_uuid": self.requested_config_uuid,
            "actual_config_uuid": self.actual_config_uuid,
            "fallback_attempt": self.attempt_number,
            "fallback_reason": self.reason,
            "fallback_model_attempt": self.model_attempt,
        }


def execute_fallbacks(
    targets,
    call,
    *,
    timeout_seconds,
    conditions=None,
    retries=0,
    availability=None,
    eligible=None,
):
    """Attempt a bounded model list within one shared time budget.

    ``call(target, timeout, attempt)`` must disable nested transport retries,
    record its own actual usage, and return only a completed result. Do not
    use this function to replay a stream after any output has been emitted.
    Backup lists are flat: a backup's own fallback list is not expanded.
    """
    targets = tuple(targets)
    if type(retries) is not int or not 0 <= retries <= 1:
        raise ValueError("Fallback retries must be zero or one.")
    allowed = set(conditions) if conditions is not None else None
    if not 1 <= len(targets) <= MAX_FALLBACK_MODELS + 1:
        raise ValueError("Select one primary and at most three backups.")
    identities = [target.config_uuid for target in targets]
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("Model configuration identities must be non-empty.")
    if len(set(identities)) != len(identities):
        raise ValueError("A model may only appear once in a fallback call.")
    for target in targets:
        _positive_timeout(target.timeout_seconds)
    deadline = time.monotonic() + _positive_timeout(timeout_seconds)
    call_id = str(uuid4())
    reason = None
    last_error = None
    attempt_number = 0
    for index, target in enumerate(targets):
        if eligible is not None and not eligible(target):
            continue
        if availability and not availability.allow(target, call_id):
            continue
        for retry in range(retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Model fallback time budget exhausted.",
                ) from last_error
            remaining_limits = target.timeout_seconds * (
                retries + 1 - retry
            ) + sum(item.timeout_seconds for item in targets[index + 1 :]) * (
                retries + 1
            )
            timeout = min(
                target.timeout_seconds,
                remaining * target.timeout_seconds / remaining_limits,
            )
            attempt_number += 1
            attempt = FallbackAttempt(
                call_id,
                targets[0].config_uuid,
                target.config_uuid,
                attempt_number,
                reason,
                retry + 1,
            )
            try:
                result = call(target, timeout, attempt)
            except Exception as exc:
                reason = fallback_reason(exc)
                if reason is None or (
                    allowed is not None and reason not in allowed
                ):
                    raise
                last_error = exc
                # No retry can restore exhausted funds on this provider.
                if retry < retries and reason != "quota_exhausted":
                    continue
                if availability:
                    availability.failure(target, reason)
                break
            else:
                if availability:
                    availability.success(target)
                return result
    if last_error is not None:
        raise last_error
    raise FallbackUnavailableError(
        "Configured models are cooling down; retry later or reset their state."
    )
