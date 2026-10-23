"""Execution context and tracker integration for bounded model failover."""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from agentcore_metering.adapters.django.models import LLMUsage
from agentcore_metering.adapters.django.services.fallback_config import (
    compatible,
    resolve_plan,
    same_upstream,
    visible_graph,
)
from agentcore_metering.adapters.django.services.fallback_health import (
    ModelAvailability,
)
from agentcore_metering.adapters.django.services.model_fallback import (
    FallbackTarget,
    execute_fallbacks,
    fallback_reason,
)

current_attempt = ContextVar("agentcore_model_fallback_attempt", default=None)
logger = logging.getLogger(__name__)


def is_fallback_attempt():
    context = current_attempt.get()
    return context is not None and context.timeout is not None


@dataclass
class AttemptContext:
    timeout: float | None
    metadata: dict
    usage_id: object | None = None
    usage_metadata: dict | None = None
    usage_success: bool = False
    state: dict | None = None
    model: str | None = None

    def prepare(self, params):
        if self.timeout is None:
            return {}
        if self.timeout is not None:
            params["timeout"] = min(
                params.get("timeout", self.timeout), self.timeout
            )
            params["num_retries"] = 0
        self.model = params.get("model")
        return dict(self.metadata, actual_model=self.model)

    def record_failure(self, exc):
        """Annotate the one actual usage row; never add a billing row."""
        if self.usage_id is None:
            return
        reason = fallback_reason(exc) or "response_validation"
        values = {
            "metadata": {
                **(self.usage_metadata or {}),
                "fallback_failure_reason": reason,
            }
        }
        if self.usage_success:
            values.update(
                success=False,
                error=f"Invalid model response ({type(exc).__name__}).",
            )
        try:
            LLMUsage.objects.filter(pk=self.usage_id).update(**values)
        except Exception as error:
            logger.warning(
                "Cannot annotate fallback attempt (%s)",
                type(error).__name__,
            )
        calls = (self.state or {}).get("llm_calls") or []
        if calls and calls[-1].get("model") == self.model:
            calls[-1]["success"] = False


@contextmanager
def primary_only():
    """Saved-model connection tests must not succeed via a backup."""
    token = current_attempt.set(AttemptContext(None, {}))
    try:
        yield
    finally:
        current_attempt.reset(token)


def run_plan(root, callback, accepts=None):
    """Call the supplied tracker adapter under one immutable flat plan."""
    policy, origin, backups = resolve_plan(root)
    if policy is None:
        token = current_attempt.set(AttemptContext(None, {}))
        try:
            return callback(root, None, None, None)
        finally:
            current_attempt.reset(token)
    rows = [root, *backups]
    by_uuid = {str(row.uuid): row for row in rows}
    targets = []
    for row in rows:
        configured = (row.config or {}).get("request_timeout_seconds")
        limit = int(configured) if configured else policy["timeout_seconds"]
        targets.append(
            FallbackTarget(
                str(row.uuid),
                min(limit, policy["timeout_seconds"]),
            )
        )
    health = ModelAvailability(root, origin, rows, policy)

    def eligible(target):
        # Recheck scope, activation and abilities just before each model.
        row = (
            visible_graph(root)
            .filter(
                uuid=target.config_uuid,
                is_active=True,
            )
            .first()
        )
        if row is None or not compatible(root, row):
            return False
        if row.uuid == root.uuid:
            if row.model_type != root.model_type:
                return False
        else:
            if same_upstream(root, row):
                return False
            if accepts is not None and not accepts(row):
                return False
        by_uuid[target.config_uuid] = row
        return True

    def call(target, timeout, attempt):
        row = by_uuid[target.config_uuid]
        context = AttemptContext(timeout, attempt.metadata())
        token = current_attempt.set(context)
        try:
            return callback(row, timeout, health, target)
        except Exception as exc:
            context.record_failure(exc)
            # Outer JSON repair/task loops must not replay a terminal
            # refusal or validation error after this bounded request.
            if fallback_reason(exc) is None:
                exc.agentcore_no_retry = True
            raise
        finally:
            current_attempt.reset(token)

    return execute_fallbacks(
        targets,
        call,
        timeout_seconds=min(
            3600,
            sum(target.timeout_seconds for target in targets)
            * (policy["retries"] + 1),
        ),
        retries=policy["retries"],
        conditions=policy["conditions"],
        availability=health,
        eligible=eligible,
    )


def _continue_stream(stream, first, deadline, health, target):
    try:
        yield first
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                stream.throw(
                    TimeoutError("Model stream time budget exhausted.")
                )
            try:
                value = next(stream)
            except StopIteration as finished:
                return finished.value
            yield value
    except Exception as exc:
        reason = fallback_reason(exc)
        if health and reason:
            health.failure(target, reason)
        raise
    finally:
        stream.close()


def _finished_stream(usage):
    yield from ()
    return usage


def dispatch_chat(root, call, arguments):
    """Switch before any stream output; never replay an emitted response."""
    original_state = arguments.get("state")
    if original_state is not None:
        original_state.setdefault("llm_calls", [])

    def invoke(row, timeout, health, target):
        state = {**(original_state or {})}
        state["metadata"] = dict(state.get("metadata") or {})
        state["llm_model_uuid"] = str(row.uuid)
        current_attempt.get().state = state
        started = time.monotonic()
        result = call(
            **{
                **arguments,
                "model_uuid": str(row.uuid),
                "state": state,
            }
        )
        if not arguments.get("stream"):
            return result
        try:
            first = next(result)
        except StopIteration as finished:
            return _finished_stream(finished.value)
        deadline = started + timeout if timeout is not None else None
        return _continue_stream(result, first, deadline, health, target)

    from agentcore_metering.adapters.django.services import (
        fallback_capabilities,
    )

    return run_plan(
        root,
        invoke,
        accepts=lambda row: fallback_capabilities.accepts_request(
            row,
            messages=arguments["messages"],
            tools=arguments.get("tools"),
            max_tokens=arguments.get("max_tokens"),
            response_format=arguments.get("response_format"),
        ),
    )
