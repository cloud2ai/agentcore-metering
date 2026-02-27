"""
LLM call tracker using LiteLLM: completion + usage and cost tracking.

Uses litellm.completion() and litellm.completion_cost() for reference pricing.
"""
import logging
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from django.db import transaction
from django.utils import timezone
import litellm
from litellm import completion_cost

from agentcore_metering.adapters.django.models import LLMUsage
from agentcore_metering.adapters.django.services.runtime_config import (
    get_litellm_params,
)
from agentcore_metering.adapters.django.utils import (
    _read_field,
    _read_nested_int,
    _safe_int,
)
from agentcore_metering.constants import DEFAULT_COST_CURRENCY

logger = logging.getLogger(__name__)

TASK_LLM_CALL = "llm_call"


def _get_cost_from_response(response: Any) -> Optional[Decimal]:
    try:
        cost = completion_cost(completion_response=response)
        if cost is not None:
            return Decimal(str(cost))
    except (TypeError, ValueError) as e:
        logger.debug("completion_cost or Decimal failed: %s", e)
    except Exception as e:
        logger.debug("completion_cost failed: %s", e)
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is not None:
        try:
            return Decimal(str(cost))
        except (TypeError, ValueError):
            model = getattr(response, "model", "unknown")
            logger.warning(
                "Invalid response_cost in hidden params "
                f"model={model} response_cost={cost}"
            )
            return None
    return None


class LLMTracker:
    """
    LLM call tracker via LiteLLM with usage and cost (reference pricing).
    """

    @staticmethod
    def call_and_track(
        messages: list,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict] = None,
        node_name: str = "unknown",
        state: Optional[Dict] = None,
        model_uuid: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call LLM via LiteLLM and persist usage + cost.

        When model_uuid is provided, uses that LLM config. Otherwise uses
        the earliest enabled LLM config (user scope then global).

        Returns:
            (response_content, usage_dict with model, prompt_tokens,
             completion_tokens, total_tokens, cached_tokens, reasoning_tokens,
             cost, cost_currency).
        """
        if not messages:
            raise ValueError("Messages cannot be empty")

        user_id = state.get("user_id") if state else None
        params = get_litellm_params(user_id=user_id, model_uuid=model_uuid)
        model = params.get("model", "unknown")

        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p

        if json_mode:
            if response_format is None:
                response_format = {"type": "json_object"}
            params["response_format"] = response_format
        elif response_format is not None:
            params["response_format"] = response_format

        params["messages"] = messages
        effective_state = {**(state or {}), "node_name": node_name}
        request_started_at = timezone.now()
        logger.info(
            f"Starting {TASK_LLM_CALL} node_name={node_name} "
            f"model={model} message_count={len(messages)}"
        )

        try:
            response = litellm.completion(**params)

            if response is None:
                logger.error(f"LiteLLM returned None; node_name={node_name}")
                raise ValueError(
                    f"[{node_name}] LLM service returned None response"
                )

            choice = (response.choices or [None])[0]
            if not choice or not getattr(choice, "message", None):
                raise ValueError("LLM returned empty response")
            msg = choice.message
            content = getattr(msg, "content", None) or ""
            if not (content and str(content).strip()):
                raise ValueError("LLM returned empty response")

            usage_obj = getattr(response, "usage", None)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cached_tokens = 0
            reasoning_tokens = 0
            if usage_obj:
                prompt_tokens = _safe_int(
                    _read_field(usage_obj, "prompt_tokens", 0)
                )
                completion_tokens = _safe_int(
                    _read_field(usage_obj, "completion_tokens", 0)
                )
                total_tokens = (
                    _safe_int(_read_field(usage_obj, "total_tokens", 0))
                    or (prompt_tokens + completion_tokens)
                )
                cached_tokens = _safe_int(
                    _read_field(usage_obj, "cached_tokens", 0)
                )
                reasoning_tokens = _safe_int(
                    _read_field(usage_obj, "reasoning_tokens", 0)
                )
                if cached_tokens == 0:
                    prompt_details = (
                        _read_field(usage_obj, "prompt_tokens_details", None)
                        or _read_field(usage_obj, "input_token_details", None)
                    )
                    cached_tokens = _read_nested_int(
                        prompt_details,
                        ("cached_tokens", "cache_read_tokens", "cache_read"),
                        0,
                    )
                if reasoning_tokens == 0:
                    completion_details = (
                        _read_field(
                            usage_obj,
                            "completion_tokens_details",
                            None,
                        )
                        or _read_field(usage_obj, "output_token_details", None)
                    )
                    reasoning_tokens = _read_nested_int(
                        completion_details,
                        ("reasoning_tokens", "reasoning"),
                        0,
                    )

            actual_model = getattr(response, "model", None) or model
            cost = _get_cost_from_response(response)
            cost_currency = DEFAULT_COST_CURRENCY

            usage = {
                "model": actual_model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cost": float(cost) if cost is not None else None,
                "cost_currency": cost_currency,
            }

            if state is not None:
                state.setdefault("llm_calls", []).append({
                    "node": effective_state.get("node_name", "unknown"),
                    "model": usage["model"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "cached_tokens": usage["cached_tokens"],
                    "reasoning_tokens": usage["reasoning_tokens"],
                    "cost": usage.get("cost"),
                    "cost_currency": usage.get("cost_currency"),
                    "success": True,
                    "error": None,
                })

            LLMTracker._save_usage_to_db(
                state=effective_state,
                model=usage["model"],
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                cost=cost,
                cost_currency=cost_currency,
                success=True,
                error=None,
                started_at=request_started_at,
            )

            logger.info(
                f"Finished {TASK_LLM_CALL} node_name={node_name} "
                f"model={usage['model']} total_tokens={usage['total_tokens']} "
                f"cost={cost} {cost_currency}"
            )
            return str(content), usage

        except Exception as e:
            node = effective_state.get("node_name", "unknown")
            logger.error(
                f"Failed {TASK_LLM_CALL} node_name={node} "
                f"error_type={type(e).__name__} error={e}"
            )
            logger.exception(e)
            if state is not None:
                state.setdefault("llm_calls", []).append({
                    "node": node,
                    "model": "unknown",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "success": False,
                    "error": str(e),
                })
            LLMTracker._save_usage_to_db(
                state=effective_state,
                model="unknown",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cached_tokens=0,
                reasoning_tokens=0,
                cost=None,
                cost_currency=DEFAULT_COST_CURRENCY,
                success=False,
                error=str(e),
                started_at=request_started_at,
            )
            raise

    @staticmethod
    def _save_usage_to_db(
        state: Optional[Dict] = None,
        node_name: str = "unknown",
        model: str = "unknown",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost: Optional[Decimal] = None,
        cost_currency: str = "USD",
        success: bool = True,
        error: Optional[str] = None,
        started_at: Optional[Any] = None,
    ) -> None:
        """Persist one LLM usage record (tokens, optional cost, started_at)."""
        try:
            state = state or {}
            user_id = state.get("user_id")
            node_name = state.get("node_name", node_name)
            metadata = {}
            if node_name and node_name != "unknown":
                metadata["node_name"] = node_name
            source_type = state.get("source_type")
            if source_type:
                metadata["source_type"] = source_type
            source_task_id = (
                state.get("source_task_id")
                or state.get("celery_task_id")
                or state.get("task_id")
            )
            if source_task_id:
                metadata["source_task_id"] = str(source_task_id)
            source_path = state.get("source_path")
            if source_path:
                metadata["source_path"] = source_path
            extra = state.get("metadata")
            if isinstance(extra, dict):
                metadata.update(extra)

            with transaction.atomic():
                LLMUsage.objects.create(
                    user_id=user_id,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    cost_currency=cost_currency or DEFAULT_COST_CURRENCY,
                    success=success,
                    error=error,
                    metadata=metadata,
                    started_at=started_at,
                )
        except Exception as e:
            logger.warning(
                f"Failed to save LLM usage; node_name={node_name}, "
                f"model={model}, user_id={user_id}, error={e}",
                exc_info=True,
            )
