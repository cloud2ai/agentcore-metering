"""
LLM call params from DB (global/per-user) or Django settings for LiteLLM.

Resolution: DB user config -> DB global config -> settings.
Returns dict of kwargs for litellm.completion(
    model=..., messages=..., **kwargs).
All providers support api_base (URL). Use official default when not set.
Supported: OpenAI, Azure, Gemini, Anthropic, Mistral, Qwen, DeepSeek, xAI,
Meta Llama, Amazon Nova, NVIDIA NIM, MiniMax, Moonshot, Z.AI, Volcengine,
OpenRouter.
"""
import logging
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from django.conf import settings
from django.utils.translation import activate, gettext as _
import litellm
from litellm import completion_cost

from agentcore_metering.adapters.django.llm_static.load import (
    get_provider_defaults,
)
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage
from agentcore_metering.adapters.django.services.config_source import (
    get_config_from_db,
)
from agentcore_metering.adapters.django.utils import (
    _read_field,
    _read_nested_int,
    _safe_int,
)
from agentcore_metering.constants import (
    DEFAULT_COST_CURRENCY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    TEST_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

TASK_VALIDATE_LLM_CONFIG = "validate_llm_config"
TASK_RUN_TEST_CALL = "run_test_call"

# Build from YAML (llm_static/providers/*.yaml). All provider defaults live
# in YAML; see providers/index.yaml and providers/*.yaml.
_yaml_defaults = get_provider_defaults()
OFFICIAL_API_BASES: Dict[str, Optional[str]] = {
    p: d.get("default_api_base") for p, d in _yaml_defaults.items()
}
DEFAULT_MODELS: Dict[str, str] = {
    p: d.get("default_model")
    for p, d in _yaml_defaults.items()
    if d.get("default_model")
}
PROVIDER_SETTINGS_KEYS: Dict[str, str] = {
    p: d.get("settings_key")
    for p, d in _yaml_defaults.items()
    if d.get("settings_key")
}
PROVIDERS_REQUIRING_API_BASE = frozenset(
    p for p, d in _yaml_defaults.items() if d.get("requires_api_base")
)

# Provider model string and parameter builders.


def _model_string(provider: str, config: dict) -> str:
    """
    Build LiteLLM model string. Default to each platform's smallest/cheapest
    model to avoid cost from misconfiguration.
    """
    provider = (provider or "openai").strip().lower()
    model = (config.get("model") or "").strip() or DEFAULT_MODELS.get(
        provider, "gpt-4o-mini"
    )
    # Azure uses deployment-based model routing in LiteLLM.
    if provider == "azure_openai":
        deployment = config.get("deployment") or model
        return f"azure/{deployment}"
    # Gemini may already include vendor prefix in model id.
    if provider == "gemini":
        return f"gemini/{model}" if "/" not in model else model
    # Most providers require a "<provider>/<model>" model string.
    # Keep pre-prefixed model id as-is except for nvidia_nim.
    if provider in (
        "anthropic",
        "mistral",
        "dashscope",
        "deepseek",
        "xai",
        "meta_llama",
        "amazon_nova",
        "nvidia_nim",
        "minimax",
        "moonshot",
        "zai",
        "volcengine",
        "openrouter",
    ):
        if "/" not in model or provider == "nvidia_nim":
            return f"{provider}/{model}"
        return model
    return model


def _litellm_kwargs_from_config(provider: str, config: dict) -> Dict[str, Any]:
    provider = (provider or "openai").strip().lower()
    config = config or {}
    model = _model_string(provider, config)
    api_base = (
        (config.get("api_base") or "").strip()
        or OFFICIAL_API_BASES.get(provider)
    )
    kwargs = {
        "model": model,
        "api_key": config.get("api_key") or None,
        "api_base": api_base,
    }
    if provider == "azure_openai":
        kwargs["api_version"] = config.get("api_version")
    # Apply defaults only when the config value is missing (None).
    # Explicit falsy values like 0 should be preserved.
    defaults = (
        ("max_tokens", DEFAULT_MAX_TOKENS),
        ("temperature", DEFAULT_TEMPERATURE),
        ("top_p", DEFAULT_TOP_P),
    )
    for key, default in defaults:
        value = config.get(key)
        kwargs[key] = default if value is None else value
    kwargs["drop_params"] = True
    return {k: v for k, v in kwargs.items() if v is not None}


def _validate_config(provider: str, config: dict) -> None:
    """Raise ValueError if required keys are missing for the provider."""
    provider = (provider or "openai").strip().lower()
    # Azure-compatible providers require api_base in addition to api_key.
    if provider in PROVIDERS_REQUIRING_API_BASE:
        if not config.get("api_key") or not config.get("api_base"):
            raise ValueError(
                "Azure OpenAI configuration is incomplete. "
                "Set api_key and api_base."
            )
        return
    if provider == "openrouter":
        if not config.get("api_key"):
            raise ValueError(
                "OpenRouter configuration is incomplete. Set api_key."
            )
        return
    if provider in DEFAULT_MODELS:
        if not config.get("api_key"):
            name = (
                "OpenAI"
                if provider == "openai"
                else provider.replace("_", " ").title()
            )
            raise ValueError(
                f"{name} configuration is incomplete. Set api_key."
            )
        return
    if not config.get("api_key"):
        raise ValueError(
            f"Provider '{provider}' requires api_key in config."
        )


def build_litellm_params_from_config(
    provider: str, config: dict
) -> Dict[str, Any]:
    """
    Build litellm.completion() kwargs from given provider and config dict.
    Same defaults as get_litellm_params. Use for validation without DB.

    Raises:
        ValueError: If required config (e.g. api_key) is missing.
    """
    provider = (provider or "openai").strip().lower()
    config = config or {}
    _validate_config(provider, config)
    return _litellm_kwargs_from_config(provider, config)


# Error keys for connection test; backend translates by request language.
VALIDATION_ERROR_KEYS = (
    "invalid_api_key",
    "auth_failed",
    "rate_limit",
    "not_found",
    "timeout",
    "network_error",
    "permission_denied",
    "unknown",
)

# Error keys for connection test; backend translates via gettext (.po files).
# Key -> default English message (msgid for gettext).
VALIDATION_MESSAGE_IDS: Dict[str, str] = {
    "invalid_api_key": (
        "Invalid or expired API key. Please check and try again."
    ),
    "auth_failed": (
        "Authentication failed. Check API key or account permissions."
    ),
    "rate_limit": "Rate limit or quota exceeded. Please try again later.",
    "not_found": (
        "Model or endpoint not found. Check model name and API base URL."
    ),
    "timeout": "Connection timed out. Check network and API base URL.",
    "network_error": (
        "Cannot reach the service. Check API base URL and network."
    ),
    "permission_denied": (
        "Access denied. Check API key or account permissions."
    ),
    "unknown": (
        "Connection test failed. Check your configuration and try again."
    ),
}


def get_validation_message(key: str, language: Optional[str] = None) -> str:
    """
    Return translated connection-test error message for the given key.
    Uses Django gettext; language is e.g. request.LANGUAGE_CODE (zh-hans, en).
    Falls back to key if unknown.
    """
    if not key:
        return key or ""
    if language:
        activate(language)
    msgid = VALIDATION_MESSAGE_IDS.get(key) or key
    return _(msgid)


def _user_friendly_validation_error(exc: Exception) -> str:
    """
    Map LiteLLM/API exceptions to an error key for connection test.
    Returns key only; view calls get_validation_message(key,
    request.LANGUAGE_CODE).
    """
    msg = str(exc).strip()
    msg_lower = msg.lower()
    type_name = type(exc).__name__

    if (
        "401" in msg
        or "authentication" in type_name
        or "authentication" in msg_lower
    ):
        if (
            "invalid" in msg_lower
            or "token" in msg_lower
            or "key" in msg_lower
            or "expired" in msg_lower
        ):
            return "invalid_api_key"
        return "auth_failed"
    if "429" in msg or "rate" in msg_lower or "limit" in msg_lower:
        return "rate_limit"
    if "404" in msg or "not found" in msg_lower or "not_found" in msg_lower:
        return "not_found"
    if "timeout" in msg_lower or "timed out" in msg_lower:
        return "timeout"
    if (
        "connection" in msg_lower
        or "network" in msg_lower
        or "unreachable" in msg_lower
    ):
        return "network_error"
    if "invalid_api_key" in msg_lower or "incorrect api key" in msg_lower:
        return "invalid_api_key"
    if "permission" in msg_lower or "forbidden" in msg_lower or "403" in msg:
        return "permission_denied"

    return "unknown"


def _extract_usage_from_response(response: Any, params: dict) -> Tuple[
    str, int, int, int, int, int, Optional[Decimal]
]:
    """
    Extract actual_model, token counts, and cost from LiteLLM completion
    response. Returns (actual_model, prompt_tokens, completion_tokens,
    total_tokens, cached_tokens, reasoning_tokens, cost).
    """
    usage_obj = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    if usage_obj:
        prompt_tokens = _safe_int(_read_field(usage_obj, "prompt_tokens", 0))
        completion_tokens = _safe_int(
            _read_field(usage_obj, "completion_tokens", 0)
        )
        total_tokens = (
            _safe_int(_read_field(usage_obj, "total_tokens", 0))
            or (prompt_tokens + completion_tokens)
        )
        cached_tokens = _safe_int(_read_field(usage_obj, "cached_tokens", 0))
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
                _read_field(usage_obj, "completion_tokens_details", None)
                or _read_field(usage_obj, "output_token_details", None)
            )
            reasoning_tokens = _read_nested_int(
                completion_details,
                ("reasoning_tokens", "reasoning"),
                0,
            )

    actual_model = (
        getattr(response, "model", None) or params.get("model", "unknown")
    )
    cost = None
    try:
        raw_cost = completion_cost(completion_response=response)
        if raw_cost is not None:
            cost = Decimal(str(raw_cost))
    except Exception:
        pass
    return (
        actual_model, prompt_tokens, completion_tokens, total_tokens,
        cached_tokens, reasoning_tokens, cost,
    )


def _string_preview(value: Any, max_len: int = 160) -> str:
    """
    Return a short single-line preview string for logging.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _response_debug_snapshot(
    response: Any,
    choice: Any,
    message: Any,
    content: Any,
) -> Dict[str, Any]:
    """
    Build safe diagnostic fields for abnormal LLM responses.
    """
    hidden = getattr(response, "_hidden_params", None) or {}
    response_id = (
        getattr(response, "id", None)
        or hidden.get("response_id")
        or hidden.get("id")
    )
    request_id = hidden.get("x_request_id") or hidden.get("request_id")
    finish_reason = getattr(choice, "finish_reason", None) if choice else None
    message_role = getattr(message, "role", None) if message else None
    tool_calls = getattr(message, "tool_calls", None) if message else None
    refusal = getattr(message, "refusal", None) if message else None
    if isinstance(message, dict):
        message_role = message_role or message.get("role")
        tool_calls = tool_calls or message.get("tool_calls")
        refusal = refusal or message.get("refusal")
    tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
    return {
        "response_id": response_id,
        "request_id": request_id,
        "finish_reason": finish_reason,
        "message_role": message_role,
        "tool_call_count": tool_call_count,
        "has_refusal": bool(refusal),
        "content_type": type(content).__name__,
        "content_preview": _string_preview(content),
    }


def _create_usage_record(
    user: Optional[Any],
    actual_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    cost: Optional[Decimal],
    metadata_node_name: str,
) -> None:
    """
    Persist one usage record for runtime validation and test calls.
    """
    LLMUsage.objects.create(
        user=user,
        model=actual_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        cost=cost,
        cost_currency=DEFAULT_COST_CURRENCY,
        success=True,
        error=None,
        metadata={"node_name": metadata_node_name},
    )


def validate_llm_config(
    provider: str,
    config: dict,
    user: Optional[Any] = None,
) -> tuple[bool, str]:
    """
    Test that the given provider + config can complete a minimal LLM call.
    Uses LiteLLM completion with TEST_MAX_TOKENS to avoid model output limits.
    On success, records token usage (and cost if available) to LLMUsage when
    user is provided.
    Returns (True, "") on success, (False, error_message) on failure.
    """
    logger.info(f"Starting {TASK_VALIDATE_LLM_CONFIG} provider={provider}")
    try:
        params = build_litellm_params_from_config(provider, config)
        params["max_tokens"] = TEST_MAX_TOKENS
        params["messages"] = [{"role": "user", "content": "Hi"}]
        user_id = getattr(user, "pk", None)
        logger.info(
            "LLM validate request "
            f"provider={provider} user_id={user_id} "
            f"model={params.get('model')} "
            f"max_tokens={params.get('max_tokens')} "
            f"has_api_key={bool(params.get('api_key'))} "
            f"api_base={params.get('api_base')}"
        )
        response = litellm.completion(**params)
        logger.info(
            "LLM validate response "
            f"provider={provider} user_id={user_id} "
            f"has_response={response is not None}"
        )

        if response is None:
            return False, "LLM returned no response"

        (
            actual_model, prompt_tokens, completion_tokens, total_tokens,
            cached_tokens, reasoning_tokens, cost,
        ) = _extract_usage_from_response(response, params)
        logger.info(
            "LLM validate usage "
            f"provider={provider} user_id={user_id} model={actual_model} "
            f"prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} "
            f"total_tokens={total_tokens}"
        )

        if user is not None:
            try:
                _create_usage_record(
                    user=user,
                    actual_model=actual_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    metadata_node_name="config_test",
                )
            except Exception as e:
                uid = getattr(user, "pk", None)
                logger.warning(
                    f"Failed to record test-call usage; user={uid}, "
                    f"model={actual_model}, error={e}",
                    exc_info=True,
                )

        logger.info(f"Finished {TASK_VALIDATE_LLM_CONFIG} provider={provider}")
        return True, ""
    except ValueError as e:
        logger.error(
            f"Failed {TASK_VALIDATE_LLM_CONFIG} provider={provider}: {e}"
        )
        return False, str(e)
    except Exception as e:
        error_key = _user_friendly_validation_error(e)
        user_id = getattr(user, "pk", None)
        logger.warning(
            "LLM validate failed "
            f"provider={provider} user_id={user_id} "
            f"error_key={error_key} error={str(e)}"
        )
        logger.debug(
            f"LLM config validation failed provider={provider}: {e}",
            exc_info=True,
        )
        return False, error_key


def run_test_call(
    config_uuid: Optional[str],
    config_id: Optional[int],
    prompt: str,
    user: Any,
    max_tokens: int = 512,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Run one completion with the given LLMConfig and record to LLMUsage.

    Uses the config identified by config_uuid (or legacy config_id).
    Prompt is the user message.
    Returns (ok, content_or_detail, usage_dict or None).
    usage_dict has model, prompt_tokens, completion_tokens, total_tokens,
    cached_tokens, reasoning_tokens, cost, cost_currency.
    """
    if not (prompt or "").strip():
        return False, "Prompt cannot be empty", None
    # Keep max_tokens in a safe range for admin test call.
    max_tokens = min(max(1, max_tokens), 4096)
    llm_config = None
    if config_uuid:
        try:
            llm_config = LLMConfig.objects.get(uuid=config_uuid)
        except (LLMConfig.DoesNotExist, ValueError, TypeError):
            llm_config = None
    if llm_config is None and config_id is not None:
        try:
            llm_config = LLMConfig.objects.get(pk=config_id)
        except (LLMConfig.DoesNotExist, ValueError, TypeError):
            llm_config = None
    if llm_config is None:
        return False, "Config not found", None

    config_uuid_value = str(getattr(llm_config, "uuid", "") or "")
    config_id_value = getattr(llm_config, "id", None)
    logger.info(
        f"Starting {TASK_RUN_TEST_CALL} "
        f"config_uuid={config_uuid_value} config_id={config_id_value}"
    )
    provider = (llm_config.provider or "openai").strip().lower()
    config = llm_config.config or {}
    user_id = getattr(user, "pk", None)
    try:
        params = build_litellm_params_from_config(provider, config)
    except ValueError as e:
        return False, str(e), None
    params["messages"] = [{"role": "user", "content": (prompt or "").strip()}]
    params["max_tokens"] = max_tokens
    logger.info(
        "LLM test-call request "
        f"config_uuid={config_uuid_value} config_id={config_id_value} "
        f"user_id={user_id} provider={provider} "
        f"model={params.get('model')} "
        f"max_tokens={params.get('max_tokens')} "
        f"prompt_len={len((prompt or '').strip())} "
        f"has_api_key={bool(params.get('api_key'))} "
        f"api_base={params.get('api_base')}"
    )
    try:
        response = litellm.completion(**params)
    except Exception as e:
        error_key = _user_friendly_validation_error(e)
        logger.warning(
            "LLM test-call failed "
            f"config_uuid={config_uuid_value} config_id={config_id_value} "
            f"user_id={user_id} provider={provider} "
            f"error_key={error_key} error={str(e)}"
        )
        logger.debug(
            f"Test call failed config_uuid={config_uuid_value}: {e}",
            exc_info=True,
        )
        return False, error_key, None

    if response is None:
        logger.warning(
            "LLM test-call empty response object "
            f"config_uuid={config_uuid_value} config_id={config_id_value} "
            f"user_id={user_id}"
        )
        return False, "LLM returned no response", None

    choices = response.choices or []
    choice = choices[0] if choices else None
    finish_reason = getattr(choice, "finish_reason", None) if choice else None
    has_message = bool(choice and getattr(choice, "message", None))
    logger.info(
        "LLM test-call response "
        f"config_uuid={config_uuid_value} config_id={config_id_value} "
        f"user_id={user_id} choices={len(choices)} "
        f"finish_reason={finish_reason} has_message={has_message}"
    )
    if not choice or not getattr(choice, "message", None):
        snapshot = _response_debug_snapshot(
            response=response,
            choice=choice,
            message=None,
            content=None,
        )
        logger.warning(
            "LLM test-call missing message "
            f"config_uuid={config_uuid_value} config_id={config_id_value} "
            f"user_id={user_id} "
            f"finish_reason={finish_reason} "
            f"response_id={snapshot['response_id']} "
            f"request_id={snapshot['request_id']}"
        )
        return False, "LLM returned empty response", None
    msg = choice.message
    content = getattr(msg, "content", None) or ""

    (
        actual_model, prompt_tokens, completion_tokens, total_tokens,
        cached_tokens, reasoning_tokens, cost,
    ) = _extract_usage_from_response(response, params)
    content_str = str(content).strip()
    logger.info(
        "LLM test-call usage "
        f"config_uuid={config_uuid_value} config_id={config_id_value} "
        f"user_id={user_id} model={actual_model} "
        f"prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} "
        f"total_tokens={total_tokens}"
    )
    if not content_str:
        snapshot = _response_debug_snapshot(
            response=response,
            choice=choice,
            message=msg,
            content=content,
        )
        logger.warning(
            "LLM test-call empty content "
            f"config_uuid={config_uuid_value} config_id={config_id_value} "
            f"user_id={user_id} model={actual_model} "
            f"finish_reason={snapshot['finish_reason']} "
            f"total_tokens={total_tokens} "
            f"response_id={snapshot['response_id']} "
            f"request_id={snapshot['request_id']} "
            f"message_role={snapshot['message_role']} "
            f"tool_call_count={snapshot['tool_call_count']} "
            f"has_refusal={snapshot['has_refusal']} "
            f"content_type={snapshot['content_type']} "
            f"content_preview={snapshot['content_preview']}"
        )
        return False, "LLM returned empty response", None
    else:
        logger.info(
            "LLM test-call content "
            f"config_uuid={config_uuid_value} config_id={config_id_value} "
            f"user_id={user_id} "
            f"content_len={len(content_str)}"
        )

    try:
        _create_usage_record(
            user=user,
            actual_model=actual_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=cost,
            metadata_node_name="admin_test_call",
        )
    except Exception as e:
        uid = getattr(user, "pk", None)
        logger.warning(
            f"Failed to record test-call usage; user={uid}, "
            f"model={actual_model}, error={e}",
            exc_info=True,
        )

    usage_dict = {
        "model": actual_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost": float(cost) if cost is not None else None,
        "cost_currency": DEFAULT_COST_CURRENCY,
    }
    logger.info(
        f"Finished {TASK_RUN_TEST_CALL} "
        f"config_uuid={config_uuid_value} config_id={config_id_value}"
    )
    return True, content_str, usage_dict


def get_provider_params_schema() -> Dict[str, Any]:
    """
    Per-provider param metadata for UI: required keys, optional keys,
    default model and default api_base. api_base is first in optional
    so UI can show it at top. editable_params lists all configurable keys.
    """
    optional_common = [
        "api_base",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
    ]
    editable_common = [
        "api_base",
        "api_key",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
    ]
    providers = {}
    for p in DEFAULT_MODELS:
        required = ["api_key"]
        if p == "azure_openai":
            required = ["api_key", "api_base", "deployment"]
            optional = [
                "api_base",
                "model",
                "api_version",
                "max_tokens",
                "temperature",
                "top_p",
            ]
            editable = [
                "api_base",
                "api_key",
                "deployment",
                "model",
                "api_version",
                "max_tokens",
                "temperature",
                "top_p",
            ]
        else:
            optional = optional_common.copy()
            editable = editable_common.copy()
        providers[p] = {
            "required": required,
            "optional": optional,
            "editable_params": editable,
            "default_model": DEFAULT_MODELS.get(p),
            "default_api_base": OFFICIAL_API_BASES.get(p),
        }
    return {"providers": providers}


def get_litellm_params(user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Build litellm.completion() kwargs from DB or settings.

    Returns:
        Dict with "model", "max_tokens", "temperature", "top_p" (defaults
        applied when not in config), and optionally "api_key", "api_base".
        Pass to litellm.completion(..., **params).

    Raises:
        ValueError: If required config (e.g. api_key) is missing.
    """
    cfg = get_config_from_db(user_id=user_id)

    if cfg is not None:
        provider = cfg["provider"]
        config = cfg["config"] or {}
        _validate_config(provider, config)
        return _litellm_kwargs_from_config(provider, config)

    provider = getattr(settings, "LLM_PROVIDER", "openai").strip().lower()
    settings_key = PROVIDER_SETTINGS_KEYS.get(provider, "OPENAI_CONFIG")
    config = dict(getattr(settings, settings_key, {}))
    _validate_config(provider, config)
    return _litellm_kwargs_from_config(provider, config)
