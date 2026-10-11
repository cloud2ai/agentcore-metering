"""Typed decisions via LiteLLM SDK or gateway, with shared usage metering."""

import math
import re
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

import httpx
from django.utils import timezone

from agentcore_metering.adapters.django.llm_static.load import (
    get_decision_providers,
)
from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services.litellm_params import (
    _litellm_kwargs_from_config,
)
from agentcore_metering.adapters.django.services.litellm_retry import (
    completion_with_retry,
)
from agentcore_metering.adapters.django.trackers.llm_usage import (
    get_cost_from_response,
    usage_dict_from_usage_obj,
)

DECISION_PROVIDERS = tuple(get_decision_providers())
GATEWAY_PROVIDERS = ("litellm_proxy", "openai_compatible")
CONNECTION_QUESTIONS = [
    {
        "type": "predicate",
        "name": "is_test",
        "instructions": "Is the input a connection test?",
    }
]


def decision_error_detail(exc, language="en"):
    """Translate gateway errors without returning URLs, keys or input."""
    from agentcore_metering.adapters.django.services.runtime_config import (
        _user_friendly_validation_error,
        get_validation_message,
    )

    if isinstance(exc, httpx.TimeoutException):
        key = "timeout"
    elif isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError)):
        key = "network_error"
    else:
        key = _user_friendly_validation_error(exc)
    return get_validation_message(key, language)


def decision_params(provider, config):
    """Resolve explicit per-config credentials; never use process env keys."""
    provider = str(provider or "").strip().lower()
    if provider not in DECISION_PROVIDERS:
        raise ValueError(
            "Unsupported decision provider. Select a listed provider."
        )
    definition = get_decision_providers()[provider]
    model = config.get("model")
    api_base = config.get("api_base") or definition["default_api_base"]
    api_key = config.get("api_key") or ""
    if not isinstance(api_key, str):
        raise ValueError("api_key must be a string.")
    if provider == "cloudflare" and not api_base:
        account_id = config.get("account_id")
        if not isinstance(account_id, str) or not re.fullmatch(
            r"[a-fA-F0-9]{32}", account_id.strip(),
        ):
            raise ValueError("Cloudflare requires a valid account_id.")
        api_base = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id.strip()}/ai/run"
        )
    required = [("model", model), ("api_base", api_base)]
    if definition.get("api_key_required", True):
        required.append(("api_key", api_key))
    for key, value in required:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Decision configuration requires {key}.")
    url = urlsplit(api_base.strip())
    if (
        url.scheme not in ("https", "http")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("api_base must be an HTTP(S) provider base URL.")
    if url.path.rstrip("/").endswith(("/decisions", "/systemone")):
        raise ValueError(
            "Set api_base to the base URL, without the decisions path."
        )
    timeout = config.get("request_timeout_seconds")
    if isinstance(timeout, bool) or (
        isinstance(timeout, float) and not timeout.is_integer()
    ):
        raise ValueError("request_timeout_seconds must be a positive integer.")
    # Reuse the existing timeout/retry parser, but send no chat parameters.
    parsed = _litellm_kwargs_from_config("openai_compatible", config)
    model = model.strip()
    if provider not in GATEWAY_PROVIDERS:
        model = (
            model if model.startswith(f"{provider}/")
            else f"{provider}/{model}"
        )
    return {
        "decision_provider": provider,
        "model": model,
        "api_base": api_base.strip().rstrip("/"),
        "api_key": api_key.strip(),
        "timeout": parsed["timeout"],
        "num_retries": parsed["num_retries"],
    }


def validate_questions(questions):
    """Validate the portable Decisions question types before any request."""
    if not isinstance(questions, list) or not 1 <= len(questions) <= 128:
        raise ValueError("questions must contain 1 to 128 questions.")
    names = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object.")
        kind = question.get("type")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("Each question requires instructions.")
        name = question.get("name")
        if name is not None:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Question names must be unique strings.")
            names.add(name)
        if kind == "predicate":
            continue
        if kind == "choice":
            choices = question.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 255:
                raise ValueError("choice requires 2 to 255 choices.")
            values = []
            for choice in choices:
                value = (
                    choice.get("value") if isinstance(choice, dict) else None
                )
                if not isinstance(value, (str, bool)):
                    raise ValueError(
                        "Choice values must be strings or booleans."
                    )
                values.append(
                    str(value).lower() if isinstance(value, bool) else value
                )
            if len(set(values)) != len(values):
                raise ValueError("Choice values must be distinct.")
        elif kind == "score":
            levels = question.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise ValueError("score requires 2 to 10 levels.")
            if any(
                not isinstance(level, dict)
                or not isinstance(level.get("label"), str)
                or not level["label"].strip()
                for level in levels
            ):
                raise ValueError("Each score level requires a label.")
        else:
            raise ValueError(
                "Question type must be predicate, choice or score."
            )


def _number(value, low, high):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and low <= value <= high
    )


def _validate_answers(payload, questions):
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise ValueError(
            "Decisions API returned missing or incomplete answers."
        )
    for answer, question in zip(answers, questions):
        if not isinstance(answer, dict):
            raise ValueError("Decisions API returned an invalid answer.")
        if question.get("name") is not None:
            if answer.get("name") != question["name"]:
                raise ValueError("Decisions API returned mismatched names.")
        kind = answer.get("type")
        if kind == "refusal":
            continue
        if kind != question["type"]:
            raise ValueError(
                "Decisions API returned a mismatched answer type."
            )
        if kind == "predicate" and not _number(
            answer.get("probability"), 0, 1
        ):
            raise ValueError("Decisions API returned an invalid probability.")
        if kind == "choice" and not any(
            type(answer.get("choice")) is type(choice["value"])
            and answer.get("choice") == choice["value"]
            for choice in question["choices"]
        ):
            raise ValueError("Decisions API returned an unknown choice.")
        if kind == "score" and not _number(
            answer.get("score"), 0, len(question["levels"]) - 1
        ):
            raise ValueError("Decisions API returned an invalid score.")


def _request_keyless_decisions(**params):
    """Keep SDK translation without its empty-Bearer/env-key fallback."""
    from litellm.decisions.main import DECISIONS_ENDPOINTS
    from litellm.llms.openai.decisions.transformation import (
        ir_to_openai_response,
        openai_request_to_ir,
    )
    from litellm.types.decisions import (
        OpenAIDecisionRequestBody,
        UnsupportedDecisionsRequest,
    )

    endpoint = DECISIONS_ENDPOINTS[params["decision_provider"]]
    model = endpoint.canonical_model(params["model"].split("/", 1)[1])
    request = openai_request_to_ir(OpenAIDecisionRequestBody(
        input=params["input"], questions=params["questions"],
    ))
    body = endpoint.request_body(endpoint.request_model(model), request)
    if isinstance(body, UnsupportedDecisionsRequest):
        raise ValueError(body.reason)
    # LiteLLM 1.104.2 sends an illegal "Bearer " for an explicit empty
    # key; None falls back to process credentials. Keep this request local
    # to its config, without mutating SDK globals or environment variables.
    with httpx.Client(
        timeout=params["timeout"], follow_redirects=False,
    ) as client:
        response = client.post(
            endpoint.endpoint_url(params["api_base"], model),
            json=dict(body),
        )
        response.raise_for_status()
        result = ir_to_openai_response(
            endpoint.parse_response(response.json(), request),
            request, params["model"],
        )
    return result.model_dump(mode="json"), None


def _request_decisions(**params):
    if params["decision_provider"] not in GATEWAY_PROVIDERS:
        definition = get_decision_providers()[params["decision_provider"]]
        if not params["api_key"] and not definition.get(
            "api_key_required", True,
        ):
            return _request_keyless_decisions(**params)
        import litellm

        # LiteLLM owns native provider translation. Agentcore owns retries.
        response = litellm.decisions(**{
            key: value for key, value in params.items()
            if key != "decision_provider"
        })
        cost = get_cost_from_response(response)
        return response.model_dump(mode="json"), cost
    base = params["api_base"]
    endpoint = base + (
        "/decisions" if base.endswith("/v1") else "/v1/decisions"
    )
    with httpx.Client(
        timeout=params["timeout"],
        follow_redirects=False,
    ) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": f"Bearer {params['api_key']}"},
            json={key: params[key] for key in ("model", "input", "questions")},
        )
        response.raise_for_status()
        return response.json(), response.headers.get("x-litellm-response-cost")


def _response_cost(raw):
    if raw is None:
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() and value >= 0 else None


class DecisionTracker:
    """Shared, non-streaming decision client; thresholds stay in callers."""

    @staticmethod
    def call_and_track(*, input, questions, model_uuid, state=None):
        state = dict(state or {})
        row = LLMConfig.objects.filter(
            uuid=model_uuid,
            model_type=LLMConfig.MODEL_TYPE_DECISION,
            is_active=True,
        ).first()
        if row is None:
            raise ValueError("Active decision model configuration not found.")
        return DecisionTracker.call_with_config(
            provider=row.provider,
            config=row.config,
            input=input,
            questions=questions,
            state=state,
            config_uuid=str(row.uuid),
        )

    @staticmethod
    def call_with_config(
        *, provider, config, input, questions, state=None, config_uuid=None
    ):
        """Call an unsaved config for admin validation, recording its usage."""
        if not isinstance(input, str) or not input.strip():
            raise ValueError("Decision input must be non-empty text.")
        validate_questions(questions)
        params = decision_params(provider, config)
        params.update(input=input, questions=questions)
        state = dict(state or {})
        state["metadata"] = {
            **(state.get("metadata") or {}),
            "model_type": "decision",
            "endpoint": "decisions",
            "question_count": len(questions),
        }
        if config_uuid:
            state["metadata"]["config_uuid"] = config_uuid
        started_at = timezone.now()
        # Use the shared persistence path without changing the chat tracker.
        from agentcore_metering.adapters.django.trackers.llm import LLMTracker

        def save_failure(exc, attempt=None):
            failure_state = dict(state)
            failure_state["metadata"] = dict(state["metadata"])
            if attempt is not None:
                failure_state["metadata"].update(
                    retry_attempt=True, attempt_number=attempt + 1
                )
            LLMTracker._save_usage_to_db(
                state=failure_state,
                model=params["model"],
                success=False,
                error=f"Decision request failed ({type(exc).__name__}).",
                started_at=started_at,
            )

        payload = None
        usage = None
        try:
            payload, raw_cost = completion_with_retry(
                _request_decisions, params, on_attempt_error=save_failure
            )
            response_model = (
                payload.get("model") if isinstance(payload, dict) else None
            )
            usage = usage_dict_from_usage_obj(
                payload.get("usage") if isinstance(payload, dict) else None,
                params["model"],
            )
            cost = _response_cost(raw_cost)
            usage["cost"] = float(cost) if cost is not None else None
            _validate_answers(payload, questions)
        except Exception as exc:
            if usage is None:
                save_failure(exc)
            else:
                LLMTracker._save_usage_to_db(
                    state=state,
                    **usage,
                    success=False,
                    error=f"Invalid decision response ({type(exc).__name__}).",
                    started_at=started_at,
                    response_model=response_model,
                )
            raise
        LLMTracker._save_usage_to_db(
            state=state,
            **usage,
            started_at=started_at,
            response_model=response_model,
        )
        return payload["answers"], usage
