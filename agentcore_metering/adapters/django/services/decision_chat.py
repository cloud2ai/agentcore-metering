"""Translate native decision questions to a metered chat response contract."""

import json
import math
from functools import lru_cache
from importlib.resources import files

import yaml

from agentcore_metering.adapters.django.services.model_fallback import (
    EmptyResponseError,
)
from agentcore_metering.adapters.django.trackers.llm_usage import (
    get_cost_from_response,
)


@lru_cache(maxsize=1)
def adapter_prompt():
    resource = files(
        "agentcore_metering.adapters.django.llm_static",
    ).joinpath("decision_adapter.yaml")
    return yaml.safe_load(resource.read_text(encoding="utf-8"))["system"]


def chat_decision_params(provider, config, input, questions, reasoning_effort):
    from agentcore_metering.adapters.django.services.litellm_params import (
        _litellm_kwargs_from_config,
        _validate_config,
        apply_reasoning_effort,
    )

    _validate_config(provider, config)
    params = _litellm_kwargs_from_config(provider, config)
    apply_reasoning_effort(
        params,
        "none" if reasoning_effort is None else reasoning_effort,
    )
    params["response_format"] = {"type": "json_object"}
    params["messages"] = [
        {"role": "system", "content": adapter_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "input": input,
                    "questions": questions,
                },
                ensure_ascii=False,
            ),
        },
    ]
    return params


def request_chat_decisions(**params):
    import litellm

    response = litellm.completion(**params)
    if response is None:
        raise EmptyResponseError("The chat provider returned no response.")
    usage = getattr(response, "usage", None)
    payload = {
        "model": getattr(response, "model", None) or params["model"],
        "usage": usage.model_dump() if usage is not None else None,
        "answers": None,
    }
    choice = (response.choices or [None])[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if (
        getattr(message, "refusal", None)
        or getattr(
            choice,
            "finish_reason",
            None,
        )
        == "content_filter"
    ):
        questions = json.loads(params["messages"][1]["content"])["questions"]
        payload["answers"] = [
            {
                "name": question.get("name"),
                "type": "refusal",
            }
            for question in questions
        ]
    elif getattr(choice, "finish_reason", None) == "length":
        pass
    elif not content or not str(content).strip():
        payload["_empty"] = True
    else:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                payload["answers"] = parsed.get("answers")
        except (ValueError, TypeError):
            pass
    return payload, get_cost_from_response(response)


def validate_chat_probabilities(payload, questions):
    """Preserve the full, typed choice distribution expected by callers."""
    for answer, question in zip(payload["answers"], questions):
        if question["type"] != "choice" or answer["type"] == "refusal":
            continue
        probabilities = answer.get("probabilities")
        options = question["choices"]
        if not isinstance(probabilities, list) or len(probabilities) != len(
            options
        ):
            raise ValueError(
                "Chat decision returned incomplete probabilities."
            )
        values = []
        seen = set()
        selected = None
        allowed = {(type(item["value"]), item["value"]) for item in options}
        for item in probabilities:
            if not isinstance(item, dict):
                raise ValueError(
                    "Chat decision returned invalid probabilities."
                )
            value = item.get("value")
            probability = item.get("probability")
            try:
                key = (type(value), value)
                valid = key in allowed and key not in seen
            except TypeError:
                valid = False
            if not valid or (
                type(probability) not in (int, float)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError(
                    "Chat decision returned invalid probabilities."
                )
            seen.add(key)
            values.append(probability)
            if (type(answer["choice"]), answer["choice"]) == key:
                selected = probability
        if not math.isclose(sum(values), 1, abs_tol=0.001):
            raise ValueError("Chat decision probabilities must sum to one.")
        if selected is None or selected < max(values):
            raise ValueError("Chat decision selected a lower probability.")
