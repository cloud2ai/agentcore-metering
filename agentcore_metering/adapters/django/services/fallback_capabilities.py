"""Reject known-incompatible backups without changing business input."""

import json

from agentcore_metering.adapters.django.services.litellm_params import (
    _model_string,
)
from agentcore_metering.adapters.django.services.model_catalog import (
    get_model_capabilities,
)
from agentcore_metering.adapters.django.services.model_info import (
    resolve_model_info,
)


def accepts_request(
    row, *, messages, tools=None, max_tokens=None, response_format=None
):
    """Use explicit metadata when known; custom gateways remain usable.

    Token counting uses the existing OpenAI-compatible local tokenizer as
    a conservative preflight estimate, not a provider-exact guarantee.
    The original material is never truncated to fit a backup.
    """
    import litellm

    config = row.config or {}
    canonical = _model_string(row.provider, config)
    model = config.get("model", "")
    registry = next(
        (
            litellm.model_cost[key]
            for key in (f"{row.provider}/{model}", canonical, model)
            if key in litellm.model_cost
        ),
        {},
    )
    catalog = get_model_capabilities(row.provider, model) or {}
    info = resolve_model_info(row.provider, config)
    required = {"text"}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind in ("image_url", "input_image"):
                required.add("image")
            elif kind in ("input_audio", "audio"):
                required.add("audio")
            elif kind in ("video_url", "input_video"):
                required.add("video")
    known_inputs = info["supported_modalities"]
    if known_inputs is not None and not required.issubset(known_inputs):
        return False
    if tools and registry.get("supports_function_calling") is False:
        return False
    if (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_schema"
        and registry.get("supports_response_schema") is False
    ):
        return False
    output_limit = registry.get("max_output_tokens") or catalog.get(
        "max_output_tokens",
    )
    if output_limit and max_tokens and max_tokens > output_limit:
        return False
    input_limit = registry.get("max_input_tokens") or catalog.get(
        "max_input_tokens",
    )
    if input_limit and required == {"text"}:
        # Count only when a known bound exists. This requires no HTTP call.
        count = litellm.token_counter(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        if count > input_limit:
            return False
    return True


def accepts_decisions(row, input, questions):
    material = json.dumps({"input": input, "questions": questions})
    messages = [{"role": "user", "content": material}]
    if row.model_type == "llm":
        from agentcore_metering.adapters.django.services.decision_chat import (
            adapter_prompt,
        )

        messages.insert(0, {"role": "system", "content": adapter_prompt()})
    return accepts_request(row, messages=messages)
