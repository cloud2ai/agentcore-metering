"""Input/output metadata, separate from provider request parameters.

Only explicit overrides are saved in config.model_info. Catalog information
is resolved on read from the local LiteLLM registry and curated directory;
no provider calls are made. Listed modalities describe known capabilities,
not a guarantee that an arbitrary gateway exposes every capability.
"""

from functools import lru_cache

from agentcore_metering.adapters.django.services.litellm_params import (
    _model_string,
)
from agentcore_metering.adapters.django.services.model_catalog import (
    get_providers_with_models,
)

INPUT_MODALITIES = ("text", "image", "audio", "video")
OUTPUT_MODALITIES = (*INPUT_MODALITIES, "code")
MODALITY_FIELDS = {
    "supported_modalities": INPUT_MODALITIES,
    "supported_output_modalities": OUTPUT_MODALITIES,
}


def validate_model_info(value):
    """Validate overrides; an empty object or null restores catalog values."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("model_info must be an object or null.")
    if value.keys() - MODALITY_FIELDS.keys():
        raise ValueError("model_info contains unsupported fields.")
    result = {}
    for key, modalities in value.items():
        if modalities is None:
            continue
        allowed = MODALITY_FIELDS[key]
        if (
            not isinstance(modalities, list) or not modalities
            or any(
                not isinstance(item, str) or item not in allowed
                for item in modalities
            )
        ):
            raise ValueError(
                f"{key} must be a non-empty list of: {', '.join(allowed)}."
            )
        result[key] = list(dict.fromkeys(modalities))
    return result


def reset_model_info_on_change(old_provider, old_config, provider, incoming):
    """Old manual overrides must not follow a different model identity."""
    if "model_info" not in incoming and (
        provider != old_provider
        or (
            "model" in incoming
            and incoming["model"] != (old_config or {}).get("model")
        )
    ):
        return {**incoming, "model_info": {}}
    return incoming


@lru_cache(maxsize=1)
def _curated_models():
    return {
        (provider["id"], model["id"]): model
        for provider in get_providers_with_models()["providers"]
        for model in provider["models"]
    }


def catalog_model_info(provider, model):
    """Resolve only explicit registry data and unambiguous directory tags."""
    import litellm

    provider = (provider or "openai").strip().lower()
    model = model.strip() if isinstance(model, str) else ""
    result = {key: None for key in MODALITY_FIELDS}
    if not model:
        return result
    canonical = _model_string(provider, {"model": model})
    registry = next((
        litellm.model_cost[key]
        for key in (f"{provider}/{model}", canonical, model)
        if key in litellm.model_cost
    ), {})
    curated = _curated_models().get((provider, model), {})
    tags = set(curated.get("capabilities") or [])
    mode = registry.get("mode") or curated.get("mode")
    for key, allowed in MODALITY_FIELDS.items():
        values = registry.get(key)
        if isinstance(values, list) and values:
            known = [item for item in allowed if item in values]
            if known:
                result[key] = known
    if result["supported_modalities"] is None:
        inputs = []
        if tags.intersection(("text-to-text", "text-to-image")) or mode in (
            "chat", "completion", "responses", "audio_speech",
            "image_generation", "video_generation",
        ):
            inputs.append("text")
        vision = registry.get("supports_vision")
        if vision is True or (vision is None and "vision" in tags):
            inputs.append("image")
        if registry.get("supports_audio_input") is True:
            inputs.append("audio")
        if registry.get("supports_video_input") is True:
            inputs.append("video")
        result["supported_modalities"] = inputs or None
    if result["supported_output_modalities"] is None:
        outputs = []
        if "text-to-text" in tags or mode in (
            "chat", "completion", "responses", "audio_transcription",
        ):
            outputs.append("text")
        if "text-to-image" in tags or mode in (
            "image_generation", "image_edit",
        ):
            outputs.append("image")
        if registry.get("supports_audio_output") is True or (
            mode == "audio_speech"
        ):
            outputs.append("audio")
        if mode == "video_generation":
            outputs.append("video")
        result["supported_output_modalities"] = outputs or None
    return result


def resolve_model_info(provider, config):
    """Read-time result; per-direction manual overrides win over catalog."""
    config = config if isinstance(config, dict) else {}
    result = catalog_model_info(provider, config.get("model"))
    overrides = config.get("model_info")
    overrides = overrides if isinstance(overrides, dict) else {}
    sources = {}
    for key, allowed in MODALITY_FIELDS.items():
        values = overrides.get(key)
        if (
            isinstance(values, list) and values
            and all(isinstance(v, str) and v in allowed for v in values)
        ):
            result[key] = list(dict.fromkeys(values))
            sources[key] = "manual"
        else:
            sources[key] = "catalog" if result[key] else "unknown"
    result["sources"] = sources
    return result
