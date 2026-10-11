"""
Provider -> model list with capability tags for LLM config UI.

Capability tags (for display/filter):
- text-to-text: natural language generation / understanding
- code: code generation / understanding
- vision: image input understanding (Vision Input / Image-to-Text)
- multimodal: text + image ± other modalities
- text-to-image: image generation
- long-context: long context (e.g. 100K+ tokens)
- low-cost: low cost, lightweight
- embedding: vector embedding
- reasoning: extended reasoning / chain-of-thought (e.g. o1,
  deepseek-reasoner). When present, reasoning *effort* or *level* may be
  controlled by API params (e.g. OpenAI reasoning.effort: low/medium/high).
  Tag = capability; param = intensity.
Data is loaded from YAML under adapters/django/llm_static/
(capability_labels.yaml, mode_to_model_type.yaml, providers/*.yaml).
See llm_static/load.py.
"""
from copy import deepcopy
from typing import Any, Dict

from agentcore_metering.adapters.django.llm_static.load import (
    get_capability_labels,
    get_decision_providers,
    get_mode_to_model_type,
)
from agentcore_metering.adapters.django.llm_static.load import (
    get_providers_with_models as _load_providers_with_models,
)

CAPABILITY_LABELS: Dict[str, str] = get_capability_labels()
MODE_TO_MODEL_TYPE: Dict[str, str] = get_mode_to_model_type()


def get_providers_with_models() -> Dict[str, Any]:
    """
    Return provider list and per-provider model list with capability tags.
    Used by GET llm-config/models/ and for deriving model_type from model id.
    """
    data = deepcopy(_load_providers_with_models())
    providers = {p["id"]: p for p in data["providers"]}
    for pid, config in get_decision_providers().items():
        if pid not in providers:
            providers[pid] = {
                "id": pid, "label": config["label"], "models": [],
            }
            data["providers"].append(providers[pid])
        provider = providers[pid]
        provider["model_types"] = sorted({
            MODE_TO_MODEL_TYPE.get(m.get("mode"), "llm")
            for m in provider["models"]
        } | {"decision"})
        for model in config["models"]:
            # A mixed provider can expose the same alias through chat and
            # decisions. Preserve its existing mode and capabilities so
            # legacy chat pickers keep seeing it. Decision choices are also
            # listed separately in the provider's decision params schema.
            if any(m["id"] == model for m in provider["models"]):
                continue
            provider["models"].append({
                "id": model, "label": model, "mode": "decisions",
                "capabilities": ["decision"],
            })
    return data


def get_model_type_for_model_id(provider: str, model_id: str) -> str:
    """
    Derive LLMConfig.model_type (llm | embedding | image_generation) from
    provider and model id using curated model list. Default llm.
    """
    # The augmented display catalog includes decision aliases on mixed
    # providers. Legacy callers selecting those IDs still mean chat unless
    # they explicitly select model_type=decision.
    data = _load_providers_with_models()
    prov_key = (provider or "").strip().lower()
    if prov_key in get_decision_providers() and prov_key not in (
        "openai", "openrouter", "openai_compatible",
    ):
        return "decision"
    mid = (model_id or "").strip()
    for p in data["providers"]:
        if (p.get("id") or "").strip().lower() != prov_key:
            continue
        for m in p.get("models") or []:
            if (m.get("id") or "").strip() == mid:
                mode = m.get("mode") or "chat"
                return MODE_TO_MODEL_TYPE.get(mode, "llm")
    return "llm"


def get_model_capabilities(
    provider: str, model_id: str
) -> Dict[str, Any] | None:
    """
    Return capability info for a given provider + model_id, or None if unknown.
    """
    data = get_providers_with_models()
    prov_key = (provider or "").strip().lower()
    mid = (model_id or "").strip()
    for p in data["providers"]:
        if (p.get("id") or "").strip().lower() != prov_key:
            continue
        for m in p.get("models") or []:
            if (m.get("id") or "").strip() == mid:
                out = {
                    "capabilities": m.get("capabilities") or [],
                    "max_input_tokens": m.get("max_input_tokens"),
                    "max_output_tokens": m.get("max_output_tokens"),
                    "mode": m.get("mode", "chat"),
                }
                if m.get("reference_pricing") is not None:
                    out["reference_pricing"] = m["reference_pricing"]
                return out
    return None
