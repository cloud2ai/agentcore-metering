"""
Load capability labels, mode->model_type mapping, and provider/model data
from YAML. Data is under this package (llm_static/*.yaml,
llm_static/providers/*.yaml). Paths are resolved relative to this file
so it works when the package is installed.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_capability_labels: Optional[Dict[str, str]] = None
_mode_to_model_type: Optional[Dict[str, str]] = None
_providers_with_models: Optional[Dict[str, Any]] = None
_provider_defaults: Optional[Dict[str, Dict[str, Any]]] = None
_providers_loaded_at = 0.0
_providers_lock = threading.Lock()

_AGIONE_MODELS_URL = "https://agione.pro/hyperone/xapi/api/models"
_AGIONE_DEFAULT_TIMEOUT = 5.0
_AGIONE_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _path(*parts: str) -> str:
    return os.path.join(_THIS_DIR, *parts)


def _load_yaml(filename: str) -> Any:
    with open(_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_capability_labels() -> Dict[str, str]:
    """
    Load capability tag id -> short label for UI. Cached after first load.
    """
    global _capability_labels
    if _capability_labels is None:
        _capability_labels = _load_yaml("capability_labels.yaml") or {}
    return _capability_labels


def get_mode_to_model_type() -> Dict[str, str]:
    """Load mode -> model_type for resolver. Cached after first load."""
    global _mode_to_model_type
    if _mode_to_model_type is None:
        _mode_to_model_type = _load_yaml("mode_to_model_type.yaml") or {}
    return _mode_to_model_type


def _normalize_model(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each model dict has id, label, capabilities, mode;
    optional max_input/max_output, reference_pricing.
    """
    out: Dict[str, Any] = {
        "id": str(m.get("id", "")),
        "label": str(m.get("label", "")),
        "capabilities": list(m.get("capabilities") or []),
        "max_input_tokens": m.get("max_input_tokens"),
        "max_output_tokens": m.get("max_output_tokens"),
        "mode": str(m.get("mode", "chat")),
    }
    rp = m.get("reference_pricing")
    if not isinstance(rp, dict):
        return out
    has_input = rp.get("input_usd_per_1m") is not None
    has_output = rp.get("output_usd_per_1m") is not None
    if has_input or has_output:
        out["reference_pricing"] = {
            "input_usd_per_1m": rp.get("input_usd_per_1m"),
            "output_usd_per_1m": rp.get("output_usd_per_1m"),
            "source": rp.get("source"),
            "updated": rp.get("updated"),
        }
    return out


def _agione_model_capabilities(model_id: str) -> tuple[str, list[str]]:
    """Infer conservative UI capabilities from an AGIOne model identifier."""

    lowered = model_id.lower()
    if "embedding" in lowered:
        return "embedding", ["embedding"]
    if any(token in lowered for token in ("image", "-t2v", "-i2v", "video")):
        return "image_generation", ["text-to-image"]
    capabilities = ["text-to-text"]
    if any(token in lowered for token in ("vision", "multimodal", "gpt-", "gemini", "claude")):
        capabilities.append("vision")
    if any(token in lowered for token in ("reason", "deepseek-r1", "o3", "o4", "gpt-6")):
        capabilities.append("reasoning")
    return "chat", capabilities


def _agione_model_label(model_id: str) -> str:
    """Build a readable label without changing the API model identifier."""

    parts = model_id.rsplit("/", 1)
    base = parts[-2] if len(parts) == 2 and len(parts[-1]) <= 8 else model_id
    return base.replace("-", " ").replace("_", " ").title()


def _fetch_agione_models() -> list[Dict[str, Any]]:
    """Fetch the public AGIOne model catalog with bounded response size."""

    timeout = float(
        os.getenv("AGIONE_MODEL_CATALOG_TIMEOUT_SECONDS", _AGIONE_DEFAULT_TIMEOUT)
    )
    request = Request(
        _AGIONE_MODELS_URL,
        headers={"Accept": "application/json", "User-Agent": "agentcore-metering"},
    )
    with urlopen(request, timeout=timeout) as response:
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length > _AGIONE_MAX_RESPONSE_BYTES:
            raise ValueError("AGIOne model catalog response is too large")
        payload = response.read(_AGIONE_MAX_RESPONSE_BYTES + 1)
    if len(payload) > _AGIONE_MAX_RESPONSE_BYTES:
        raise ValueError("AGIOne model catalog response is too large")
    body = json.loads(payload.decode("utf-8"))
    entries = body.get("data") if isinstance(body, dict) else None
    if not isinstance(entries, list):
        raise ValueError("AGIOne model catalog response has no data list")
    models = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or entry.get("model") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        mode, capabilities = _agione_model_capabilities(model_id)
        models.append(
            {
                "id": model_id,
                "label": _agione_model_label(model_id),
                "capabilities": capabilities,
                "max_input_tokens": None,
                "max_output_tokens": None,
                "mode": mode,
            }
        )
    return models


def _dynamic_models(provider_id: str, fallback: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Return dynamic provider models, falling back when the provider is unavailable."""

    if provider_id != "agione":
        return fallback
    try:
        models = _fetch_agione_models()
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return fallback
    return models or fallback


def get_providers_with_models() -> Dict[str, Any]:
    """
    Load providers list (in index order) with models and capability_labels.
    Same shape as before: { "providers": [...], "capability_labels": {...} }.
    Cached for the configured model catalog TTL. Dynamic providers may refresh
    while static provider entries are reloaded from package data.
    """
    global _providers_loaded_at, _providers_with_models
    ttl = float(os.getenv("AGENTCORE_MODEL_CATALOG_TTL_SECONDS", "300"))
    now = time.monotonic()
    if _providers_with_models is not None and now - _providers_loaded_at < ttl:
        return _providers_with_models
    with _providers_lock:
        now = time.monotonic()
        if _providers_with_models is not None and now - _providers_loaded_at < ttl:
            return _providers_with_models
        index = _load_yaml("providers/index.yaml") or {}
        provider_ids = index.get("provider_ids") or []
        providers = []
        for pid in provider_ids:
            path = os.path.join("providers", f"{pid}.yaml")
            if not os.path.isfile(_path(path)):
                continue
            data = _load_yaml(path) or {}
            static_models = [_normalize_model(m) for m in (data.get("models") or [])]
            models = _dynamic_models(str(data.get("id", pid)), static_models)
            providers.append({
                "id": str(data.get("id", pid)),
                "label": str(data.get("label", pid)),
                "models": models,
            })
        _providers_with_models = {
            "providers": providers,
            "capability_labels": get_capability_labels(),
        }
        _providers_loaded_at = time.monotonic()
        return _providers_with_models


def get_provider_defaults() -> Dict[str, Dict[str, Any]]:
    """
    Load per-provider defaults: default_api_base, default_model,
    default_temperature, default_top_p, default_max_tokens, settings_key,
    requires_api_base. Keys are provider ids. Cached. Only includes providers
    that have a YAML file in providers/.
    """
    global _provider_defaults
    if _provider_defaults is not None:
        return _provider_defaults
    index = _load_yaml("providers/index.yaml") or {}
    provider_ids = index.get("provider_ids") or []
    out: Dict[str, Dict[str, Any]] = {}
    for pid in provider_ids:
        path = os.path.join("providers", f"{pid}.yaml")
        if not os.path.isfile(_path(path)):
            continue
        data = _load_yaml(path) or {}
        out[pid] = {
            "default_api_base": data.get("default_api_base"),
            "default_model": data.get("default_model"),
            "default_temperature": data.get("default_temperature"),
            "default_top_p": data.get("default_top_p"),
            "default_max_tokens": data.get("default_max_tokens"),
            "settings_key": data.get("settings_key"),
            "requires_api_base": bool(data.get("requires_api_base", False)),
        }
    _provider_defaults = out
    return _provider_defaults
