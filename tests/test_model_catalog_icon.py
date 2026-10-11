"""
Provider catalog exposes optional brand icon metadata.

The provider list is the single source for provider branding (label + icon)
so every consumer does not have to maintain its own provider->logo map.
"""
import pytest

from agentcore_metering.adapters.django.llm_static import load

AGIONE_ICON = "https://agione.pro/static/images/favicon.png"


@pytest.fixture(autouse=True)
def _reset_catalog_cache(monkeypatch):
    monkeypatch.setattr(load, "_providers_with_models", None)
    monkeypatch.setattr(load, "_providers_loaded_at", 0.0)
    # Keep the dynamic AGIOne fetch off the network in unit tests.
    monkeypatch.setattr(load, "_fetch_agione_models", lambda: [])


@pytest.mark.unit
def test_agione_provider_exposes_icon_url():
    data = load.get_providers_with_models()
    providers = {p["id"]: p for p in data["providers"]}
    assert providers["agione"]["icon"] == AGIONE_ICON


@pytest.mark.unit
def test_provider_without_icon_omits_the_key():
    data = load.get_providers_with_models()
    providers = {p["id"]: p for p in data["providers"]}
    # Providers whose logo comes from the frontend icon set carry no icon,
    # keeping the payload unchanged for every existing provider.
    assert "icon" not in providers["openai"]
    assert "icon" not in providers["deepseek"]
