"""Capability metadata round trips without changing provider requests."""

import litellm
import pytest
from rest_framework.test import APIClient

from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services.litellm_params import (
    _litellm_kwargs_from_config,
)
from agentcore_metering.adapters.django.services.model_info import (
    catalog_model_info,
    resolve_model_info,
    validate_model_info,
)

BASE = "/api/v1/admin/llm-config/"
INFO = {
    "supported_modalities": ["text", "image"],
    "supported_output_modalities": ["text", "audio"],
}


@pytest.mark.parametrize("value", [
    [], "vision", True,
    {"supports_vision": True},
    {"supported_modalities": []},
    {"supported_modalities": "image"},
    {"supported_modalities": ["pdf"]},
    {"supported_modalities": ["code"]},
    {"supported_output_modalities": [True]},
    {"supported_output_modalities": [{"image": True}]},
])
def test_reject_invalid_metadata(value):
    with pytest.raises((ValueError, TypeError)):
        validate_model_info(value)


def test_reset_and_deduplication():
    assert validate_model_info(None) == {}
    assert validate_model_info({"supported_modalities": None}) == {}
    assert validate_model_info({
        "supported_modalities": ["text", "text", "image"],
    }) == {"supported_modalities": ["text", "image"]}


def test_unknown_and_partial_overrides():
    result = resolve_model_info("openai_compatible", {"model": "custom-937"})
    assert result["supported_modalities"] is None
    assert result["supported_output_modalities"] is None
    assert set(result["sources"].values()) == {"unknown"}
    result = resolve_model_info("openai_compatible", {
        "model": "custom-937",
        "model_info": {"supported_modalities": ["image"]},
    })
    assert result["supported_modalities"] == ["image"]
    assert result["supported_output_modalities"] is None
    assert result["sources"]["supported_modalities"] == "manual"


def test_registry_input_and_output_are_independent(monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "openai/modal-test", {
        "mode": "chat", "supported_modalities": ["text", "audio"],
        "supported_output_modalities": ["audio"],
        "supports_vision": True,
    })
    result = catalog_model_info("openai", "modal-test")
    assert result["supported_modalities"] == ["text", "audio"]
    assert result["supported_output_modalities"] == ["audio"]
    result = resolve_model_info("openai", {
        "model": "modal-test",
        "model_info": {"supported_modalities": ["text"]},
    })
    assert result["supported_modalities"] == ["text"]
    assert result["supported_output_modalities"] == ["audio"]
    assert result["sources"]["supported_output_modalities"] == "catalog"


def test_vision_does_not_imply_image_output(monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "openai/modal-vision", {
        "mode": "chat", "supports_vision": True,
    })
    result = catalog_model_info("openai", "modal-vision")
    assert result["supported_modalities"] == ["text", "image"]
    assert result["supported_output_modalities"] == ["text"]


def test_explicit_registry_flag_overrides_curated_vision(monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "openai/gpt-4o", {
        "mode": "chat", "supports_vision": False,
    })
    result = catalog_model_info("openai", "gpt-4o")
    assert result["supported_modalities"] == ["text"]


def test_generic_multimodal_tag_does_not_invent_modalities(monkeypatch):
    from agentcore_metering.adapters.django.services import model_info

    monkeypatch.setattr(model_info, "_curated_models", lambda: {
        ("custom", "generic"): {"capabilities": ["multimodal"]},
    })
    result = catalog_model_info("custom", "generic")
    assert result == {
        "supported_modalities": None, "supported_output_modalities": None,
    }


def test_metadata_does_not_change_request_parameters():
    config = {"model": "custom-model", "api_key": "secret"}
    assert _litellm_kwargs_from_config("openai", config) == (
        _litellm_kwargs_from_config("openai", {
            **config, "model_info": INFO,
        })
    )


@pytest.mark.django_db
@pytest.mark.parametrize("scope", ["global", "user"])
def test_admin_crud_metadata(admin_client, normal_user, scope):
    response = admin_client.post(BASE, {
        "scope": scope, "user_id": normal_user.pk,
        "provider": "openai_compatible", "model_type": "llm",
        "config": {
            "model": "custom-937", "api_key": "secret-test-key",
            "model_info": INFO,
        },
    }, format="json")
    assert response.status_code == 201, response.data
    row = LLMConfig.objects.get(uuid=response.data["uuid"])
    url = f"{BASE}{row.uuid}/"
    assert row.config["model_info"] == INFO
    assert response.data["model_info"]["supported_modalities"] == [
        "text", "image",
    ]
    assert "secret-test-key" not in str(response.data)
    assert admin_client.get(url).data["config"]["model_info"] == INFO
    # An ordinary edit preserves metadata and credentials.
    response = admin_client.put(url, {
        "config": {"temperature": 0.3},
    }, format="json")
    assert response.data["config"]["model_info"] == INFO
    # Replacing the namespace can reset one direction independently.
    response = admin_client.put(url, {
        "config": {"model_info": {"supported_modalities": ["text"]}},
    }, format="json")
    assert response.data["model_info"]["supported_modalities"] == ["text"]
    assert response.data["model_info"]["supported_output_modalities"] is None
    response = admin_client.put(url, {
        "config": {"model_info": None},
    }, format="json")
    assert response.data["config"]["model_info"] == {}
    assert response.data["model_info"]["supported_modalities"] is None
    row.refresh_from_db()
    assert row.config["api_key"] == "secret-test-key"
    assert admin_client.delete(url).status_code == 204
    assert not LLMConfig.objects.filter(pk=row.pk).exists()


@pytest.mark.django_db
def test_changed_model_clears_old_overrides(admin_client):
    row = LLMConfig.objects.create(
        scope="global", provider="openai", config={
            "model": "custom-old", "model_info": INFO,
        },
    )
    response = admin_client.put(f"{BASE}{row.uuid}/", {
        "config": {"model": "custom-new"},
    }, format="json")
    assert response.status_code == 200
    assert response.data["config"]["model_info"] == {}


@pytest.mark.django_db
def test_legacy_user_endpoint_clears_old_overrides(admin_client, normal_user):
    row = LLMConfig.objects.create(
        scope="user", user=normal_user, provider="openai", config={
            "model": "custom-old", "model_info": INFO,
        },
    )
    response = admin_client.put(f"{BASE}users/{normal_user.pk}/", {
        "config": {"model": "custom-new"},
    }, format="json")
    assert response.status_code == 200
    row.refresh_from_db()
    assert row.config["model_info"] == {}


@pytest.mark.django_db
def test_invalid_metadata_is_not_persisted(admin_client):
    response = admin_client.post(BASE, {
        "config": {"model": "custom", "model_info": {
            "supported_modalities": ["bad"],
        }},
    }, format="json")
    assert response.status_code == 400
    assert not LLMConfig.objects.exists()


@pytest.mark.django_db
def test_user_catalog_has_no_saved_secrets(authenticated_client):
    LLMConfig.objects.create(scope="global", config={
        "model": "custom-private", "api_key": "catalog-private-key",
    })
    url = "/api/v1/user/llm-config/models"
    response = authenticated_client.get(url)
    assert response.status_code == 200
    assert "catalog-private-key" not in str(response.data)
    assert "custom-private" not in str(response.data)
    model = next(
        model for provider in response.data["providers"]
        if provider["id"] == "openai"
        for model in provider["models"] if model["id"] == "gpt-4o"
    )
    assert model["model_info"]["supported_modalities"] == ["text", "image"]
    assert APIClient().get(url).status_code == 403
