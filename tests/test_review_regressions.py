"""Regression cases found while auditing the shared model configuration."""

from types import SimpleNamespace

import pytest

from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.trackers.llm_usage import (
    usage_dict_from_usage_obj,
)

BASE = "/api/v1/admin/llm-config/"


@pytest.mark.django_db
@pytest.mark.parametrize("model_type", ["embedding", "image_generation"])
@pytest.mark.parametrize("payload", [
    {"is_active": False},
    {"config": {"model_info": {"supported_modalities": ["text"]}}},
    {"config": {"model": "custom-model", "api_key": "new-key"}},
])
def test_edit_preserves_custom_model_type(admin_client, model_type, payload):
    row = LLMConfig.objects.create(
        provider="openai_compatible", model_type=model_type,
        config={"model": "custom-model", "api_key": "old-key"},
    )
    response = admin_client.put(
        f"{BASE}{row.uuid}/", payload, format="json",
    )
    assert response.status_code == 200
    row.refresh_from_db()
    assert row.model_type == model_type


@pytest.mark.django_db
@pytest.mark.parametrize("model", [{"bad": "value"}, ["model"], 42, True])
def test_invalid_model_type_returns_validation_error(admin_client, model):
    response = admin_client.post(
        BASE, {"provider": "openai", "config": {"model": model}},
        format="json",
    )
    assert response.status_code == 400
    assert "model" in str(response.data)
    assert not LLMConfig.objects.exists()


@pytest.mark.parametrize("as_object", [False, True])
def test_null_chat_token_fields_fall_back_to_decision_usage(as_object):
    raw = {
        "prompt_tokens": None, "completion_tokens": None,
        "input_tokens": 123, "output_tokens": 4,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 2},
    }
    usage = usage_dict_from_usage_obj(
        SimpleNamespace(**raw) if as_object else raw, "decision-model",
    )
    assert usage["prompt_tokens"] == 123
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 127
    assert usage["cached_tokens"] == 20
    assert usage["reasoning_tokens"] == 2
