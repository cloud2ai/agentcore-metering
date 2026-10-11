"""Masked-key testing must not let users forward another config's key."""

from unittest.mock import patch

import pytest

from agentcore_metering.adapters.django.models import LLMConfig

pytestmark = pytest.mark.django_db
URL = "/api/v1/user/llm-config/test"
TARGET = (
    "agentcore_metering.adapters.django.views.config_validation"
    ".validate_llm_config"
)


@pytest.mark.parametrize("selector", ["config_uuid", "config_id"])
@pytest.mark.parametrize("ownership", ["own", "other", "global", "disabled"])
def test_user_masked_key_resolution_is_scoped(
    authenticated_client, normal_user, django_user_model, selector, ownership,
):
    other = django_user_model.objects.create_user(username="other")
    row = LLMConfig.objects.create(
        scope="global" if ownership == "global" else "user",
        user=None if ownership == "global" else (
            other if ownership == "other" else normal_user
        ),
        provider="openai", is_active=ownership != "disabled",
        config={"model": "gpt-4o-mini", "api_key": "stored-secret-key"},
    )
    payload = {
        selector: str(row.uuid) if selector == "config_uuid" else row.pk,
        "provider": "openai", "config": {
            "model": "gpt-4o-mini", "api_key": "stor***-key",
            "api_base": "https://caller-selected.example",
        },
    }
    with patch(TARGET, return_value=(True, "")) as validate:
        response = authenticated_client.post(URL, payload, format="json")
    if ownership == "own":
        assert response.status_code == 200
        assert validate.call_args.args[1]["api_key"] == "stored-secret-key"
    else:
        assert response.status_code == 404
        validate.assert_not_called()


def test_user_mismatched_selectors_do_not_resolve_stored_credentials(
    authenticated_client, normal_user, django_user_model,
):
    own = LLMConfig.objects.create(
        scope="user", user=normal_user, config={"api_key": "own-key"},
    )
    other = LLMConfig.objects.create(
        scope="user",
        user=django_user_model.objects.create_user(username="other"),
        config={"api_key": "foreign-key"},
    )
    with patch(TARGET) as validate:
        response = authenticated_client.post(URL, {
            "config_uuid": str(own.uuid), "config_id": other.pk,
            "config": {"model": "gpt-4o-mini", "api_key": "******"},
        }, format="json")
    assert response.status_code == 404
    validate.assert_not_called()


def test_staff_can_validate_global_masked_credentials(admin_client):
    row = LLMConfig.objects.create(
        scope="global", config={"api_key": "shared-secret-key"},
    )
    with patch(TARGET, return_value=(True, "")) as validate:
        response = admin_client.post(URL, {
            "config_uuid": str(row.uuid),
            "config": {"model": "gpt-4o-mini", "api_key": "******"},
        }, format="json")
    assert response.status_code == 200
    assert validate.call_args.args[1]["api_key"] == "shared-secret-key"
