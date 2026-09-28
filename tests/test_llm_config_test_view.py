"""
Testing an edited config must not send the masked api_key.

The admin UI fills the edit form from the masked read serializer, so its
api_key is a placeholder (``ak-5***d60f``). The test endpoint receives the
config id and has to resolve masked secrets from storage before validating,
otherwise the placeholder is sent as the real key and auth fails.
"""
from unittest.mock import patch

import pytest

from agentcore_metering.adapters.django.models import LLMConfig

REAL_KEY = "ak-5c807bd992874225abf981c35964d60f"
MASKED_KEY = "ak-5***d60f"
API_BASE = "https://agione.pro/hyperone/xapi/api"
STORED_MODEL = "deepseek/deepseek-v4-pro/d3462"
NEW_MODEL = "openai/gpt-5.6-luna/68499"

TEST_URL = "/api/v1/admin/llm-config/test/"
VALIDATE_TARGET = (
    "agentcore_metering.adapters.django.views.config_validation"
    ".validate_llm_config"
)


def _stored_config():
    return {
        "api_key": REAL_KEY,
        "api_base": API_BASE,
        "model": STORED_MODEL,
    }


def _form_config(model=NEW_MODEL):
    return {
        "api_key": MASKED_KEY,
        "api_base": API_BASE,
        "model": model,
    }


@pytest.fixture
def stored_config():
    return LLMConfig.objects.create(
        scope=LLMConfig.Scope.GLOBAL,
        user=None,
        model_type=LLMConfig.MODEL_TYPE_LLM,
        provider="agione",
        config=_stored_config(),
        is_active=True,
    )


@pytest.mark.api
@pytest.mark.django_db
class TestLLMConfigTestMaskedSecret:
    def test_masked_api_key_resolved_from_stored_config(
        self, admin_client, stored_config
    ):
        payload = {
            "provider": "agione",
            "config": _form_config(),
            "config_uuid": str(stored_config.uuid),
        }
        with patch(VALIDATE_TARGET) as validate:
            validate.return_value = (True, "")
            response = admin_client.post(TEST_URL, payload, format="json")

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        sent = validate.call_args.args[1]
        # Mask is replaced by the stored secret; the edited model is kept.
        assert sent["api_key"] == REAL_KEY
        assert sent["model"] == NEW_MODEL

    def test_masked_api_key_resolved_via_deprecated_config_id(
        self, admin_client, stored_config
    ):
        payload = {
            "provider": "agione",
            "config": _form_config(),
            "config_id": stored_config.id,
        }
        with patch(VALIDATE_TARGET) as validate:
            validate.return_value = (True, "")
            admin_client.post(TEST_URL, payload, format="json")

        assert validate.call_args.args[1]["api_key"] == REAL_KEY

    def test_without_config_id_the_form_value_is_used(
        self, admin_client, stored_config
    ):
        # No config_uuid/config_id: the payload is validated verbatim, which
        # is how a brand-new config (real key typed in) is tested.
        payload = {"provider": "agione", "config": _form_config()}
        with patch(VALIDATE_TARGET) as validate:
            validate.return_value = (True, "")
            admin_client.post(TEST_URL, payload, format="json")

        assert validate.call_args.args[1]["api_key"] == MASKED_KEY

    def test_a_real_key_in_the_form_still_wins(
        self, admin_client, stored_config
    ):
        # User rotated the key in the form: the new real value is not masked,
        # so it must override the stored one.
        rotated = "ak-newkey0123456789abcdefghijklmno"
        payload = {
            "provider": "agione",
            "config": {**_form_config(), "api_key": rotated},
            "config_uuid": str(stored_config.uuid),
        }
        with patch(VALIDATE_TARGET) as validate:
            validate.return_value = (True, "")
            admin_client.post(TEST_URL, payload, format="json")

        assert validate.call_args.args[1]["api_key"] == rotated
