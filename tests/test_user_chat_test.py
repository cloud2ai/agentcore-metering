"""Saved-key chat tests must enforce ownership before calling a provider."""

from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import litellm
import pytest
from rest_framework.test import APIClient

from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage

pytestmark = pytest.mark.django_db
URL = "/api/v1/user/llm-config/test-call"


@pytest.fixture
def completion(monkeypatch):
    response = SimpleNamespace(
        model="openai/gpt-4o-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=2, total_tokens=12,
            cached_tokens=3, reasoning_tokens=1,
        ),
    )
    completion = Mock(return_value=response)
    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.00000042)
    return completion


def config(user=None, **kwargs):
    return LLMConfig.objects.create(
        scope="user" if user else "global", user=user,
        provider="openai", config={"model": "gpt-4o-mini", "api_key": "k"},
        **kwargs,
    )


@pytest.mark.parametrize("by_id", [False, True])
def test_personal_chat_test_tracks_usage(
    authenticated_client, normal_user, completion, by_id,
):
    row = config(normal_user)
    selector = {"config_id": row.pk} if by_id else {"config_uuid": row.uuid}
    response = authenticated_client.post(
        URL, {**selector, "prompt": "Hello", "max_tokens": 64}, format="json",
    )
    assert response.status_code == 200
    assert response.data["ok"] is True
    assert response.data["content"] == "Hello"
    assert response.data["usage"]["total_tokens"] == 12
    assert completion.call_count == 1
    assert completion.call_args.kwargs["max_tokens"] == 64
    usage = LLMUsage.objects.get(user=normal_user)
    assert usage.metadata["node_name"] == "user_test_call"
    assert float(usage.cost) == 0.00000042


@pytest.mark.parametrize("stream", [False, True])
def test_deleted_owned_uuid_cannot_fall_back_to_foreign_id(
    authenticated_client, normal_user, admin_user, completion,
    monkeypatch, stream,
):
    from agentcore_metering.adapters.django.views.config_validation import (
        UserLLMConfigTestCallView,
    )

    owned = config(normal_user)
    foreign = config(admin_user)
    owned_uuid = str(owned.uuid)
    check = UserLLMConfigTestCallView.can_test

    def delete_after_check(self, request, data):
        allowed = check(self, request, data)
        if allowed:
            owned.delete()
        return allowed

    monkeypatch.setattr(
        UserLLMConfigTestCallView, "can_test", delete_after_check,
    )
    response = authenticated_client.post(URL, {
        "config_uuid": owned_uuid, "config_id": foreign.pk,
        "prompt": "Hello", "stream": stream,
    }, format="json")
    if response.streaming:
        list(response.streaming_content)
    assert completion.call_count == 0
    assert not LLMUsage.objects.exists()


@pytest.mark.parametrize("kind", [
    "other_user", "global", "disabled", "decision", "embedding",
    "image", "missing", "foreign_uuid_with_own_id",
])
@pytest.mark.parametrize("stream", [False, True])
def test_inaccessible_configs_never_call_provider(
    authenticated_client, normal_user, admin_user, completion, kind, stream,
):
    if kind == "other_user":
        row = config(admin_user)
    elif kind == "global":
        row = config()
    elif kind == "disabled":
        row = config(normal_user, is_active=False)
    elif kind in ("decision", "embedding", "image"):
        model_type = "image_generation" if kind == "image" else kind
        row = config(normal_user, model_type=model_type)
    else:
        row = config(normal_user)
    payload = {"config_uuid": row.uuid, "prompt": "Hello", "stream": stream}
    if kind in ("missing", "foreign_uuid_with_own_id"):
        payload["config_uuid"] = uuid4()
    if kind == "foreign_uuid_with_own_id":
        payload["config_id"] = row.pk
    response = authenticated_client.post(URL, payload, format="json")
    assert response.status_code == 404
    completion.assert_not_called()
    assert not LLMUsage.objects.exists()


def test_staff_can_test_shared_but_not_other_personal_models(
    admin_client, normal_user, completion,
):
    shared = config()
    own = config(normal_user)
    response = admin_client.post(
        URL, {"config_uuid": shared.uuid, "prompt": "Hi"}, format="json",
    )
    assert response.status_code == 200 and response.data["ok"]
    response = admin_client.post(
        URL, {"config_uuid": own.uuid, "prompt": "Hi"}, format="json",
    )
    assert response.status_code == 404
    assert completion.call_count == 1


def test_anonymous_and_invalid_requests_never_call_provider(
    authenticated_client, normal_user, completion,
):
    row = config(normal_user)
    payload = {"config_uuid": row.uuid, "prompt": "Hi"}
    assert APIClient().post(URL, payload, format="json").status_code == 403
    payload["max_tokens"] = 4097
    assert authenticated_client.post(
        URL, payload, format="json",
    ).status_code == 400
    completion.assert_not_called()
