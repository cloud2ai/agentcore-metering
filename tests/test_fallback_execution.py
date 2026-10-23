"""Tracker-level fallover, billing and real-request recovery contracts."""

import json
from decimal import Decimal
from unittest.mock import patch

import httpx
import pytest
from django.core.cache import cache
from litellm import ModelResponse

from agentcore_metering.adapters.django import DecisionTracker, LLMTracker
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage
from agentcore_metering.adapters.django.services.fallback_config import (
    save_policy,
)
from agentcore_metering.adapters.django.services.fallback_health import (
    ModelAvailability,
)
from agentcore_metering.adapters.django.services.model_fallback import (
    FallbackTarget,
    FallbackUnavailableError,
)

pytestmark = pytest.mark.django_db
MESSAGES = [{"role": "user", "content": "Test a backup request."}]
QUESTIONS = [
    {
        "type": "predicate",
        "name": "relevant",
        "instructions": "Is it AI?",
    }
]
ANSWERS = [{"type": "predicate", "name": "relevant", "probability": 0.9}]
COMPLETION = "litellm.completion"
NATIVE = (
    "agentcore_metering.adapters.django.services.decisions._request_decisions"
)


@pytest.fixture(autouse=True)
def clean_cooldowns():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def chain():
    def create(model, kind="llm"):
        return LLMConfig.objects.create(
            scope="global",
            provider="openai_compatible",
            model_type=kind,
            config={
                "model": model,
                "api_key": "test-only-key",
                "api_base": "https://test.invalid/v1",
                "num_retries": 5,
            },
        )

    primary, backup = create("primary"), create("backup")
    return primary, backup, create


def response(content="ok", finish="stop", refusal=None):
    result = ModelResponse(
        model="served-chat",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                },
                "finish_reason": finish,
            }
        ],
        usage={
            "prompt_tokens": 30,
            "completion_tokens": 7,
            "total_tokens": 37,
        },
    )
    result._hidden_params["response_cost"] = 0.00012
    return result


def configure(primary, backup, **options):
    return save_policy(
        primary,
        {
            "model_uuids": [str(backup.uuid)],
            **options,
        },
    )


def chat(primary, **kwargs):
    return LLMTracker.call_and_track(
        messages=MESSAGES,
        model_uuid=str(primary.uuid),
        **kwargs,
    )


def test_success_preserves_primary_and_one_usage(chain):
    primary, backup, _ = chain
    primary = configure(primary, backup)
    with patch(COMPLETION, return_value=response()) as request:
        assert chat(primary)[0] == "ok"
    assert request.call_count == 1
    row = LLMUsage.objects.get()
    assert row.success and row.cost == Decimal("0.00012")
    assert row.metadata["requested_config_uuid"] == str(primary.uuid)
    assert row.metadata["actual_config_uuid"] == str(primary.uuid)
    assert row.metadata["fallback_attempt"] == 1
    assert "test-only-key" not in str(row.metadata)


def test_timeout_switches_with_actual_usage_and_no_nested_retry(chain):
    primary, backup, _ = chain
    primary = configure(primary, backup)
    state = {"metadata": {"workflow": "test"}}
    with patch(
        COMPLETION, side_effect=[TimeoutError("offline"), response()]
    ) as request:
        content, usage = chat(primary, state=state, reasoning_effort="none")
    assert content == "ok" and usage["total_tokens"] == 37
    assert [call.kwargs["model"] for call in request.call_args_list] == [
        "openai/primary",
        "openai/backup",
    ]
    for call in request.call_args_list:
        assert call.kwargs["num_retries"] == 0
        assert call.kwargs["max_retries"] == 0
        assert 0 < call.kwargs["timeout"] <= 120
        assert call.kwargs["reasoning_effort"] == "none"
    failed, passed = list(LLMUsage.objects.order_by("created_at"))
    assert not failed.success and failed.cost is None
    assert failed.model == "openai/primary"
    assert failed.metadata["fallback_failure_reason"] == "timeout"
    assert passed.success and passed.cost == Decimal("0.00012")
    assert passed.metadata["actual_config_uuid"] == str(backup.uuid)
    assert passed.metadata["fallback_reason"] == "timeout"
    assert (
        failed.metadata["fallback_call_id"]
        == passed.metadata["fallback_call_id"]
    )
    assert len(state["llm_calls"]) == 2
    assert state["metadata"] == {"workflow": "test"}


@pytest.mark.parametrize(
    "content,finish,refusal",
    [
        ("", "content_filter", None),
        ("", "stop", "policy refusal"),
        ('{"ok":', "length", None),
        ("not JSON", "stop", None),
    ],
)
def test_refusal_truncation_and_invalid_json_never_switch(
    chain,
    content,
    finish,
    refusal,
):
    primary, backup, _ = chain
    primary = configure(primary, backup, conditions=["empty_response"])
    with patch(
        COMPLETION, return_value=response(content, finish, refusal)
    ) as request:
        with pytest.raises(ValueError):
            chat(primary, json_mode=True)
    assert request.call_count == 1
    row = LLMUsage.objects.get()
    assert not row.success and row.total_tokens == 37
    assert row.cost == Decimal("0.00012")


@pytest.mark.parametrize("enabled,expected", [(True, 2), (False, 1)])
def test_empty_answer_opt_in_preserves_billed_failure(
    chain, enabled, expected
):
    primary, backup, _ = chain
    conditions = ["timeout", *(["empty_response"] if enabled else [])]
    primary = configure(primary, backup, conditions=conditions)
    with patch(COMPLETION, side_effect=[response(""), response()]) as request:
        if enabled:
            assert chat(primary)[0] == "ok"
        else:
            with pytest.raises(RuntimeError):
                chat(primary)
    assert request.call_count == expected
    first = LLMUsage.objects.order_by("created_at").first()
    assert not first.success and first.cost == Decimal("0.00012")
    assert LLMUsage.objects.count() == expected


def test_cooldown_skips_then_real_traffic_recovers(chain):
    primary, backup, _ = chain
    primary = configure(primary, backup, cooldown_seconds=60)
    with patch(
        COMPLETION,
        side_effect=[TimeoutError(), response(), response(), response()],
    ) as request:
        chat(primary)
        chat(primary)
        assert [call.kwargs["model"] for call in request.call_args_list] == [
            "openai/primary",
            "openai/backup",
            "openai/backup",
        ]
        with patch(
            "agentcore_metering.adapters.django.services.fallback_health.time.time",
            return_value=10**11,
        ):
            chat(primary)
    assert request.call_args_list[-1].kwargs["model"] == "openai/primary"
    assert LLMUsage.objects.count() == 4


def test_only_one_real_request_checks_recovery(chain):
    primary, backup, _ = chain
    primary = configure(primary, backup, cooldown_seconds=0)
    health = ModelAvailability(
        primary, primary, [primary, backup], primary.fallback
    )
    target = FallbackTarget(str(primary.uuid), 120)
    health.failure(target, "timeout")
    assert health.allow(target, "one")
    assert not health.allow(target, "two")
    health.success(target)
    assert health.allow(target, "three")


def test_manual_recovery_and_reset_are_owner_scoped(
    chain,
    normal_user,
    authenticated_client,
    admin_client,
):
    primary, backup, _ = chain
    primary = configure(primary, backup, auto_recover=False)
    health = ModelAvailability(
        primary, primary, [primary, backup], primary.fallback
    )
    target = FallbackTarget(str(primary.uuid), 120)
    health.failure(target, "timeout")
    assert not health.allow(target, "one")
    user_url = f"/api/v1/user/llm-config/{primary.uuid}/fallback/reset"
    authenticated_client.force_authenticate(user=normal_user)
    assert authenticated_client.post(user_url).status_code == 404
    from rest_framework.test import APIClient

    admin_client = APIClient()
    from django.contrib.auth import get_user_model

    staff = get_user_model().objects.get(username="adminuser")
    admin_client.force_authenticate(user=staff)
    admin_url = f"/api/v1/admin/llm-config/{primary.uuid}/fallback/reset"
    assert admin_client.post(admin_url).status_code == 200
    assert health.allow(target, "two")


def test_all_cooling_fails_without_extra_provider_requests(chain):
    primary, backup, _ = chain
    primary = configure(primary, backup)
    with patch(COMPLETION, side_effect=TimeoutError()) as request:
        with pytest.raises(TimeoutError):
            chat(primary)
        with pytest.raises(FallbackUnavailableError):
            chat(primary)
    assert request.call_count == 2
    assert LLMUsage.objects.count() == 2


@pytest.mark.parametrize("llm_backup", [False, True])
def test_decision_native_and_chat_backups_keep_same_contract(
    chain, llm_backup
):
    _, _, create = chain
    primary = create("primary-decision", "decision")
    backup = create("backup", "llm" if llm_backup else "decision")
    primary = configure(primary, backup)
    payload = {
        "answers": ANSWERS,
        "usage": {"input_tokens": 40, "total_tokens": 40},
    }
    side_effect = (
        [TimeoutError()] if llm_backup else [TimeoutError(), (payload, None)]
    )
    with (
        patch(NATIVE, side_effect=side_effect) as native,
        patch(
            COMPLETION,
            return_value=response(json.dumps({"answers": ANSWERS})),
        ) as completion,
    ):
        answers, usage = DecisionTracker.call_and_track(
            input="AI test",
            questions=QUESTIONS,
            model_uuid=str(primary.uuid),
        )
    assert answers == ANSWERS
    assert native.call_count == (1 if llm_backup else 2)
    assert completion.call_count == int(llm_backup)
    if llm_backup:
        kwargs = completion.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["reasoning_effort"] == "none"
        assert "input" not in kwargs and "questions" not in kwargs
        material = json.loads(kwargs["messages"][1]["content"])
        assert material == {"input": "AI test", "questions": QUESTIONS}
    failed, passed = list(LLMUsage.objects.order_by("created_at"))
    assert not failed.success and passed.success
    assert passed.metadata["actual_config_uuid"] == str(backup.uuid)
    assert passed.metadata["model_type"] == (
        "llm" if llm_backup else "decision"
    )


def test_decision_invalid_chat_response_is_billed_once_without_next_backup(
    chain,
):
    _, _, create = chain
    primary = create("primary-decision", "decision")
    backup, third = create("backup"), create("third")
    primary = save_policy(
        primary, {"model_uuids": [str(backup.uuid), str(third.uuid)]}
    )
    with (
        patch(NATIVE, side_effect=TimeoutError()),
        patch(
            COMPLETION,
            return_value=response("invalid JSON"),
        ) as completion,
    ):
        with pytest.raises(ValueError):
            DecisionTracker.call_and_track(
                input="AI test",
                questions=QUESTIONS,
                model_uuid=str(primary.uuid),
            )
    assert completion.call_count == 1
    failed = LLMUsage.objects.order_by("created_at").last()
    assert not failed.success and failed.cost == Decimal("0.00012")
    assert LLMUsage.objects.count() == 2


def stream_chunk(text="ok", finish="stop", *, usage=True, refusal=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, refusal=refusal),
                finish_reason=finish,
            )
        ],
        model="served-chat",
        usage=(
            SimpleNamespace(
                prompt_tokens=30, completion_tokens=7, total_tokens=37
            )
            if usage
            else None
        ),
    )


def test_stream_switches_only_before_first_output(chain):
    primary, backup, _ = chain
    primary = configure(primary, backup)

    def failed():
        raise TimeoutError("before any output")
        yield

    with patch(
        COMPLETION, side_effect=[failed(), iter([stream_chunk()])]
    ) as request:
        assert list(chat(primary, stream=True)) == [("content", "ok")]
    assert request.call_count == 2
    assert list(
        LLMUsage.objects.order_by("created_at").values_list(
            "success",
            flat=True,
        )
    ) == [False, True]


@pytest.mark.parametrize("kind", ["content", "reasoning"])
def test_stream_never_replays_after_content_or_reasoning(chain, kind):
    primary, backup, _ = chain
    primary = configure(primary, backup)

    def failed():
        chunk = stream_chunk("partial", None, usage=False)
        if kind == "reasoning":
            chunk.choices[0].delta.content = None
            chunk.choices[0].delta.reasoning_content = "partial"
        yield chunk
        raise TimeoutError("after output")

    with patch(COMPLETION, return_value=failed()) as request:
        stream = chat(primary, stream=True)
        assert next(stream) == (kind, "partial")
        with pytest.raises(TimeoutError):
            next(stream)
    assert request.call_count == 1
    row = LLMUsage.objects.get()
    assert not row.success and row.cost is None
    assert row.metadata["usage_may_be_incomplete"] is True


@pytest.mark.parametrize(
    "finish,refusal",
    [("length", None), ("content_filter", None), ("stop", "no")],
)
def test_stream_refusal_or_truncation_does_not_switch(chain, finish, refusal):
    primary, backup, _ = chain
    primary = configure(primary, backup, conditions=["empty_response"])
    with patch(
        COMPLETION,
        return_value=iter(
            [
                stream_chunk("", finish, refusal=refusal),
            ]
        ),
    ) as request:
        with pytest.raises(ValueError):
            list(chat(primary, stream=True))
    assert request.call_count == 1
    assert not LLMUsage.objects.get().success


def test_empty_stream_uses_opt_in_backup(chain):
    primary, backup, _ = chain
    primary = configure(primary, backup, conditions=["empty_response"])
    with patch(
        COMPLETION,
        side_effect=[
            iter([stream_chunk("")]),
            iter([stream_chunk()]),
        ],
    ) as request:
        assert list(chat(primary, stream=True)) == [("content", "ok")]
    assert request.call_count == 2
    assert LLMUsage.objects.count() == 2


def test_known_incompatible_backup_skipped_without_provider_call(chain):
    primary, backup, create = chain
    backup.config["model_info"] = {"supported_modalities": ["text"]}
    backup.save(update_fields=["config", "updated_at"])
    third = create("vision-backup")
    third.config["model_info"] = {"supported_modalities": ["text", "image"]}
    third.save(update_fields=["config", "updated_at"])
    primary = save_policy(
        primary,
        {
            "model_uuids": [str(backup.uuid), str(third.uuid)],
        },
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://image.invalid/a.png"},
                },
            ],
        }
    ]
    with patch(
        COMPLETION, side_effect=[TimeoutError(), response()]
    ) as request:
        LLMTracker.call_and_track(
            messages=messages, model_uuid=str(primary.uuid)
        )
    assert [call.kwargs["model"] for call in request.call_args_list] == [
        "openai/primary",
        "openai/vision-backup",
    ]
    assert LLMUsage.objects.count() == 2


def test_connection_test_never_masks_primary_failure(chain):
    from agentcore_metering.adapters.django.services.runtime_config import (
        run_test_call,
    )

    primary, backup, _ = chain
    primary = configure(primary, backup)
    primary.config["num_retries"] = 0
    primary.save(update_fields=["config"])
    with patch(COMPLETION, side_effect=TimeoutError()) as request:
        ok, _, _ = run_test_call(str(primary.uuid), None, "test", None)
    assert not ok and request.call_count == 1


def test_choice_adapter_preserves_boolean_and_string_identities(chain):
    _, _, create = chain
    primary, backup = create("native", "decision"), create("chat")
    primary = configure(primary, backup)
    question = [
        {
            "name": "choice",
            "type": "choice",
            "instructions": "Choose.",
            "choices": [{"value": True}, {"value": "false"}],
        }
    ]
    answers = [
        {
            "name": "choice",
            "type": "choice",
            "choice": True,
            "probabilities": [
                {"value": True, "probability": 0.8},
                {"value": "false", "probability": 0.2},
            ],
        }
    ]
    with (
        patch(NATIVE, side_effect=TimeoutError()),
        patch(
            COMPLETION,
            return_value=response(json.dumps({"answers": answers})),
        ),
    ):
        actual, _ = DecisionTracker.call_and_track(
            input="test",
            questions=question,
            model_uuid=str(primary.uuid),
        )
    assert actual == answers


@pytest.mark.parametrize(
    "probabilities",
    [
        [{"value": "a", "probability": 0.8}],
        [
            {"value": "a", "probability": 0.8},
            {"value": "b", "probability": 0.8},
        ],
        [
            {"value": "a", "probability": 0.2},
            {"value": "b", "probability": 0.8},
        ],
        [
            {"value": "a", "probability": 0.8},
            {"value": "a", "probability": 0.2},
        ],
    ],
)
def test_choice_adapter_rejects_incomplete_or_wrong_distributions(
    chain, probabilities
):
    _, _, create = chain
    primary, backup = create("native", "decision"), create("chat")
    primary = configure(primary, backup)
    question = [
        {
            "name": "c",
            "type": "choice",
            "instructions": "Choose.",
            "choices": [{"value": "a"}, {"value": "b"}],
        }
    ]
    answer = {
        "name": "c",
        "type": "choice",
        "choice": "a",
        "probabilities": probabilities,
    }
    with (
        patch(NATIVE, side_effect=TimeoutError()),
        patch(
            COMPLETION,
            return_value=response(json.dumps({"answers": [answer]})),
        ),
    ):
        with pytest.raises(ValueError):
            DecisionTracker.call_and_track(
                input="test",
                questions=question,
                model_uuid=str(primary.uuid),
            )
    assert LLMUsage.objects.count() == 2
    assert not LLMUsage.objects.order_by("created_at").last().success


def test_real_gateway_quota_body_switches_without_retry(chain, monkeypatch):
    _, _, create = chain
    primary, backup = (
        create("native-primary", "decision"),
        create("native-backup", "decision"),
    )
    primary = configure(primary, backup, retries=1)
    calls = []
    original = httpx.Client

    def handler(request):
        data = json.loads(request.content)
        calls.append(data["model"])
        if data["model"] == "native-primary":
            return httpx.Response(
                500,
                json={
                    "error": {"code": "insufficient_quota"},
                },
            )
        return httpx.Response(
            200, json={"answers": ANSWERS, "usage": {"input_tokens": 40}}
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )
    answers, _ = DecisionTracker.call_and_track(
        model_uuid=str(primary.uuid),
        input="test",
        questions=QUESTIONS,
    )
    assert answers == ANSWERS
    assert calls == ["native-primary", "native-backup"]
    failed = LLMUsage.objects.order_by("created_at").first()
    assert failed.metadata["fallback_failure_reason"] == "quota_exhausted"
