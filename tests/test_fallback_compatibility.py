"""Legacy calls and mixed-version database access remain valid."""

import copy
import json
from unittest.mock import patch

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from litellm import ModelResponse

from agentcore_metering.adapters.django import DecisionTracker, LLMTracker
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage
from agentcore_metering.adapters.django.services.config_source import (
    get_config_from_db,
)
from agentcore_metering.adapters.django.services.fallback_config import (
    save_policy,
)
from agentcore_metering.adapters.django.services.runtime_config import (
    get_litellm_params,
)

pytestmark = pytest.mark.django_db
PREVIOUS = ("agentcore_metering", "0019_increase_cost_precision")
CURRENT = ("agentcore_metering", "0020_llmconfig_fallback")


def create_model(name, *, user=None, kind="llm"):
    return LLMConfig.objects.create(
        scope="user" if user else "global",
        user=user,
        provider="openai_compatible",
        model_type=kind,
        config={
            "model": name,
            "api_base": "https://fixture.invalid/v1",
            "api_key": "fixture-key",
            "request_timeout_seconds": 23,
            "num_retries": 0,
        },
    )


def test_internal_config_metadata_preserves_mapping_contract():
    row = create_model("legacy-chat")
    resolved = get_config_from_db(model_uuid=str(row.uuid))
    assert isinstance(resolved, dict)
    assert set(resolved) == {"provider", "config"}
    assert json.loads(json.dumps(resolved)) == dict(resolved)
    assert copy.deepcopy(resolved) == dict(resolved)

    params = get_litellm_params(model_uuid=str(row.uuid))
    assert isinstance(params, dict)
    assert "model_config" not in params
    assert "fallback" not in params
    assert "fallback" not in params["model"]
    assert json.loads(json.dumps(params)) == dict(params)
    assert copy.deepcopy(params) == dict(params)


@pytest.mark.parametrize("kind", ["llm", "decision"])
@pytest.mark.parametrize("failure", [False, True])
def test_personal_models_do_not_implicitly_inherit_global_policy(
    normal_user,
    kind,
    failure,
):
    global_model = create_model("global-primary", kind=kind)
    backup = create_model("global-backup", kind=kind)
    save_policy(global_model, {"model_uuids": [str(backup.uuid)]})
    personal = create_model("personal-primary", user=normal_user, kind=kind)
    assert personal.fallback is None
    error = RuntimeError("legacy provider failure")
    if kind == "llm":
        request_path = "litellm.completion"
        result = ModelResponse(
            model="personal-primary",
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 2, "completion_tokens": 1},
        )
    else:
        request_path = (
            "agentcore_metering.adapters.django.services.decisions."
            "_request_decisions"
        )
        result = (
            {
                "answers": [
                    {
                        "name": "keep",
                        "type": "predicate",
                        "probability": 0.8,
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            None,
        )
    with patch(
        request_path,
        side_effect=error if failure else None,
        return_value=result,
    ) as request:

        def invoke():
            state = {"user_id": normal_user.pk}
            if kind == "llm":
                return LLMTracker.call_and_track(
                    messages=[{"role": "user", "content": "Hello"}],
                    state=state,
                )
            return DecisionTracker.call_and_track(
                input="News",
                questions=[
                    {
                        "name": "keep",
                        "type": "predicate",
                        "instructions": "Keep it?",
                    }
                ],
                model_uuid=str(personal.uuid),
                state=state,
            )

        if failure:
            with pytest.raises(RuntimeError) as caught:
                invoke()
            assert caught.value is error
            assert not hasattr(error, "agentcore_no_retry")
        else:
            invoke()
    assert request.call_count == 1
    assert request.call_args.kwargs["model"].endswith("personal-primary")
    # The existing transport subtracts setup time from its timeout budget.
    assert 0 < request.call_args.kwargs["timeout"] <= 23
    usage = LLMUsage.objects.get()
    assert usage.success is not failure
    assert "fallback_call_id" not in usage.metadata


@pytest.mark.django_db(transaction=True)
def test_upgrade_and_outgoing_orm_preserve_existing_data():
    executor = MigrationExecutor(connection)
    restore_targets = executor.loader.graph.leaf_nodes()
    try:
        executor.migrate([PREVIOUS])
        old_apps = executor.loader.project_state([PREVIOUS]).apps
        old_model = old_apps.get_model("agentcore_metering", "LLMConfig")
        old_usage = old_apps.get_model("agentcore_metering", "LLMUsage")
        old_row = old_model.objects.create(
            scope="global",
            provider="openai_compatible",
            config={"model": "existing", "api_key": "fixture-key"},
        )
        usage = old_usage.objects.create(model="existing", total_tokens=42)
        executor = MigrationExecutor(connection)
        executor.migrate([CURRENT])
        row = LLMConfig.objects.get(pk=old_row.pk)
        assert row.fallback is None
        assert row.config == old_row.config
        assert LLMUsage.objects.get(pk=usage.pk).total_tokens == 42

        # During blue/green overlap, old ORM inserts omit the new column.
        outgoing = old_model.objects.create(
            scope="global",
            provider="openai_compatible",
            config={"model": "outgoing", "api_key": "fixture-key"},
        )
        assert LLMConfig.objects.get(pk=outgoing.pk).fallback is None
        backup = create_model("backup")
        save_policy(row, {"model_uuids": [str(backup.uuid)]})
        outgoing_view = old_model.objects.get(pk=row.pk)
        outgoing_view.is_active = False
        outgoing_view.save()
        row.refresh_from_db()
        assert not row.is_active
        assert row.fallback["model_uuids"] == [str(backup.uuid)]
    finally:
        MigrationExecutor(connection).migrate(restore_targets)
