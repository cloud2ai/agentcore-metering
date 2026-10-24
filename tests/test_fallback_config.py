"""Ownership, compatibility and graph validation for backup settings."""

from uuid import uuid4

import pytest

from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services.fallback_config import (
    normalize_policy,
    resolve_plan,
    save_policy,
)

pytestmark = pytest.mark.django_db
BASE = "/api/v1/admin/llm-config/"
USER_BASE = "/api/v1/user/llm-config/"


@pytest.fixture
def models(normal_user, django_user_model):
    other = django_user_model.objects.create_user(username="other-owner")

    def create(user=None, kind="llm", active=True):
        return LLMConfig.objects.create(
            scope="user" if user else "global",
            user=user,
            model_type=kind,
            provider="openai_compatible",
            config={
                "model": "test-chat",
                "api_key": "private-key",
                "api_base": (
                    f"https://fixture-{LLMConfig.objects.count()}.invalid/v1"
                ),
            },
            is_active=active,
        )

    return {
        "global": create(),
        "backup": create(),
        "personal": create(normal_user),
        "own_backup": create(normal_user),
        "foreign": create(other),
        "decision": create(kind="decision"),
        "inactive": create(active=False),
    }


def settings(*rows, **options):
    return {"model_uuids": [str(row.uuid) for row in rows], **options}


def test_admin_read_write_and_clear_preserve_credentials(admin_client, models):
    root, backup = models["global"], models["backup"]
    url = f"{BASE}{root.uuid}/fallback"
    reply = admin_client.put(
        url, {"fallback": settings(backup)}, format="json"
    )
    assert reply.status_code == 200
    assert reply.data["fallback"]["model_uuids"] == [str(backup.uuid)]
    assert "private-key" not in str(reply.data)
    assert reply.data["backups"][0]["model"] == "test-chat"
    root.refresh_from_db()
    assert root.config["api_key"] == "private-key"
    assert (
        admin_client.put(url, {"fallback": None}, format="json").status_code
        == 200
    )
    root.refresh_from_db()
    assert root.fallback is None


def test_user_can_select_own_backup(
    authenticated_client,
    models,
):
    root = models["personal"]
    reply = authenticated_client.put(
        f"{USER_BASE}{root.uuid}/fallback",
        {"fallback": settings(models["own_backup"])},
        format="json",
    )
    assert reply.status_code == 200
    assert len(reply.data["backups"]) == 1


@pytest.mark.parametrize(
    "target", ["global", "foreign", "inactive", "decision"],
)
def test_bad_backup_cannot_be_saved_by_user(
    authenticated_client,
    models,
    target,
):
    root = models["personal"]
    reply = authenticated_client.put(
        f"{USER_BASE}{root.uuid}/fallback",
        {"fallback": settings(models[target])},
        format="json",
    )
    assert reply.status_code == 400
    root.refresh_from_db()
    assert root.fallback is None


@pytest.mark.parametrize("root_key", ["foreign", "global"])
def test_user_cannot_edit_foreign_or_shared_policy(
    authenticated_client,
    models,
    root_key,
):
    reply = authenticated_client.put(
        f"{USER_BASE}{models[root_key].uuid}/fallback",
        {"fallback": None},
        format="json",
    )
    assert reply.status_code == 404


def test_admin_cannot_reference_another_users_credentials(models):
    with pytest.raises(ValueError, match="scope"):
        save_policy(models["personal"], settings(models["foreign"]))
    with pytest.raises(ValueError, match="scope"):
        save_policy(models["global"], settings(models["personal"]))


def test_self_reference_and_indirect_cycle_rejected(models):
    first, second, third = (
        models["global"],
        models["backup"],
        models["decision"],
    )
    third.model_type = "llm"
    third.save(update_fields=["model_type"])
    with pytest.raises(ValueError, match="cycle"):
        save_policy(first, settings(first))
    save_policy(first, settings(second))
    save_policy(second, settings(third))
    with pytest.raises(ValueError, match="cycle"):
        save_policy(third, settings(first))
    third.refresh_from_db()
    assert third.fallback is None


@pytest.mark.parametrize("base", [BASE, USER_BASE])
def test_identical_personal_destination_is_not_a_backup(
    request,
    models,
    base,
):
    root, duplicate = models["personal"], models["own_backup"]
    duplicate.config = {
        **root.config,
        "api_base": root.config["api_base"] + "/",
        "max_tokens": 1000,
        "num_retries": 1,
    }
    duplicate.save(update_fields=["config"])
    client = request.getfixturevalue(
        "admin_client" if base == BASE else "authenticated_client"
    )
    url = f"{base}{root.uuid}/fallback"
    reply = client.get(url)
    assert reply.status_code == 200
    candidates = {item["uuid"] for item in reply.data["candidates"]}
    assert str(root.uuid) not in candidates
    assert str(duplicate.uuid) not in candidates
    reply = client.put(url, {"fallback": settings(duplicate)}, format="json")
    assert reply.status_code == 400
    assert "private-key" not in str(reply.data)
    root.refresh_from_db()
    assert root.fallback is None


@pytest.mark.parametrize(
    "base, root_key, expected",
    [
        (BASE, "global", {"backup"}),
        (BASE, "personal", {"own_backup"}),
        (BASE, "foreign", set()),
        (USER_BASE, "personal", {"own_backup"}),
    ],
)
def test_candidates_follow_model_owner_not_viewer(
    request, models, base, root_key, expected,
):
    client = request.getfixturevalue(
        "admin_client" if base == BASE else "authenticated_client"
    )
    reply = client.get(f"{base}{models[root_key].uuid}/fallback")
    assert reply.status_code == 200
    assert {row["uuid"] for row in reply.data["candidates"]} == {
        str(models[key].uuid) for key in expected
    }


@pytest.mark.parametrize(
    "root_key, target_key",
    [("global", "personal"), ("personal", "global"),
     ("personal", "foreign")],
)
def test_admin_cannot_bypass_local_backup_scope(
    admin_client, models, root_key, target_key,
):
    root = models[root_key]
    reply = admin_client.put(
        f"{BASE}{root.uuid}/fallback",
        {"fallback": settings(models[target_key])}, format="json",
    )
    assert reply.status_code == 400
    assert "scope" in reply.data["detail"]
    root.refresh_from_db()
    assert root.fallback is None


def test_legacy_mixed_policy_remains_readable_until_reconfigured(
    authenticated_client, models,
):
    root, backup = models["personal"], models["global"]
    # Older releases allowed this direct edge. Do not rewrite stored policy
    # or silently change running calls during a configuration-only fix.
    LLMConfig.objects.filter(pk=root.pk).update(fallback=settings(backup))
    root.refresh_from_db()
    assert [row.pk for row in resolve_plan(root)[2]] == [backup.pk]
    url = f"{USER_BASE}{root.uuid}/fallback"
    reply = authenticated_client.get(url)
    assert reply.status_code == 200
    assert reply.data["backups"][0]["uuid"] == str(backup.uuid)
    assert str(backup.uuid) not in {
        row["uuid"] for row in reply.data["candidates"]
    }
    reply = authenticated_client.put(
        url, {"fallback": settings(backup)}, format="json",
    )
    assert reply.status_code == 400
    reply = authenticated_client.put(
        url, {"fallback": settings(models["own_backup"])}, format="json",
    )
    assert reply.status_code == 200
    assert reply.data["backups"][0]["uuid"] == str(models["own_backup"].uuid)


@pytest.mark.parametrize("base", [BASE, USER_BASE])
def test_explicit_inheritance_is_separate_from_local_candidates(
    request, models, base,
):
    global_root, backup = models["global"], models["backup"]
    global_root.is_default = True
    global_root.save(update_fields=["is_default"])
    save_policy(global_root, settings(backup))
    client = request.getfixturevalue(
        "admin_client" if base == BASE else "authenticated_client"
    )
    reply = client.put(
        f"{base}{models['personal'].uuid}/fallback",
        {"fallback": {"inherit_global": True}}, format="json",
    )
    assert reply.status_code == 200
    assert reply.data["inherited_from"] == str(global_root.uuid)
    assert reply.data["backups"][0]["uuid"] == str(backup.uuid)
    assert {row["uuid"] for row in reply.data["candidates"]} == {
        str(models["own_backup"].uuid),
    }


def test_own_backup_can_explicitly_inherit_global_policy(models):
    save_policy(models["global"], settings(models["backup"]))
    save_policy(models["own_backup"], {"inherit_global": True})
    root = save_policy(models["personal"], settings(models["own_backup"]))
    assert [row.pk for row in resolve_plan(root)[2]] == [
        models["own_backup"].pk,
    ]


@pytest.mark.parametrize("field", ["api_key", "api_base", "model"])
def test_same_name_with_distinct_route_can_be_a_backup(models, field):
    root, target = models["personal"], models["own_backup"]
    target.config = {**root.config, field: "different-value"}
    target.save(update_fields=["config"])
    root = save_policy(root, settings(target))
    assert [row.pk for row in resolve_plan(root)[2]] == [target.pk]


def test_later_credential_edit_cannot_make_primary_its_own_backup(models):
    root, target = models["personal"], models["own_backup"]
    root = save_policy(root, settings(target))
    target.config = root.config.copy()
    target.save(update_fields=["config"])
    assert resolve_plan(root)[2] == []


def test_deleted_or_disabled_backup_is_never_resolved(models):
    root, first, second = (
        models["global"],
        models["inactive"],
        models["backup"],
    )
    first.is_active = True
    first.save(update_fields=["is_active"])
    save_policy(root, settings(first, second))
    first.delete()
    second.is_active = False
    second.save(update_fields=["is_active"])
    root.refresh_from_db()
    policy, _, targets = resolve_plan(root)
    assert policy["model_uuids"]
    assert targets == []


def test_inherit_matching_global_policy_without_copying_secrets(models):
    root, global_root, backup = (
        models["personal"],
        models["global"],
        models["backup"],
    )
    global_root.is_default = True
    global_root.save(update_fields=["is_default"])
    save_policy(global_root, settings(backup, timeout_seconds=45))
    root = save_policy(root, {"inherit_global": True})
    policy, origin, targets = resolve_plan(root)
    assert origin.pk == global_root.pk
    assert policy["timeout_seconds"] == 45
    assert [row.pk for row in targets] == [backup.pk]
    with pytest.raises(ValueError):
        save_policy(global_root, {"inherit_global": True})


def test_inherit_requires_existing_global_backup_policy(models):
    with pytest.raises(ValueError, match="global backup"):
        save_policy(models["personal"], {"inherit_global": True})


def test_native_decision_can_use_chat_backup_but_not_reverse(models):
    root = save_policy(models["decision"], settings(models["global"]))
    assert resolve_plan(root)[2][0].pk == models["global"].pk
    with pytest.raises(ValueError, match="incompatible"):
        save_policy(models["global"], settings(models["decision"]))


def test_ordinary_admin_edit_preserves_backup_configuration(
    admin_client, models
):
    root = save_policy(models["global"], settings(models["backup"]))
    reply = admin_client.put(
        f"{BASE}{root.uuid}/",
        {"is_active": False},
        format="json",
    )
    assert reply.status_code == 200
    root.refresh_from_db()
    assert root.fallback["model_uuids"] == [str(models["backup"].uuid)]


@pytest.mark.parametrize(
    "policy",
    [
        [],
        {"api_key": "must-not-be-stored"},
        {"model_uuids": ["not-a-uuid"]},
        {"model_uuids": [str(uuid4())] * 2},
        {"retries": 5},
        {"timeout_seconds": True},
        {"cooldown_seconds": -1},
        {"conditions": ["content_refusal"]},
        {"auto_recover": "false"},
        {"model_uuids": [str(uuid4())], "conditions": []},
    ],
)
def test_invalid_configuration_is_rejected(policy):
    with pytest.raises(ValueError):
        normalize_policy(policy)
