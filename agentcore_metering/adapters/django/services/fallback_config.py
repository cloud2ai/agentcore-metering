"""Scope-safe configuration and flat backup plans for model calls."""

from uuid import UUID

from django.db import transaction
from django.db.models import Q

from agentcore_metering.adapters.django.llm_static.load import (
    get_decision_providers,
)
from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services.litellm_params import (
    OFFICIAL_API_BASES,
)

REASONS = (
    "timeout",
    "rate_limit",
    "upstream_error",
    "connection_error",
    "quota_exhausted",
    "empty_response",
)
DEFAULT_POLICY = {
    "model_uuids": [],
    "inherit_global": False,
    "conditions": list(REASONS[:-1]),
    "timeout_seconds": 120,
    "retries": 0,
    "cooldown_seconds": 60,
    "auto_recover": True,
}
_GRAPH_FIELDS = (
    "id",
    "uuid",
    "scope",
    "user_id",
    "model_type",
    "is_active",
    "is_default",
    "created_at",
    "updated_at",
    "fallback",
    "config",
)


def same_upstream(root, target):
    """Compare request destinations locally without exposing credentials."""

    def identity(row):
        config = row.config or {}
        provider = row.provider.strip().lower()
        default_base = OFFICIAL_API_BASES.get(provider, "")
        if row.model_type == LLMConfig.MODEL_TYPE_DECISION:
            definition = get_decision_providers().get(provider, {})
            default_base = definition.get("default_api_base", "")
        return (
            row.model_type,
            provider,
            str(config.get("model") or "").strip(),
            str(config.get("api_base") or default_base or "")
            .strip()
            .rstrip("/"),
            str(config.get("api_key") or "").strip(),
            str(config.get("account_id") or "").strip(),
            str(config.get("api_version") or "").strip(),
        )

    return identity(root) == identity(target)


def normalize_policy(value):
    """Validate a complete policy without ever accepting credentials."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - set(DEFAULT_POLICY):
        raise ValueError("Invalid fallback configuration fields.")
    policy = {**DEFAULT_POLICY, **value}
    for key in ("inherit_global", "auto_recover"):
        if not isinstance(policy[key], bool):
            raise ValueError(f"{key} must be a boolean.")
    limits = {
        "timeout_seconds": (5, 600),
        "retries": (0, 1),
        "cooldown_seconds": (0, 3600),
    }
    for key, (low, high) in limits.items():
        number = policy[key]
        if type(number) is not int or not low <= number <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    identities = policy["model_uuids"]
    if not isinstance(identities, list) or len(identities) > 3:
        raise ValueError("Select at most three backup models.")
    try:
        identities = [str(UUID(value)) for value in identities]
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Backup models must be configuration UUIDs.")
    if len(set(identities)) != len(identities):
        raise ValueError("Backup models must be distinct.")
    policy["model_uuids"] = identities
    conditions = policy["conditions"]
    if (
        not isinstance(conditions, list)
        or any(
            not isinstance(key, str) or key not in REASONS
            for key in conditions
        )
        or len(set(conditions)) != len(conditions)
    ):
        raise ValueError("Select supported, distinct failure conditions.")
    if identities and not conditions:
        raise ValueError("Select at least one fallback condition.")
    policy["conditions"] = list(conditions)
    if policy["inherit_global"] and identities:
        raise ValueError("An inherited policy cannot have local backups.")
    return policy


def visible_graph(root):
    scopes = Q(scope=LLMConfig.Scope.GLOBAL, user__isnull=True)
    if root.scope == LLMConfig.Scope.USER and root.user_id is not None:
        scopes |= Q(scope=LLMConfig.Scope.USER, user_id=root.user_id)
    return LLMConfig.objects.filter(scopes)


def compatible(root, target):
    allowed = {
        LLMConfig.MODEL_TYPE_LLM: {LLMConfig.MODEL_TYPE_LLM},
        LLMConfig.MODEL_TYPE_DECISION: {
            LLMConfig.MODEL_TYPE_LLM,
            LLMConfig.MODEL_TYPE_DECISION,
        },
    }
    return target.model_type in allowed.get(root.model_type, set())


def inherited_origin(root, rows):
    candidates = [
        row
        for row in rows.values()
        if (
            row.scope == LLMConfig.Scope.GLOBAL
            and row.user_id is None
            and row.model_type == root.model_type
            and row.is_active
        )
    ]
    candidates.sort(
        key=lambda row: (
            not row.is_default,
            row.created_at,
            row.pk,
        )
    )
    return candidates[0] if candidates else None


def effective_policy(root, rows, *, strict=False):
    policy = normalize_policy(root.fallback)
    origin = root
    if policy and policy["inherit_global"]:
        if root.scope != LLMConfig.Scope.USER:
            raise ValueError("Only personal models can inherit a policy.")
        origin = inherited_origin(root, rows)
        policy = normalize_policy(origin.fallback) if origin else None
        if not policy or not policy["model_uuids"]:
            if strict:
                raise ValueError("No global backup policy is configured.")
            return None, origin
    return policy, origin


def _validate_graph(root, rows):
    if root.model_type not in ("llm", "decision") and root.fallback:
        raise ValueError("Fallback supports chat and decision models only.")
    visited = set()

    def visit(row, path):
        identity = str(row.uuid)
        if identity in path:
            raise ValueError("Fallback model relationships contain a cycle.")
        if identity in visited:
            return
        if len(visited) >= 256:
            raise ValueError("The fallback graph is too large.")
        visited.add(identity)
        policy, _ = effective_policy(row, rows, strict=True)
        for target_id in (policy or {}).get("model_uuids", []):
            target = rows.get(target_id)
            if target is None or not target.is_active:
                raise ValueError(
                    "A backup is unavailable or outside the model's scope."
                )
            # Scope is checked per edge, including existing relationships.
            if target.scope == LLMConfig.Scope.USER and (
                row.scope != LLMConfig.Scope.USER
                or row.user_id != target.user_id
            ):
                raise ValueError("A backup is outside the model's scope.")
            if not compatible(row, target):
                raise ValueError("A backup model has incompatible abilities.")
            if target.pk != row.pk and same_upstream(row, target):
                raise ValueError(
                    "A backup must use a different model, endpoint or "
                    "credential from its primary."
                )
            visit(target, path | {identity})

    visit(root, set())


def save_policy(root, value):
    """Serialize graph writes, including simultaneous inverse references."""
    policy = normalize_policy(value)
    with transaction.atomic():
        # Every graph writer locks the same scoped rows in PK order. Lock
        # first, then read again so validation sees the latest committed
        # graph after a competing writer has released its locks.
        list(
            visible_graph(root)
            .order_by("pk")
            .select_for_update()
            .values_list(
                "pk",
                flat=True,
            )
        )
        rows = {
            str(row.uuid): row
            for row in visible_graph(root).only(*_GRAPH_FIELDS)
        }
        current = rows.get(str(root.uuid))
        if current is None:
            raise ValueError("Model configuration is no longer available.")
        current.fallback = policy
        _validate_graph(current, rows)
        current.save(update_fields=["fallback", "updated_at"])
    return current


def resolve_plan(root):
    """Only resolve explicit backups; never execute recursive backup lists."""
    if not root.fallback:
        return None, root, []
    rows = {
        str(row.uuid): row for row in visible_graph(root).only(*_GRAPH_FIELDS)
    }
    policy, origin = effective_policy(root, rows)
    if not policy or not policy["model_uuids"]:
        return None, origin, []
    selected = []
    for identity in policy["model_uuids"]:
        row = rows.get(identity)
        if (
            row
            and row.pk != root.pk
            and row.is_active
            and compatible(root, row)
            and not same_upstream(root, row)
        ):
            selected.append(row)
    full_rows = LLMConfig.objects.in_bulk([row.pk for row in selected])
    return (
        policy,
        origin,
        [full_rows[row.pk] for row in selected if row.pk in full_rows],
    )
