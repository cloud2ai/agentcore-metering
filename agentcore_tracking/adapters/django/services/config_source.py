"""
Resolve LLM configuration from DB (global or per-user) or Django settings.

Multiple configs per scope: active configs ordered by `order`; use the first.
User configs first, then global.
"""
from typing import Any, Dict, List, Optional

from ..models import LLMConfig


def _get_active_configs(scope: str, user_id: Optional[int]) -> List[LLMConfig]:
    qs = (
        LLMConfig.objects.filter(
            scope=scope,
            model_type=LLMConfig.MODEL_TYPE_LLM,
            is_active=True,
        )
        .order_by("order", "id")
    )
    if scope == LLMConfig.Scope.USER and user_id is not None:
        qs = qs.filter(user_id=user_id)
    elif scope == LLMConfig.Scope.GLOBAL:
        qs = qs.filter(user__isnull=True)
    return list(qs)


def _config_to_dict(row: LLMConfig) -> Dict[str, Any]:
    return {
        "provider": (row.provider or "openai").strip().lower(),
        "config": row.config or {},
    }


def get_config_from_db(
    user_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load one LLM config from DB: per-user pool first, then global pool.
    Uses the first active config by order.

    Returns:
        Dict with keys: provider (str), config (dict).
        None if no active config in DB.
    """
    configs = _get_active_configs(LLMConfig.Scope.USER, user_id)
    if not configs and user_id is not None:
        configs = _get_active_configs(LLMConfig.Scope.GLOBAL, None)
    if not configs:
        configs = _get_active_configs(LLMConfig.Scope.GLOBAL, None)
    if not configs:
        return None
    return _config_to_dict(configs[0])


def get_config_list_from_db(
    scope: str,
    user_id: Optional[int] = None,
    model_type: str = LLMConfig.MODEL_TYPE_LLM,
) -> List[Dict[str, Any]]:
    """
    List all configs for a scope (and optionally user), for API list response.
    Does not filter by is_active so UI can show all.
    """
    qs = (
        LLMConfig.objects.filter(scope=scope, model_type=model_type)
        .order_by("order", "id")
    )
    if scope == LLMConfig.Scope.USER:
        if user_id is None:
            return []
        qs = qs.filter(user_id=user_id)
    else:
        qs = qs.filter(user__isnull=True)
    return [
        {
            "id": row.id,
            "uuid": str(row.uuid),
            "scope": row.scope,
            "user_id": row.user_id,
            "model_type": row.model_type,
            "provider": row.provider,
            "config": row.config,
            "is_active": row.is_active,
            "order": row.order,
            "created_at": (
                row.created_at.isoformat() if row.created_at else None
            ),
            "updated_at": (
                row.updated_at.isoformat() if row.updated_at else None
            ),
        }
        for row in qs
    ]
