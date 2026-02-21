"""
LLM usage records listing for admin (read-only, paginated).
"""
from typing import Any, Dict, List, Optional

from ..models import LLMUsage

from .usage_stats import _parse_date, _parse_end_date


def get_llm_usage_list(
    page: int = 1,
    page_size: int = 20,
    user_id: Optional[str] = None,
    model_filter: Optional[str] = None,
    success_filter: Optional[str] = None,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Return paginated LLM usage records with filters.
    Applied filters: user_id, model (icontains), success, start_date, end_date.
    """
    page_size = min(max(1, page_size), 100)
    qs = (
        LLMUsage.objects.select_related("user")
        .order_by("-created_at")
    )
    if user_id:
        qs = qs.filter(user_id=user_id)
    if model_filter:
        qs = qs.filter(model__icontains=model_filter)
    if success_filter and success_filter.strip():
        low = success_filter.strip().lower()
        if low == "true":
            qs = qs.filter(success=True)
        elif low == "false":
            qs = qs.filter(success=False)
    if start_date:
        qs = qs.filter(created_at__gte=start_date)
    if end_date:
        qs = qs.filter(created_at__lte=end_date)

    total = qs.count()
    start = (page - 1) * page_size
    usages = qs[start: start + page_size]

    items: List[Dict[str, Any]] = []
    for u in usages:
        created_at = u.created_at.isoformat() if u.created_at else None
        username = u.user.username if u.user else None
        cost = None
        if u.cost is not None:
            cost = float(u.cost)
        items.append({
            "id": str(u.id),
            "user_id": u.user_id,
            "username": username,
            "model": u.model,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
            "cost": cost,
            "cost_currency": u.cost_currency or "USD",
            "success": u.success,
            "error": u.error,
            "created_at": created_at,
            "metadata": u.metadata,
        })

    return {
        "results": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def get_llm_usage_list_from_query(params: Any) -> Dict[str, Any]:
    """
    Build paginated LLM usage list from a query params dict
    (e.g. request.query_params).
    """
    page = int(params.get("page", 1))
    page_size = min(int(params.get("page_size", 20)), 100)
    user_id = (params.get("user_id") or "").strip() or None
    model_filter = (params.get("model") or "").strip() or None
    success_filter = (params.get("success") or "").strip() or None
    start_date = _parse_date(params.get("start_date"))
    end_date = _parse_end_date(params.get("end_date"))
    return get_llm_usage_list(
        page=page,
        page_size=page_size,
        user_id=user_id,
        model_filter=model_filter,
        success_filter=success_filter,
        start_date=start_date,
        end_date=end_date,
    )
