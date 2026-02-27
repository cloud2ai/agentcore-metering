"""
LLM Usage Statistics utilities for admin cost analysis.

All date/time ranges and time-series buckets are interpreted and returned
in the server's local timezone (see _ensure_aware_datetime). Use consistent
start_date/end_date in that timezone when querying.
"""
from datetime import date, datetime, timedelta
from typing import Any, Optional

from django.db.models import Count, Q, Sum
from django.db.models.functions import TruncDate, TruncHour, TruncMonth
from django.utils.dateparse import parse_datetime
from django.utils import timezone

from agentcore_metering.adapters.django.models import LLMUsage


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parse_datetime(value)
        if dt:
            return _ensure_aware_datetime(dt)
        return _ensure_aware_datetime(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        )
    except (ValueError, TypeError):
        return None


def _ensure_aware_datetime(dt: datetime) -> datetime:
    """
    Normalize to timezone-aware datetime in local timezone.
    Stats (buckets, ranges) are reported in local timezone for consistency.
    """
    if timezone.is_naive(dt):
        return timezone.make_aware(dt, timezone.get_current_timezone())
    return timezone.localtime(dt, timezone.get_current_timezone())


def _parse_end_date(value: Optional[str]) -> Optional[datetime]:
    """
    Parse end_date; if value is date-only (no time part), return end of that
    day so that the whole day is included in range filters.
    """
    dt = _parse_date(value)
    if dt is None:
        return None
    value = (value or "").strip()
    if "T" not in value and " " not in value and len(value) <= 10:
        return dt.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    return dt


def get_summary_stats(start_date=None, end_date=None, user_id=None):
    """
    Return aggregate token and cost stats for the given date range and user.

    Optional filters: start_date, end_date (timezone-aware), user_id.
    Returns dict with total_* tokens, total_calls, successful_calls,
    failed_calls, total_cost, total_cost_currency.
    """
    qs = LLMUsage.objects.all()
    if user_id:
        qs = qs.filter(user_id=user_id)
    if start_date:
        qs = qs.filter(created_at__gte=start_date)
    if end_date:
        qs = qs.filter(created_at__lte=end_date)
    agg = qs.aggregate(
        total_calls=Count("id"),
        total_prompt_tokens=Sum("prompt_tokens"),
        total_completion_tokens=Sum("completion_tokens"),
        total_tokens=Sum("total_tokens"),
        total_cached_tokens=Sum("cached_tokens"),
        total_reasoning_tokens=Sum("reasoning_tokens"),
        total_cost=Sum("cost"),
        successful_calls=Count("id", filter=Q(success=True)),
        failed_calls=Count("id", filter=Q(success=False)),
    )
    total_cost = agg.get("total_cost")
    if total_cost is not None:
        total_cost = float(total_cost)
    else:
        total_cost = 0
    return {
        "total_prompt_tokens": agg["total_prompt_tokens"] or 0,
        "total_completion_tokens": agg["total_completion_tokens"] or 0,
        "total_tokens": agg["total_tokens"] or 0,
        "total_cached_tokens": agg["total_cached_tokens"] or 0,
        "total_reasoning_tokens": agg["total_reasoning_tokens"] or 0,
        "total_cost": total_cost,
        "total_cost_currency": "USD",
        "total_calls": agg["total_calls"] or 0,
        "successful_calls": agg["successful_calls"] or 0,
        "failed_calls": agg["failed_calls"] or 0,
    }


def get_stats_by_model(start_date=None, end_date=None, user_id=None):
    """
    Return per-model aggregate stats (calls, tokens, cost) for the given range.

    Optional filters: start_date, end_date, user_id.
    Ordered by total_tokens desc.
    """
    qs = LLMUsage.objects.values("model").annotate(
        total_calls=Count("id"),
        total_prompt_tokens=Sum("prompt_tokens"),
        total_completion_tokens=Sum("completion_tokens"),
        total_tokens=Sum("total_tokens"),
        total_cached_tokens=Sum("cached_tokens"),
        total_reasoning_tokens=Sum("reasoning_tokens"),
        total_cost=Sum("cost"),
    ).order_by("-total_tokens")
    if user_id:
        qs = qs.filter(user_id=user_id)
    if start_date:
        qs = qs.filter(created_at__gte=start_date)
    if end_date:
        qs = qs.filter(created_at__lte=end_date)
    return list(qs)


def get_time_series_stats(
    granularity: str,
    start_date=None,
    end_date=None,
    user_id=None,
):
    """
    Returns time-series token usage points for charting.

    Supported granularity:
    - day: points bucketed by hour
    - month: points bucketed by day
    - year: points bucketed by month
    """
    granularity = (granularity or "").strip().lower()
    if granularity == "day":
        trunc = TruncHour("created_at")
    elif granularity == "month":
        trunc = TruncDate("created_at")
    elif granularity == "year":
        trunc = TruncMonth("created_at")
    else:
        raise ValueError(
            "Unsupported granularity. Use one of: day, month, year."
        )

    qs = (
        LLMUsage.objects.annotate(bucket=trunc)
        .values("bucket")
        .annotate(
            total_calls=Count("id"),
            total_prompt_tokens=Sum("prompt_tokens"),
            total_completion_tokens=Sum("completion_tokens"),
            total_tokens=Sum("total_tokens"),
            total_cached_tokens=Sum("cached_tokens"),
            total_reasoning_tokens=Sum("reasoning_tokens"),
            total_cost=Sum("cost"),
        )
        .order_by("bucket")
    )
    if user_id:
        qs = qs.filter(user_id=user_id)
    if start_date:
        qs = qs.filter(created_at__gte=start_date)
    if end_date:
        qs = qs.filter(created_at__lte=end_date)

    def _row(i):
        bucket = i["bucket"]
        if bucket is None:
            bucket_str = None
        else:
            bucket_str = bucket.isoformat()
        cost = i.get("total_cost")
        cost = float(cost) if cost is not None else 0
        return {
            "bucket": bucket_str,
            "total_calls": i["total_calls"],
            "total_prompt_tokens": i["total_prompt_tokens"] or 0,
            "total_completion_tokens": i["total_completion_tokens"] or 0,
            "total_tokens": i["total_tokens"] or 0,
            "total_cached_tokens": i["total_cached_tokens"] or 0,
            "total_reasoning_tokens": i["total_reasoning_tokens"] or 0,
            "total_cost": cost,
        }

    rows = [_row(r) for r in qs]
    if not start_date or not end_date:
        return rows

    # Fill missing buckets so the series is contiguous. Bucket key must match
    # the annotation: day -> TruncHour (datetime), month -> TruncDate (date),
    # year -> TruncMonth (datetime, first day of month).
    # _build_expected_buckets returns the same types so bucket.isoformat()
    # aligns with by_bucket keys.
    filled = []
    by_bucket = {row["bucket"]: row for row in rows}
    for bucket in _build_expected_buckets(
        granularity=granularity,
        start_date=start_date,
        end_date=end_date,
    ):
        bucket_str = bucket.isoformat()
        filled.append(
            by_bucket.get(bucket_str)
            or {
                "bucket": bucket_str,
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cached_tokens": 0,
                "total_reasoning_tokens": 0,
                "total_cost": 0,
            }
        )
    return filled


def _build_expected_buckets(
    granularity: str,
    start_date: datetime,
    end_date: datetime,
) -> list[date | datetime]:
    """
    Build ordered bucket values for fill. Return type matches the annotation:
    day -> datetime (TruncHour), month -> date (TruncDate),
    year -> datetime first-of-month (TruncMonth).
    """
    if granularity == "day":
        return _hour_buckets(start_date, end_date)
    if granularity == "month":
        return _day_buckets(start_date, end_date)
    if granularity == "year":
        return _month_buckets(start_date, end_date)
    return []


def _hour_buckets(start_date: datetime, end_date: datetime) -> list[datetime]:
    start_date = _ensure_aware_datetime(start_date)
    end_date = _ensure_aware_datetime(end_date)
    current = start_date.replace(minute=0, second=0, microsecond=0)
    end = end_date.replace(minute=0, second=0, microsecond=0)
    out = []
    while current <= end:
        out.append(current)
        current = current + timedelta(hours=1)
    return out


def _day_buckets(start_date: datetime, end_date: datetime) -> list[date]:
    current = start_date.date()
    end = end_date.date()
    out = []
    while current <= end:
        out.append(current)
        current = current + timedelta(days=1)
    return out


def _month_buckets(start_date: datetime, end_date: datetime) -> list[datetime]:
    start_date = _ensure_aware_datetime(start_date)
    end_date = _ensure_aware_datetime(end_date)
    current = start_date.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = end_date.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    out = []
    while current <= end:
        out.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return out


def get_token_stats_from_query(params: Any) -> dict:
    """
    Build token stats dict from query params (e.g. request.query_params).
    Raises ValueError for unsupported granularity.
    """
    start_date = _parse_date(params.get("start_date"))
    end_date = _parse_end_date(params.get("end_date"))
    user_id = params.get("user_id") or None
    if user_id is not None and str(user_id).strip() == "":
        user_id = None
    granularity = params.get("granularity")

    summary = get_summary_stats(
        start_date=start_date, end_date=end_date, user_id=user_id
    )
    by_model = get_stats_by_model(
        start_date=start_date, end_date=end_date, user_id=user_id
    )
    for i in by_model:
        cost = i.get("total_cost")
        i["total_cost"] = float(cost) if cost is not None else 0
        i["total_cost_currency"] = "USD"
    series = None
    if granularity:
        series_items = get_time_series_stats(
            granularity=granularity,
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
        )
        series = {
            "granularity": (granularity or "").strip().lower(),
            "items": series_items,
        }
    return {
        "summary": summary,
        "by_model": by_model,
        "series": series,
    }
