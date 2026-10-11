"""Low-cost calls must survive storage and existing aggregation APIs."""

from decimal import Decimal

import pytest
from django.utils import timezone

from agentcore_metering.adapters.django.models import LLMUsage, LLMUsageSeries
from agentcore_metering.adapters.django.services.usage_aggregation import (
    get_token_stats_from_query,
)
from agentcore_metering.adapters.django.services.usage_list import (
    get_llm_usage_list,
)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "cost",
    [
        Decimal("0.00000042"),
        Decimal("0.000000000001"),
        Decimal("123.456789"),
        Decimal("999999"),
        Decimal("0"),
        None,
    ],
)
def test_cost_round_trip_and_aggregation(cost):
    usage = LLMUsage.objects.create(model="precision-test", cost=cost)
    series = LLMUsageSeries.objects.create(
        granularity="hour",
        bucket=timezone.now(),
        model=usage.model,
        total_cost=cost,
    )
    usage.refresh_from_db()
    series.refresh_from_db()
    assert usage.cost == cost
    assert series.total_cost == cost

    record = get_llm_usage_list()["results"][0]
    if cost is None:
        assert record["cost"] is None
    else:
        assert record["cost"] == float(cost)
    response = get_token_stats_from_query({})
    for stats in (response["summary"], response["by_model"][0]):
        expected = float(cost) if cost is not None else 0
        assert stats["total_cost"] == expected
        expected_status = "priced" if cost is not None else "unavailable"
        assert stats["cost_status"] == expected_status
