"""
Tests for llm_usage: get_llm_usage_list and get_llm_usage_list_from_query.

Paginated listing with filters (user_id, model, success, dates).
"""
import pytest
from django.utils import timezone as django_tz

from agentcore_metering.adapters.django.services.usage import (
    get_llm_usage_list,
    get_llm_usage_list_from_query,
)
from agentcore_metering.adapters.django.models import LLMUsage


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmUsageList:
    """
    get_llm_usage_list returns paginated results with filters.
    """

    def test_empty_results_when_no_records(self):
        out = get_llm_usage_list(page=1, page_size=20)
        assert out["results"] == []
        assert out["total"] == 0
        assert out["page"] == 1
        assert out["page_size"] == 20

    def test_returns_records_with_expected_shape(self, django_user_model):
        user = django_user_model.objects.create_user(
            username="u1", password="p", email="u1@example.com"
        )
        LLMUsage.objects.create(
            user=user,
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            success=True,
        )
        out = get_llm_usage_list(page=1, page_size=20)
        assert out["total"] == 1
        assert len(out["results"]) == 1
        item = out["results"][0]
        assert item["model"] == "gpt-4"
        assert item["prompt_tokens"] == 10
        assert item["completion_tokens"] == 20
        assert item["total_tokens"] == 30
        assert item["success"] is True
        assert item["username"] == "u1"
        assert "id" in item
        assert "created_at" in item
        assert "metadata" in item

    def test_pagination_respects_page_and_page_size(self):
        for i in range(5):
            LLMUsage.objects.create(
                model=f"m{i}",
                total_tokens=1,
            )
        out = get_llm_usage_list(page=2, page_size=2)
        assert out["total"] == 5
        assert out["page"] == 2
        assert out["page_size"] == 2
        assert len(out["results"]) == 2

    def test_page_size_capped_at_100(self):
        out = get_llm_usage_list(page=1, page_size=200)
        assert out["page_size"] == 100

    def test_filter_by_user_id(self, django_user_model):
        u1 = django_user_model.objects.create_user(
            username="u1", password="p", email="u1@example.com"
        )
        u2 = django_user_model.objects.create_user(
            username="u2", password="p", email="u2@example.com"
        )
        LLMUsage.objects.create(user=u1, model="m1", total_tokens=1)
        LLMUsage.objects.create(user=u2, model="m2", total_tokens=1)
        out = get_llm_usage_list(user_id=str(u1.id))
        assert out["total"] == 1
        assert out["results"][0]["user_id"] == u1.id

    def test_filter_by_model_icontains(self):
        LLMUsage.objects.create(model="gpt-4-turbo", total_tokens=1)
        LLMUsage.objects.create(model="gpt-3.5", total_tokens=1)
        LLMUsage.objects.create(model="claude-2", total_tokens=1)
        out = get_llm_usage_list(model_filter="gpt")
        assert out["total"] == 2

    def test_filter_by_success_true(self):
        LLMUsage.objects.create(model="m1", total_tokens=1, success=True)
        LLMUsage.objects.create(model="m2", total_tokens=1, success=False)
        out = get_llm_usage_list(success_filter="true")
        assert out["total"] == 1
        assert out["results"][0]["success"] is True

    def test_filter_by_success_false(self):
        LLMUsage.objects.create(model="m1", total_tokens=1, success=True)
        LLMUsage.objects.create(model="m2", total_tokens=1, success=False)
        out = get_llm_usage_list(success_filter="false")
        assert out["total"] == 1
        assert out["results"][0]["success"] is False

    def test_filter_by_start_and_end_date(self):
        base = django_tz.now()
        LLMUsage.objects.create(
            model="m1",
            total_tokens=1,
            created_at=base,
        )
        start = (base - django_tz.timedelta(days=2)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = (base + django_tz.timedelta(days=1)).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        out = get_llm_usage_list(start_date=start, end_date=end)
        assert out["total"] == 1


@pytest.mark.unit
@pytest.mark.django_db
class TestGetLlmUsageListFromQuery:
    """
    get_llm_usage_list_from_query parses params and delegates to list.
    """

    def test_default_params(self):
        out = get_llm_usage_list_from_query({})
        assert out["page"] == 1
        assert out["page_size"] == 20
        assert out["results"] == []

    def test_parses_page_and_page_size(self):
        LLMUsage.objects.create(model="m1", total_tokens=1)
        out = get_llm_usage_list_from_query({
            "page": "2",
            "page_size": "5",
        })
        assert out["page"] == 2
        assert out["page_size"] == 5

    def test_parses_user_id_model_success_from_params(self, django_user_model):
        user = django_user_model.objects.create_user(
            username="u1", password="p", email="u1@example.com"
        )
        LLMUsage.objects.create(user=user, model="gpt-4", total_tokens=1)
        out = get_llm_usage_list_from_query({
            "user_id": str(user.id),
            "model": "gpt",
            "success": "true",
        })
        assert out["total"] == 1
        assert out["results"][0]["model"] == "gpt-4"

    def test_parses_start_date_and_end_date(self):
        LLMUsage.objects.create(model="m1", total_tokens=1)
        out = get_llm_usage_list_from_query({
            "start_date": "2025-01-01T00:00:00+00:00",
            "end_date": "2025-12-31T23:59:59+00:00",
        })
        assert "results" in out
        assert "total" in out

    def test_invalid_page_raises_value_error(self):
        with pytest.raises(ValueError):
            get_llm_usage_list_from_query({"page": "not-a-number"})

    def test_invalid_page_size_raises_value_error(self):
        with pytest.raises(ValueError):
            get_llm_usage_list_from_query({"page_size": "nope"})
