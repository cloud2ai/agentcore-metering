"""Mount under /api/v1/user/ for authenticated personal model tests."""

from django.urls import path

from agentcore_metering.adapters.django.views.config_catalog import (
    UserLLMConfigModelsView,
    UserLLMConfigProvidersView,
)
from agentcore_metering.adapters.django.views.config_validation import (
    UserLLMConfigTestCallView,
    UserLLMConfigTestView,
)
from agentcore_metering.adapters.django.views.decisions import (
    UserDecisionTestView,
)

app_name = "agentcore_metering_user"

urlpatterns = [
    path(
        "llm-config/models", UserLLMConfigModelsView.as_view(), name="models",
    ),
    path(
        "llm-config/providers", UserLLMConfigProvidersView.as_view(),
        name="providers",
    ),
    path("llm-config/test", UserLLMConfigTestView.as_view(), name="test"),
    path(
        "llm-config/test-call",
        UserLLMConfigTestCallView.as_view(),
        name="test-call",
    ),
    path(
        "llm-config/decision-test",
        UserDecisionTestView.as_view(),
        name="decision-test",
    ),
]
