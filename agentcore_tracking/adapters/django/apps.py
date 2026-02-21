"""Django app config for agentcore_tracking (LLM observability)."""
from django.apps import AppConfig


class AgentcoreTrackingDjangoConfig(AppConfig):
    """App config for agentcore_tracking Django adapter."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "agentcore_tracking.adapters.django"
    label = "agentcore_tracking"
    verbose_name = "Agentcore Tracking (LLM)"
