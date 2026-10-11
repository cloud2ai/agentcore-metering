"""Shared visibility and ownership rules for personal model settings."""

from django.db.models import Q

from agentcore_metering.adapters.django.models import LLMConfig


def user_model_configs(user, *, writable=False):
    """Personal models; globals are readable, writable only by staff.

    Staff manage other users' models through the separate admin API.
    """
    scopes = Q(scope=LLMConfig.Scope.USER, user=user)
    if not writable or user.is_staff:
        scopes |= Q(scope=LLMConfig.Scope.GLOBAL, user__isnull=True)
    return LLMConfig.objects.filter(scopes)
