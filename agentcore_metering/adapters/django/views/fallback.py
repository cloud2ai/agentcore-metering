"""Admin and personal backup-model configuration resources."""

from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from agentcore_metering.adapters.django.fallback_serializers import (
    FallbackReadSerializer,
    FallbackWriteSerializer,
)
from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services.config_access import (
    user_model_configs,
)
from agentcore_metering.adapters.django.services.fallback_config import (
    DEFAULT_POLICY,
    compatible,
    inherited_origin,
    normalize_policy,
    resolve_plan,
    same_upstream,
    save_policy,
    visible_graph,
)
from agentcore_metering.adapters.django.services.fallback_health import (
    ModelAvailability,
)
from agentcore_metering.adapters.django.services.model_fallback import (
    FallbackTarget,
)


def policy_representation(row):
    policy, origin, targets = resolve_plan(row)
    available = list(visible_graph(row).filter(is_active=True))
    source = inherited_origin(
        row, {str(item.uuid): item for item in available}
    )
    global_policy = normalize_policy(source.fallback) if source else None
    health = (
        ModelAvailability(row, origin, [row, *targets], policy)
        if policy
        else None
    )
    return {
        "fallback": row.fallback,
        "defaults": DEFAULT_POLICY,
        "effective_policy": policy,
        "inherited_from": (
            str(origin.uuid) if origin and origin != row else None
        ),
        "backups": [
            {
                "uuid": str(target.uuid),
                "provider": target.provider,
                "model_type": target.model_type,
                "model": (target.config or {}).get("model"),
            }
            for target in targets
        ],
        "candidates": [
            {
                "uuid": str(item.uuid),
                "provider": item.provider,
                "model": (item.config or {}).get("model"),
                "model_type": item.model_type,
                "scope": item.scope,
                "fallback": item.fallback,
            }
            for item in available
            if (
                item.pk != row.pk
                and compatible(row, item)
                and not same_upstream(row, item)
            )
        ],
        "inheritance": {
            "available": bool(global_policy and global_policy["model_uuids"]),
            "source_uuid": str(source.uuid) if source else None,
            "policy": global_policy,
        },
        "health": [
            {
                "uuid": str(item.uuid),
                **health.status(
                    FallbackTarget(
                        str(item.uuid),
                        policy["timeout_seconds"],
                    )
                ),
            }
            for item in [row, *targets]
        ]
        if health
        else [],
    }


class AdminModelFallbackView(APIView):
    permission_classes = [IsAdminUser]

    def queryset(self, request):
        return LLMConfig.objects.all()

    @extend_schema(tags=["llm-metering"], responses=FallbackReadSerializer)
    def get(self, request, config_uuid):
        row = get_object_or_404(self.queryset(request), uuid=config_uuid)
        return Response(policy_representation(row))

    @extend_schema(
        tags=["llm-metering"],
        request=FallbackWriteSerializer,
        responses=FallbackReadSerializer,
    )
    def put(self, request, config_uuid):
        row = get_object_or_404(self.queryset(request), uuid=config_uuid)
        if not isinstance(request.data, dict) or set(request.data) != {
            "fallback",
        }:
            return Response(
                {"detail": "Provide a fallback object or null."}, status=400
            )
        try:
            row = save_policy(row, request.data["fallback"])
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=400)
        return Response(policy_representation(row))

    def reset(self, request, config_uuid):
        row = get_object_or_404(self.queryset(request), uuid=config_uuid)
        policy, origin, targets = resolve_plan(row)
        if policy:
            health = ModelAvailability(row, origin, [row, *targets], policy)
            for item in [row, *targets]:
                health.success(
                    FallbackTarget(
                        str(item.uuid),
                        policy["timeout_seconds"],
                    )
                )
        return Response(policy_representation(row))


class UserModelFallbackView(AdminModelFallbackView):
    permission_classes = [IsAuthenticated]

    def queryset(self, request):
        return user_model_configs(request.user, writable=True)


class AdminModelFallbackResetView(AdminModelFallbackView):
    http_method_names = ["post", "options"]

    @extend_schema(
        tags=["llm-metering"],
        request=None,
        responses=FallbackReadSerializer,
    )
    def post(self, request, config_uuid):
        return self.reset(request, config_uuid)


class UserModelFallbackResetView(UserModelFallbackView):
    http_method_names = ["post", "options"]

    @extend_schema(
        tags=["llm-metering"],
        request=None,
        responses=FallbackReadSerializer,
    )
    def post(self, request, config_uuid):
        return self.reset(request, config_uuid)
