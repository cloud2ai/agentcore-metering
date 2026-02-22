from django.contrib.auth import get_user_model
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import LLMConfig
from ..serializers import (
    ErrorDetailSerializer,
    LLMConfigSerializer,
    LLMConfigWriteSerializer,
)
from ..services.model_catalog import get_model_type_for_model_id

User = get_user_model()


class AdminLLMConfigAllListView(APIView):
    """
    GET: List all LLM configs (global + user) in one list.
    Query param scope: all (default) | global | user.
    Optional user_id when scope=user.
    """

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["llm-metering"],
        summary="List all LLM configs",
        description=(
            "List all LLM configs (global + user). scope=all returns both; "
            "scope=global or user filters. When scope=user, use user_id."
        ),
        parameters=[
            OpenApiParameter(
                "scope",
                OpenApiTypes.STR,
                OpenApiParameter.QUERY,
                description="Filter: all (default), global, or user",
                enum=["all", "global", "user"],
                default="all",
            ),
            OpenApiParameter(
                "user_id",
                OpenApiTypes.INT,
                OpenApiParameter.QUERY,
                description="Filter by user id when scope=user",
            ),
        ],
        responses={200: LLMConfigSerializer(many=True)},
    )
    def get(self, request):
        scope_param = (
            request.query_params.get("scope") or "all"
        ).strip().lower()
        user_id_param = request.query_params.get("user_id")
        qs = (
            LLMConfig.objects.filter(model_type=LLMConfig.MODEL_TYPE_LLM)
            .select_related("user")
            .order_by("scope", "order", "id")
        )
        if scope_param == "global":
            qs = qs.filter(scope=LLMConfig.Scope.GLOBAL)
        elif scope_param == "user":
            qs = qs.filter(scope=LLMConfig.Scope.USER)
            if user_id_param is not None and str(user_id_param).strip():
                qs = qs.filter(user_id=user_id_param)
        return Response(LLMConfigSerializer(qs, many=True).data)


class AdminLLMConfigGlobalView(APIView):
    """
    GET: List global LLM configs (model_type=llm, ordered).
    POST: Add one config. Body may include scope (global|user) and
    user_id for user config.
    """

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["llm-metering"],
        summary="List global LLM configs",
        description=(
            "List global LLM configs (model_type=llm, ordered by order, id)."
        ),
        responses={200: LLMConfigSerializer(many=True)},
    )
    def get(self, request):
        qs = (
            LLMConfig.objects.filter(
                scope=LLMConfig.Scope.GLOBAL,
                model_type=LLMConfig.MODEL_TYPE_LLM,
            )
            .order_by("order", "id")
        )
        return Response(LLMConfigSerializer(qs, many=True).data)

    @extend_schema(
        tags=["llm-metering"],
        summary="Create LLM config",
        description=(
            "Add one global or user config. Body: provider, config; optional "
            "scope (global|user), user_id, is_active, order."
        ),
        request=LLMConfigWriteSerializer,
        responses={201: LLMConfigSerializer, 400: ErrorDetailSerializer},
    )
    def post(self, request):
        ser = LLMConfigWriteSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        data = ser.validated_data
        scope_raw = (request.data.get("scope") or "global").strip().lower()
        user_id_raw = request.data.get("user_id")
        provider = (data.get("provider") or "openai").strip().lower()
        config = data.get("config") or {}
        model_id = (config.get("model") or "").strip()
        model_type_raw = (
            (request.data.get("model_type") or "").strip()
            or get_model_type_for_model_id(provider, model_id or None)
            or LLMConfig.MODEL_TYPE_LLM
        )
        if model_type_raw not in LLMConfig.MODEL_TYPES:
            model_type_raw = LLMConfig.MODEL_TYPE_LLM
        if scope_raw == "user" and user_id_raw is not None:
            user = None
            try:
                user = User.objects.get(pk=user_id_raw)
            except (User.DoesNotExist, ValueError, TypeError):
                return Response(
                    {"detail": "User not found."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            obj = LLMConfig.objects.create(
                scope=LLMConfig.Scope.USER,
                user=user,
                model_type=model_type_raw,
                provider=(data.get("provider") or "openai").strip().lower(),
                config=data.get("config") or {},
                is_active=data.get("is_active", True),
                order=data.get("order", 0),
            )
        else:
            obj = LLMConfig.objects.create(
                scope=LLMConfig.Scope.GLOBAL,
                user=None,
                model_type=model_type_raw,
                provider=(data.get("provider") or "openai").strip().lower(),
                config=data.get("config") or {},
                is_active=data.get("is_active", True),
                order=data.get("order", 0),
            )
        return Response(
            LLMConfigSerializer(obj).data,
            status=status.HTTP_201_CREATED,
        )


class AdminLLMConfigDetailView(APIView):
    """GET/PUT/DELETE one LLM config by uuid."""

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["llm-metering"],
        summary="Get LLM config",
        description="Get one LLM config by uuid.",
        responses={200: LLMConfigSerializer, 404: ErrorDetailSerializer},
    )
    def get(self, request, config_ref):
        obj = self._get_obj(config_ref)
        if obj is None:
            return Response(
                {"detail": "Not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(LLMConfigSerializer(obj).data)

    @extend_schema(
        tags=["llm-metering"],
        summary="Update LLM config",
        description=(
            "Update one LLM config by uuid. Body: optional provider, config, "
            "is_active, order."
        ),
        request=LLMConfigWriteSerializer,
        responses={
            200: LLMConfigSerializer,
            400: ErrorDetailSerializer,
            404: ErrorDetailSerializer,
        },
    )
    def put(self, request, config_ref):
        obj = self._get_obj(config_ref)
        if obj is None:
            return Response(
                {"detail": "Not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        ser = LLMConfigWriteSerializer(data=request.data, partial=True)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        data = ser.validated_data
        if "provider" in data:
            obj.provider = (data["provider"] or "openai").strip().lower()
        if "config" in data:
            obj.config = data["config"]
        if "is_active" in data:
            obj.is_active = data["is_active"]
        if "order" in data:
            obj.order = data["order"]
        model_type_raw = request.data.get("model_type")
        mt_ok = (
            model_type_raw is not None
            and str(model_type_raw).strip() in LLMConfig.MODEL_TYPES
        )
        if mt_ok:
            obj.model_type = str(model_type_raw).strip()
        else:
            provider = (
                data.get("provider") or obj.provider or "openai"
            ).strip().lower()
            config = data.get("config") if "config" in data else obj.config
            model_id = ((config or {}).get("model") or "").strip()
            if model_id:
                derived = get_model_type_for_model_id(provider, model_id)
                if derived:
                    obj.model_type = derived
        obj.save()
        return Response(LLMConfigSerializer(obj).data)

    @extend_schema(
        tags=["llm-metering"],
        summary="Delete LLM config",
        description="Delete one LLM config by uuid.",
        responses={204: None, 404: ErrorDetailSerializer},
    )
    def delete(self, request, config_ref):
        obj = self._get_obj(config_ref)
        if obj is None:
            return Response(
                {"detail": "Not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        obj.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    def _get_obj(self, config_ref):
        qs = LLMConfig.objects.select_related("user")
        try:
            return qs.get(uuid=config_ref)
        except (LLMConfig.DoesNotExist, ValueError, TypeError):
            if str(config_ref).isdigit():
                try:
                    return qs.get(pk=int(config_ref))
                except (LLMConfig.DoesNotExist, ValueError, TypeError):
                    return None
            return None


class AdminLLMConfigUserListView(APIView):
    """
    GET: List per-user LLM configs. Optional query param user_id to filter
    by one user.
    """

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["llm-metering"],
        summary="List per-user LLM configs",
        description=(
            "List per-user LLM configs. Optional user_id to filter."
        ),
        parameters=[
            OpenApiParameter(
                "user_id",
                OpenApiTypes.INT,
                OpenApiParameter.QUERY,
                description="Filter by user id (optional)",
            ),
        ],
        responses={200: LLMConfigSerializer(many=True)},
    )
    def get(self, request):
        user_id = request.query_params.get("user_id")
        qs = (
            LLMConfig.objects.filter(scope=LLMConfig.Scope.USER)
            .select_related("user")
        )
        if user_id is not None and str(user_id).strip():
            qs = qs.filter(user_id=user_id)
        qs = qs.order_by("order", "id")
        return Response(LLMConfigSerializer(qs, many=True).data)


class AdminLLMConfigUserDetailView(APIView):
    """
    GET: Return one user's LLM config (404 if not set).
    PUT: Create or update that user's LLM config.
    DELETE: Remove user config so they fall back to global/settings.
    """

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["llm-metering"],
        summary="Get user LLM config",
        description="Get one user's LLM config. 404 if not set.",
        responses={200: LLMConfigSerializer, 404: ErrorDetailSerializer},
    )
    def get(self, request, user_id):
        user = self._get_user(user_id)
        if user is None:
            return Response(
                {"detail": "User not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        obj = (
            LLMConfig.objects.filter(
                scope=LLMConfig.Scope.USER, user_id=user.pk
            )
            .select_related("user")
            .first()
        )
        if obj is None:
            return Response(
                {"detail": "No LLM config for this user. Use PUT to create."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(LLMConfigSerializer(obj).data)

    @extend_schema(
        tags=["llm-metering"],
        summary="Create or update user LLM config",
        description=(
            "Create or update that user's LLM config. Body: provider, config."
        ),
        request=LLMConfigWriteSerializer,
        responses={
            200: LLMConfigSerializer,
            400: ErrorDetailSerializer,
            404: ErrorDetailSerializer,
        },
    )
    def put(self, request, user_id):
        user = self._get_user(user_id)
        if user is None:
            return Response(
                {"detail": "User not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        ser = LLMConfigWriteSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        data = ser.validated_data
        obj, _ = LLMConfig.objects.update_or_create(
            scope=LLMConfig.Scope.USER,
            user=user,
            defaults={
                "provider": (data.get("provider") or "openai").strip().lower(),
                "config": data.get("config") or {},
            },
        )
        return Response(LLMConfigSerializer(obj).data)

    @extend_schema(
        tags=["llm-metering"],
        summary="Delete user LLM config",
        description="Remove user config so they fall back to global/settings.",
        responses={204: None, 404: ErrorDetailSerializer},
    )
    def delete(self, request, user_id):
        user = self._get_user(user_id)
        if user is None:
            return Response(
                {"detail": "User not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        deleted, _ = LLMConfig.objects.filter(
            scope=LLMConfig.Scope.USER, user=user
        ).delete()
        if deleted:
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(
            {"detail": "No user config to delete."},
            status=status.HTTP_404_NOT_FOUND,
        )

    def _get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except (User.DoesNotExist, ValueError, TypeError):
            return None
