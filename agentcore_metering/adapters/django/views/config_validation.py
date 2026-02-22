from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from ..serializers import (
    ConfigTestResponseSerializer,
    LLMConfigWriteSerializer,
    TestCallRequestSerializer,
    TestCallResponseSerializer,
)
from ..services.runtime_config import (
    VALIDATION_MESSAGE_IDS,
    get_validation_message,
    run_test_call,
    validate_llm_config,
)


class AdminLLMConfigTestView(APIView):
    """
    POST: Validate provider + config without saving. Body: provider, config.
    Runs minimal completion to verify api_key and endpoint.
    Returns {"ok": true} or {"ok": false, "detail": "error message"}.
    """

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["llm-metering"],
        summary="Test LLM config",
        description=(
            "Validate provider and config without saving. Runs minimal "
            "completion to verify api_key and endpoint. Returns 200 with "
            "ok=true on success, ok=false and detail on failure."
        ),
        request=LLMConfigWriteSerializer,
        responses={
            200: ConfigTestResponseSerializer,
            400: ConfigTestResponseSerializer,
        },
    )
    def post(self, request):
        ser = LLMConfigWriteSerializer(data=request.data)
        if not ser.is_valid():
            return Response(
                {"ok": False, "detail": ser.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )
        data = ser.validated_data
        provider = (data.get("provider") or "openai").strip().lower()
        config = data.get("config") or {}
        ok, message = validate_llm_config(provider, config, user=request.user)
        if ok:
            return Response({"ok": True})
        if message in VALIDATION_MESSAGE_IDS:
            detail = get_validation_message(message, request.LANGUAGE_CODE)
        else:
            detail = message
        return Response(
            {"ok": False, "detail": detail},
            status=status.HTTP_200_OK,
        )


class AdminLLMConfigTestCallView(APIView):
    """
    POST: Run one completion with a saved LLMConfig and record to LLMUsage.

    Body: config_uuid, prompt; optional max_tokens (default 512, max 4096).
    Returns { ok, content?, detail?, usage? }. Call is synchronous and
    recorded in admin_test_call usage.
    """

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["llm-metering"],
        summary="Test call with config",
        description=(
            "Run one completion using the given LLMConfig uuid and prompt. "
            "Synchronous; records the call to LLM usage. Returns content "
            "and usage when ok is true."
        ),
        request=TestCallRequestSerializer,
        responses={200: TestCallResponseSerializer},
    )
    def post(self, request):
        ser = TestCallRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(
                {"ok": False, "detail": ser.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )
        data = ser.validated_data
        config_uuid = data.get("config_uuid")
        config_id = data.get("config_id")
        prompt = (data.get("prompt") or "").strip()
        max_tokens = data.get("max_tokens") or 512
        ok, content_or_detail, usage_dict = run_test_call(
            config_uuid=str(config_uuid) if config_uuid else None,
            config_id=config_id,
            prompt=prompt,
            user=request.user,
            max_tokens=max_tokens,
        )
        if ok:
            return Response({
                "ok": True,
                "content": content_or_detail,
                "usage": usage_dict,
            })
        return Response({
            "ok": False,
            "detail": content_or_detail,
        }, status=status.HTTP_200_OK)
