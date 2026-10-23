"""Permission-scoped test calls for the shared Decisions client."""

from drf_spectacular.utils import extend_schema
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from agentcore_metering.adapters.django.serializers import (
    DecisionTestRequestSerializer,
)
from agentcore_metering.adapters.django.services.decisions import (
    DecisionTracker,
    decision_error_detail,
)
from agentcore_metering.adapters.django.services.fallback_runtime import (
    primary_only,
)
from agentcore_metering.adapters.django.services.config_access import (
    user_model_configs,
)


class AdminDecisionTestView(APIView):
    permission_classes = [IsAdminUser]
    node_name = "admin_decision_test"

    def can_test(self, request, config_uuid):
        return True

    @extend_schema(
        tags=["llm-metering"],
        summary="Test a decision model",
        request=DecisionTestRequestSerializer,
    )
    def post(self, request):
        ser = DecisionTestRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        if data.get("config_uuid") and not self.can_test(
            request, data["config_uuid"]
        ):
            return Response({"detail": "Not found."}, status=404)
        params = {
            "input": data["input"],
            "questions": data["questions"],
            "state": {"user_id": request.user.pk, "node_name": self.node_name},
        }
        try:
            if data.get("config_uuid"):
                with primary_only():
                    answers, usage = DecisionTracker.call_and_track(
                        model_uuid=data["config_uuid"], **params
                    )
            else:
                answers, usage = DecisionTracker.call_with_config(
                    provider=data["provider"], config=data["config"], **params
                )
        except ValueError as exc:
            return Response({"ok": False, "detail": str(exc)}, status=400)
        except Exception as exc:
            return Response(
                {
                    "ok": False,
                    "detail": decision_error_detail(
                        exc, getattr(request, "LANGUAGE_CODE", "en")
                    ),
                }
            )
        return Response({"ok": True, "answers": answers, "usage": usage})


class UserDecisionTestView(AdminDecisionTestView):
    permission_classes = [IsAuthenticated]
    node_name = "user_decision_test"

    def can_test(self, request, config_uuid):
        # A saved-key test spends its owner's credits. Read access to a
        # shared config does not grant permission to test with its key.
        return (
            user_model_configs(request.user, writable=True)
            .filter(
                uuid=config_uuid,
                model_type="decision",
                is_active=True,
            )
            .exists()
        )
