"""Public schemas for model-specific backup configuration resources."""

from rest_framework import serializers


class FallbackPolicySerializer(serializers.Serializer):
    model_uuids = serializers.ListField(
        child=serializers.UUIDField(),
        max_length=3,
        required=False,
    )
    inherit_global = serializers.BooleanField(required=False)
    conditions = serializers.ListField(
        child=serializers.ChoiceField(
            choices=(
                "timeout",
                "rate_limit",
                "upstream_error",
                "connection_error",
                "quota_exhausted",
                "empty_response",
            )
        ),
        required=False,
    )
    timeout_seconds = serializers.IntegerField(
        min_value=5,
        max_value=600,
        required=False,
    )
    retries = serializers.IntegerField(
        min_value=0, max_value=1, required=False
    )
    cooldown_seconds = serializers.IntegerField(
        min_value=0,
        max_value=3600,
        required=False,
    )
    auto_recover = serializers.BooleanField(required=False)


class FallbackWriteSerializer(serializers.Serializer):
    fallback = FallbackPolicySerializer(allow_null=True)


class FallbackReadSerializer(serializers.Serializer):
    fallback = FallbackPolicySerializer(allow_null=True)
    defaults = FallbackPolicySerializer()
    effective_policy = FallbackPolicySerializer(allow_null=True)
    inherited_from = serializers.UUIDField(allow_null=True)
    backups = serializers.ListField(child=serializers.DictField())
    candidates = serializers.ListField(child=serializers.DictField())
    inheritance = serializers.DictField()
    health = serializers.ListField(child=serializers.DictField())
