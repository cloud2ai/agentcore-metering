"""
Serializers for agentcore_metering admin API.

Includes read/write serializers for LLM config and response serializers
for OpenAPI/Swagger documentation (token stats, usage list, errors).
"""

from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from agentcore_metering.adapters.django.llm_static.load import (
    get_decision_providers,
)
from agentcore_metering.adapters.django.models import LLMConfig
from agentcore_metering.adapters.django.services.model_info import (
    INPUT_MODALITIES,
    OUTPUT_MODALITIES,
    reset_model_info_on_change,
    resolve_model_info,
    validate_model_info,
)


def _mask_secrets(config: dict) -> dict:
    if not config or not isinstance(config, dict):
        return config or {}
    out = {}
    for k, v in config.items():
        if isinstance(v, dict):
            out[k] = _mask_secrets(v)
        elif k in ("api_key", "key") and v and isinstance(v, str):
            if len(v) <= 8:
                out[k] = "***"
            else:
                out[k] = v[:4] + "***" + v[-4:]
        else:
            out[k] = v
    return out


class ModelInfoSerializer(serializers.Serializer):
    """Known input/output modalities; null means unknown, not unsupported."""

    supported_modalities = serializers.ListField(
        child=serializers.ChoiceField(choices=INPUT_MODALITIES),
        allow_null=True,
    )
    supported_output_modalities = serializers.ListField(
        child=serializers.ChoiceField(choices=OUTPUT_MODALITIES),
        allow_null=True,
    )
    sources = serializers.DictField(
        child=serializers.ChoiceField(
            choices=("catalog", "manual", "unknown"),
        ),
        required=False,
    )


class LLMConfigSerializer(serializers.ModelSerializer):
    """
    Read LLM config. On read, api_key (and key) in config are masked.
    is_default is True when this config is the default (earliest enabled
    global config by created_at); frontend can use it to show the default.
    """

    model_info = serializers.SerializerMethodField()

    @extend_schema_field(ModelInfoSerializer)
    def get_model_info(self, instance):
        return resolve_model_info(instance.provider, instance.config)

    is_default = serializers.BooleanField(
        read_only=True,
        help_text=(
            "True if this config is the one used when model_uuid is not set "
            "(current default). When there is only one active global config, "
            "it is the default."
        ),
    )

    class Meta:
        model = LLMConfig
        fields = [
            "id",
            "uuid",
            "scope",
            "user",
            "model_type",
            "provider",
            "config",
            "model_info",
            "fallback",
            "is_active",
            "is_default",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "uuid",
            "is_default",
            "fallback",
            "created_at",
            "updated_at",
        ]

    def to_representation(self, instance):
        data = super().to_representation(instance)
        if instance.config:
            data["config"] = _mask_secrets(instance.config)
        if instance.user_id:
            data["user_id"] = instance.user_id
            data["username"] = getattr(instance.user, "username", None)
        default_uuid = self.context.get("default_config_uuid")
        data["is_default"] = default_uuid is not None and str(
            instance.uuid
        ) == str(default_uuid)
        return data


class LLMConfigWriteSerializer(serializers.Serializer):
    """
    Payload for creating/updating LLM config: provider + single config dict.
    """

    model_type = serializers.ChoiceField(
        choices=LLMConfig.MODEL_TYPES, required=False,
        help_text="decision uses LiteLLM Decisions, not chat completions.",
    )
    provider = serializers.CharField(
        default="openai",
        max_length=50,
        help_text=(
            "Provider id (e.g. openai, anthropic, azure_openai). "
            "Default: openai."
        ),
    )
    config = serializers.JSONField(
        required=False,
        default=dict,
        help_text=(
            "Provider-specific config: api_key (required for most), model, "
            "api_base, deployment (Azure), max_tokens, temperature, top_p, "
            "request_timeout_seconds, num_retries, api_version (Azure). "
            "Optional model_info contains supported_modalities and "
            "supported_output_modalities overrides; {} restores automatic "
            "detection. Omit model_info to preserve it on update. "
            "Use GET "
            ".../llm-config/providers/ for per-provider required/optional "
            "keys."
        ),
    )
    is_active = serializers.BooleanField(
        required=False,
        default=True,
        help_text="Whether this config is active for resolution.",
    )
    is_default = serializers.BooleanField(
        required=False,
        default=False,
        help_text=(
            "If true, set this global config as the default (clears default "
            "on other global configs). Ignored for user-scope configs."
        ),
    )
    config_uuid = serializers.UUIDField(
        required=False,
        allow_null=True,
        help_text=(
            "Optional. When testing an existing config, its uuid lets the "
            "server resolve masked secret fields (e.g. api_key) from storage "
            "instead of sending the mask as the key."
        ),
    )
    config_id = serializers.IntegerField(
        required=False,
        allow_null=True,
        help_text="Deprecated: integer id of the config being tested.",
    )

    def validate_config(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("config must be an object.")
        model = value.get("model")
        if model is not None and not isinstance(model, str):
            raise serializers.ValidationError({
                "model": "model must be a string or null."
            })
        if "model_info" in value:
            value = dict(value)
            try:
                value["model_info"] = validate_model_info(value["model_info"])
            except (ValueError, TypeError) as exc:
                raise serializers.ValidationError({"model_info": str(exc)})
        return value

    def validate(self, attrs):
        from agentcore_metering.adapters.django.services.decisions import (
            decision_params,
        )
        from agentcore_metering.adapters.django.services.model_catalog import (
            get_model_type_for_model_id,
        )

        instance = self.context.get("instance")
        provider = attrs.get(
            "provider", getattr(instance, "provider", "openai")
        )
        provider = provider.strip().lower()
        if "provider" in attrs:
            attrs["provider"] = provider
        incoming = attrs.get("config", {})
        if instance:
            updated = reset_model_info_on_change(
                instance.provider, instance.config, provider, incoming,
            )
            if updated is not incoming:
                attrs["config"] = updated
        config = {**(getattr(instance, "config", None) or {}),
                  **attrs.get("config", {})}
        inferred = get_model_type_for_model_id(provider, config.get("model"))
        existing_type = getattr(instance, "model_type", None)
        model_type = attrs.get("model_type")
        if not model_type:
            identity_unchanged = instance and (
                provider == instance.provider
                and config.get("model") == (instance.config or {}).get("model")
            )
            if existing_type == "decision":
                model_type = existing_type
            elif identity_unchanged or (instance and not config.get("model")):
                model_type = existing_type
            else:
                model_type = inferred
        if (
            instance and model_type != existing_type
            and "decision" in (model_type, existing_type)
        ):
            raise serializers.ValidationError({
                "model_type": "Create a new config to change model type."
            })
        decision_only = set(get_decision_providers()) - {
            "openai", "openrouter", "openai_compatible",
        }
        if provider in decision_only and model_type != "decision":
            raise serializers.ValidationError({
                "model_type": "This provider requires decision model type."
            })
        if model_type == "decision":
            try:
                decision_params(provider, config)
            except (ValueError, TypeError) as exc:
                raise serializers.ValidationError({"config": str(exc)})
        if model_type != "llm" and attrs.get("is_default"):
            raise serializers.ValidationError({
                "is_default": "Only chat models can be the default LLM."
            })
        attrs["model_type"] = model_type
        return attrs


class ErrorDetailSerializer(serializers.Serializer):
    """Standard error body for 400/404 responses in API docs."""

    detail = serializers.CharField(
        allow_blank=True,
        help_text="Human-readable error message.",
    )


class DecisionTestRequestSerializer(serializers.Serializer):
    config_uuid = serializers.UUIDField(required=False)
    provider = serializers.ChoiceField(
        choices=tuple(get_decision_providers()),
        default="litellm_proxy",
    )
    config = serializers.DictField(required=False)
    input = serializers.CharField(allow_blank=False, max_length=500000)
    questions = serializers.ListField(
        child=serializers.DictField(), min_length=1, max_length=128,
    )
    stream = serializers.BooleanField(required=False, default=False)

    def validate(self, attrs):
        from agentcore_metering.adapters.django.services.decisions import (
            decision_params,
            validate_questions,
        )

        if attrs["stream"]:
            raise serializers.ValidationError(
                "Decisions do not support streaming."
            )
        try:
            validate_questions(attrs["questions"])
            if not attrs.get("config_uuid"):
                decision_params(attrs["provider"], attrs.get("config") or {})
        except ValueError as exc:
            raise serializers.ValidationError(str(exc))
        return attrs


class TokenStatsSummarySerializer(serializers.Serializer):
    """Summary block of token statistics (totals in date range)."""

    total_prompt_tokens = serializers.IntegerField(
        help_text="Sum of prompt tokens",
    )
    total_completion_tokens = serializers.IntegerField(
        help_text="Sum of completion tokens",
    )
    total_tokens = serializers.IntegerField(help_text="Sum of total tokens")
    total_cached_tokens = serializers.IntegerField(
        help_text="Sum of cached tokens",
    )
    total_reasoning_tokens = serializers.IntegerField(
        help_text="Sum of reasoning tokens (if any)",
    )
    total_cost = serializers.FloatField(help_text="Estimated cost in USD")
    total_cost_currency = serializers.CharField(
        help_text="Currency code (e.g. USD)",
    )
    priced_calls = serializers.IntegerField(
        help_text="Calls with an explicit cost value, including zero",
    )
    unpriced_calls = serializers.IntegerField(
        help_text="Calls whose cost could not be determined",
    )
    cost_status = serializers.ChoiceField(
        choices=["empty", "unavailable", "partial", "priced"],
        help_text="Cost coverage state for the aggregate",
    )
    total_calls = serializers.IntegerField(
        help_text="Total number of LLM calls",
    )
    successful_calls = serializers.IntegerField(
        help_text="Number of successful calls",
    )
    failed_calls = serializers.IntegerField(help_text="Number of failed calls")


class TokenStatsByModelItemSerializer(serializers.Serializer):
    """One row of token stats grouped by model."""

    model = serializers.CharField(help_text="Model identifier")
    total_calls = serializers.IntegerField()
    total_prompt_tokens = serializers.IntegerField()
    total_completion_tokens = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    total_cached_tokens = serializers.IntegerField()
    total_reasoning_tokens = serializers.IntegerField()
    total_cost = serializers.FloatField()
    total_cost_currency = serializers.CharField()
    priced_calls = serializers.IntegerField()
    unpriced_calls = serializers.IntegerField()
    cost_status = serializers.ChoiceField(
        choices=["empty", "unavailable", "partial", "priced"],
    )


class TokenStatsSeriesItemSerializer(serializers.Serializer):
    """One time bucket in the series (hour/day/month)."""

    bucket = serializers.CharField(
        allow_null=True,
        help_text="ISO datetime or date string for the bucket",
    )
    total_calls = serializers.IntegerField()
    total_prompt_tokens = serializers.IntegerField()
    total_completion_tokens = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    total_cached_tokens = serializers.IntegerField()
    total_reasoning_tokens = serializers.IntegerField()
    total_cost = serializers.FloatField()


class TokenStatsSeriesSerializer(serializers.Serializer):
    """Optional time series when granularity query param is set."""

    granularity = serializers.CharField(help_text="day | month | year")
    items = TokenStatsSeriesItemSerializer(many=True)


class TokenStatsSeriesByModelItemSerializer(serializers.Serializer):
    """
    One (bucket, model) row from pre-aggregated LLMUsageSeries.
    Used for performance curves (e2e, ttft, output_tps), token and cost trend.
    """

    bucket = serializers.CharField(allow_null=True)
    model = serializers.CharField()
    call_count = serializers.IntegerField()
    success_count = serializers.IntegerField()
    avg_e2e_latency_sec = serializers.FloatField(allow_null=True)
    avg_ttft_sec = serializers.FloatField(allow_null=True)
    avg_output_tps = serializers.FloatField(allow_null=True)
    total_prompt_tokens = serializers.IntegerField()
    total_completion_tokens = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    total_cached_tokens = serializers.IntegerField()
    total_reasoning_tokens = serializers.IntegerField()
    total_cost = serializers.FloatField(allow_null=True)
    cost_currency = serializers.CharField()


class TokenStatsResponseSerializer(serializers.Serializer):
    """Full response shape for GET .../token-stats/."""

    summary = TokenStatsSummarySerializer(
        help_text="Aggregate stats in the date range",
    )
    by_model = TokenStatsByModelItemSerializer(
        many=True,
        help_text="Stats per model, ordered by total_tokens desc",
    )
    series = TokenStatsSeriesSerializer(
        allow_null=True,
        required=False,
        help_text=(
            "Present only when granularity is set "
            "(day=hourly, month=daily, year=monthly)",
        ),
    )
    series_by_model = TokenStatsSeriesByModelItemSerializer(
        many=True,
        required=False,
        allow_null=True,
        help_text=(
            "Pre-aggregated (bucket, model) series when use_series=1 and "
            "granularity + date range set; for performance and cost charts.",
        ),
    )
    expected_buckets = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_null=True,
        help_text=(
            "Full ordered bucket list (ISO) for granularity and range; "
            "charts use this for complete x-axis."
        ),
    )


class MeteringConfigSerializer(serializers.Serializer):
    """Effective metering config (GET response)."""

    retention_days = serializers.IntegerField(min_value=1)
    cleanup_enabled = serializers.BooleanField()
    cleanup_crontab = serializers.CharField()
    aggregation_crontab = serializers.CharField()


class MeteringConfigUpdateSerializer(serializers.Serializer):
    """Request body for PATCH metering config (optional fields)."""

    retention_days = serializers.IntegerField(
        min_value=1, max_value=3650, required=False
    )
    cleanup_enabled = serializers.BooleanField(required=False)
    cleanup_crontab = serializers.CharField(required=False, allow_blank=False)
    aggregation_crontab = serializers.CharField(
        required=False, allow_blank=False
    )


class LLMUsageItemSerializer(serializers.Serializer):
    """One LLM usage record in the paginated list."""

    id = serializers.CharField(help_text="Usage record id")
    user_id = serializers.IntegerField(allow_null=True)
    username = serializers.CharField(allow_null=True)
    model = serializers.CharField()
    prompt_tokens = serializers.IntegerField(allow_null=True)
    completion_tokens = serializers.IntegerField(allow_null=True)
    total_tokens = serializers.IntegerField(allow_null=True)
    cached_tokens = serializers.IntegerField(allow_null=True)
    reasoning_tokens = serializers.IntegerField(allow_null=True)
    cost = serializers.FloatField(allow_null=True)
    cost_currency = serializers.CharField(allow_null=True)
    success = serializers.BooleanField()
    error = serializers.CharField(allow_null=True)
    created_at = serializers.CharField(allow_null=True)
    metadata = serializers.JSONField(allow_null=True)


class LLMUsageListResponseSerializer(serializers.Serializer):
    """Response shape for GET .../llm-usage/ (paginated)."""

    results = LLMUsageItemSerializer(
        many=True,
        help_text="Page of usage records",
    )
    total = serializers.IntegerField(help_text="Total count matching filters")
    page = serializers.IntegerField(help_text="Current page (1-based)")
    page_size = serializers.IntegerField(help_text="Page size used")


class TestCallRequestSerializer(serializers.Serializer):
    """Request body for POST .../llm-config/test-call/."""

    config_uuid = serializers.UUIDField(
        required=False,
        help_text="LLMConfig uuid to use for the call",
    )
    config_id = serializers.IntegerField(
        required=False,
        help_text="Deprecated: LLMConfig integer id (use config_uuid instead)",
    )
    prompt = serializers.CharField(
        allow_blank=False,
        help_text="User message to send to the model",
    )
    max_tokens = serializers.IntegerField(
        required=False,
        default=512,
        min_value=1,
        max_value=4096,
        help_text="Max tokens for the completion (default 512, max 4096)",
    )
    stream = serializers.BooleanField(
        required=False,
        default=False,
        help_text=(
            "If true, response is SSE stream "
            "(chunk events then done with usage)."
        ),
    )

    def validate(self, attrs):
        if not attrs.get("config_uuid") and attrs.get("config_id") is None:
            raise serializers.ValidationError(
                "Either config_uuid or config_id is required; "
                "config_uuid is preferred (config_id is deprecated)."
            )
        return attrs


class TestCallUsageSerializer(serializers.Serializer):
    """Usage block in test-call response."""

    model = serializers.CharField()
    prompt_tokens = serializers.IntegerField()
    completion_tokens = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    cached_tokens = serializers.IntegerField(required=False, default=0)
    reasoning_tokens = serializers.IntegerField(required=False, default=0)
    cost = serializers.FloatField(allow_null=True)
    cost_currency = serializers.CharField(allow_null=True)


class TestCallResponseSerializer(serializers.Serializer):
    """Response shape for POST .../llm-config/test-call/ (200)."""

    ok = serializers.BooleanField(
        help_text="True if the completion succeeded",
    )
    content = serializers.CharField(
        allow_null=True,
        required=False,
        help_text="Model reply when ok is true",
    )
    detail = serializers.CharField(
        allow_null=True,
        required=False,
        help_text="Error message when ok is false",
    )
    usage = TestCallUsageSerializer(
        allow_null=True,
        required=False,
        help_text="Token and cost info when ok is true",
    )


class ConfigTestResponseSerializer(serializers.Serializer):
    """Response shape for POST .../llm-config/test/ (200)."""

    ok = serializers.BooleanField(
        help_text="True if validation/completion succeeded",
    )
    detail = serializers.CharField(
        allow_null=True,
        required=False,
        help_text="Error message when ok is false",
    )


class ProviderParamSchemaEntrySerializer(serializers.Serializer):
    """Per-provider param schema (one entry in GET .../providers/)."""

    required = serializers.ListField(
        child=serializers.CharField(),
        help_text=(
            "Required config keys (e.g. api_key, or "
            "api_key+api_base+deployment for Azure)",
        ),
    )
    optional = serializers.ListField(
        child=serializers.CharField(),
        help_text=(
            "Optional config keys (e.g. model, api_base, max_tokens, "
            "temperature, top_p)",
        ),
    )
    editable_params = serializers.ListField(
        child=serializers.CharField(),
        help_text="All configurable keys in suggested order",
    )
    default_model = serializers.CharField(
        allow_null=True,
        help_text="Default model id when not set in config",
    )
    default_api_base = serializers.CharField(
        allow_null=True,
        help_text="Official API base URL when not set in config",
    )
    default_temperature = serializers.FloatField(
        allow_null=True,
        required=False,
        help_text="Provider default temperature",
    )
    default_top_p = serializers.FloatField(
        allow_null=True,
        required=False,
        help_text="Provider default top_p",
    )
    default_max_tokens = serializers.IntegerField(
        allow_null=True,
        required=False,
        help_text="Provider default max_tokens",
    )
    default_num_retries = serializers.IntegerField(
        required=False,
        help_text="Default retries after the initial provider attempt",
    )
    model_types = serializers.ListField(
        child=serializers.CharField(), required=False,
    )
    decision = serializers.DictField(
        required=False,
        help_text="Decision-specific credentials, defaults and model choices.",
    )


class ProvidersResponseSerializer(serializers.Serializer):
    """Response for GET .../llm-config/providers/. Keys are provider ids."""

    providers = serializers.DictField(
        child=ProviderParamSchemaEntrySerializer(),
        help_text=(
            "Map of provider id to param schema (required, optional, "
            "default_model, default_api_base)",
        ),
    )


class ModelCapabilitySerializer(serializers.Serializer):
    """One capability tag (e.g. text-to-text, vision, reasoning)."""

    id = serializers.CharField()
    name = serializers.CharField(allow_null=True, required=False)


class ModelEntrySerializer(serializers.Serializer):
    """One model under a provider in GET .../llm-config/models/."""

    id = serializers.CharField(help_text="Model id (e.g. gpt-4o-mini)")
    label = serializers.CharField(required=False)
    mode = serializers.CharField(required=False)
    model_info = ModelInfoSerializer(required=False)
    name = serializers.CharField(allow_null=True, required=False)
    capabilities = ModelCapabilitySerializer(many=True, required=False)


class ProviderModelsEntrySerializer(serializers.Serializer):
    """One provider block in GET .../llm-config/models/."""

    id = serializers.CharField(help_text="Provider id")
    label = serializers.CharField(required=False)
    name = serializers.CharField(allow_null=True, required=False)
    icon = serializers.CharField(
        allow_null=True,
        required=False,
        help_text="Optional brand logo URL for UI",
    )
    models = ModelEntrySerializer(many=True)
    default_api_base = serializers.CharField(allow_null=True, required=False)
    model_types = serializers.ListField(
        child=serializers.CharField(), required=False,
    )


class ModelsResponseSerializer(serializers.Serializer):
    """Response shape for GET .../llm-config/models/."""

    providers = serializers.ListField(
        child=ProviderModelsEntrySerializer(),
        help_text=(
            "List of providers with models and capability tags "
            "(e.g. text-to-text, vision, code, reasoning)",
        ),
    )
