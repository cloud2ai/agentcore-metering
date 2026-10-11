# Decision models

Decision calls are owned by **agentcore-metering**, independently of any
application. The client uses LiteLLM SDK's Decisions API for TypeSafe Jev, Cloudflare
Clef, Perplexity, OpenAI, OpenRouter and self-hosted Strands Decider.
Existing LiteLLM Gateway / OpenAI-compatible Decisions configurations remain
supported; their deployment aliases are passed unchanged.

## Configure

Choose a native provider (`typesafe`, `cloudflare`, `perplexity`, `openai`,
`openrouter`, `strands_decider`) and mark `model_type=decision`. Enter that
provider's API key and model name. Preset model and endpoint suggestions are
returned by the provider catalog; model names remain editable. Cloudflare
accepts a 32-character `account_id` or an explicit `api_base`. Strands requires
an explicit endpoint, with an optional key. Credentials and endpoints are
resolved per configuration, never from global provider environment variables.
LiteLLM performs native protocol translation; this package does not duplicate
System One or Cloudflare adapters.

For keyless Strands endpoints, LiteLLM 1.104.2's HTTP wrapper generates an
invalid empty Bearer header; passing `None` instead would borrow environment
credentials. Only this keyless path uses a direct HTTP request with no
Authorization header, reusing LiteLLM's provider request/response conversion.
Authenticated native providers retain the SDK call path. This workaround does
not modify process environment variables or shared SDK state. Self-hosted
keyless calls record token usage with unknown cost rather than an assumed zero.

For an existing gateway, use `litellm_proxy` or `openai_compatible`:

1. Use a gateway release supporting the OpenAI Decisions format: LiteLLM
   `1.104.2+` on the `1.104.x` line, or `1.105.0rc3+`. Some development builds
   served the older System One format at the same path; those are incompatible.
   See <https://docs.litellm.ai/docs/decisions>.
2. Add the upstream decision provider/model on that gateway.
3. Create a configuration using the existing admin API:

```json
{
  "model_type": "decision",
  "scope": "global",
  "provider": "litellm_proxy",
  "config": {
    "api_base": "https://gateway.example/v1",
    "api_key": "gateway-virtual-key",
    "model": "your-jev-deployment",
    "request_timeout_seconds": 60,
    "num_retries": 0
  },
  "is_active": true
}
```

`api_base` accepts the gateway origin or its `/v1` base, including path prefixes.
Do not append `/decisions`. Model aliases are sent unchanged, including any
provider prefix in the alias. No provider-specific adapter is duplicated here.
`openai_compatible` is also accepted when explicitly marked `decision`.

Existing config lists remain chat-only by default. Request
`GET .../llm-config/all/?model_type=all` to manage all types, or
`?model_type=decision` for decision models. Decision models never participate
in default chat selection and cannot be set as the default LLM. Editing or
toggling a decision config preserves its type. No table or data migration is
needed: the existing indexed `model_type` column accepts the new value.

## Call from any Django application

```python
from agentcore_metering.adapters.django import DecisionTracker

answers, usage = DecisionTracker.call_and_track(
    model_uuid=decision_config_uuid,
    input="The content to evaluate",
    questions=[
        {
            "type": "predicate",
            "name": "relevant",
            "instructions": "Is the content about artificial intelligence?",
        },
        {
            "type": "choice",
            "name": "category",
            "instructions": "Choose the best category.",
            "choices": [{"value": "model"}, {"value": "product"}],
        },
    ],
    state={"user_id": user_id, "node_name": "my_decision_step"},
)
```

The UUID must identify an active decision configuration. Explicit selection
does not silently fall back to another configuration. This is a trusted server
library: the host application must authorize user input and configuration
access before calling it, as for `LLMTracker`.

Up to 128 questions can share one input. Supported question types are
`predicate`, `choice` and `score`. This first version accepts **text input**
and does not stream. Business thresholds, fallbacks and subsequent actions
belong to callers. A refusal is returned as a refusal, never converted to yes
or no. Existing application pipelines are not switched to decisions by adding
a configuration.

## Test and meter

`POST /api/v1/admin/llm-config/decision-test` requires admin access. Send
`config_uuid`, `input` and `questions`; it returns `ok`, `answers`, `usage`.
For an unsaved configuration, send `provider` and `config` instead of the UUID.
`POST .../llm-config/test/` dispatches a minimal decision test when the config
is marked `decision`. Ordinary chat testing keeps its existing endpoint.

Every completed request is recorded in the existing usage table, with
`model_type=decision`, configured model alias, actual response model, question
count, timestamps and normalized token counts. Native calls use the shared LiteLLM cost extractor; gateway calls use the
`x-litellm-response-cost` header in USD. Missing cost remains `null`, not zero.
Migration `0019_increase_cost_precision` widens usage and series costs to
12 decimal places while preserving the existing integer range. Apply it in
each consuming application. Previously rounded costs cannot be recovered.
Usage record responses include `cached_tokens` and `reasoning_tokens`.
Input text, answers and API keys are not written to usage metadata.

Authentication and invalid-request errors are not retried. Network failures,
408, 429 and server failures use the shared bounded retry policy. Each failed
retry is recorded. Unknown usage/cost on a lost response stays unknown;
malformed answers with returned usage still preserve that billed usage.

## Personal model settings

Mount `agentcore_metering.adapters.django.user_urls` under `/api/v1/user/`.
`POST /api/v1/user/llm-config/test-call` tests a saved chat model using
`config_uuid`, `prompt` and optional `max_tokens` / `stream`. It reuses the
existing completion and metering flow with `node_name=user_test_call`.
Personal tests require ownership, an active configuration and the correct
model type. Only staff can spend a shared model's key on a test call.
`GET /api/v1/user/llm-config/providers` returns the same public provider
metadata as admin settings, with no saved model configurations or secrets.
All new endpoint paths omit trailing slashes. It provides:

- `POST /api/v1/user/llm-config/test`: test caller-supplied credentials.
- `POST /api/v1/user/llm-config/decision-test`: test an active saved decision
  config or caller-supplied credentials, with multiple questions in one call.

Both require authentication. A saved-key test is permitted for the owner's
personal config, or a global config when the caller is staff. Merely reading
a global model does not permit spending its shared key on test calls. Unknown,
disabled and unauthorized UUIDs return 404 without calling the gateway.
Admins can test other users' configs through the separate admin endpoint.

Hosts implementing personal CRUD reuse `LLMConfigWriteSerializer` and
`services.config_access.user_model_configs` for validation and visibility.
NewsHub exposes this at `/api/v1/user/llm-config` (list/create) and
`/api/v1/user/llm-config/<uuid>` (edit/delete); its previous personal URLs remain
compatible. Lists default to chat models; settings screens request
`?model_type=all`. User and admin screens both mark decisions explicitly.
Conversion to or from decision requires a separate config to avoid breaking
workflow references or default bindings. Legacy chat/embedding/image type
updates retain their existing behavior. Mixed providers such as OpenAI and
OpenRouter require explicit `model_type=decision`; adding a decision alias to
the catalog does not reclassify legacy chat requests. Changing credentials,
endpoint, alias, timeout, retries and enabled state remains supported.

## Dependencies and rollout

Native providers require LiteLLM `>=1.104.2,<1.105`. The previous SDK-only
function-tool import regression is covered by the upgrade smoke test.
HTTPX remains an explicit dependency for existing gateway configurations.
Publish a new agentcore-metering package version before updating consuming
projects' frozen locks. For local integration, use the host project's documented
read-only source mount and `PYTHONPATH`; do not editable-install into containers.

LiteLLM now refreshes its remote price catalog with retries during import.
Hosts without access to GitHub's raw content endpoint should set
`LITELLM_LOCAL_MODEL_COST_MAP=True` before importing LiteLLM, including in
workers and schedulers. This uses the SDK's bundled price snapshot; pricing
then updates with SDK releases. Agentcore does not override the host's choice.
