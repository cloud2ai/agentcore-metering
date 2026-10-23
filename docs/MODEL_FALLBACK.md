# Model fallback (#486)

AgentCore Metering owns model resolution, switching, cooldown and per-request
metering. NewsHub supplies the admin and personal configuration UI. Article
classification, risk thresholds, persistence and publishing remain in their
existing business workflows.

## Configuration and compatibility

Migration `0020_llmconfig_fallback` adds a nullable JSON field to `LLMConfig`.
It is separate from provider parameters and is never passed to LiteLLM. An
omitted/null policy or an empty backup list preserves the previous invocation,
retry and return-value behavior. Existing rows are not opted in on upgrade.
The nullable column also permits inserts from the outgoing blue/green version.

Upgrading a Django consumer requires applying `0020` before new application
processes query model configurations. The outgoing application can keep using
the expanded schema; rolling application code back does not require dropping
the new column. The previous call signatures remain valid. Decision methods
only add optional keyword arguments, and configuration responses add a read-only
`fallback` field. Internal dict subclasses preserve the mapping/JSON contract;
their model-resolution attributes are not provider request fields.

The bounded retry policy and stricter terminal-response checks apply only to
explicitly enabled chains. Unconfigured personal models do not automatically
inherit a configured global chain. Compatibility tests cover legacy calls,
upgrade from `0019`, mixed-version ORM writes and preservation of stored policy.

Both API families expose the same resources, without a trailing slash:

| Operation | Resource |
|---|---|
| Read policy, candidates and circuit state | `GET /api/v1/{admin,user}/llm-config/{uuid}/fallback` |
| Replace policy; `{"fallback": null}` clears it | `PUT /api/v1/{admin,user}/llm-config/{uuid}/fallback` |
| Manually reset this chain's circuit state | `POST /api/v1/{admin,user}/llm-config/{uuid}/fallback/reset` |

A complete example (fields other than `fallback` use the defaults below when
omitted):

```json
{
  "fallback": {
    "model_uuids": ["backup-configuration-uuid"],
    "inherit_global": false,
    "conditions": [
      "timeout", "rate_limit", "upstream_error", "connection_error",
      "quota_exhausted"
    ],
    "timeout_seconds": 120,
    "retries": 0,
    "cooldown_seconds": 60,
    "auto_recover": true
  }
}
```

- The ordered list accepts at most three distinct, existing, active models.
  Use actual UUIDs in requests, not the placeholder in this example.
- `timeout_seconds`: integer 5–600, upper bound per model; the model's own
  shorter configured timeout still applies.
- `retries`: 0 or 1, before proceeding to the next model. Quota exhaustion
  always switches immediately without retrying the depleted model.
- `cooldown_seconds`: integer 0–3600. `auto_recover` defaults to true.
- Optional `empty_response` is off by default. Refusal, truncation, invalid
  JSON, invalid decisions and other business validation errors never switch.
- Global models reference global models only. Personal models reference their
  owner's models or global models, even when edited by an administrator.
  Foreign models are neither candidates nor valid writable references.
- Graph writes lock visible configurations in PK order and reread after
  acquiring the locks. Self-reference and indirect cycles are rejected.
  Runtime rechecks access, active state and model type before each model.
- A personal policy can instead use `{"inherit_global": true}`. This inherits
  the active same-type global default's backup order and policy, not its
  primary model. It cannot also specify local backups. It is explicit opt-in;
  existing personal configurations do not inherit automatically.
- If an inherited source or backup is later deleted/disabled, it is not called.
  An absent inherited policy leaves the original primary behavior in place.
  The UI displays unavailable local references so they can be removed.
- Ordinary config edits use explicit update fields, so a stale edit cannot
  overwrite a concurrently saved fallback policy.
- A personal and a global configuration pointing to the same provider,
  model, endpoint and credential cannot back each other up. Tuning differences
  such as output limits and retries do not make a different destination.
  Candidates exclude these duplicates, saves reject them, and execution skips
  backups made identical by later edits. The same model with another endpoint
  or credential is still allowed; the UI labels personal and global choices.

## Execution

```text
Resolve the primary and its explicit, flat list (never expand a backup chain)
  -> recheck current scope, active state and known required capabilities
  -> primary request, unless its circuit is cooling down
       -> success: return the existing result contract
       -> refusal / validation / authentication failure: stop
       -> selected availability error: bounded retry or next backup
  -> backups in order, under the same total deadline
  -> return a successful result, or preserve the terminal error
```

The total budget is the sum of bounded model timeouts times `(retries + 1)`,
capped at 3600 seconds, with at most eight requests. Each attempt reserves a
proportion of remaining time for later attempts. SDK/provider retries are
explicitly zero; JSON repair makes at most one request inside an attempt.
The request transport must obey its supplied timeout; this is not thread or
process cancellation for an arbitrary provider that ignores timeouts.

Terminal errors carry `agentcore_no_retry` so NewsHub's JSON repair and digest
Celery loops do not replay refusals/validation after the bounded call. Other
business retry/checkpoint policies are retained. All-cooling calls fail without
issuing requests. Existing business retries remain separate calls; this policy
does not add Celery retries. Its attempt limit applies to one tracker call,
not an entire long-running business workflow.

A saved model's connection/test action deliberately bypasses its backup policy:
a working backup must not make an unavailable primary appear healthy.

## Real-request recovery

Circuit state lives in Django's configured shared cache (Redis in NewsHub),
scoped to the requested configuration, effective policy version and target.
Changing a configuration/policy starts a new state namespace.

- A selected service failure opens that target's circuit for the cooldown.
- Other calls use the remaining available models rather than waiting.
- After cooldown, one real request acquires an atomic cache lease and checks
  recovery. Concurrent calls continue to backups. There is no background probe
  task or separate paid connection test.
- A successful request closes the circuit. A failed check opens it again.
  The recovery lease expires if its worker disappears.
- With automatic recovery off, manual reset is required.
- A cache outage fails open to normal bounded provider calls; it does not make
  a model permanently unavailable. No credentials or prompts enter cache keys.

## Capabilities and streams

Chat primaries accept only chat backups. Native decisions accept native decision
backups or chat models through `decision_chat.py` and the packaged
`llm_static/decision_adapter.yaml` prompt. The adapter preserves question names,
predicate/choice/score types, typed option identities and complete normalized
choice probabilities; invalid output is metered once and rejected. It never
sends a `/decisions` request to a chat model.

Each target resolves its own credentials and provider-specific parameters.
Chat calls reapply the caller's reasoning, tools and structured output options.
Native-decision-to-chat defaults to reasoning off unless explicitly supplied.
Known modality/tool/schema/output incompatibilities and oversized text input
are skipped before requesting a backup. Text size uses a local OpenAI-compatible
estimate; unknown/custom gateway limits and exact provider tokenization still
require provider validation. Original input is never truncated to fit a backup.

For streaming, switching is permitted only before the first emitted content or
reasoning chunk. Once output escapes, a failure is surfaced, never replayed on
another model. Closing the consumer closes the active generator. Midstream
failures are marked as potentially incomplete usage; unknown cost stays null.

## Metering

Every actual provider attempt uses the existing `LLMUsage` row, with its own
actual model, tokens, duration, success and cost. No logical-call billing row
is added. Validation of an already billed response updates that same row rather
than creating another charge. Failed requests with no received usage remain
unknown cost; they are not presented as free.

Safe metadata links rows using `fallback_call_id`, `requested_config_uuid`,
`actual_config_uuid`, `fallback_attempt`, `fallback_model_attempt`,
`fallback_reason` and `fallback_failure_reason`. The final attempted row records
the logical result; response-model provenance is also retained. Credentials,
article text and raw upstream errors are not copied into these provenance keys.

## Verification and release order

The tests cover configuration ownership/cycles, PostgreSQL concurrent inverse
writes, chat/native decisions/chat adapters, SDK wire requests, retry budgets,
quota handling, refusal/truncation/invalid responses, circuit recovery, streams,
per-attempt costs and old unconfigured behavior. NewsHub integration tests use
its real business adapters and persistence boundary with simulated provider
outages; browser checks exercise admin and personal settings on desktop/mobile.

1. Review and release AgentCore Metering to PyPI first.
2. Confirm that version exists; update NewsHub's minimum dependency and frozen
   `uv.lock`. Do not point a production lock at an unreleased version/path.
3. Rebuild and recreate dev API/workers with the published package, remove the
   temporary read-only source override, and run both project release gates.
4. Apply the additive migration during the normal Hub deployment. Configure
   backups deliberately after upgrade; existing policies remain off by default.

Local development can use the documented read-only `AGENTCORE_SRC`/`PYTHONPATH`
override. It does not change the published package or production configuration.

## LiteLLM assessment

The tested NewsHub dependency is LiteLLM 1.104.2. Its Router has chat fallback,
but no Router `decisions`/`adecisions` entry point. The shared policy therefore
sits above the existing LiteLLM chat/native transport adapters, rather than
adding a second Router retry layer or forwarding unsupported fields.

- https://docs.litellm.ai/docs/proxy/reliability
- https://docs.litellm.ai/docs/decisions
