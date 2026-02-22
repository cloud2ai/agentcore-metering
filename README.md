# Agentcore Metering

[中文](README.zh-CN.md)

Unified module for AI service invocation and statistics in Django projects.

- Uses **LiteLLM** for completion and cost metering.
- Stores cost as amount + currency (e.g. `cost`, `cost_currency`).
- Uses USD as default cost currency from LiteLLM.

---

## Install

- **Not on PyPI**, install only from GitHub.

**From GitHub** (editable after clone):
```bash
pip install -e git+https://github.com/cloud2ai/agentcore-metering.git
```
Or, when the host project uses it as a submodule, from repo root:
```bash
pip install -e path/to/agentcore-metering
```
- The host project Dockerfile should iterate over `agentcore/`
  submodules and run `pip install -e`.
- See the host project README for details.

---

## Testing

Install with dev extras, then run pytest from the package root:

```bash
pip install -e ".[dev]"
pytest tests -v
```

- Unit tests cover:
  - `llm_usage_stats` (parse, summary, by_model, series)
  - `llm_usage` (paginated list and query parsing)
- API tests cover:
  - `GET .../token-stats/`
  - `GET .../llm-usage/`
  - `llm-config` endpoints (auth and response shape)

---

## Backend: usage

1. **Register**
   - Add `'agentcore_metering.adapters.django'` to `INSTALLED_APPS`
   - Add `path('api/v1/admin/', include('agentcore_metering.adapters.django.urls'))` to root URLconf
2. **Call LLM and persist usage**:
   ```python
   from agentcore_metering.adapters.django import LLMTracker
   content, usage = LLMTracker.call_and_track(
       messages=[{"role": "user", "content": "..."}],
       node_name="my_node",
       state=request_state,
   )
   ```
   - `state` is optional (task/user context).
   - If `state` contains `user_id`, per-user LLM config (if set) is used.
3. **Config**
   - All config can be managed by admin APIs (global defaults + optional per-user overrides).
   - DB resolution picks the first active config by `order`: user scope -> global scope -> Django `settings` fallback.
   - LLM calls go through **LiteLLM**.
   - Cost (USD) is estimated from LiteLLM reference pricing and stored per call and in stats.

### Supported providers

- Each provider uses a smallest/cheapest default model to reduce
  misconfiguration cost.
- You can override `model` and optionally `api_base`/`deployment`
  in `config`.
- **All providers support `api_base` (URL)**:
  - official endpoint when omitted
  - proxy/forwarding URL when needed

| Provider        | Default model / notes |
|-----------------|------------------------|
| `openai`        | gpt-4o-mini (official default URL) |
| `azure_openai`  | gpt-4o-mini (requires `api_base` and `deployment`) |
| `gemini`        | gemini-2.0-flash |
| `anthropic`     | claude-3-5-haiku |
| `mistral`       | mistral-tiny |
| `dashscope`     | qwen-turbo (Alibaba Qwen) |
| `deepseek`      | deepseek-chat |
| `xai`           | grok-3-mini-beta (Grok) |
| `meta_llama`    | Llama-3.3-8B-Instruct |
| `amazon_nova`   | nova-micro-v1 |
| `nvidia_nim`    | meta/llama3-8b (Nemotron / NIM) |
| `minimax`       | MiniMax-M2.1 |
| `moonshot`      | moonshot-v1-8k (Kimi) |
| `zai`           | glm-4.5-flash (Z.AI GLM) |
| `volcengine`    | doubao-pro-32k (ByteDance Doubao) |
| `openrouter`    | google/gemma-2-9b-it:free |

- Settings fallback:
  - set `LLM_PROVIDER` and matching config key (e.g. `OPENAI_CONFIG`, `ANTHROPIC_CONFIG`)
  - each config dict should include at least `api_key`
  - all providers accept optional `api_base` (official by default, proxy when needed)
  - Azure requires `api_base` and typically `deployment`

---

## API reference

- Mount under an admin prefix (e.g. `api/v1/admin/`).
- **Auth**: `IsAdminUser` (staff or superuser), otherwise 403.
- If the main project uses drf-spectacular, endpoints appear in
  Swagger UI (e.g. `/swagger`) under **llm-metering**.

### LLM configuration (global and per-user)

| Method | Path | Description |
|--------|------|-------------|
| GET | `.../llm-config/` | List global LLM configs (ordered by `order`, `id`) |
| POST | `.../llm-config/` | Create one config. Body: `provider`, `config` (optional `scope`, `user_id`, `is_active`, `order`) |
| GET | `.../llm-config/all/` | List all LLM configs (global + user). Optional `scope=all|global|user`, `user_id` |
| GET | `.../llm-config/<pk>/` | Get one config by id |
| PUT | `.../llm-config/<pk>/` | Update one config by id |
| DELETE | `.../llm-config/<pk>/` | Delete one config by id |
| GET | `.../llm-config/providers/` | Per-provider param schema (required/optional/editable keys, default model and api_base) for building provider-specific forms |
| GET | `.../llm-config/models/` | Provider list and model list with capability tags (text-to-text / vision / code / reasoning, etc.) |
| POST | `.../llm-config/test/` | Validate credentials without saving. Body: `provider`, `config` |
| POST | `.../llm-config/test-call/` | Run one synchronous completion by saved config id and persist usage. Body: `config_id`, `prompt`, optional `max_tokens` |
| GET | `.../llm-config/users/` | List per-user configs (optional `?user_id=` filter) |
| GET | `.../llm-config/users/<user_id>/` | Get one user's config (404 if not set) |
| PUT | `.../llm-config/users/<user_id>/` | Create or update that user's config |
| DELETE | `.../llm-config/users/<user_id>/` | Remove user config (they fall back to global/settings) |

- **POST/PUT body**
  - `provider` (default `openai`)
  - `config` (single JSON object, e.g. `api_key`, `model`, `api_base`,
    `deployment`, `max_tokens`, `temperature`, `top_p`)
  - for create/list workflows: optional `scope`, `user_id`, `is_active`,
    `order`, optional `model_type`
  - required/optional keys differ by provider (e.g. Azure needs
    `api_base` and `deployment`)
  - use `GET .../llm-config/providers/` to fetch schema
  - on GET, `config.api_key` and `config.key` are masked (e.g. `sk-**xxxx`)

- **GET `.../llm-config/providers/`**
  - returns `{ "providers": { "<provider>": { "required": [...], "optional": [...], "editable_params": [...], "default_model": "...", "default_api_base": "..." } } }`
  - use it to render provider-specific forms and placeholders

- **POST `.../llm-config/test/`**
  - body: `provider`, `config` (same as PUT)
  - runs a minimal completion to verify key and endpoint
  - success: `200` + `{ "ok": true }`
  - validation/completion failure: `200` + `{ "ok": false, "detail": "..." }`
  - invalid payload: `400`

- **POST `.../llm-config/test-call/`**
  - body: `config_id`, `prompt`, optional `max_tokens` (default 512, max 4096)
  - success: `{ "ok": true, "content": "...", "usage": { ... } }`
  - failure: `{ "ok": false, "detail": "..." }`
  - call is persisted to usage records

### GET `.../token-stats/`

- Returns aggregates and optional time series.
- Query params are all optional:

| Param        | Type   | Description |
|--------------|--------|-------------|
| start_date   | string | Start time, ISO or date-only (e.g. 2025-01-01) |
| end_date     | string | End time; date-only is end of that day |
| user_id      | int    | Filter by user id |
| granularity  | string | Time bucket: `day` (hour), `month` (day), `year` (month); omit for no series |

**200** (JSON):

```json
{
  "summary": {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "total_cached_tokens": 0,
    "total_reasoning_tokens": 0,
    "total_cost": 0,
    "total_cost_currency": "USD",
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0
  },
  "by_model": [
    {
      "model": "gpt-4",
      "total_calls": 10,
      "total_prompt_tokens": 1000,
      "total_completion_tokens": 500,
      "total_tokens": 1500,
      "total_cached_tokens": 0,
      "total_reasoning_tokens": 0,
      "total_cost": 0.012,
      "total_cost_currency": "USD"
    }
  ],
  "series": null
}
```

- `series` is non-null only when `granularity` is set:
  `{ "granularity": "day", "items": [ { "bucket": "...", ... } ] }`
- invalid `granularity` returns `400` + `{ "detail": "..." }`

---

### GET `.../llm-usage/`

- Returns a paginated usage list.
- Query params:

| Param      | Type   | Description |
|------------|--------|-------------|
| page       | int    | Page number, default 1 |
| page_size  | int    | Page size, default 20, max 100 |
| user_id    | int    | Filter by user id |
| model      | string | Model name (icontains) |
| success    | string | `true` / `false` |
| start_date | string | Start time |
| end_date   | string | End time |

**200** (JSON):

```json
{
  "results": [
    {
      "id": "uuid",
      "user_id": "uuid-or-null",
      "username": "string-or-null",
      "model": "gpt-4",
      "prompt_tokens": 100,
      "completion_tokens": 50,
      "total_tokens": 150,
      "cost": 0.0012,
      "cost_currency": "USD",
      "success": true,
      "error": null,
      "created_at": "2025-01-01T12:00:00+00:00",
      "metadata": {}
    }
  ],
  "total": 100,
  "page": 1,
  "page_size": 20
}
```

---

- **UX**
  - use `token-stats` for dashboards and trend charts
  - use `llm-usage` for filterable, paginated call logs

---

## Data and layout

- **Tables**: `llm_tracker_usage` (usage records), `agentcore_metering_llm_config` (LLM provider config).
- **Package**: `agentcore_metering.adapters.django` is a full Django app (models, views, urls, admin, migrations). The public API for LLM metering lives under `trackers/` (e.g. `trackers/llm.py`); import `LLMTracker` from `agentcore_metering.adapters.django` or `agentcore_metering.adapters.django.trackers.llm`. Additional tracker types may be added under `trackers/` (e.g. `trackers/other.py`).
