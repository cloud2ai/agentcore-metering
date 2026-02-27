# Ray Review Python – agentcore-metering

**Rule:** ray-review-python (+ python-expert, python-code-review)  
**Scope:** Hand-written Python in `agentcore_metering/` and `tests/` (migrations excluded)  
**Date:** 2026-02-27

---

## 1. Findings (by severity)

### Medium (fixed)

| File | Line | Finding | Fix |
|------|------|---------|-----|
| `agentcore_metering/adapters/django/services/runtime_config.py` | ~410 | Import inside `run_test_call()`; Ray: imports at file top only. | Moved `from ...trackers.llm import LLMTracker` to top with other local imports. |
| `agentcore_metering/adapters/django/models.py` | 102 | Line length 82 chars (max 79). | Wrapped `help_text` in parentheses and split across lines. |
| `tests/test_llm_tracker.py` | 95, 137 | Patch path string 86 chars. | Split string across two lines with implicit concatenation. |
| `agentcore_metering/adapters/django/trackers/llm.py` | 36–37, 43–44 | Bare `except Exception: pass` swallows all errors. | First block: catch `(TypeError, ValueError)` and `Exception`, log at debug. Second: catch `(TypeError, ValueError)` only. |
| `agentcore_metering/adapters/django/services/runtime_config.py` | 198–199 | Bare `except Exception: pass` in `_extract_usage_from_response`. | Catch `(TypeError, ValueError)` and `Exception`, log at debug. |

### Low (fixed)

| File | Finding | Fix |
|------|---------|-----|
| `agentcore_metering/constants.py` | No module docstring. | Added triple-quoted module docstring (English). |
| `agentcore_metering/adapters/django/views/config_management.py` | No module docstring. | Added module docstring describing admin LLM config API. |
| `agentcore_metering/adapters/django/views/usage.py` | No module docstring. | Added module docstring for usage list and token stats views. |

### Advisory (no change)

- **Import order:** `runtime_config.py` has stdlib → third-party (django, litellm) → local; groups are correct. No change.
- **Logging:** Long tasks already use paired "Starting …" / "Finished …" (e.g. `TASK_RUN_TEST_CALL`, `TASK_VALIDATE_LLM_CONFIG`). OK.
- **usage_stats.get_summary_stats:** No type hints or docstring; acceptable for internal helper; consider adding in a later pass.

---

## 2. Open questions / assumptions

- **LiteLLM `completion_cost`:** Exception types are unspecified; kept a broad `except Exception` with debug log so unknown errors are visible without breaking call path.
- **Tests:** `test_llm_tracker.py` patch paths use string concatenation for line length; behaviour unchanged.

---

## 3. Summary

- **Applied:** Ray max line length 79, imports at top only, no bare `except Exception: pass` (narrowed or logged), and module docstrings for constants and the two view modules.
- **Residual risks:** None identified; cost fallback and validation paths are unchanged.
- **Testing:** Existing tests remain valid; no new tests added for the style/logging changes.
