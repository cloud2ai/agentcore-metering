"""Gateway envelopes must preserve availability and policy distinctions."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from litellm.exceptions import AuthenticationError, BadRequestError

from agentcore_metering.adapters.django import LLMTracker
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage
from agentcore_metering.adapters.django.services.fallback_config import (
    save_policy,
)
from agentcore_metering.adapters.django.services.model_fallback import (
    fallback_reason,
)


def gateway_body(code="invalid_request_error", message="Insufficient Balance"):
    return {
        "message": "The model service is temporarily unavailable.",
        "status": 502,
        "type": "provider_error",
        "code": "provider_unstable",
        "providerError": {
            "source": "model_source",
            "httpStatus": 402,
            "body": {
                "error": {
                    "message": message,
                    "type": "insufficient_balance_error",
                    "code": code,
                },
            },
        },
    }


def sdk_error(body, status=502):
    request = httpx.Request("POST", "https://gateway.invalid/completions")
    # SDKs can retain the parsed body while leaving response.content empty.
    response = httpx.Response(status, request=request)
    return BadRequestError(
        message="The model service is temporarily unavailable.",
        model="primary",
        llm_provider="openai",
        response=response,
        body=body,
    )


@pytest.mark.parametrize("encoded", [False, True])
def test_nested_quota_survives_sdk_bad_request_mapping(encoded):
    body = gateway_body()
    if encoded:
        body["providerError"]["body"] = json.dumps(
            body["providerError"]["body"],
        )
    assert fallback_reason(sdk_error(body)) == "quota_exhausted"


def test_nested_quota_code_without_a_human_message_is_recognized():
    body = gateway_body(code="insufficient_balance", message="Unavailable")
    assert fallback_reason(sdk_error(body)) == "quota_exhausted"


def test_gateway_response_body_without_sdk_body_is_recognized():
    request = httpx.Request("POST", "https://gateway.invalid/completions")
    response = httpx.Response(502, request=request, json=gateway_body())
    error = httpx.HTTPStatusError(
        "Unavailable", request=request, response=response,
    )
    assert fallback_reason(error) == "quota_exhausted"


@pytest.mark.parametrize("code", ["content_policy_violation", "model_refusal"])
def test_nested_refusal_still_wins_over_quota(code):
    body = gateway_body(code=code)
    assert fallback_reason(sdk_error(body)) is None


def test_authentication_failure_does_not_switch_even_with_a_quota_body():
    error = AuthenticationError(
        message="Invalid API key", model="primary", llm_provider="openai",
    )
    error.body = gateway_body()
    assert fallback_reason(error) is None


@pytest.mark.parametrize("status", [400, 502])
def test_invalid_parameters_stay_terminal(status):
    body = gateway_body(message="Invalid max_tokens parameter")
    body["providerError"]["httpStatus"] = 400
    body["providerError"]["body"]["error"]["type"] = "invalid_request_error"
    assert fallback_reason(sdk_error(body, status)) is None


def test_request_echo_is_not_used_as_a_quota_signal():
    body = {
        "message": "Invalid request",
        "request": {"message": "insufficient balance"},
    }
    assert fallback_reason(sdk_error(body, 400)) is None


def test_cyclic_gateway_body_has_bounded_inspection():
    body = {"message": "Invalid request"}
    body["body"] = body
    assert fallback_reason(sdk_error(body, 400)) is None


@pytest.mark.parametrize("body", ["not-json", "[" * 4096])
def test_malformed_or_deep_json_does_not_break_error_handling(body):
    assert fallback_reason(sdk_error(body, 400)) is None


@pytest.mark.parametrize("kind", [ValueError, TypeError])
def test_local_validation_errors_remain_terminal(kind):
    error = kind("Invalid article")
    error.body = gateway_body()
    assert fallback_reason(error) is None


@pytest.mark.django_db
def test_real_sdk_gateway_quota_switches_once_and_records_usage():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(
                self.headers["Content-Length"],
            )))
            calls.append(data["model"])
            if data["model"] == "primary":
                status, body = 502, gateway_body()
            else:
                status = 200
                body = {
                    "id": "backup-result",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "backup",
                    "choices": [{
                        "index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }],
                    "usage": {
                        "prompt_tokens": 12, "completion_tokens": 2,
                        "total_tokens": 14,
                    },
                }
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        configs = [
            LLMConfig.objects.create(
                scope="global", provider="openai_compatible", config={
                    "model": model, "api_key": "test-only-key",
                    "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                },
            )
            for model in ("primary", "backup")
        ]
        primary, backup = configs
        primary = save_policy(primary, {
            "model_uuids": [str(backup.uuid)],
            "conditions": ["quota_exhausted"], "retries": 1,
        })
        content, usage = LLMTracker.call_and_track(
            model_uuid=str(primary.uuid),
            messages=[{"role": "user", "content": "Reply OK."}],
        )
        assert content == "ok" and usage["total_tokens"] == 14
        assert calls == ["primary", "backup"]
        failed, passed = LLMUsage.objects.order_by("created_at")
        assert not failed.success and failed.cost is None
        assert passed.success
        assert failed.metadata["fallback_failure_reason"] == "quota_exhausted"
        assert passed.metadata["fallback_reason"] == "quota_exhausted"
        assert passed.metadata["actual_config_uuid"] == str(backup.uuid)
        assert passed.metadata["fallback_attempt"] == 2
        assert failed.metadata["fallback_call_id"] == (
            passed.metadata["fallback_call_id"]
        )
        assert "test-only-key" not in str(failed.metadata)
        assert "Insufficient Balance" not in str(failed.metadata)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
