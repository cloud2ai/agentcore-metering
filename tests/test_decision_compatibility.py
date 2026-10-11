"""Legacy configuration and real SDK chat contracts alongside Decisions."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from agentcore_metering.adapters.django import LLMTracker
from agentcore_metering.adapters.django.llm_static.load import (
    get_providers_with_models as load_legacy_models,
)
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage
from agentcore_metering.adapters.django.services.litellm_params import (
    _litellm_kwargs_from_config,
)
from agentcore_metering.adapters.django.services.model_catalog import (
    get_model_type_for_model_id,
    get_providers_with_models,
)

pytestmark = pytest.mark.django_db
BASE = "/api/v1/admin/llm-config/"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {}},
    },
}]
USAGE = {
    "prompt_tokens": 20,
    "completion_tokens": 6,
    "total_tokens": 26,
    "prompt_tokens_details": {"cached_tokens": 4},
    "completion_tokens_details": {"reasoning_tokens": 2},
}


def test_decision_catalog_preserves_legacy_entries():
    providers = {
        p["id"]: p for p in get_providers_with_models()["providers"]
    }
    for provider in load_legacy_models()["providers"]:
        for model in provider["models"]:
            assert model in providers[provider["id"]]["models"]


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-6-luna"),
    ("openrouter", "typesafe/jev-1.13"),
])
def test_mixed_provider_requires_explicit_decision_type(provider, model):
    # Existing callers do not send model_type for ordinary chat models.
    assert get_model_type_for_model_id(provider, model) == "llm"


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_model_type_update(admin_client, explicit):
    row = LLMConfig.objects.create(
        provider="openai", config={"model": "gpt-4o-mini"},
    )
    payload = {"config": {"model": "text-embedding-3-small"}}
    if explicit:
        payload["model_type"] = "embedding"
    response = admin_client.put(
        f"{BASE}{row.uuid}/", payload, format="json",
    )
    assert response.status_code == 200
    row.refresh_from_db()
    assert row.model_type == "embedding"


@pytest.fixture
def chat_endpoint():
    """Exercise LiteLLM/OpenAI over loopback; no paid provider requests."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(
                self.headers["Content-Length"]
            )))
            requests.append(body)
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer test-only-key"
            prompt = str(body["messages"][-1]["content"])
            if "unauthorized" in prompt:
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"bad key"}}')
                return
            if "retry" in prompt and len(requests) == 1:
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"rate limit"}}')
                return
            message = {"role": "assistant", "content": "ok"}
            if body.get("tools"):
                message.update(content=None, tool_calls=[{
                    "id": "call_test", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }])
            if body.get("response_format"):
                message["content"] = '{"ok": true}'
            self.send_response(200)
            stream = body.get("stream")
            self.send_header(
                "Content-Type",
                "text/event-stream" if stream else "application/json",
            )
            self.end_headers()
            envelope = {
                "id": "chat_test", "created": 1,
                "model": "gpt-4o-mini",
            }
            if stream:
                chunks = [
                    {"choices": [{"index": 0, "delta": {
                        "role": "assistant", "content": "ok",
                    }, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {},
                                  "finish_reason": "stop"}]},
                    {"choices": [], "usage": USAGE},
                ]
                for chunk in chunks:
                    value = {**envelope, "object": "chat.completion.chunk",
                             **chunk}
                    self.wfile.write(
                        f"data: {json.dumps(value)}\n\n".encode()
                    )
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps({
                    **envelope, "object": "chat.completion", "usage": USAGE,
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": (
                                     "tool_calls" if body.get("tools")
                                     else "stop"
                                 )}],
                }).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    params = _litellm_kwargs_from_config("openai_compatible", {
        "api_base": f"http://127.0.0.1:{server.server_port}/v1",
        "api_key": "test-only-key", "model": "gpt-4o-mini",
        "num_retries": 1, "request_timeout_seconds": 10,
    })
    try:
        with patch(
            "agentcore_metering.adapters.django.trackers."
            "llm.get_litellm_params", return_value=params,
        ):
            yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("mode", ["text", "json", "vision", "tools"])
def test_sdk_chat_http_contract(chat_endpoint, mode):
    content = "hello"
    kwargs = {}
    if mode == "json":
        kwargs["json_mode"] = True
    elif mode == "vision":
        content = [{"type": "text", "text": "describe"}, {
            "type": "image_url", "image_url": {
                "url": "data:image/png;base64,aGVsbG8=",
            },
        }]
    elif mode == "tools":
        kwargs.update(tools=TOOLS, tool_choice="auto", return_message=True)
    response, usage = LLMTracker.call_and_track(
        messages=[{"role": "user", "content": content}], **kwargs,
    )
    assert len(chat_endpoint) == 1
    assert chat_endpoint[0]["messages"][-1]["content"] == content
    if mode == "tools":
        assert chat_endpoint[0]["tools"] == TOOLS
        assert response["tool_calls"][0]["function"]["name"] == "lookup"
    elif mode == "json":
        assert json.loads(response) == {"ok": True}
    else:
        assert response == "ok"
    assert usage["prompt_tokens"] == 20
    assert usage["cached_tokens"] == 4
    assert usage["reasoning_tokens"] == 2
    assert usage["cost"] > 0
    assert LLMUsage.objects.get().success


def test_sdk_stream_http_contract(chat_endpoint):
    stream = LLMTracker.call_and_track(
        messages=[{"role": "user", "content": "hello"}], stream=True,
    )
    output = []
    while True:
        try:
            output.append(next(stream))
        except StopIteration as done:
            usage = done.value
            break
    assert ("content", "ok") in output
    assert usage["total_tokens"] == 26
    assert LLMUsage.objects.get().is_streaming
    assert len(chat_endpoint) == 1


@pytest.mark.parametrize("prompt,count,success", [
    ("retry", 2, True), ("unauthorized", 1, False),
])
def test_sdk_retry_budget(chat_endpoint, prompt, count, success):
    import litellm

    # Process-global SDK retries must not multiply Agentcore's budget.
    with patch.object(litellm, "num_retries", 5):
        if success:
            response, _ = LLMTracker.call_and_track(
                messages=[{"role": "user", "content": prompt}],
            )
            assert response == "ok"
        else:
            with pytest.raises(litellm.AuthenticationError):
                LLMTracker.call_and_track(
                    messages=[{"role": "user", "content": prompt}],
                )
    assert len(chat_endpoint) == count
    assert LLMUsage.objects.count() == count
    assert LLMUsage.objects.filter(success=True).exists() is success
