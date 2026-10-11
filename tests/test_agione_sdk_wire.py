"""The SDK upgrade must preserve all three existing AGIOne protocols."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentcore_metering.adapters.django import LLMTracker
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("model,base_path,path", [
    ("openai/gpt-5.4-nano/d688d", "/v1", "/v1/responses"),
    ("anthropic/claude-sonnet-4.6/e6887", "", "/v1/messages"),
    ("deepseek/deepseek-v4-flash/02acd", "/v1", "/v1/chat/completions"),
])
def test_agione_sdk_uses_configured_model_and_protocol(model, base_path, path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(
                self.headers["Content-Length"],
            )))
            calls.append((self.path, body))
            if self.path == "/v1/responses":
                payload = {
                    "id": "resp_test", "object": "response",
                    "created_at": 1, "status": "completed", "model": model,
                    "output": [{
                        "id": "msg_test", "type": "message",
                        "role": "assistant", "status": "completed",
                        "content": [{
                            "type": "output_text", "text": "ok",
                            "annotations": [],
                        }],
                    }],
                    "usage": {"input_tokens": 12, "output_tokens": 2,
                              "total_tokens": 14},
                }
            elif self.path == "/v1/messages":
                payload = {
                    "id": "msg_test", "type": "message", "model": model,
                    "role": "assistant", "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 12, "output_tokens": 2},
                }
            else:
                payload = {
                    "id": "chat_test", "object": "chat.completion",
                    "created": 1, "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "ok"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2,
                              "total_tokens": 14},
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        row = LLMConfig.objects.create(
            scope="global", provider="agione", config={
                "model": model, "api_key": "test-only-key",
                "api_base": f"http://127.0.0.1:{server.server_port}{base_path}",
                "num_retries": 0, "request_timeout_seconds": 5,
            },
        )
        content, usage = LLMTracker.call_and_track(
            messages=[{"role": "user", "content": "hello"}],
            model_uuid=str(row.uuid),
        )
        assert content == "ok"
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 2
        assert len(calls) == 1
        assert calls[0][0] == path
        assert calls[0][1]["model"] == model
        assert LLMUsage.objects.get().success
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
