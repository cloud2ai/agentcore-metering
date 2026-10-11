"""Real HTTP proves keyless decisions never borrow process credentials."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agentcore_metering.adapters.django import DecisionTracker
from agentcore_metering.adapters.django.models import LLMUsage

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("ambient_key", [False, True])
@pytest.mark.parametrize("first_status", [200, 429, 401])
def test_keyless_strands_wire_and_metering(
    monkeypatch, ambient_key, first_status,
):
    if ambient_key:
        monkeypatch.setenv("STRANDS_DECIDER_API_KEY", "foreign-process-key")
    else:
        monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    monkeypatch.setenv(
        "STRANDS_DECIDER_API_BASE", "https://must-not-call.invalid",
    )
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(
                self.headers["Content-Length"],
            )))
            calls.append((self.path, dict(self.headers), body))
            status = first_status if len(calls) == 1 else 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {
                "model": "local-model",
                "answers": {"relevant": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 12, "output_tokens": 0},
            } if status == 200 else {"error": "test-only failure"}
            self.wfile.write(json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = {
            "model": "local-model",
            "api_base": f"http://127.0.0.1:{server.server_port}",
            "request_timeout_seconds": 2, "num_retries": 1,
        }

        def call():
            return DecisionTracker.call_with_config(
                provider="strands_decider", config=config, input="AI news",
                questions=[{
                    "type": "predicate", "name": "relevant",
                    "instructions": "Is this about AI?",
                }],
            )

        if first_status == 401:
            with pytest.raises(httpx.HTTPStatusError):
                call()
            assert not LLMUsage.objects.get().success
        else:
            answers, usage = call()
            assert answers[0]["probability"] == 0.9
            assert usage["prompt_tokens"] == 12
            assert usage["completion_tokens"] == 0
            assert usage["cost"] is None
            assert LLMUsage.objects.filter(success=True).count() == 1
        assert len(calls) == (2 if first_status == 429 else 1)
        assert LLMUsage.objects.count() == len(calls)
        for path, headers, body in calls:
            assert path == "/v1/systemone"
            assert "authorization" not in {key.lower() for key in headers}
            assert body["model"] == "local-model"
            assert body["state"] == "AI news"
            assert body["questions"]["relevant"]["type"] == "noul"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
