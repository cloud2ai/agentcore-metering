"""Decision configuration, gateway wire contract, and usage isolation."""

from decimal import Decimal
from unittest.mock import patch

import httpx
import pytest

from agentcore_metering.adapters.django import DecisionTracker
from agentcore_metering.adapters.django.models import LLMConfig, LLMUsage
from agentcore_metering.adapters.django.services.config_source import (
    get_config_from_db,
    get_default_llm_config_uuid,
)
from agentcore_metering.adapters.django.services.decisions import (
    decision_error_detail,
    decision_params,
    validate_questions,
)

pytestmark = pytest.mark.django_db
BASE = "/api/v1/admin/llm-config/"
CONFIG = {
    "api_base": "https://gateway.example/v1",
    "api_key": "secret-test-key",
    "model": "typesafe/jev-latest",
    "num_retries": 0,
}
QUESTIONS = [
    {
        "type": "predicate",
        "name": "relevant",
        "instructions": "Is this about AI?",
    },
    {
        "type": "choice",
        "name": "category",
        "instructions": "Choose category.",
        "choices": [{"value": "model"}, {"value": "product"}],
    },
    {
        "type": "score",
        "name": "quality",
        "instructions": "Score quality.",
        "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}],
    },
]
ANSWERS = [
    {"type": "predicate", "name": "relevant", "probability": 0.9},
    {"type": "choice", "name": "category", "choice": "model"},
    {"type": "score", "name": "quality", "score": 1.7},
]
PAYLOAD = {
    "model": "jev-1.13.0",
    "answers": ANSWERS,
    "usage": {
        "input_tokens": 123,
        "output_tokens": 0,
        "total_tokens": 123,
        "input_tokens_details": {"cached_tokens": 20},
    },
}


@pytest.fixture
def decision_config():
    return LLMConfig.objects.create(
        scope="global",
        provider="litellm_proxy",
        model_type="decision",
        config=dict(CONFIG),
    )


@pytest.fixture
def gateway(monkeypatch):
    original = httpx.Client

    def install(handler):
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kwargs: original(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )

    return install


def test_gateway_contract_and_metering(decision_config, gateway, admin_user):
    calls = []

    def handler(request):
        import json

        calls.append(request)
        body = json.loads(request.content)
        assert set(body) == {"model", "input", "questions"}
        assert body["model"] == CONFIG["model"]
        assert body["questions"] == QUESTIONS
        assert str(request.url) == "https://gateway.example/v1/decisions"
        assert request.headers["authorization"] == "Bearer secret-test-key"
        return httpx.Response(
            200, json=PAYLOAD, headers={"x-litellm-response-cost": "0.000123"}
        )

    gateway(handler)
    answers, usage = DecisionTracker.call_and_track(
        model_uuid=decision_config.uuid,
        input="AI model released.",
        questions=QUESTIONS,
        state={"user_id": admin_user.pk, "node_name": "test_decision"},
    )
    assert len(calls) == 1
    assert answers == ANSWERS
    assert usage["prompt_tokens"] == 123
    assert usage["completion_tokens"] == 0
    assert usage["cached_tokens"] == 20
    row = LLMUsage.objects.get()
    assert row.success and row.cost == Decimal("0.000123")
    assert row.user_id == admin_user.pk and row.started_at
    assert row.metadata["model_type"] == "decision"
    assert row.metadata["response_model"] == "jev-1.13.0"
    assert "secret-test-key" not in str(row.metadata)


@pytest.mark.parametrize(
    "status,expected_calls", [(401, 1), (400, 1), (429, 2)]
)
def test_gateway_retries_and_failure_audit(
    decision_config, gateway, status, expected_calls
):
    decision_config.config["num_retries"] = 1
    decision_config.save()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "test"})

    gateway(handler)
    with (
        patch(
            "agentcore_metering.adapters.django.services."
            "litellm_retry.time.sleep"
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        DecisionTracker.call_and_track(
            model_uuid=decision_config.uuid,
            input="AI",
            questions=QUESTIONS,
        )
    assert len(calls) == expected_calls
    assert LLMUsage.objects.filter(success=False).count() == expected_calls


def test_missing_answers_keep_billed_usage(decision_config, gateway):
    gateway(
        lambda _: httpx.Response(
            200,
            json={**PAYLOAD, "answers": []},
            headers={"x-litellm-response-cost": "0.000123"},
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        DecisionTracker.call_and_track(
            model_uuid=decision_config.uuid,
            input="AI",
            questions=QUESTIONS,
        )
    row = LLMUsage.objects.get()
    assert not row.success and row.prompt_tokens == 123
    assert row.cost == Decimal("0.000123")


def test_missing_cost_is_unknown_not_zero(decision_config, gateway):
    gateway(lambda _: httpx.Response(200, json=PAYLOAD))
    _, usage = DecisionTracker.call_and_track(
        model_uuid=decision_config.uuid,
        input="AI",
        questions=QUESTIONS,
    )
    assert usage["cost"] is None
    assert LLMUsage.objects.get().cost is None


def test_decision_cannot_replace_chat_default(admin_client, decision_config):
    chat = LLMConfig.objects.create(
        scope="global",
        provider="openai",
        config={"model": "gpt-4"},
        is_default=True,
    )
    assert get_default_llm_config_uuid() == str(chat.uuid)
    assert get_config_from_db(model_uuid=decision_config.uuid) is None
    response = admin_client.put(
        f"{BASE}{decision_config.uuid}/", {"is_default": True}, format="json"
    )
    assert response.status_code == 400
    chat.refresh_from_db()
    assert chat.is_default


def test_create_edit_list_decision(admin_client):
    response = admin_client.post(
        BASE,
        {
            "provider": "litellm_proxy",
            "model_type": "decision",
            "config": CONFIG,
        },
        format="json",
    )
    assert response.status_code == 201
    uuid = response.json()["uuid"]
    assert "***" in response.json()["config"]["api_key"]
    assert admin_client.get(BASE).json() == []
    assert len(admin_client.get(BASE + "?model_type=decision").json()) == 1
    assert len(admin_client.get(BASE + "all/?model_type=all").json()) == 1
    response = admin_client.put(
        f"{BASE}{uuid}/", {"is_active": False}, format="json"
    )
    assert response.status_code == 200
    assert response.json()["model_type"] == "decision"
    with pytest.raises(ValueError, match="not found"):
        DecisionTracker.call_and_track(
            model_uuid=uuid,
            input="AI",
            questions=QUESTIONS,
        )


@pytest.mark.parametrize(
    "body",
    [
        {"model_type": "invalid"},
        {"model_type": "decision", "provider": "anthropic", "config": CONFIG},
        {"model_type": "decision", "provider": "litellm_proxy", "config": {}},
        {"model_type": "llm", "provider": "litellm_proxy", "config": CONFIG},
        {
            "model_type": "decision",
            "provider": "litellm_proxy",
            "config": CONFIG,
            "is_default": True,
        },
    ],
)
def test_invalid_config_rejected(admin_client, body):
    assert admin_client.post(BASE, body, format="json").status_code == 400
    assert not LLMConfig.objects.exists()


def test_decision_test_api(admin_client, decision_config, gateway):
    gateway(lambda _: httpx.Response(200, json=PAYLOAD))
    response = admin_client.post(
        BASE + "decision-test",
        {
            "config_uuid": str(decision_config.uuid),
            "input": "AI",
            "questions": QUESTIONS,
        },
        format="json",
    )
    assert response.status_code == 200
    assert response.json()["answers"] == ANSWERS
    row = LLMUsage.objects.get()
    assert row.metadata["node_name"] == "admin_decision_test"


def test_decision_test_requires_admin(authenticated_client, decision_config):
    response = authenticated_client.post(
        BASE + "decision-test",
        {
            "config_uuid": str(decision_config.uuid),
            "input": "AI",
            "questions": QUESTIONS,
        },
        format="json",
    )
    assert response.status_code in (401, 403)
    assert not LLMUsage.objects.exists()


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [{"type": "predicate"}],
        [
            {
                "type": "choice",
                "instructions": "Pick",
                "choices": [{"value": "a"}],
            }
        ],
        [{"type": "score", "instructions": "Score", "levels": ["a", "b"]}],
        QUESTIONS + [QUESTIONS[0]],
    ],
)
def test_invalid_questions_fail_before_transport(questions):
    with pytest.raises(ValueError):
        validate_questions(questions)


def test_base_url_rejects_embedded_secrets():
    with pytest.raises(ValueError):
        decision_params(
            "litellm_proxy",
            {
                **CONFIG,
                "api_base": "https://user:secret@example.com/v1",
            },
        )


def test_user_decision_test_ownership(
    authenticated_client, normal_user, admin_user, gateway
):
    mine = LLMConfig.objects.create(
        scope="user",
        user=normal_user,
        provider="litellm_proxy",
        model_type="decision",
        config=CONFIG,
    )
    other = LLMConfig.objects.create(
        scope="user",
        user=admin_user,
        provider="litellm_proxy",
        model_type="decision",
        config=CONFIG,
    )
    shared = LLMConfig.objects.create(
        scope="global",
        provider="litellm_proxy",
        model_type="decision",
        config=CONFIG,
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=PAYLOAD)

    gateway(handler)
    url = "/api/v1/user/llm-config/decision-test"
    for row, expected in ((mine, 200), (other, 404), (shared, 404)):
        response = authenticated_client.post(
            url,
            {
                "config_uuid": str(row.uuid),
                "input": "AI",
                "questions": QUESTIONS,
            },
            format="json",
        )
        assert response.status_code == expected
    assert len(calls) == 1
    usage = LLMUsage.objects.get()
    assert usage.user_id == normal_user.pk
    assert usage.metadata["node_name"] == "user_decision_test"
    mine.is_active = False
    mine.save()
    assert (
        authenticated_client.post(
            url,
            {
                "config_uuid": str(mine.uuid),
                "input": "AI",
                "questions": QUESTIONS,
            },
            format="json",
        ).status_code
        == 404
    )
    assert len(calls) == 1


def test_user_can_validate_unsaved_decision_config(
    authenticated_client, gateway
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "answers": [
                    {
                        "name": "is_test",
                        "type": "predicate",
                        "probability": 1.0,
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

    gateway(handler)
    response = authenticated_client.post(
        "/api/v1/user/llm-config/test",
        {
            "model_type": "decision",
            "provider": "litellm_proxy",
            "config": CONFIG,
        },
        format="json",
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert len(calls) == 1
    assert str(calls[0].url).endswith("/v1/decisions")


def test_user_decision_test_requires_login(api_client):
    response = api_client.post(
        "/api/v1/user/llm-config/decision-test",
        {
            "provider": "litellm_proxy",
            "config": CONFIG,
            "input": "AI",
            "questions": QUESTIONS,
        },
        format="json",
    )
    assert response.status_code in (401, 403)
    assert not LLMUsage.objects.exists()


def test_model_type_is_immutable(admin_client, decision_config):
    response = admin_client.put(
        f"{BASE}{decision_config.uuid}/",
        {
            "model_type": "llm",
            "provider": "openai_compatible",
        },
        format="json",
    )
    assert response.status_code == 400
    decision_config.refresh_from_db()
    assert decision_config.model_type == "decision"


@pytest.mark.parametrize(
    "exception, expected",
    [
        (httpx.ReadTimeout(""), "timeout"),
        (httpx.ConnectError(""), "network_error"),
    ],
)
def test_empty_gateway_errors_have_actionable_details(exception, expected):
    from agentcore_metering.adapters.django.services.runtime_config import (
        get_validation_message,
    )

    assert decision_error_detail(exception) == get_validation_message(
        expected, "en"
    )


@pytest.mark.parametrize(
    'provider,model,base,path',
    [
        ('typesafe', 'jev-latest', 'https://api.typesafe.ai', '/v1/systemone'),
        ('cloudflare', 'clef', '', '/ai/run/@cf/cloudflare/clef'),
        ('perplexity', 'pplx-decider-v1-27b', 'https://api.perplexity.ai',
         '/v1/decisions'),
        ('openrouter', 'typesafe/jev-1.13', 'https://openrouter.ai/api',
         '/alpha/decisions'),
        ('strands_decider', 'strands-decider-2B-hobson-v19',
         'https://selfhost.example', '/v1/systemone'),
        ('openai', 'gpt-6-luna', 'https://api.openai.com/v1', '/decisions'),
    ],
)
def test_native_provider_wire_and_metering(provider, model, base, path):
    """Exercise real LiteLLM translation, mocking only its outbound HTTP."""
    config = {
        'model': model, 'api_key': 'native-test-key',
        'api_base': base, 'account_id': 'a' * 32, 'num_retries': 0,
    }
    calls = []

    def post(url, *, json, headers, **kwargs):
        calls.append((url, json))
        assert url.endswith(path)
        assert headers['Authorization'] == 'Bearer native-test-key'
        if provider == 'openai':
            assert json['input'] == 'AI news'
            assert json['questions'] == QUESTIONS
            response = {
                **PAYLOAD,
                'answers': [
                    ANSWERS[0],
                    {**ANSWERS[1], 'confidence': 0.9, 'probabilities': []},
                    {**ANSWERS[2], 'confidence': 0.9, 'probabilities': []},
                ],
            }
        else:
            assert json['state'] == 'AI news'
            assert json['questions']['relevant']['type'] == 'noul'
            assert set(json['questions']) == {
                'relevant', 'category', 'quality',
            }
            assert json['model'] == model
            response = {
                'model': model,
                'answers': {
                    'relevant': {'type': 'noul', 'noul': 0.9},
                    'category': {
                        'type': 'choice', 'choice': 'model',
                        'confidence': 0.9,
                        'probabilities': {'model': 0.9, 'product': 0.1},
                    },
                    'quality': {
                        'type': 'score', 'score': 1.7, 'confidence': 0.9,
                        'probabilities': {'0': 0.05, '1': 0.2, '2': 0.75},
                        'legend': {'0': 'low', '1': 'medium', '2': 'high'},
                    },
                },
                'usage': {'input_tokens': 123, 'output_tokens': 0},
            }
            if provider == 'cloudflare':
                response = {'success': True, 'result': response}
        return httpx.Response(
            200, json=response, request=httpx.Request('POST', url),
        )

    from unittest.mock import Mock
    with patch(
        'litellm.decisions.main._get_httpx_client',
        return_value=Mock(post=post),
    ):
        answers, usage = DecisionTracker.call_with_config(
            provider=provider, config=config, input='AI news',
            questions=QUESTIONS,
        )
    assert len(calls) == 1
    assert answers[0]['probability'] == 0.9
    assert answers[1]['choice'] == 'model'
    assert answers[2]['score'] == 1.7
    assert usage['prompt_tokens'] == 123
    assert usage['completion_tokens'] == 0
    if provider in ('typesafe', 'cloudflare', 'perplexity', 'openrouter'):
        assert usage['cost'] is not None and usage['cost'] > 0
    assert LLMUsage.objects.get().success


@pytest.mark.parametrize('provider', [
    'typesafe', 'cloudflare', 'perplexity', 'openai', 'openrouter',
    'strands_decider',
])
def test_native_config_admin_create(provider, admin_client):
    from agentcore_metering.adapters.django.llm_static.load import (
        get_decision_providers,
    )
    config = get_decision_providers()[provider]
    response = admin_client.post(BASE, {
        'provider': provider, 'model_type': 'decision',
        'config': {
            'model': config['default_model'], 'api_key': 'test-key',
            'api_base': config['default_api_base'] or (
                '' if provider == 'cloudflare' else 'https://selfhost.example'
            ),
            'account_id': 'a' * 32,
        },
    }, format='json')
    assert response.status_code == 201, response.json()
    assert response.json()['model_type'] == 'decision'


def test_catalog_available_to_user_and_admin(
    authenticated_client, admin_client,
):
    mine = authenticated_client.get('/api/v1/user/llm-config/providers')
    admin = admin_client.get(BASE + 'providers/')
    assert mine.status_code == admin.status_code == 200
    assert mine.json() == admin.json()
    providers = mine.json()['providers']
    for pid in (
        'typesafe', 'cloudflare', 'perplexity', 'openai', 'openrouter',
    ):
        assert providers[pid]['decision']['models']
    assert 'decision' not in providers['anthropic']
    models = admin_client.get(BASE + 'models/').json()['providers']
    jev = next(p for p in models if p['id'] == 'typesafe')
    assert jev['model_types'] == ['decision']
    assert jev['models'][0]['mode'] == 'decisions'


def test_cloudflare_config_never_reads_global_credentials(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'a' * 32)
    monkeypatch.setenv('CLOUDFLARE_API_KEY', 'global-secret')
    with pytest.raises(ValueError):
        decision_params('cloudflare', {'model': 'clef'})
    with pytest.raises(ValueError):
        decision_params('cloudflare', {
            'model': 'clef', 'account_id': 'a' * 32,
        })
    with pytest.raises(ValueError):
        decision_params('cloudflare', {
            'model': 'clef', 'api_key': 'test', 'account_id': '../invalid',
        })


def test_strands_supports_explicit_no_auth():
    params = decision_params('strands_decider', {
        'model': 'local-model', 'api_base': 'http://localhost:3000',
    })
    assert params['api_key'] == ''
    assert params['model'] == 'strands_decider/local-model'


def test_sdk_upgrade_preserves_function_tools():
    """Guard the optional-FastAPI regression behind the previous SDK cap."""
    import litellm

    response = litellm.completion(
        model='openai/gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Say OK'}],
        tools=[{
            'type': 'function',
            'function': {
                'name': 'noop', 'description': 'No operation',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }],
        mock_response='OK',
    )
    assert response.choices[0].message.content == 'OK'
