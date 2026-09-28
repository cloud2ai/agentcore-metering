"""
AGIOne dynamic model labels use the vendor's official casing.

Ids look like ``<family>/<name>/<hash>``; the label drops the family and the
trailing hash and cases the name correctly (GPT/GLM/DeepSeek/MiniMax/...),
instead of the naive title-case that produced "Openai/Gpt 5.4 Nano".
"""
import pytest

from agentcore_metering.adapters.django.llm_static.load import (
    _agione_model_label,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("openai/gpt-5.4-nano/d688d", "GPT-5.4 Nano"),
        ("openai/gpt-6-astra/adc3e", "GPT-6 Astra"),
        ("openai/gpt-5.3-codex/b687c", "GPT-5.3 Codex"),
        ("openai/GPT-5.5/c6fbe", "GPT-5.5"),
        ("z-ai/glm-5/e577e", "GLM-5"),
        ("z-ai/glm-5.3/08ff4", "GLM-5.3"),
        ("deepseek/deepseek-v4-pro/d3462", "DeepSeek V4 Pro"),
        ("minimax/minimax-m2.7/2f893", "MiniMax M2.7"),
        ("moonshotai/kimi-k2.5/aaa1e", "Kimi K2.5"),
        ("qwen/qwen3.5-122b-a10b/f6403", "Qwen3.5 122B A10B"),
        ("qwen/qwen3.8-max/4ad01", "Qwen3.8 Max"),
        ("anthropic/claude-opus-4.6/a6b6f", "Claude Opus 4.6"),
        ("anthropic/claude-haiku-4.5/27a29", "Claude Haiku 4.5"),
        ("gemini/gemini-3.5-flash/a17fd", "Gemini 3.5 Flash"),
        ("gemini/Gemini-3.1-Pro-Preview/b3a6b", "Gemini 3.1 Pro Preview"),
        ("qwen/wan2.7-i2v/3c76f", "Wan2.7 I2V"),
    ],
)
def test_official_casing(model_id, expected):
    assert _agione_model_label(model_id) == expected


@pytest.mark.unit
def test_trailing_hash_is_stripped_even_with_non_hex_letters():
    # The version hash is not always hex ("0000n", "0000g").
    assert _agione_model_label("deepseek/deepseek-v3.2/0000n") == "DeepSeek V3.2"
    assert _agione_model_label("deepseek/deepseek-r1-0528/0000g") == (
        "DeepSeek R1 0528"
    )


@pytest.mark.unit
def test_family_prefix_is_dropped():
    label = _agione_model_label("openai/gpt-5.4-nano/d688d")
    assert "/" not in label
    assert "Openai" not in label
