"""
AGIOne capability inference must not classify non-chat models as chat.

The model pickers (install wizard and admin UI) keep only ``mode == "chat"``
models, so a video/image/embedding id inferred as chat leaks into the picker.
Video brands without an ``-i2v/-t2v`` suffix (ByteDance Seedance/Dreamina,
Alibaba Wan) are the ones that used to slip through.
"""
import pytest

from agentcore_metering.adapters.django.llm_static.load import (
    _agione_model_capabilities,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id,expected_mode",
    [
        ("qwen/wan2.7-r2v/472a8", "image_generation"),
        ("qwen/wan2.7-i2v/3c76f", "image_generation"),
        ("qwen/wan2.7-t2v/06480", "image_generation"),
        ("qwen/wan2.7-videoedit/142f3", "image_generation"),
        ("bytedance/doubao-seedance-2-0-260128/f726e", "image_generation"),
        ("bytedance/dreamina-seedance-2-0-fast-260128/eebda", "image_generation"),
        ("qwen/qwen3.7-text-embedding/fa445", "embedding"),
        ("openai/gpt-image-2/x", "image_generation"),
    ],
)
def test_non_chat_models_are_not_chat(model_id, expected_mode):
    mode, _capabilities = _agione_model_capabilities(model_id)
    assert mode == expected_mode


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id",
    [
        "openai/gpt-5.6-luna/68499",
        "z-ai/glm-5/e577e",
        "deepseek/deepseek-v4-pro/d3462",
        "anthropic/claude-opus-4.6/a6b6f",
        "qwen/qwen3.5-122b-a10b/f6403",
        "gemini/gemini-3.5-flash/a17fd",
    ],
)
def test_chat_models_stay_chat(model_id):
    mode, capabilities = _agione_model_capabilities(model_id)
    assert mode == "chat"
    assert "text-to-text" in capabilities
