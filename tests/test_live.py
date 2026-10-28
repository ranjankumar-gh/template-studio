"""Live smoke test against a real tokenizer. Skipped automatically when transformers
is missing or the model cannot be downloaded (offline CI)."""

from __future__ import annotations

import pytest

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    pytest.importorskip("transformers")
    from templatestudio.render import load_tokenizer

    try:
        return load_tokenizer(MODEL)
    except Exception as exc:  # network/download/auth failure -> skip, don't fail
        pytest.skip(f"could not load {MODEL}: {exc}")


def test_describe_template_on_real_model(qwen_tokenizer):
    from templatestudio.anatomy import describe_template

    report = describe_template(qwen_tokenizer)
    assert report.has_system_role is True  # Qwen3 has a system role
    assert "assistant" in report.generation_prompt


def test_safe_render_round_trips(qwen_tokenizer):
    from templatestudio.render import safe_render

    convo = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    out = safe_render(qwen_tokenizer, convo)
    assert "<|im_start|>" in out
    assert out.rstrip().endswith("assistant") or out.endswith("assistant\n")


def test_authoring_checks_on_a_real_tokenizer():
    """Chapter 8 end to end: the support markers are text to Qwen3 until they are
    added, and once added the mask covers exactly content plus <|end|>."""
    pytest.importorskip("transformers")
    from templatestudio.authoring import (
        SUPPORT_TEMPLATE,
        check_assistant_mask,
        check_marker_tokens,
        check_stop_alignment,
        install_template,
    )
    from templatestudio.render import load_tokenizer

    try:
        tok = load_tokenizer(MODEL)  # a fresh copy: this test adds tokens
    except Exception as exc:
        pytest.skip(f"could not load {MODEL}: {exc}")
    markers = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    install_template(tok, SUPPORT_TEMPLATE)

    assert set(check_marker_tokens(tok, markers)) == set(markers)  # all split
    tok.add_tokens(markers, special_tokens=True)
    assert check_marker_tokens(tok, markers) == {}

    convo = [
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "Let me check."},
    ]
    assert check_assistant_mask(tok, convo) == []

    end_id = tok.convert_tokens_to_ids("<|end|>")
    assert check_stop_alignment(tok, {"eos_token_id": [151645, 151643]})
    assert check_stop_alignment(tok, {"eos_token_id": [151645, end_id]}) == []


def test_llama_cpp_matches_transformers_token_for_token(qwen_tokenizer):
    """Chapter 10's live comparison. Opt in by pointing TSH_LLAMA_CPP_URL at a
    llama.cpp server running Qwen/Qwen3-0.6B-GGUF."""
    import os

    url = os.environ.get("TSH_LLAMA_CPP_URL")
    if not url:
        pytest.skip("set TSH_LLAMA_CPP_URL to a llama.cpp server serving Qwen3-0.6B")
    from templatestudio.prefix import assert_prefix_stable
    from templatestudio.serving import (
        LlamaCppTokens,
        TransformersTokens,
        compare_token_renders,
    )

    engines = [TransformersTokens(qwen_tokenizer), LlamaCppTokens(url)]
    convo = [
        {"role": "system", "content": "You are a terse support agent."},
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "What is the order id?"},
        {"role": "user", "content": "#4471"},
    ]
    assert compare_token_renders(engines, convo) == []
    assert compare_token_renders(engines, convo, enable_thinking=False) == []
    assert_prefix_stable(engines[1].ids, convo)
