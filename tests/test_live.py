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
