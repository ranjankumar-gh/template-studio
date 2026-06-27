from __future__ import annotations

import pytest
from jinja2.exceptions import TemplateError

from templatestudio.jinja_env import build_template_env, render_template_string


def test_renders_chatml_without_stray_whitespace(chatml):
    convo = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    out = render_template_string(chatml, convo)
    expected = (
        "<|im_start|>system\nbe terse<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert out == expected


def test_no_generation_prompt_when_disabled(chatml):
    convo = [{"role": "user", "content": "hi"}]
    out = render_template_string(chatml, convo, add_generation_prompt=False)
    assert out == "<|im_start|>user\nhi<|im_end|>\n"


def test_raise_exception_global_is_available():
    env = build_template_env()
    template = env.from_string("{{ raise_exception('boom') }}")
    with pytest.raises(TemplateError):
        template.render()


def test_tojson_filter_does_not_escape_html():
    env = build_template_env()
    template = env.from_string("{{ value | tojson }}")
    assert template.render(value={"a": "<b>"}) == '{"a": "<b>"}'
