"""Shared fixtures. A FakeTokenizer renders via the faithful Jinja env, so the
golden and studio logic is testable offline with no model download."""

from __future__ import annotations

import pytest

from templatestudio.jinja_env import render_template_string

# A minimal ChatML template used across the offline tests.
CHATML = (
    "{% for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


class FakeTokenizer:
    """A stand-in for a Transformers tokenizer that renders a fixed template
    through Template Studio's faithful environment."""

    def __init__(self, template: str = CHATML) -> None:
        self.template = template
        self.chat_template = template

    def apply_chat_template(
        self,
        conversation,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ) -> str:
        return render_template_string(
            self.template,
            conversation,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


@pytest.fixture
def chatml() -> str:
    return CHATML


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
