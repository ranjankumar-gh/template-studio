"""Authoring and shipping a template for your own model (Chapter 8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from templatestudio.jinja_env import render_template_string
from templatestudio.messages import Conversation

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Derived from a training format - a transcription, not a guess.
SUPPORT_TEMPLATE = (
    "{% for message in messages %}"
    "{{- '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|assistant|>\n' }}{%- endif %}"
)


def install_template(
    tokenizer: "PreTrainedTokenizerBase", template_source: str
) -> "PreTrainedTokenizerBase":
    """Install a chat template on a tokenizer. After this, apply_chat_template uses
    it, and save_pretrained writes it to chat_template.jinja in the tokenizer dir."""
    tokenizer.chat_template = template_source
    return tokenizer


def assert_reproduces_training_format(
    template_source: str,
    conversation: Conversation,
    expected_wire_format: str,
    *,
    add_generation_prompt: bool = False,
) -> None:
    """Render `conversation` through the candidate template and assert it equals the
    exact string the model trained on. The core test of the training-format contract."""
    rendered = render_template_string(
        template_source,
        conversation,
        add_generation_prompt=add_generation_prompt,
    )
    if rendered != expected_wire_format:
        raise AssertionError(
            "template does not reproduce the training format.\n"
            f"expected:\n{expected_wire_format!r}\n"
            f"got:\n{rendered!r}"
        )
