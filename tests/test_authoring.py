from __future__ import annotations

import pytest

from templatestudio.authoring import (
    SUPPORT_TEMPLATE,
    assert_reproduces_training_format,
    install_template,
)


def test_support_template_round_trips_training_format():
    convo = [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "Let me check."},
    ]
    expected = (
        "<|system|>\nYou are a support agent.<|end|>\n"
        "<|user|>\nWhere is my order?<|end|>\n"
        "<|assistant|>\nLet me check.<|end|>\n"
    )
    # add_generation_prompt=False: assert the history reproduces the training format.
    assert_reproduces_training_format(
        SUPPORT_TEMPLATE, convo, expected, add_generation_prompt=False
    )


def test_round_trip_catches_whitespace_drift():
    convo = [{"role": "user", "content": "hi"}]
    wrong = "<|user|>\nhi<|end|>"  # missing trailing newline the template emits
    with pytest.raises(AssertionError):
        assert_reproduces_training_format(
            SUPPORT_TEMPLATE, convo, wrong, add_generation_prompt=False
        )


def test_every_turn_closes_with_the_end_of_turn_marker():
    """A format with no end-of-turn marker trains a model that never stops."""
    from templatestudio.jinja_env import render_template_string

    convo = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    rendered = render_template_string(
        SUPPORT_TEMPLATE, convo, add_generation_prompt=False
    )
    assert rendered.count("<|end|>") == len(convo)
    assert rendered.endswith("hello<|end|>\n")


def test_install_template_sets_attribute():
    class Dummy:
        chat_template = None

    tok = Dummy()
    install_template(tok, SUPPORT_TEMPLATE)
    assert tok.chat_template == SUPPORT_TEMPLATE
