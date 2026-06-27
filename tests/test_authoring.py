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
        "<|system|>\nYou are a support agent.\n"
        "<|user|>\nWhere is my order?\n"
        "<|assistant|>\nLet me check.\n"
    )
    # add_generation_prompt=False: assert the history reproduces the training format.
    assert_reproduces_training_format(
        SUPPORT_TEMPLATE, convo, expected, add_generation_prompt=False
    )


def test_round_trip_catches_whitespace_drift():
    convo = [{"role": "user", "content": "hi"}]
    wrong = "<|user|>\nhi"  # missing trailing newline the template emits
    with pytest.raises(AssertionError):
        assert_reproduces_training_format(
            SUPPORT_TEMPLATE, convo, wrong, add_generation_prompt=False
        )


def test_install_template_sets_attribute():
    class Dummy:
        chat_template = None

    tok = Dummy()
    install_template(tok, SUPPORT_TEMPLATE)
    assert tok.chat_template == SUPPORT_TEMPLATE
