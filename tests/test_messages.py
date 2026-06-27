from __future__ import annotations

import pytest

from templatestudio.messages import (
    InvalidConversation,
    assert_valid,
    validate_conversation,
)


def test_valid_conversation_has_no_problems():
    convo = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    assert validate_conversation(convo) == []


def test_empty_conversation_flagged():
    assert validate_conversation([]) == ["conversation is empty"]


def test_unknown_role_flagged():
    problems = validate_conversation([{"role": "User", "content": "hi"}])
    assert any("unknown role 'User'" in p for p in problems)


def test_none_content_flagged():
    problems = validate_conversation([{"role": "user", "content": None}])
    assert any("content is None" in p for p in problems)


def test_system_must_be_first():
    convo = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
    ]
    problems = validate_conversation(convo)
    assert any("system message must be first" in p for p in problems)


def test_assert_valid_raises_on_bad_conversation():
    with pytest.raises(InvalidConversation):
        assert_valid([{"role": "nope", "content": "x"}])


def test_assert_valid_passes_on_good_conversation():
    assert_valid([{"role": "user", "content": "hi"}])  # does not raise


def test_malformed_multimodal_part_flagged():
    convo = [{"role": "user", "content": [{"type": "image"}]}]
    problems = validate_conversation(convo)
    assert any("missing 'url'" in p for p in problems)


def test_valid_multimodal_content_has_no_problems():
    convo = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "url": "http://example.com/cat.png"},
            ],
        }
    ]
    assert validate_conversation(convo) == []
