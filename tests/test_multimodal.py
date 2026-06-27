from __future__ import annotations

from templatestudio.multimodal import (
    count_media_parts,
    image_part,
    text_part,
    user_message,
    validate_content_parts,
)


def test_valid_content_parts_pass():
    content = [text_part("what is this?"), image_part("http://x/y.jpg")]
    assert validate_content_parts(content) == []


def test_unknown_part_type_flagged():
    problems = validate_content_parts([{"type": "hologram", "data": 1}])
    assert any("unknown type" in p for p in problems)


def test_missing_required_field_flagged():
    problems = validate_content_parts([{"type": "image"}])
    assert any("missing 'url'" in p for p in problems)


def test_count_media_parts_counts_only_media():
    convo = [
        user_message([text_part("a"), image_part("u1"), image_part("u2")]),
        {"role": "assistant", "content": "ok"},
    ]
    assert count_media_parts(convo) == 2


def test_user_message_shape():
    msg = user_message([text_part("hi")])
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
