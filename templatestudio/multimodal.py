"""Multimodal templates: typed content and the processor boundary (Chapter 7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from templatestudio.messages import Conversation

if TYPE_CHECKING:
    pass  # the processor is a duck-typed object with apply_chat_template


def text_part(text: str) -> dict:
    """A text part of a multimodal content list."""
    return {"type": "text", "text": text}


def image_part(url: str) -> dict:
    """An image part. The url may be a remote URL, a local path, or a data URI."""
    return {"type": "image", "url": url}


def user_message(parts: list[dict]) -> dict:
    """A user turn whose content is an ordered list of typed parts."""
    return {"role": "user", "content": parts}


def render_multimodal(
    processor: object,
    conversation: Conversation,
    *,
    add_generation_prompt: bool = True,
    **processor_kwargs: object,
) -> str:
    """Render a multimodal conversation through the processor-level template."""
    return processor.apply_chat_template(  # type: ignore[attr-defined]
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **processor_kwargs,
    )


_PART_REQUIRED_FIELDS = {"text": "text", "image": "url", "video": "url", "audio": "url"}


def validate_content_parts(content: list[dict]) -> list[str]:
    """Validate a typed content list. Empty list means well-formed."""
    problems: list[str] = []
    for i, part in enumerate(content):
        part_type = part.get("type")
        if part_type not in _PART_REQUIRED_FIELDS:
            problems.append(f"part {i}: unknown type {part_type!r}")
            continue
        required = _PART_REQUIRED_FIELDS[part_type]
        if required not in part:
            problems.append(f"part {i}: {part_type!r} part missing {required!r}")
    return problems


def count_media_parts(conversation: Conversation) -> int:
    """Count media (non-text) parts across a conversation."""
    total = 0
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            total += sum(1 for p in content if p.get("type") != "text")
    return total
