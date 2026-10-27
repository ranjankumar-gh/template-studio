"""Multimodal templates: typed content and the processor boundary (Chapter 7)."""

from __future__ import annotations

import math
from collections.abc import Sequence
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


# Two-stage rendering (Chapter 7): the template emits one placeholder per
# image; the processor expands it into a data-dependent number of tokens.

GEMMA3_IMAGE_TOKENS = 256  # Gemma 3 processor: image_seq_length, every image


def qwen_vl_image_tokens(
    height: int,
    width: int,
    *,
    patch_size: int = 14,
    merge_size: int = 2,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> int:
    """How many <|image_pad|> tokens one image becomes on Qwen2-VL / Qwen2.5-VL
    (patch_size 14) or Qwen3-VL (patch_size 16): the processor resizes each
    side to a multiple of patch_size * merge_size inside the pixel bounds, then
    every merge_size x merge_size block of patches becomes one token."""
    factor = patch_size * merge_size
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return (h // patch_size) * (w // patch_size) // merge_size**2


def count_expanded_placeholders(
    input_ids: Sequence[int], placeholder_id: int
) -> int:
    """Count a media placeholder token in tokenized ids. After the processor
    has run, this is the number of tokens the images actually cost."""
    return sum(1 for token in input_ids if token == placeholder_id)
