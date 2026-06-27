"""Reasoning and thinking templates (Chapter 6)."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_EMPTY_THINK = re.compile(r"<think>\s*</think>")


def strip_reasoning(generated: str) -> str:
    """Remove all <think>...</think> blocks from the current generated turn, returning
    the visible answer. This is the display-side strip. Do not hand-strip history - that
    is the template checkpoint's job, and it must match native generation spacing (Ch6)."""
    return _THINK_BLOCK.sub("", generated).lstrip()


def has_empty_think_blocks(rendered: str) -> bool:
    """Detect empty historical <think></think> blocks that invalidate the KV cache."""
    return _EMPTY_THINK.search(rendered) is not None


def assistant_message(content: str, reasoning: str | None = None) -> dict:
    """An assistant turn that optionally carries its reasoning in a dedicated field,
    so the template's checkpoint logic can keep or strip it."""
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message
