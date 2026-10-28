"""Reasoning and thinking templates (Chapter 6)."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_EMPTY_THINK = re.compile(r"<think>\s*</think>")
_CLOSE = "</think>"


def strip_reasoning(generated: str) -> str:
    """Remove all <think>...</think> blocks from the current generated turn,
    returning the visible answer. This is the display-side strip. Do not
    hand-strip history - that is the template checkpoint's job, and it must
    match native generation spacing."""
    close, opening = generated.find(_CLOSE), generated.find("<think>")
    if close != -1 and (opening == -1 or close < opening):
        # The template opened the block in the generation prompt, so the
        # output starts mid-reasoning and carries only the closing tag.
        generated = generated[close + len(_CLOSE):]
    return _THINK_BLOCK.sub("", generated).lstrip()


def has_empty_think_blocks(rendered: str) -> bool:
    """Detect empty <think></think> blocks in a rendered history. Inside
    an agent loop a kept step must reproduce the reasoning the model
    generated; an empty block there is a mismatch that re-prefills the
    step on every request."""
    return _EMPTY_THINK.search(rendered) is not None


def assistant_message(content: str, reasoning: str | None = None) -> dict:
    """An assistant turn that optionally carries its reasoning in a
    dedicated field, so the template's checkpoint logic can keep or strip
    it - not your concatenation."""
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message
