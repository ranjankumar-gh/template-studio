"""Reasoning and thinking templates (Chapter 6)."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_EMPTY_THINK = re.compile(r"<think>\s*</think>")
_CLOSE = "</think>"


def strip_reasoning(generated: str, prompt_opened: bool = False) -> str:
    """Remove all <think>...</think> blocks from the current generated turn,
    returning the visible answer. This is the display-side strip. Do not
    hand-strip history - that is the template checkpoint's job, and it must
    match native generation spacing. Pass prompt_opened=True when the
    generation prompt ends inside an open <think>, as Qwen3.5's does with
    thinking on: output with no </think> was then cut off mid-reasoning."""
    close, opening = generated.find(_CLOSE), generated.find("<think>")
    if prompt_opened or (close != -1 and (opening == -1 or close < opening)):
        # The output starts mid-reasoning and carries only the closing tag.
        if close == -1:
            return ""  # truncated before the reasoning ended: no answer yet
        generated = generated[close + len(_CLOSE):]
    visible = _THINK_BLOCK.sub("", generated)
    unclosed = visible.find("<think>")
    if unclosed != -1:
        visible = visible[:unclosed]  # opened, never closed: reasoning to the end
    return visible.lstrip()


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
