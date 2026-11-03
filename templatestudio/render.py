"""Rendering a conversation into a model's wire format (Chapter 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

Message = dict[str, str]
Conversation = list[Message]


def load_tokenizer(model_id: str) -> "PreTrainedTokenizerBase":
    """Load a tokenizer along with the chat template it ships with."""
    from transformers import AutoTokenizer  # lazy: only needed for live rendering

    return AutoTokenizer.from_pretrained(model_id)


def render(
    tokenizer: "PreTrainedTokenizerBase",
    conversation: Conversation,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render a conversation into its model-specific wire format.

    The only sanctioned way to build a prompt: delegate to the model's own template.
    """
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def visible_whitespace(text: str) -> str:
    """Render a wire-format string with whitespace made visible, for inspection.

    Newlines become a marker plus a real break; tabs and spaces become glyphs.
    Use this whenever a template 'looks right' but behaves wrong - the bug is
    almost always in characters you cannot see.
    """
    return (
        text.replace("\t", "[TAB]")
        .replace(" ", "·")
        .replace("\n", "\\n\n")
    )

