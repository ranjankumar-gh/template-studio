"""Rendering a conversation into a model's wire format (Chapters 1, 2, 5, 6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from templatestudio.messages import Conversation, assert_valid

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def load_tokenizer(model_id: str) -> "PreTrainedTokenizerBase":
    """Load a tokenizer along with the chat template it ships with."""
    from transformers import AutoTokenizer  # lazy: only needed for live rendering

    return AutoTokenizer.from_pretrained(model_id)


def render(
    tokenizer: "PreTrainedTokenizerBase",
    conversation: Conversation,
    *,
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **template_kwargs: object,
) -> str:
    """Render a conversation into its model-specific wire format.

    The only sanctioned way to build a prompt: delegate to the model's own template.
    Extra keyword arguments (e.g. enable_thinking) are forwarded to the template.
    """
    return tokenizer.apply_chat_template(
        conversation,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )


def visible_whitespace(text: str) -> str:
    """Render a wire-format string with whitespace made visible, for inspection."""
    return (
        text.replace("\t", "[TAB]")
        .replace(" ", "·")
        .replace("\n", "\\n\n")
    )


def safe_render(
    tokenizer: "PreTrainedTokenizerBase",
    conversation: Conversation,
    *,
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **template_kwargs: object,
) -> str:
    """Validate the input contract, then render. The only call you should ship."""
    assert_valid(conversation)
    return render(
        tokenizer,
        conversation,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **template_kwargs,
    )
