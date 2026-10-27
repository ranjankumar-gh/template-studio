"""Token ids from the engine itself (Chapter 10): what the model receives
after the server's half of the render - normalize, merge kwargs, render,
tokenize - which a template-string test never runs."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from templatestudio.messages import Conversation, assert_valid

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# (url, json payload) -> parsed json response
Transport = Callable[[str, dict], dict]


def http_post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST a JSON body and return the parsed JSON reply (stdlib only)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class TokenRenderer(Protocol):
    """One engine's answer to: which token ids does this request become?"""

    name: str

    def ids(
        self, conversation: Conversation, **template_kwargs: Any
    ) -> list[int]:
        ...


def _ids(encoded: Any) -> list[int]:
    # Transformers v5 returns a BatchEncoding here; 4.x returns a list.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return [int(i) for i in encoded]


class TransformersTokens:
    """The reference: apply_chat_template with tokenize=True, which tokenizes
    without adding special tokens a second time."""

    name = "transformers"

    def __init__(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        self.tokenizer = tokenizer

    def ids(
        self, conversation: Conversation, **template_kwargs: Any
    ) -> list[int]:
        return _ids(
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
        )


class LlamaCppTokens:
    """llama.cpp's server: /apply-template renders with the GGUF's template,
    /tokenize turns the prompt into ids. add_special and parse_special are on
    because the server's own chat path tokenizes that way - which is how a
    GGUF that adds BOS and a template that writes BOS produce two."""

    name = "llama.cpp"

    def __init__(self, base_url: str, transport: Transport = http_post) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    def ids(
        self, conversation: Conversation, **template_kwargs: Any
    ) -> list[int]:
        body: dict[str, Any] = {"messages": conversation}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        rendered = self.transport(f"{self.base_url}/apply-template", body)
        prompt = rendered["prompt"]
        reply = self.transport(
            f"{self.base_url}/tokenize",
            {"content": prompt, "add_special": True, "parse_special": True},
        )
        return [int(t) for t in reply["tokens"]]


class VLLMTokens:
    """vLLM's /tokenize with messages: the chat path's render and its
    tokenization (add_special_tokens defaults to False there, as in chat)."""

    name = "vllm"

    def __init__(
        self, base_url: str, model: str, transport: Transport = http_post
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport

    def ids(
        self, conversation: Conversation, **template_kwargs: Any
    ) -> list[int]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "add_generation_prompt": True,
        }
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        reply = self.transport(f"{self.base_url}/tokenize", body)
        return [int(t) for t in reply["tokens"]]


@dataclass(frozen=True)
class TokenDivergence:
    reference: str
    engine: str
    position: int          # first token index where the two differ
    reference_len: int
    engine_len: int


def compare_token_renders(
    renderers: list[TokenRenderer],
    conversation: Conversation,
    **template_kwargs: Any,
) -> list[TokenDivergence]:
    """Render one conversation through every engine and report each engine
    whose token ids differ from the first (the reference). Ids are only
    comparable when the engines share a vocabulary - the same model."""
    assert_valid(conversation)
    reference, *others = renderers
    expected = reference.ids(conversation, **template_kwargs)
    divergences: list[TokenDivergence] = []
    for engine in others:
        actual = engine.ids(conversation, **template_kwargs)
        if actual == expected:
            continue
        position = next(
            (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
            min(len(expected), len(actual)),
        )
        divergences.append(
            TokenDivergence(
                reference=reference.name,
                engine=engine.name,
                position=position,
                reference_len=len(expected),
                engine_len=len(actual),
            )
        )
    return divergences
