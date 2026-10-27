"""Authoring and training through a template for your own model (Chapter 8)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from templatestudio.jinja_env import render_template_string
from templatestudio.messages import Conversation

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

END_OF_TURN = "<|end|>"

# Derived from a training format - a transcription, not a guess. The assistant
# body and its end-of-turn marker sit inside {% generation %}: that is what the
# model produces, so it is what the loss covers. The newline after the marker
# is template-inserted, never generated, so it stays outside.
SUPPORT_TEMPLATE = (
    "{% for message in messages %}"
    "{{- '<|' + message['role'] + '|>\n' }}"
    "{%- if message['role'] == 'assistant' %}"
    "{% generation %}{{- message['content'] + '<|end|>' }}{% endgeneration %}"
    "{{- '\n' }}"
    "{%- else %}"
    "{{- message['content'] + '<|end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|assistant|>\n' }}{%- endif %}"
)


def install_template(
    tokenizer: "PreTrainedTokenizerBase", template_source: str
) -> "PreTrainedTokenizerBase":
    """Install a chat template on a tokenizer. After this,
    apply_chat_template uses it, and save_pretrained writes it to
    chat_template.jinja in the tokenizer dir."""
    tokenizer.chat_template = template_source
    return tokenizer


def assert_reproduces_training_format(
    template_source: str,
    conversation: Conversation,
    expected_wire_format: str,
    *,
    add_generation_prompt: bool = False,
) -> None:
    """Render `conversation` through the candidate template and assert it
    equals the exact string the model trained on. The string-level check of
    the training-format contract."""
    rendered = render_template_string(
        template_source,
        conversation,
        add_generation_prompt=add_generation_prompt,
    )
    if rendered != expected_wire_format:
        raise AssertionError(
            "template does not reproduce the training format.\n"
            f"expected:\n{expected_wire_format!r}\n"
            f"got:\n{rendered!r}"
        )


def _ids(encoded: Any) -> list[int]:
    # Transformers v5 returns a BatchEncoding here; 4.x returns a list.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return [int(i) for i in encoded]


def assert_token_roundtrip(
    tokenizer: "PreTrainedTokenizerBase",
    conversation: Conversation,
    trainer_input_ids: Sequence[int],
    *,
    add_generation_prompt: bool = False,
) -> None:
    """Assert the served template tokenizes `conversation` to exactly the ids
    the trainer fed the model. The token-level check of the contract: it sees
    what a string comparison cannot - a doubled BOS, a marker in pieces."""
    served = _ids(
        tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )
    trained = [int(i) for i in trainer_input_ids]
    if served == trained:
        return
    at = next(
        (i for i, (s, t) in enumerate(zip(served, trained)) if s != t),
        min(len(served), len(trained)),
    )
    window = slice(max(at - 3, 0), at + 3)
    raise AssertionError(
        f"token streams diverge at position {at} "
        f"(served {len(served)} ids, trained {len(trained)}).\n"
        f"served:  {tokenizer.convert_ids_to_tokens(served[window])}\n"
        f"trained: {tokenizer.convert_ids_to_tokens(trained[window])}"
    )


def check_marker_tokens(
    tokenizer: "PreTrainedTokenizerBase", markers: Iterable[str]
) -> dict[str, list[int]]:
    """Return every marker that does not encode to exactly one token id, with
    the ids it does encode to. Empty means every marker is real vocabulary."""
    split: dict[str, list[int]] = {}
    for marker in markers:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1:
            split[marker] = list(ids)
    return split


def _stop_ids(generation_config: Any) -> set[int]:
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None and isinstance(generation_config, Mapping):
        eos = generation_config.get("eos_token_id")
    if eos is None:
        return set()
    return {int(eos)} if isinstance(eos, int) else {int(i) for i in eos}


def check_stop_alignment(
    tokenizer: "PreTrainedTokenizerBase",
    generation_config: Any,
    end_of_turn: str = END_OF_TURN,
) -> list[str]:
    """List the ways the end-of-turn marker fails to stop generation. Empty
    means the engine stops where the model learned to stop."""
    ids = tokenizer.encode(end_of_turn, add_special_tokens=False)
    if len(ids) != 1:
        return [f"{end_of_turn!r} is {len(ids)} tokens; it cannot be a stop id"]
    eot = ids[0]
    problems: list[str] = []
    stops = _stop_ids(generation_config)
    if eot not in stops:
        problems.append(
            f"{end_of_turn!r} (id {eot}) is not in eos_token_id {sorted(stops)}"
        )
    if getattr(tokenizer, "pad_token_id", None) == eot:
        problems.append(
            f"pad_token_id is {end_of_turn!r}: a collator that masks padding "
            "masks the stop token out of the loss"
        )
    return problems


def check_assistant_mask(
    tokenizer: "PreTrainedTokenizerBase",
    conversation: Conversation,
    end_of_turn: str = END_OF_TURN,
) -> list[str]:
    """Check that the assistant-token mask covers each assistant turn, ending on
    the end-of-turn marker, and nothing else. Empty means the loss will cover
    exactly what the model is supposed to learn to produce."""
    out = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = list(out["input_ids"]), list(out["assistant_masks"])
    if not any(mask):
        return ["assistant mask is all zeros: no {% generation %} block"]
    runs: list[list[int]] = []
    current: list[int] = []
    for token, flag in zip(ids, mask):
        if flag:
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    turns = [m["content"] for m in conversation if m["role"] == "assistant"]
    if len(runs) != len(turns):
        return [f"mask has {len(runs)} spans for {len(turns)} assistant turns"]
    problems: list[str] = []
    for i, (run, content) in enumerate(zip(runs, turns)):
        span = tokenizer.decode(run).strip()
        if span != content + end_of_turn:
            problems.append(
                f"span {i} is {span!r}, expected {content + end_of_turn!r}"
            )
    return problems
