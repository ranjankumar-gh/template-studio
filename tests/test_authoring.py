from __future__ import annotations

import pytest

from templatestudio.authoring import (
    SUPPORT_TEMPLATE,
    assert_reproduces_training_format,
    install_template,
)


def test_support_template_round_trips_training_format():
    convo = [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "Let me check."},
    ]
    expected = (
        "<|system|>\nYou are a support agent.<|end|>\n"
        "<|user|>\nWhere is my order?<|end|>\n"
        "<|assistant|>\nLet me check.<|end|>\n"
    )
    # add_generation_prompt=False: assert the history reproduces the training format.
    assert_reproduces_training_format(
        SUPPORT_TEMPLATE, convo, expected, add_generation_prompt=False
    )


def test_round_trip_catches_whitespace_drift():
    convo = [{"role": "user", "content": "hi"}]
    wrong = "<|user|>\nhi<|end|>"  # missing trailing newline the template emits
    with pytest.raises(AssertionError):
        assert_reproduces_training_format(
            SUPPORT_TEMPLATE, convo, wrong, add_generation_prompt=False
        )


def test_every_turn_closes_with_the_end_of_turn_marker():
    """A format with no end-of-turn marker trains a model that never stops."""
    from templatestudio.jinja_env import render_template_string

    convo = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    rendered = render_template_string(
        SUPPORT_TEMPLATE, convo, add_generation_prompt=False
    )
    assert rendered.count("<|end|>") == len(convo)
    assert rendered.endswith("hello<|end|>\n")


def test_install_template_sets_attribute():
    class Dummy:
        chat_template = None

    tok = Dummy()
    install_template(tok, SUPPORT_TEMPLATE)
    assert tok.chat_template == SUPPORT_TEMPLATE


# --- token-level checks (offline, with a character-level fake tokenizer) ---

from templatestudio.authoring import (  # noqa: E402
    assert_token_roundtrip,
    check_assistant_mask,
    check_marker_tokens,
    check_stop_alignment,
)
from templatestudio.jinja_env import render_template_string  # noqa: E402

MARKERS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class CharTokenizer:
    """One id per character, except registered markers, which are one id each."""

    def __init__(self, specials: list[str], pad: str | None = None) -> None:
        self.specials = {s: 1000 + i for i, s in enumerate(specials)}
        self.pad_token_id = self.specials.get(pad) if pad else None
        self.chat_template = SUPPORT_TEMPLATE

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            hit = next((s for s in self.specials if text.startswith(s, i)), None)
            if hit:
                ids.append(self.specials[hit])
                i += len(hit)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        names = {v: k for k, v in self.specials.items()}
        return "".join(names.get(i, chr(i)) for i in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.decode([i]) for i in ids]

    def apply_chat_template(self, conversation, *, tokenize=False,
                            add_generation_prompt=False, **_):
        text = render_template_string(
            self.chat_template, conversation,
            add_generation_prompt=add_generation_prompt,
        )
        return {"input_ids": self.encode(text)} if tokenize else text


CONVO = [
    {"role": "user", "content": "Where is my order?"},
    {"role": "assistant", "content": "Let me check."},
]


def test_markers_that_are_not_vocabulary_are_reported():
    assert set(check_marker_tokens(CharTokenizer([]), MARKERS)) == set(MARKERS)
    assert check_marker_tokens(CharTokenizer(MARKERS), MARKERS) == {}


def test_token_roundtrip_passes_on_the_same_ids():
    tok = CharTokenizer(MARKERS)
    trained = tok.encode(render_template_string(
        SUPPORT_TEMPLATE, CONVO, add_generation_prompt=False))
    assert_token_roundtrip(tok, CONVO, trained)


def test_token_roundtrip_catches_a_doubled_bos():
    tok = CharTokenizer(MARKERS + ["<s>"])
    served = tok.encode(render_template_string(
        SUPPORT_TEMPLATE, CONVO, add_generation_prompt=False))
    with pytest.raises(AssertionError, match="position 0"):
        assert_token_roundtrip(tok, CONVO, [tok.specials["<s>"]] + served)


def test_stop_alignment_finds_the_missing_end_of_turn_id():
    tok = CharTokenizer(MARKERS)
    eot = tok.specials["<|end|>"]
    assert check_stop_alignment(tok, {"eos_token_id": [7, 8]})
    assert check_stop_alignment(tok, {"eos_token_id": [7, eot]}) == []
    assert check_stop_alignment(tok, {"eos_token_id": eot}) == []


def test_stop_alignment_flags_pad_equal_to_end_of_turn():
    tok = CharTokenizer(MARKERS, pad="<|end|>")
    problems = check_stop_alignment(tok, {"eos_token_id": tok.specials["<|end|>"]})
    assert any("pad_token_id" in p for p in problems)


def test_stop_alignment_rejects_a_marker_that_is_text():
    problems = check_stop_alignment(CharTokenizer([]), {"eos_token_id": 2})
    assert "cannot be a stop id" in problems[0]


class MaskTokenizer(CharTokenizer):
    """Returns a prepared mask, to test the span logic on its own."""

    def __init__(self, mask_for) -> None:
        super().__init__(MARKERS)
        self.mask_for = mask_for

    def apply_chat_template(self, conversation, **_):
        ids = self.encode(render_template_string(
            SUPPORT_TEMPLATE, conversation, add_generation_prompt=False))
        return {"input_ids": ids, "assistant_masks": self.mask_for(self, ids)}


def _span_mask(tok, ids, text):
    target = tok.encode(text)
    mask = [0] * len(ids)
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            mask[i:i + len(target)] = [1] * len(target)
    return mask


def test_mask_check_passes_when_span_is_content_plus_end_of_turn():
    tok = MaskTokenizer(lambda t, ids: _span_mask(t, ids, "Let me check.<|end|>"))
    assert check_assistant_mask(tok, CONVO) == []


def test_mask_check_reports_all_zeros():
    tok = MaskTokenizer(lambda t, ids: [0] * len(ids))
    assert "all zeros" in check_assistant_mask(tok, CONVO)[0]


def test_mask_check_reports_a_span_missing_the_end_of_turn():
    tok = MaskTokenizer(lambda t, ids: _span_mask(t, ids, "Let me check."))
    assert "expected" in check_assistant_mask(tok, CONVO)[0]


def test_mask_check_reports_a_span_that_includes_the_header():
    tok = MaskTokenizer(
        lambda t, ids: _span_mask(t, ids, "<|assistant|>\nLet me check.<|end|>")
    )
    assert "expected" in check_assistant_mask(tok, CONVO)[0]
