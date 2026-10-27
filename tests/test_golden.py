from __future__ import annotations

from dataclasses import replace

from templatestudio.golden import (
    GoldenCase,
    check_goldens,
    record_goldens,
    render_case,
)


def _cases(tokenizer) -> list[GoldenCase]:
    convo = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    expected = render_case(tokenizer, GoldenCase("seed", convo, expected=""))
    return [GoldenCase("plain", convo, expected=expected)]


def test_recorded_goldens_pass_check(fake_tokenizer):
    convo = [{"role": "user", "content": "hi"}]
    recorded = record_goldens(
        fake_tokenizer, [GoldenCase("a", convo, expected="")]
    )
    cases = [GoldenCase("a", convo, expected=recorded["a"])]
    assert check_goldens(fake_tokenizer, cases) == []


def test_changed_output_fails_with_divergence(fake_tokenizer):
    convo = [{"role": "user", "content": "hi"}]
    good = record_goldens(fake_tokenizer, [GoldenCase("a", convo, expected="")])["a"]
    tampered = GoldenCase("a", convo, expected=good.replace("user", "uSer"))
    failures = check_goldens(fake_tokenizer, [tampered])
    assert len(failures) == 1
    assert failures[0].name == "a"
    assert failures[0].first_divergence >= 0


def test_replace_keeps_case_frozen(fake_tokenizer):
    convo = [{"role": "user", "content": "hi"}]
    case = GoldenCase("a", convo, expected="x")
    updated = replace(case, expected="y")
    assert case.expected == "x" and updated.expected == "y"


# --- token goldens recorded from a serving engine ---

from templatestudio.golden import record_golden_ids  # noqa: E402


class FakeEngine:
    """Returns fixed ids per conversation; `extra` simulates a server that
    adds one more token (a second BOS) in front."""

    name = "fake"

    def __init__(self, extra: list[int] | None = None) -> None:
        self.extra = extra or []

    def ids(self, conversation, **kwargs):
        text = "|".join(m["content"] for m in conversation)
        flag = [9] if kwargs.get("enable_thinking") is False else []
        return self.extra + [1] + flag + [ord(c) for c in text]


def _token_case(fake_tokenizer, engine):
    convo = [{"role": "user", "content": "hi"}]
    base = GoldenCase("t", convo, expected="")
    string = record_goldens(fake_tokenizer, [base])["t"]
    ids = record_golden_ids(engine, [base])["t"]
    return replace(base, expected=string, expected_ids=ids, engine=engine.name)


def test_token_golden_passes_on_the_same_engine(fake_tokenizer):
    case = _token_case(fake_tokenizer, FakeEngine())
    assert check_goldens(fake_tokenizer, [case], engine=FakeEngine()) == []


def test_token_golden_catches_what_the_string_cannot(fake_tokenizer):
    case = _token_case(fake_tokenizer, FakeEngine())
    (failure,) = check_goldens(fake_tokenizer, [case], engine=FakeEngine(extra=[1]))
    assert failure.unit == "token" and failure.first_divergence == 1
    # the string golden alone passes: the template output did not change
    assert check_goldens(fake_tokenizer, [case]) == []


def test_string_failures_still_report_chars(fake_tokenizer):
    case = replace(_token_case(fake_tokenizer, FakeEngine()), expected="other")
    (failure,) = check_goldens(fake_tokenizer, [case])
    assert failure.unit == "char"
