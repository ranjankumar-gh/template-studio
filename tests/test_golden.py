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
