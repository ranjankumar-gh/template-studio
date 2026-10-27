from __future__ import annotations

import templatestudio.parity as parity
from templatestudio.parity import (
    CPythonJinja,
    compare_engines,
    first_divergence,
    lint_portability,
)


def test_identical_engines_report_no_divergence(chatml):
    convo = [{"role": "user", "content": "hi"}]
    reports = compare_engines(chatml, convo, [CPythonJinja(), CPythonJinja()])
    assert len(reports) == 1
    assert reports[0].identical is True
    assert reports[0].first_divergence is None


def test_first_divergence_finds_index():
    assert first_divergence("abc", "abx") == 2
    assert first_divergence("abc", "abc") is None


def test_first_divergence_is_public_and_the_private_name_is_gone():
    """A helper that crosses a module boundary - golden.py imports it - stops
    being private. No alias: a leading underscore means no external dependant."""
    assert not hasattr(parity, "_first_divergence")


def test_lint_flags_risky_filter():
    template = "{{ messages | reject('equalto', x) }}"
    problems = lint_portability(template)
    assert any("reject" in p for p in problems)


def test_lint_message_is_not_keyed_to_a_named_engine():
    """A portability heuristic keyed to one engine's name goes stale the day
    that engine changes; ch09 now prints the engine-neutral wording."""
    template = "{{ messages | selectattr('role', 'equalto', 'user') }}"
    problems = lint_portability(template)
    assert problems == [
        "filter 'selectattr' is a known cross-engine risk "
        "(verify on the engine you deploy)"
    ]
    assert "minja" not in problems[0]


def test_lint_passes_plain_template(chatml):
    assert lint_portability(chatml) == []
