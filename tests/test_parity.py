from __future__ import annotations

from templatestudio.parity import (
    CPythonJinja,
    _first_divergence,
    compare_engines,
    lint_portability,
)


def test_identical_engines_report_no_divergence(chatml):
    convo = [{"role": "user", "content": "hi"}]
    reports = compare_engines(chatml, convo, [CPythonJinja(), CPythonJinja()])
    assert len(reports) == 1
    assert reports[0].identical is True
    assert reports[0].first_divergence is None


def test_first_divergence_finds_index():
    assert _first_divergence("abc", "abx") == 2
    assert _first_divergence("abc", "abc") is None


def test_lint_flags_risky_filter():
    template = "{{ messages | reject('equalto', x) }}"
    problems = lint_portability(template)
    assert any("reject" in p for p in problems)


def test_lint_passes_plain_template(chatml):
    assert lint_portability(chatml) == []
