from __future__ import annotations

from templatestudio.golden import GoldenCase, record_goldens
from templatestudio.parity import CPythonJinja
from templatestudio.security import template_fingerprint
from templatestudio.studio import audit_template, ci_gate


def _goldens(tokenizer) -> list[GoldenCase]:
    convo = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    recorded = record_goldens(tokenizer, [GoldenCase("plain", convo, expected="")])
    return [GoldenCase("plain", convo, expected=recorded["plain"])]


def test_clean_template_passes_audit_and_gate(chatml, fake_tokenizer):
    goldens = _goldens(fake_tokenizer)
    report = audit_template(
        chatml,
        fake_tokenizer,
        goldens=goldens,
        engines=[CPythonJinja()],
        pinned_fingerprint=template_fingerprint(chatml),
        probe=goldens[0],
    )
    assert report.ok is True
    assert ci_gate(report) == 0


def test_risky_filter_and_pin_mismatch_fail_audit(chatml, fake_tokenizer):
    goldens = _goldens(fake_tokenizer)
    dirty = "{{ messages | reject('equalto', x) }}"
    report = audit_template(
        dirty,
        fake_tokenizer,
        goldens=goldens,
        engines=[CPythonJinja()],
        pinned_fingerprint=template_fingerprint(chatml),  # does not match `dirty`
        probe=goldens[0],
    )
    assert report.ok is False
    assert report.portability  # reject flagged
    assert report.pin_ok is False
    assert ci_gate(report) == 1
