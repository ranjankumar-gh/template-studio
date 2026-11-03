from __future__ import annotations

import pytest

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


BRANCHY = (
    "{% if 'launch' in messages[-1]['content'] %}x{% endif %}"
    "{{ messages[0]['content'] }}"
)
FINDING = "branch on message content - possible trigger condition"


def test_a_reviewed_finding_passes_for_exactly_that_template(fake_tokenizer):
    reviewed = {template_fingerprint(BRANCHY): [FINDING]}
    report = audit_template(BRANCHY, fake_tokenizer, goldens=[], engines=[],
                            reviewed=reviewed)
    assert report.ok is True and report.reviewed == [FINDING]
    assert ci_gate(report) == 0


def test_the_review_does_not_survive_a_template_change(fake_tokenizer):
    reviewed = {template_fingerprint(BRANCHY): [FINDING]}
    changed = BRANCHY.replace("launch", "detonate")
    report = audit_template(changed, fake_tokenizer, goldens=[], engines=[],
                            reviewed=reviewed)
    assert report.ok is False and report.security == [FINDING]


def test_stock_qwen3_passes_the_gate_once_its_findings_are_reviewed():
    """Chapter 13's exercise on a real model: unreviewed, Qwen3's legitimate
    template trips the scanner; reviewed against its fingerprint, it passes."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen/Qwen3-0.6B is not in the local Hugging Face cache")
    source = tok.chat_template
    unreviewed = audit_template(source, tok, goldens=[], engines=[])
    assert unreviewed.ok is False
    reviewed = {template_fingerprint(source): list(unreviewed.security)}
    report = audit_template(source, tok, goldens=[], engines=[], reviewed=reviewed,
                            pinned_fingerprint=template_fingerprint(source))
    assert report.ok is True
