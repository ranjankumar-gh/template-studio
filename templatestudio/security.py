"""Templates as an attack surface: provenance, scanning, injection (Chapter 11)."""

from __future__ import annotations

import hashlib
import re


def template_fingerprint(template_source: str) -> str:
    """A stable fingerprint of a template's source. Record the fingerprint of the
    template you reviewed; a changed fingerprint means re-review before trusting."""
    return hashlib.sha256(template_source.encode("utf-8")).hexdigest()


def assert_pinned(template_source: str, expected_fingerprint: str) -> None:
    """Fail if a template's source no longer matches the reviewed-and-pinned version."""
    actual = template_fingerprint(template_source)
    if actual != expected_fingerprint:
        raise ValueError(
            f"template fingerprint changed: expected {expected_fingerprint[:12]}..., "
            f"got {actual[:12]}... - re-review before trusting"
        )


_SUSPICIOUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"https?://"), "hardcoded URL in template source"),
    (re.compile(r"\{\{\s*['\"][^'\"]{40,}", re.S), "long literal string injected into output"),
    (
        re.compile(r"\{%[^%]*\bif\b[^%]*\bin\b[^%]*\bcontent\b", re.I),
        "branch on message content - possible trigger condition",
    ),
]


def scan_template(template_source: str) -> list[str]:
    """Flag backdoor-shaped constructs in a template's source. A hit means review,
    not proof of malice; a clean scan is not proof of safety."""
    findings: list[str] = []
    for pattern, description in _SUSPICIOUS_PATTERNS:
        if pattern.search(template_source):
            findings.append(description)
    return findings


# Turn/control markers across the families from Chapter 4.
_CONTROL_MARKERS = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|start_header_id\|>|<\|eot_id\|>"
    r"|<start_of_turn>|<end_of_turn>"
)


def find_control_tokens(content: str) -> list[str]:
    """Find control/turn markers embedded in message content - a structure-forging
    injection attempt. Catch these before the template renders them as real tokens."""
    return _CONTROL_MARKERS.findall(content)
