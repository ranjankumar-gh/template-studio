"""The capstone: composing every check into one audit and a CI gate (Chapter 12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from templatestudio.golden import GoldenCase, GoldenFailure, check_goldens
from templatestudio.parity import (
    EngineRenderer,
    ParityReport,
    compare_engines,
    lint_portability,
)
from templatestudio.security import scan_template, template_fingerprint

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class StudioReport:
    portability: list[str] = field(default_factory=list)        # Ch9
    security: list[str] = field(default_factory=list)           # Ch11
    golden_failures: list[GoldenFailure] = field(default_factory=list)   # Ch10
    parity: list[ParityReport] = field(default_factory=list)    # Ch9
    pin_ok: bool = True                                         # Ch11

    @property
    def ok(self) -> bool:
        """The template is clean only if every check passed."""
        return (
            not self.portability
            and not self.security
            and not self.golden_failures
            and all(p.identical for p in self.parity)
            and self.pin_ok
        )


def audit_template(
    template_source: str,
    tokenizer: "PreTrainedTokenizerBase",
    *,
    goldens: list[GoldenCase],
    engines: list[EngineRenderer],
    pinned_fingerprint: str | None = None,
    probe: GoldenCase | None = None,
) -> StudioReport:
    """Run every Template Studio check and collect the findings into one report."""
    report = StudioReport()
    report.portability = lint_portability(template_source)          # Ch9
    report.security = scan_template(template_source)                # Ch11
    report.golden_failures = check_goldens(tokenizer, goldens)      # Ch10

    if pinned_fingerprint is not None:                             # Ch11
        report.pin_ok = template_fingerprint(template_source) == pinned_fingerprint

    if probe is not None and engines:                             # Ch9
        report.parity = compare_engines(
            template_source,
            probe.conversation,
            engines,
            add_generation_prompt=probe.add_generation_prompt,
            **probe.kwargs,
        )
    return report


def ci_gate(report: StudioReport) -> int:
    """Print findings and return a process exit code: 0 clean, 1 failed."""
    if report.ok:
        print("template audit: PASS")
        return 0

    print("template audit: FAIL")
    for finding in report.portability:
        print(f"  [portability] {finding}")
    for finding in report.security:
        print(f"  [security] {finding}")
    if not report.pin_ok:
        print("  [provenance] template fingerprint does not match the pinned value")
    for failure in report.golden_failures:
        print(f"  [golden] {failure.name}: diverged at char {failure.first_divergence}")
    for parity in report.parity:
        if not parity.identical:
            print(
                f"  [parity] {parity.reference} vs {parity.other}: "
                f"diverged at char {parity.first_divergence}"
            )
    return 1
