"""The cross-engine problem: the parity gap and a diff harness (Chapter 9)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from templatestudio.jinja_env import render_template_string
from templatestudio.messages import Conversation


class EngineRenderer(Protocol):
    """One adapter per engine. render() returns the wire format that engine produces."""

    name: str

    def render(
        self, template_source: str, conversation: Conversation, **kwargs: object
    ) -> str: ...


class CPythonJinja:
    """Transformers / vLLM / SGLang family: full CPython Jinja2."""

    name = "cpython-jinja2"

    def render(
        self, template_source: str, conversation: Conversation, **kwargs: object
    ) -> str:
        return render_template_string(template_source, conversation, **kwargs)


@dataclass
class ParityReport:
    reference: str          # the reference engine's name
    other: str              # the compared engine's name
    identical: bool
    first_divergence: int | None   # char index of first difference, or None


def _first_divergence(a: str, b: str) -> int | None:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    if i == len(a) == len(b):
        return None
    return i


def compare_engines(
    template_source: str,
    conversation: Conversation,
    renderers: list[EngineRenderer],
    **kwargs: object,
) -> list[ParityReport]:
    """Render the same conversation on every engine and diff each against the first."""
    if not renderers:
        return []
    reference = renderers[0]
    ref_output = reference.render(template_source, conversation, **kwargs)

    reports: list[ParityReport] = []
    for engine in renderers[1:]:
        output = engine.render(template_source, conversation, **kwargs)
        divergence = _first_divergence(ref_output, output)
        reports.append(
            ParityReport(
                reference=reference.name,
                other=engine.name,
                identical=divergence is None,
                first_divergence=divergence,
            )
        )
    return reports


# Filters and constructs commonly absent from or divergent in minja. Heuristic.
_RISKY_FILTERS = {"reject", "select", "map", "groupby", "rejectattr", "selectattr"}


def lint_portability(template_source: str) -> list[str]:
    """Flag template constructs unlikely to survive the move to minja."""
    problems: list[str] = []
    for match in re.finditer(r"\|\s*([a-zA-Z_]+)", template_source):
        filter_name = match.group(1)
        if filter_name in _RISKY_FILTERS:
            problems.append(
                f"filter {filter_name!r} is a known cross-engine risk (often absent in minja)"
            )
    return problems
