"""Reading a template's behavior by differential probing (Chapter 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from templatestudio.messages import Conversation
from templatestudio.render import render

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# A probe uses distinctive markers so we can find them unambiguously in the output.
_PROBE: Conversation = [
    {"role": "system", "content": "SYS_MARKER"},
    {"role": "user", "content": "USER_MARKER"},
    {"role": "assistant", "content": "ASST_MARKER"},
]


@dataclass
class TemplateReport:
    has_system_role: bool      # does a system message survive into the wire format?
    generation_prompt: str     # exactly what add_generation_prompt appends
    declares_tools: bool       # does the template reference a tools variable?
    sample_render: str         # the probe render, for eyeballing the skeleton


def _suffix_diff(shorter: str, longer: str) -> str:
    """Return the tail of `longer` that `shorter` does not share from the front."""
    i = 0
    while i < len(shorter) and i < len(longer) and shorter[i] == longer[i]:
        i += 1
    return longer[i:]


def describe_template(tokenizer: "PreTrainedTokenizerBase") -> TemplateReport:
    """Report a template's behavior by differential probing - no source reading."""
    without_gen = render(tokenizer, _PROBE, add_generation_prompt=False)
    with_gen = render(tokenizer, _PROBE, add_generation_prompt=True)

    template_source = getattr(tokenizer, "chat_template", None) or ""
    return TemplateReport(
        has_system_role="SYS_MARKER" in without_gen,
        generation_prompt=_suffix_diff(without_gen, with_gen),
        declares_tools="tools" in template_source,
        sample_render=with_gen,
    )
