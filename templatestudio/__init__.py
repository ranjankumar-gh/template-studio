"""Template Studio - the toolkit built across *The Chat Templates Handbook*.

A render/validate/probe/diff/golden/security toolkit for LLM chat templates.
transformers is an optional runtime dependency: it is only needed to render with a
real tokenizer or processor. The pure logic (validation, parity, golden, security,
reasoning, multimodal, authoring round-trips) runs with only jinja2 installed.
"""

from __future__ import annotations

__version__ = "1.2.0"

__all__ = ["__version__"]
