"""The declared dependency floors and the shipped version, pinned.

`transformers>=4.5` predates `apply_chat_template` entirely: a resolver
honouring that floor can install a Transformers with no chat-template support at
all, against a package whose whole subject is chat templates.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
_REQUIREMENTS = (_ROOT / "requirements.txt").read_text(encoding="utf-8")

# apply_chat_template landed in Transformers 4.34; the tool/reasoning behaviour
# this package probes needs the 4.51 line.
_MINIMUM_TRANSFORMERS = (4, 51)


def _floor(text: str) -> tuple[int, ...]:
    match = re.search(r"transformers>=([0-9]+(?:\.[0-9]+)*)", text)
    assert match is not None, "no transformers floor declared"
    return tuple(int(part) for part in match.group(1).split("."))


def test_no_source_line_exceeds_88_columns():
    """The house limit the book states for code, pinned so it cannot regress."""
    over: list[str] = []
    for path in sorted(_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if len(line) > 88:
                rel = path.relative_to(_ROOT).as_posix()
                over.append(f"{rel}:{number} is {len(line)} columns")
    assert over == []


def test_pyproject_transformers_floor_postdates_apply_chat_template():
    assert _floor(_PYPROJECT) >= _MINIMUM_TRANSFORMERS


def test_requirements_transformers_floor_matches_pyproject():
    assert _floor(_REQUIREMENTS) == _floor(_PYPROJECT)


def test_version_matches_the_book_edition():
    """The package version tracks the book edition."""
    match = re.search(r'^version = "([^"]+)"', _PYPROJECT, re.M)
    assert match is not None
    version = tuple(int(part) for part in match.group(1).split("."))
    assert version[:2] == (1, 2)


def test_dunder_version_matches_pyproject():
    """Two declarations of one version drift silently; pin them together."""
    import templatestudio

    match = re.search(r'^version = "([^"]+)"', _PYPROJECT, re.M)
    assert match is not None
    assert templatestudio.__version__ == match.group(1)
