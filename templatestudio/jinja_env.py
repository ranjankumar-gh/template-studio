"""The faithful environment and a model-free template workbench (Chapter 3)."""

from __future__ import annotations

import json
from datetime import datetime

from jinja2 import nodes
from jinja2.exceptions import TemplateError
from jinja2.ext import Extension
from jinja2.parser import Parser
from jinja2.sandbox import ImmutableSandboxedEnvironment

from templatestudio.messages import Conversation, assert_valid


class GenerationExtension(Extension):
    """A no-op {% generation %} tag. Transformers installs AssistantTracker,
    which uses this tag to record which characters the assistant generated
    (how return_assistant_tokens_mask works, Chapters 8 and 13). For a
    string render the body passes through unchanged - but some extension has
    to claim the tag, or a template that uses it is a syntax error."""

    tags = {"generation"}

    def parse(self, parser: Parser) -> nodes.Node:
        line = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=line)


def raise_exception(message: str) -> None:
    """Let a template fail loudly. Transformers injects this so a template
    can reject malformed input itself - e.g.
    {{ raise_exception('no system role') }}."""
    raise TemplateError(message)


def tojson(
    x: object,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    """Transformers' JSON filter, signature and all. It keeps the schema's
    key order and does not escape HTML characters (the built-in sorts keys
    and turns < into \\u003c), and templates call it with keyword
    arguments the built-in lacks: GLM writes tojson(ensure_ascii=False)."""
    return json.dumps(
        x,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def strftime_now(format_str: str) -> str:
    """datetime.now().strftime(...), exposed to templates that stamp the
    date into a system message (a surprising number do)."""
    return datetime.now().strftime(format_str)


def build_template_env() -> ImmutableSandboxedEnvironment:
    """A Jinja environment configured like the one Transformers compiles chat
    templates in: immutable sandbox, trim_blocks, lstrip_blocks, the two extensions
    it installs, and the three helpers it injects. Render here and you get
    production's string."""
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols", GenerationExtension],
    )
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    env.filters["tojson"] = tojson
    return env


def render_template_string(
    template_source: str,
    conversation: Conversation,
    *,
    add_generation_prompt: bool = True,
    **kwargs: object,
) -> str:
    """Render a raw chat-template string against a conversation, faithfully.

    This is the workbench for the rest of the book: experiment with a template,
    author one (Ch8), or reproduce a bug (Ch11) without loading a model.
    """
    assert_valid(conversation)
    env = build_template_env()
    template = env.from_string(template_source)
    return template.render(
        messages=conversation,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
