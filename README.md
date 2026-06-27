# Template Studio

The companion toolkit for [*The Chat Templates Handbook*](https://the-chat-templates-handbook.ranjankumar.in/)
by Ranjan Kumar. A render / validate / probe / diff / golden / security toolkit for LLM chat templates,
built one module per chapter.

## Why

A chat template is the contract between a list of structured messages and the exact token stream a model
was trained on. When it is wrong, divergent, or hostile, the model still answers - just worse - and
nothing raises. Template Studio makes the wire format a first-class, tested, reviewed artifact.

## Install

```bash
pip install -e .            # core (jinja2 only)
pip install -e ".[live]"    # + transformers, for rendering with a real tokenizer
pip install -e ".[dev]"     # + pytest
```

`transformers` is **optional**: it is needed only to render with a real tokenizer or processor
(`render`, `anatomy`, live `golden`). All the pure logic - validation, parity, golden checks,
security, reasoning, multimodal, authoring round-trips - runs with only `jinja2`.

## Modules (one per chapter)

| Module | Chapter | Provides |
|---|---|---|
| `messages` | 2 | `Message`/`Role`/`Conversation` types, `validate_conversation`, `assert_valid` |
| `render` | 1,2,5,6 | `load_tokenizer`, `render`, `safe_render`, `visible_whitespace` |
| `jinja_env` | 3 | `build_template_env`, `render_template_string` (the faithful environment) |
| `anatomy` | 4 | `describe_template` via differential probing |
| `tools` | 5 | `tool_schema`, message builders, `JSONBlockParser` |
| `reasoning` | 6 | `strip_reasoning`, `has_empty_think_blocks`, `assistant_message` |
| `multimodal` | 7 | typed-content builders, `validate_content_parts`, `count_media_parts` |
| `authoring` | 8 | `install_template`, `assert_reproduces_training_format` |
| `parity` | 9 | `compare_engines`, `lint_portability` (the parity gap) |
| `golden` | 10 | `GoldenCase`, `record_goldens`, `check_goldens` (the golden-token test) |
| `security` | 11 | `template_fingerprint`, `scan_template`, `find_control_tokens` |
| `studio` | 12 | `audit_template`, `ci_gate` (the whole suite + CI gate) |

## Quick start

```python
from templatestudio.jinja_env import render_template_string

CHATML = (
    "{% for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)

convo = [{"role": "user", "content": "hi"}]
print(render_template_string(CHATML, convo))
```

## The CI gate

```python
from templatestudio.studio import audit_template, ci_gate

report = audit_template(template_source, tokenizer,
                        goldens=goldens, engines=engines,
                        pinned_fingerprint=pinned, probe=probe)
raise SystemExit(ci_gate(report))   # exit 0 clean, 1 on any finding
```

## Test

```bash
pytest            # offline logic (no model download)
```

The live smoke test (`tests/test_live.py`) renders against a real Qwen3 tokenizer and **skips
automatically** when `transformers` is missing or the model cannot be downloaded.

## Note on book vs package

The book prints the modules as teaching snippets. This package is the runnable form: it imports
`transformers` lazily / under `TYPE_CHECKING`, so the toolkit installs and its logic tests with just
`jinja2`. The behavior is identical; only the import hygiene differs.

## License

MIT.
