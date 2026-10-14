"""Every class of this package that implements ``generate`` has ONE row saying what it does with a
schema — and the row is what the class is.

The list is not written by hand. It is derived from the package by AST: every class whose body
defines ``generate`` (or inherits one from a class of this package). A new backend or wrapper that
arrives without a row in ``cogno_synapse.schema.BACKEND_SCHEMA_SUPPORT`` fails here, and so does
one whose row no longer matches its body. The two ways a sink goes wrong in silence are exactly
"structured in the table, half in the code" and "nobody listed it":

* ``"structured"`` — the class body defines BOTH ``schema_dialect`` and ``generate_structured``,
  and an instance satisfies ``StructuredBackend``;
* ``"plain: <reason>"`` — the class body defines NO ``generate_structured`` (it may define
  ``schema_dialect``, answering ``None``), and an instance does NOT satisfy the protocol.

Never half: ``generate_structured`` without ``schema_dialect``, or the reverse on a row that says
``structured``, is a failure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import cogno_synapse
from cogno_synapse import (AnthropicBackend, BedrockBackend, FallbackBackend, GeminiBackend,
                           GroqBackend, OllamaBackend, OpenAIBackend, StructuredBackend)
from cogno_synapse.schema import BACKEND_SCHEMA_SUPPORT, SCHEMA_FORMS

PACKAGE = Path(cogno_synapse.__file__).resolve().parent


def classes_of(source_dir: Path) -> "dict[str, dict]":
    """name → {module, methods, bases} for every class defined in the package."""
    found: "dict[str, dict]" = {}
    for path in sorted(source_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {item.name for item in node.body
                       if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
            bases = [base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                     for base in node.bases]
            module = ".".join(path.relative_to(source_dir.parent).with_suffix("").parts)
            assert node.name not in found, f"two classes named {node.name}"
            found[node.name] = {"module": module, "methods": methods, "bases": bases}
    return found


def _has(name: str, method: str, classes: "dict[str, dict]") -> bool:
    """``method`` is defined on ``name`` or on a base that is a class of this package."""
    info = classes.get(name)
    if info is None:
        return False
    return method in info["methods"] or any(_has(base, method, classes) for base in info["bases"])


def generate_sinks(classes: "dict[str, dict]") -> "set[str]":
    """Classes with a ``generate`` of their own or an inherited one — minus the Protocols, which
    declare the method and implement nothing."""
    return {name for name, info in classes.items()
            if _has(name, "generate", classes) and "Protocol" not in info["bases"]}


CLASSES = classes_of(PACKAGE)
SINKS = generate_sinks(CLASSES)


def build(name: str):
    """One instance of each sink — no SDK is imported and no network is touched by a constructor.
    A sink this table cannot build is a sink nobody probed."""
    return {
        "OpenAIBackend": lambda: OpenAIBackend(api_key="k"),
        "OllamaBackend": lambda: OllamaBackend("m"),
        "FallbackBackend": lambda: FallbackBackend([OllamaBackend("m")]),
        "AnthropicBackend": lambda: AnthropicBackend(api_key="k"),
        "GeminiBackend": lambda: GeminiBackend(api_key="k"),
        "GroqBackend": lambda: GroqBackend(api_key="k"),
        "BedrockBackend": lambda: BedrockBackend("m"),
    }[name]()


def test_the_derivation_sees_the_backends():
    """The control of everything below: an enumeration that found nothing would agree with any
    table."""
    assert {"OpenAIBackend", "OllamaBackend", "FallbackBackend", "AnthropicBackend",
            "GeminiBackend", "GroqBackend", "BedrockBackend"} <= SINKS
    assert "LLMBackend" not in SINKS and "StructuredBackend" not in SINKS
    assert "OllamaEmbedder" not in SINKS and "CachingEmbedder" not in SINKS


def test_every_sink_has_a_row_and_every_row_names_a_sink():
    assert SINKS == set(BACKEND_SCHEMA_SUPPORT), (
        f"without a row: {sorted(SINKS - set(BACKEND_SCHEMA_SUPPORT))}; "
        f"rows that name no class: {sorted(set(BACKEND_SCHEMA_SUPPORT) - SINKS)}")


def test_a_new_sink_without_a_row_is_caught(tmp_path):
    """The mutation, written as a test: a wrapper nobody listed."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "listed.py").write_text(
        "class OpenAIBackend:\n    async def generate(self, system, prompt):\n        return ('', 0, 0)\n")
    assert generate_sinks(classes_of(package)) == {"OpenAIBackend"}
    (package / "new.py").write_text(
        "from listed import OpenAIBackend\n\n"
        "class QuietWrapper:\n    async def generate(self, system, prompt):\n        return ('', 0, 0)\n\n"
        "class Inheriting(OpenAIBackend):\n    pass\n")
    found = generate_sinks(classes_of(package))
    assert found - set(BACKEND_SCHEMA_SUPPORT) == {"QuietWrapper", "Inheriting"}


@pytest.mark.parametrize("name", sorted(BACKEND_SCHEMA_SUPPORT))
def test_the_row_is_what_the_class_is(name):
    row = BACKEND_SCHEMA_SUPPORT[name]
    info = CLASSES[name]
    announces = _has(name, "schema_dialect", CLASSES)
    carries = _has(name, "generate_structured", CLASSES)
    backend = build(name)
    assert type(backend).__module__ == info["module"]
    if row == "structured":
        assert announces and carries, f"{name}: structured in the table, half in the code"
        assert isinstance(backend, StructuredBackend)
    else:
        assert row.startswith("plain: ") and len(row) > len("plain: ") + 10, (
            f"{name}: a plain row carries its reason")
        assert not carries, f"{name}: carries a schema but the table says plain"
        assert not isinstance(backend, StructuredBackend)
        # the method answers "no"; its absence would say nothing
        assert announces, f"{name}: no schema_dialect to answer None"
    assert carries == (row == "structured")


def test_no_sink_is_half_structured():
    for name in SINKS:
        carries = _has(name, "generate_structured", CLASSES)
        announces = _has(name, "schema_dialect", CLASSES)
        assert not (carries and not announces), name


def test_a_plain_backend_announces_none_for_every_form():
    plain = sorted(name for name, row in BACKEND_SCHEMA_SUPPORT.items() if row != "structured")
    assert len(plain) == 4                                              # the harness has subjects
    for name in plain:
        assert [build(name).schema_dialect(form) for form in SCHEMA_FORMS] == [None, None], name
    # control: a structured one answers something
    assert build("OpenAIBackend").schema_dialect("open") is not None
