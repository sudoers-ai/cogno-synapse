"""The schema on its way to a provider: the closed alphabets, the form → dialect table, the PURE
conversion, the digest and the refusal memory.

No network, no SDK, no model. The schema used here is INVENTED but has the shape a stage's reply
schema has — an open root, a required enum, a nullable enum, optional keys, an open sub-object
with optional arrays and one sub-object that is already closed — because the conversion's rules
are about exactly those shapes.

Every refusal below is paired with a control that converts: a test of an absence that cannot first
show the presence proves only that the harness is inert.
"""

from __future__ import annotations

import copy
import json

import pytest

from cogno_synapse import (CHAIN_DIALECT, SCHEMA_ANNOUNCEMENTS, SCHEMA_DIALECTS, SCHEMA_FORMS,
                           SCHEMA_RECEIPTS, SCHEMA_REFUSED, schema_digest, to_dialect)
from cogno_synapse.schema import (DIALECT_FOR_FORM, DIALECT_KEYWORDS, SCHEMA_REFUSAL_TTL_S,
                                  SchemaRefusals, dialect_for, is_invalid_request)

OPEN_DIALECTS = ("openai_json_schema", "ollama_format")


def reply_schema() -> dict:
    """A stage-shaped reply schema. ``note`` and the ``parts`` sub-keys are OPTIONAL."""
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["ALPHA", "BETA"]},
            "mood": {"type": ["string", "null"], "enum": ["CALM", "TENSE", None]},
            "score": {"type": "number"},
            "flag": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["X", "Y"]}},
            "note": {"type": "string"},
            "parts": {"type": "object",
                      "properties": {"people": {"type": "array", "items": {"type": "string"}},
                                     "things": {"type": "array", "items": {"type": "string"}}}},
            "facets": {"type": "object",
                       "properties": {"cause": {"type": "string"}, "place": {"type": "string"}},
                       "additionalProperties": False},
        },
        "required": ["facets", "flag", "kind", "mood", "parts", "score", "tags"],
    }


def _objects(node, path=""):
    """Every object schema in the tree, with its path."""
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "object" or (isinstance(kind, list) and "object" in kind):
            yield path, node
        for key, sub in node.get("properties", {}).items():
            yield from _objects(sub, f"{path}.{key}")
        if "items" in node:
            yield from _objects(node["items"], f"{path}[]")


def _enums(node, path=""):
    if isinstance(node, dict):
        if "enum" in node:
            yield path, node["enum"]
        for key, sub in node.get("properties", {}).items():
            yield from _enums(sub, f"{path}.{key}")
        if "items" in node:
            yield from _enums(node["items"], f"{path}[]")


# ── the alphabets are closed, and they are these ──────────────────────────────────────────────

def test_the_alphabets():
    assert SCHEMA_DIALECTS == {"openai_strict", "openai_json_schema", "ollama_format"}
    assert SCHEMA_FORMS == ("open", "strict")
    assert CHAIN_DIALECT == "chain" and CHAIN_DIALECT not in SCHEMA_DIALECTS
    assert SCHEMA_ANNOUNCEMENTS == SCHEMA_DIALECTS | {"chain"}
    assert SCHEMA_REFUSED == "refused" and SCHEMA_REFUSED not in SCHEMA_DIALECTS
    assert SCHEMA_RECEIPTS == SCHEMA_DIALECTS | {"refused"}
    # a chain's announcement is never a receipt: a chain is served by a link, and the receipt is
    # the link's
    assert CHAIN_DIALECT not in SCHEMA_RECEIPTS
    assert set(DIALECT_KEYWORDS) == set(SCHEMA_DIALECTS)


def test_every_dialect_of_the_table_is_in_the_alphabet_and_every_form_is_known():
    for family, row in DIALECT_FOR_FORM.items():
        assert set(row) <= set(SCHEMA_FORMS), family
        assert set(row.values()) <= SCHEMA_DIALECTS, family


# ── form → dialect: no demotion ───────────────────────────────────────────────────────────────

def test_openai_serves_each_form_with_its_own_dialect():
    assert dialect_for("openai", "open") == "openai_json_schema"
    assert dialect_for("openai", "strict") == "openai_strict"


def test_ollama_answers_the_open_form_only():
    """The twin: asked for ``strict``, the answer is ``None`` — never the open dialect. Were it
    ``ollama_format`` for both, the two arms of a comparison would send one identical request."""
    assert dialect_for("ollama", "open") == "ollama_format"        # the control: it can answer
    assert dialect_for("ollama", "strict") is None


@pytest.mark.parametrize("form", [None, "", "OPEN", "Strict", "closed", 1, True, ["open"], "chain"])
def test_a_form_outside_the_alphabet_is_served_by_nothing(form):
    for family in DIALECT_FOR_FORM:
        assert dialect_for(family, form) is None


def test_an_unknown_family_serves_nothing():
    assert dialect_for("anthropic", "open") is None
    assert dialect_for("openai", "open") is not None               # control


# ── the open dialects: the schema as it is ────────────────────────────────────────────────────

@pytest.mark.parametrize("dialect", OPEN_DIALECTS)
def test_an_open_dialect_carries_the_schema_as_it_is(dialect):
    schema = reply_schema()
    before = copy.deepcopy(schema)
    out = to_dialect(schema, dialect)
    assert out == before
    assert json.dumps(out, sort_keys=True) == json.dumps(before, sort_keys=True)
    assert schema == before, "the input was touched"


@pytest.mark.parametrize("dialect", sorted(SCHEMA_DIALECTS))
def test_the_result_shares_nothing_with_the_input(dialect):
    schema = reply_schema()
    before = copy.deepcopy(schema)
    out = to_dialect(schema, dialect)
    out["properties"]["kind"]["enum"].append("GAMMA")
    out["properties"]["parts"]["properties"]["people"]["items"]["type"] = "number"
    out["required"].append("zzz")
    assert schema == before
    assert to_dialect(schema, dialect) != out


# ── the strict dialect: closed, all required, optional becomes nullable ───────────────────────

def test_strict_closes_every_object():
    out = to_dialect(reply_schema(), "openai_strict")
    paths = [path for path, _ in _objects(out)]
    assert paths == ["", ".parts", ".facets"]                       # the harness sees all three
    for path, node in _objects(out):
        assert node["additionalProperties"] is False, path
    # control: two of the three were OPEN in the canonical schema
    assert [path for path, node in _objects(reply_schema())
            if node.get("additionalProperties") is not False] == ["", ".parts"]


def test_strict_requires_every_key():
    out = to_dialect(reply_schema(), "openai_strict")
    for path, node in _objects(out):
        assert node["required"] == list(node["properties"]), path
    assert "note" in out["required"]
    assert "note" not in reply_schema()["required"]                 # control: it was optional


def test_a_key_that_was_optional_becomes_nullable_and_one_that_was_required_does_not():
    out = to_dialect(reply_schema(), "openai_strict")
    props = out["properties"]
    assert props["note"]["type"] == ["string", "null"]
    assert props["parts"]["properties"]["people"] == {"type": ["array", "null"],
                                                      "items": {"type": "string"}}
    assert props["facets"]["properties"]["cause"]["type"] == ["string", "null"]
    # controls: required keys keep the type the stage reads
    assert props["kind"] == {"type": "string", "enum": ["ALPHA", "BETA"]}
    assert props["score"] == {"type": "number"}
    assert props["flag"] == {"type": "boolean"}
    assert props["parts"]["type"] == "object"
    # already nullable and required: unchanged, not doubled
    assert props["mood"] == {"type": ["string", "null"], "enum": ["CALM", "TENSE", None]}


def test_an_optional_enum_gains_null_in_its_enum_too():
    schema = reply_schema()
    schema["required"].remove("kind")
    out = to_dialect(schema, "openai_strict")
    assert out["properties"]["kind"] == {"type": ["string", "null"], "enum": ["ALPHA", "BETA", None]}


def test_strict_is_idempotent():
    once = to_dialect(reply_schema(), "openai_strict")
    assert to_dialect(once, "openai_strict") == once


def test_strict_does_not_touch_its_input():
    schema = reply_schema()
    before = copy.deepcopy(schema)
    to_dialect(schema, "openai_strict")
    assert schema == before


@pytest.mark.parametrize("dialect", sorted(SCHEMA_DIALECTS))
def test_no_enumeration_is_copied_or_lost(dialect):
    """Every ``enum`` of the result is the input's own, in order — plus ``null`` only where the
    strict form made an optional key nullable. A conversion that carried its own list would pass
    a spot check and drift the day a vocabulary gained a word."""
    schema = reply_schema()
    schema["required"].remove("kind")                               # one optional enum
    source = dict(_enums(schema))
    result = dict(_enums(to_dialect(schema, dialect)))
    assert set(result) == set(source) and len(source) == 3
    for path, values in result.items():
        widened = dialect == "openai_strict" and path == ".kind"
        assert values == source[path] + ([None] if widened else []), path
    # the derivation is live: a word added to the source is a word in the result
    schema["properties"]["tags"]["items"]["enum"].append("Z")
    assert dict(_enums(to_dialect(schema, dialect)))[".tags[]"] == ["X", "Y", "Z"]


# ── what is refused, BEFORE any request ───────────────────────────────────────────────────────

@pytest.mark.parametrize("dialect", sorted(SCHEMA_DIALECTS))
@pytest.mark.parametrize("keyword,value", [
    ("minLength", 1), ("maxLength", 9), ("pattern", "^a"), ("minimum", 0), ("maximum", 1),
    ("uniqueItems", True), ("format", "email"), ("anyOf", [{"type": "string"}]),
    ("$ref", "#/x"), ("description", "text"), ("default", "x"), ("const", "x"),
])
def test_a_keyword_outside_the_dialect_is_refused(dialect, keyword, value):
    assert to_dialect(reply_schema(), dialect)                      # control: without it, it converts
    schema = reply_schema()
    schema["properties"]["note"][keyword] = value
    with pytest.raises(ValueError, match="outside dialect"):
        to_dialect(schema, dialect)


@pytest.mark.parametrize("dialect", sorted(SCHEMA_DIALECTS))
def test_the_refusal_reaches_the_bottom_of_the_tree(dialect):
    for place in ("items", "nested", "closed"):
        schema = reply_schema()
        target = {"items": schema["properties"]["tags"]["items"],
                  "nested": schema["properties"]["parts"]["properties"]["people"]["items"],
                  "closed": schema["properties"]["facets"]["properties"]["place"]}[place]
        target["maxLength"] = 3
        with pytest.raises(ValueError, match="maxLength"):
            to_dialect(schema, dialect)


@pytest.mark.parametrize("dialect", ["chain", "openai", "OPENAI_STRICT", "", None, "refused"])
def test_an_unknown_dialect_is_refused_and_a_chain_is_not_a_dialect(dialect):
    with pytest.raises(ValueError, match="unknown schema dialect"):
        to_dialect(reply_schema(), dialect)


@pytest.mark.parametrize("dialect", sorted(SCHEMA_DIALECTS))
@pytest.mark.parametrize("root", [{"type": "array", "items": {"type": "string"}}, {"type": "string"},
                                  {}, [], "x", None])
def test_a_root_that_is_not_an_object_schema_is_refused(dialect, root):
    with pytest.raises(ValueError):
        to_dialect(root, dialect)


def test_extra_keys_that_carry_a_schema_cannot_be_closed_but_travel_open():
    schema = reply_schema()
    schema["properties"]["parts"]["additionalProperties"] = {"type": "string"}
    for dialect in OPEN_DIALECTS:                                   # control: the open forms say it
        assert to_dialect(schema, dialect)["properties"]["parts"]["additionalProperties"] == {
            "type": "string"}
    with pytest.raises(ValueError, match="cannot close"):
        to_dialect(schema, "openai_strict")


def test_strict_refuses_what_it_cannot_say():
    schema = reply_schema()
    schema["properties"]["note"] = {"enum": ["a", "b"]}             # optional, and no type to widen
    with pytest.raises(ValueError, match="no 'type'"):
        to_dialect(schema, "openai_strict")
    schema = reply_schema()
    schema["required"].append("ghost")                              # required, and not a property
    with pytest.raises(ValueError, match="ghost"):
        to_dialect(schema, "openai_strict")
    schema = reply_schema()
    schema["properties"]["note"]["required"] = ["x"]                # object keyword on a string
    with pytest.raises(ValueError, match="non-object"):
        to_dialect(schema, "openai_strict")


@pytest.mark.parametrize("dialect", sorted(SCHEMA_DIALECTS))
def test_malformed_containers_are_refused(dialect):
    for key, bad in (("properties", ["a"]), ("required", "kind"), ("additionalProperties", "no")):
        schema = reply_schema()
        schema[key] = bad
        with pytest.raises(ValueError):
            to_dialect(schema, dialect)


# ── the digest ────────────────────────────────────────────────────────────────────────────────

def test_the_digest_names_the_schema_not_the_order_it_was_built_in():
    schema = reply_schema()
    shuffled = {key: schema[key] for key in reversed(list(schema))}
    shuffled["properties"] = {k: schema["properties"][k] for k in reversed(list(schema["properties"]))}
    digest = schema_digest(schema)
    assert digest == schema_digest(shuffled)
    assert len(digest) == 12 and set(digest) <= set("0123456789abcdef")
    schema["properties"]["kind"]["enum"].append("GAMMA")
    assert schema_digest(schema) != digest
    assert schema_digest(to_dialect(reply_schema(), "openai_strict")) != digest


# ── the refusal memory: a window, on an injected clock ────────────────────────────────────────

def test_a_refusal_is_remembered_for_the_window_and_then_forgotten():
    now = [1000.0]
    memory = SchemaRefusals(clock=lambda: now[0])
    assert not memory.refused("m", "abc")                           # control: nothing yet
    memory.remember("m", "abc")
    assert memory.refused("m", "abc")
    now[0] += SCHEMA_REFUSAL_TTL_S - 0.001
    assert memory.refused("m", "abc")
    now[0] += 0.001
    assert not memory.refused("m", "abc")
    assert SCHEMA_REFUSAL_TTL_S == 900


def test_the_memory_is_per_model_and_per_schema():
    memory = SchemaRefusals(clock=lambda: 0.0)
    memory.remember("m", "abc")
    assert memory.refused("m", "abc")
    assert not memory.refused("other", "abc")
    assert not memory.refused("m", "abd")


# ── which failure is a refusal of the REQUEST ─────────────────────────────────────────────────

class _Status(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


class _ViaResponse(Exception):
    def __init__(self, status):
        super().__init__("http")
        self.response = type("R", (), {"status_code": status})()


def test_only_a_400_is_a_refusal_of_the_request():
    assert is_invalid_request(_Status(400)) and is_invalid_request(_ViaResponse(400))
    for status in (401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504):
        assert not is_invalid_request(_Status(status)), status
        assert not is_invalid_request(_ViaResponse(status)), status
    for exc in (TimeoutError("slow"), ConnectionError("down"), ValueError("400"), Exception()):
        assert not is_invalid_request(exc)
