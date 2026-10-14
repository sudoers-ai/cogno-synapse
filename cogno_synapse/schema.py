"""
cogno_synapse.schema — a JSON Schema on its way to a provider: the dialects, the conversion, the receipt.

Some providers take a JSON Schema WITH the request and constrain the reply to it. They do not take
the same schema: the shape each one accepts differs, and that shape is the thing a measurement of
"does a schema help" varies. This module is everything about that which is not a network call:

* the closed alphabets — which DIALECTS exist (:data:`SCHEMA_DIALECTS`), which FORMS a caller may
  ask for (:data:`SCHEMA_FORMS`), what a backend may ANNOUNCE (:data:`SCHEMA_ANNOUNCEMENTS`) and
  what a call may leave as its RECEIPT (:data:`SCHEMA_RECEIPTS`);
* :func:`to_dialect`, the PURE conversion of one canonical schema into the shape a dialect takes;
* :func:`schema_digest`, the name a schema goes by in a log line (never the schema itself);
* :class:`SchemaRefusals`, what an instance remembers about a schema its provider refused;
* :data:`BACKEND_SCHEMA_SUPPORT`, one row per class of this package that implements ``generate``.

**The caller hands over ONE schema, the canonical one, and never converts it.** A stage derives the
schema of the reply it reads (open objects, optional keys) and passes that. The LEAF backend — the
one that talks to a provider — converts it to its own dialect. A chain passes it on untouched: a
chain inside a chain would otherwise convert twice, and a mixed chain would hand one provider the
shape built for another.

**Which keywords a dialect accepts is a WORKING HYPOTHESIS, not a measurement.**
:data:`DIALECT_KEYWORDS` lists, per dialect, the keywords this module lets through, and it is
deliberately the small set a derived stage schema uses. Nothing here has been checked against a
provider; a capability probe is what confirms or corrects the table. The design does not depend on
the table being right: a keyword outside it is refused HERE, before any request (``ValueError``),
and a schema the provider refuses anyway degrades to a call without one (see ``StructuredBackend``
in :mod:`cogno_synapse.base`) — counted, never fatal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from typing import Any, Callable, Mapping, Optional

__all__ = [
    "BACKEND_SCHEMA_SUPPORT",
    "CHAIN_DIALECT",
    "DIALECT_FOR_FORM",
    "DIALECT_KEYWORDS",
    "SCHEMA_ANNOUNCEMENTS",
    "SCHEMA_DIALECTS",
    "SCHEMA_FORMS",
    "SCHEMA_NAME_RE",
    "SCHEMA_RECEIPTS",
    "SCHEMA_REFUSED",
    "SCHEMA_REFUSAL_TTL_S",
    "SchemaRefusals",
    "dialect_for",
    "is_invalid_request",
    "schema_digest",
    "to_dialect",
]

#: The shapes a provider takes a schema in — the variable of the measurement, closed.
#:
#: * ``openai_strict`` — OpenAI ``response_format`` of type ``json_schema`` with ``strict: true``.
#:   Every object closed, every key required, a key that was optional becomes ``type | null``.
#: * ``openai_json_schema`` — the same parameter with ``strict: false``. The schema as it is.
#: * ``ollama_format`` — Ollama's ``format`` field carrying the schema. The schema as it is.
SCHEMA_DIALECTS: "frozenset[str]" = frozenset({"openai_strict", "openai_json_schema", "ollama_format"})

#: What a CHAIN announces when at least one of its links takes a schema. Not a dialect: a chain
#: converts nothing, and :func:`to_dialect` refuses it.
CHAIN_DIALECT = "chain"

#: Everything ``schema_dialect()`` may answer besides ``None``.
SCHEMA_ANNOUNCEMENTS: "frozenset[str]" = SCHEMA_DIALECTS | {CHAIN_DIALECT}

#: The receipt of a call whose schema did NOT travel although one was asked for: the provider
#: refused it (and the call was repeated without), the instance remembered such a refusal, or the
#: schema could not be converted to the dialect at all.
SCHEMA_REFUSED = "refused"

#: Everything a call may leave as its receipt besides ``None`` (the call carried no schema).
SCHEMA_RECEIPTS: "frozenset[str]" = SCHEMA_DIALECTS | {SCHEMA_REFUSED}

#: What a caller may ASK for, per call. ``open`` is the canonical schema as the stage derived it;
#: ``strict`` is the provider's closed form. There is no default anywhere: a caller that forgets
#: the form fails at the call instead of getting one chosen for it.
SCHEMA_FORMS: "tuple[str, ...]" = ("open", "strict")

#: Per provider FAMILY, the dialect that serves each form. A form with no row is a form that
#: family cannot serve: a backend asked for it announces ``None`` and serves the call through
#: plain ``generate``. THERE IS NO DEMOTION — a strict request is never quietly served by the open
#: dialect, because a strict arm measured over non-strict requests is not an arm.
#:
#: Ollama answers the open form only. Its ``format=<schema>`` already is constrained decoding and
#: it has no second mode; ``strict`` names OpenAI's closed form. Were it to answer both, the two
#: arms of a comparison would send the same request and be counted as two.
DIALECT_FOR_FORM: "dict[str, dict[str, str]]" = {
    "openai": {"open": "openai_json_schema", "strict": "openai_strict"},
    "ollama": {"open": "ollama_format"},
}

def dialect_for(family: str, form: object) -> Optional[str]:
    """The dialect ``family`` serves ``form`` with, or ``None`` — the single reading of
    :data:`DIALECT_FOR_FORM`. A form outside :data:`SCHEMA_FORMS`, or one the family has no row
    for, answers ``None``: nothing is announced, and nothing else is substituted."""
    if not isinstance(form, str) or form not in SCHEMA_FORMS:
        return None
    return DIALECT_FOR_FORM.get(family, {}).get(form)


#: The keywords :func:`to_dialect` lets through, per dialect. A WORKING HYPOTHESIS (see the module
#: docstring): the set a derived stage schema uses, and nothing else. A keyword outside it raises
#: ``ValueError`` at conversion — a request that would fail is refused before it is made.
_CORE_KEYWORDS: "frozenset[str]" = frozenset(
    {"type", "properties", "required", "items", "enum", "additionalProperties"})
DIALECT_KEYWORDS: "dict[str, frozenset[str]]" = {dialect: _CORE_KEYWORDS for dialect in SCHEMA_DIALECTS}

#: The name a schema travels under (OpenAI's own rule for ``json_schema.name``).
SCHEMA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: How long an instance remembers that its provider refused a schema, in seconds.
SCHEMA_REFUSAL_TTL_S = 15 * 60

#: One row per class of this package that implements ``generate`` — ``"structured"`` (it has both
#: members of ``StructuredBackend`` and carries the schema to whatever serves the call) or
#: ``"plain: <reason>"`` (it has neither, and a stage calls its ``generate`` as it always did).
#: Never half: ``tests/unit/test_schema_sinks.py`` derives the class list from the package by AST
#: and fails a new class that arrives without a row, or with a row that is not what the class is.
BACKEND_SCHEMA_SUPPORT: "dict[str, str]" = {
    "OpenAIBackend": "structured",
    "OllamaBackend": "structured",
    "FallbackBackend": "structured",
    "AnthropicBackend": "plain: no structured mode is wired into generate for this provider",
    "GeminiBackend": "plain: no structured mode is wired into generate for this provider",
    "GroqBackend": "plain: no structured mode is wired into generate for this provider",
    "BedrockBackend": "plain: no structured mode is wired into generate for this provider",
}


# ── the conversion ────────────────────────────────────────────────────────────────────────────

def _is_object(node: "Mapping[str, Any]") -> bool:
    kind = node.get("type")
    return kind == "object" or (isinstance(kind, list) and "object" in kind)


def _check(node: Any, dialect: str, path: str) -> None:
    """Refuse anything the dialect is not known to take — the whole tree, before any request."""
    if not isinstance(node, Mapping):
        raise ValueError(f"schema node at {path or '<root>'} is not an object")
    allowed = DIALECT_KEYWORDS[dialect]
    for keyword in node:
        if keyword not in allowed:
            raise ValueError(f"keyword {keyword!r} at {path or '<root>'} is outside dialect {dialect!r}")
    properties = node.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError(f"'properties' at {path or '<root>'} is not an object")
    for key, sub in properties.items():
        _check(sub, dialect, f"{path}.{key}" if path else str(key))
    if "items" in node:
        _check(node["items"], dialect, f"{path}[]")
    extra = node.get("additionalProperties")
    if isinstance(extra, Mapping):
        _check(extra, dialect, f"{path}{{}}")
    elif extra is not None and not isinstance(extra, bool):
        raise ValueError(f"'additionalProperties' at {path or '<root>'} is neither a boolean nor a schema")
    required = node.get("required", [])
    if not isinstance(required, (list, tuple)) or any(not isinstance(k, str) for k in required):
        raise ValueError(f"'required' at {path or '<root>'} is not a list of keys")


def _nullable(node: "dict[str, Any]", path: str) -> "dict[str, Any]":
    """``node`` or ``null``. A key that was optional has to be SAYABLE as absent once required."""
    kind = node.get("type")
    if isinstance(kind, str):
        node["type"] = [kind, "null"]
    elif isinstance(kind, list):
        if "null" not in kind:
            node["type"] = [*kind, "null"]
    else:
        raise ValueError(f"optional key at {path} has no 'type' to widen with null")
    if "enum" in node and None not in node["enum"]:
        node["enum"] = [*node["enum"], None]
    return node


def _strict(node: "Mapping[str, Any]", path: str) -> "dict[str, Any]":
    """Every object closed, every key required, a key that was optional becomes ``type | null``."""
    out: "dict[str, Any]" = {k: copy.deepcopy(v) for k, v in node.items()
                             if k not in ("properties", "items", "required", "additionalProperties")}
    if "items" in node:
        out["items"] = _strict(node["items"], f"{path}[]")
    if not _is_object(node):
        if "properties" in node or "required" in node or "additionalProperties" in node:
            raise ValueError(f"object keywords on a non-object at {path or '<root>'}")
        return out
    if isinstance(node.get("additionalProperties"), Mapping):
        raise ValueError(f"cannot close the object at {path or '<root>'}: its extra keys carry a schema")
    properties = node.get("properties", {})
    was_required = set(node.get("required", ()))
    missing = was_required - set(properties)
    if missing:
        raise ValueError(f"required key(s) {sorted(missing)} at {path or '<root>'} have no property")
    converted: "dict[str, Any]" = {}
    for key, sub in properties.items():
        where = f"{path}.{key}" if path else str(key)
        child = _strict(sub, where)
        converted[key] = child if key in was_required else _nullable(child, where)
    out["properties"] = converted
    out["required"] = list(converted)
    out["additionalProperties"] = False
    return out


def to_dialect(schema: "Mapping[str, Any]", dialect: str) -> "dict[str, Any]":
    """The canonical ``schema`` in the shape ``dialect`` takes. PURE: a new object every call, the
    input never touched, no I/O, the same bytes for the same input.

    * ``ollama_format`` and ``openai_json_schema`` — the schema as it is (a deep copy);
    * ``openai_strict`` — every object closed (``additionalProperties: false``), every key of
      ``properties`` in ``required``, and a key that was optional widened to ``type | null`` (and
      its ``enum``, when it has one, to include ``null``). Idempotent.

    Raises ``ValueError`` — BEFORE any request is built — for a dialect outside
    :data:`SCHEMA_DIALECTS` (``"chain"`` included: a chain converts nothing), for a root that is
    not an object schema, for a keyword outside :data:`DIALECT_KEYWORDS`, and for a shape the
    strict form cannot say (extra keys that carry a schema, an optional key with no ``type``).
    """
    if dialect not in SCHEMA_DIALECTS:
        raise ValueError(f"unknown schema dialect {dialect!r}")
    _check(schema, dialect, "")
    if not _is_object(schema):
        raise ValueError("the root of a reply schema must be an object")
    if dialect == "openai_strict":
        return _strict(schema, "")
    return copy.deepcopy(dict(schema))


def schema_digest(schema: "Mapping[str, Any]") -> str:
    """The 12 hex characters a schema goes by in a log line or a refusal memory.

    SHA-256 over the canonical JSON (sorted keys, no whitespace), so two workers name the same
    schema the same way whatever order they built it in. A schema is derived from vocabularies and
    holds no contact data, but it is long: the digest is what a line can afford to carry.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ── a refused schema ──────────────────────────────────────────────────────────────────────────

def is_invalid_request(exc: BaseException) -> bool:
    """The provider answered 400: THIS request is malformed, and sending it again changes nothing.

    The one failure a schema can cause by itself. Told apart from everything a chain is for — a
    429, a timeout, a 5xx — by the status alone; those are never read as a refusal of the schema
    and travel on to the failover loop exactly as before.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 400


class SchemaRefusals:
    """What ONE backend instance remembers about schemas its provider refused.

    Keyed by ``(model, digest of the schema as it was sent)`` and forgotten after ``ttl_s``. It
    exists so a schema the provider refuses costs one failed round trip per window instead of one
    per turn. The clock is injected: a memory with a window is otherwise a test that is green
    until the day it is not.
    """

    def __init__(self, *, ttl_s: float = SCHEMA_REFUSAL_TTL_S,
                 clock: "Callable[[], float]" = time.monotonic) -> None:
        self._ttl_s = float(ttl_s)
        self._clock = clock
        self._until: "dict[tuple[str, str], float]" = {}

    def remember(self, model: str, digest: str) -> None:
        self._until[(model, digest)] = self._clock() + self._ttl_s

    def refused(self, model: str, digest: str) -> bool:
        until: Optional[float] = self._until.get((model, digest))
        if until is None:
            return False
        if self._clock() >= until:
            del self._until[(model, digest)]
            return False
        return True
