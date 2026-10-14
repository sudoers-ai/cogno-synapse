"""A schema travels with the CALL, to whichever backend serves it — or the receipt says it did not.

Two ways of failing in silence are what this file exists to close:

* a comparison measuring nothing, in green — the schema was asked for and some layer between the
  stage and the provider dropped it without a word;
* a dead turn — a layer handed a keyword it did not expect to a two-argument ``generate``.

So the rule is per SINK: a class that implements ``generate`` either has BOTH members of
``StructuredBackend`` and carries the schema to whatever serves the call, or has neither and is
called exactly as it always was. Each sink below has its twin (the schema arrives) and its control
(without one, the request is today's).

No network, no SDK, no model: the OpenAI client and ``httpx`` are replaced by doubles. Every
"→ None" assertion sits beside one that produces a real value from the same harness.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import sys

import httpx
import pytest
from cogno_homeo import CircuitBreaker, RetryPolicy

import cogno_synapse.openai_backend as openai_module
from cogno_synapse import (AnthropicBackend, BedrockBackend, FallbackBackend, GeminiBackend,
                           GroqBackend, LLMBackend, OllamaBackend, OpenAIBackend,
                           StructuredBackend, schema_applied_of, schema_dialect_of, schema_digest,
                           to_dialect)
from cogno_synapse.schema import SCHEMA_FORMS, SCHEMA_REFUSAL_TTL_S, SchemaRefusals

SYSTEM = "Reply with a json object."
PROMPT = "the caller's text — ação, 'quotes' and a {brace}"


def canonical() -> dict:
    """An invented, stage-shaped reply schema: open root, one optional key."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "boolean"},
            "kind": {"type": "string", "enum": ["ALPHA", "BETA"]},
            "note": {"type": "string"},
        },
        "required": ["kind", "verdict"],
    }


# ── doubles ───────────────────────────────────────────────────────────────────────────────────

class _Status(Exception):
    """An SDK error carrying an HTTP status, with a body that must never reach a log line."""

    def __init__(self, status: int, text: str = "PROVIDER-ERROR-TEXT quoting the request") -> None:
        super().__init__(text)
        self.status_code = status


async def _released(event: asyncio.Event) -> None:
    """Wait for a held request to be released — BOUNDED. A test that holds a request open must
    fail when the choreography breaks (the call it meant to hold never reached the provider, so
    another call took its place), not wait on a gate nobody will open."""
    await asyncio.wait_for(event.wait(), timeout=5)


def _response(content="{}", refusal=None):
    message = type("M", (), {"content": content, "tool_calls": None, "refusal": refusal})()
    return type("R", (), {
        "choices": [type("C", (), {"finish_reason": "stop", "message": message})()],
        "usage": type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})(),
        "system_fingerprint": "fp_x", "model": "served-x"})()


def openai_backend(script=None, *, model="gpt-4o-mini", gate=None, **kwargs):
    """An ``OpenAIBackend`` whose client is a double. ``script`` is consumed one entry per request:
    an exception is raised, anything else is the response. ``gate`` (request index → Event) holds
    a request open until the test releases it. The client's ``close`` SUSPENDS, as a real one
    does — the receipt has to survive that."""
    script = list(script or [])
    seen: list = []

    class _Completions:
        async def create(self, **kw):
            index = len(seen)
            seen.append(kw)
            if gate and index in gate:
                await _released(gate[index])
            step = script.pop(0) if script else _response()
            if isinstance(step, Exception):
                raise step
            return step

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            await asyncio.sleep(0)

    backend = OpenAIBackend(model=model, api_key="sk-x", **kwargs)
    backend._client = _Client  # type: ignore[method-assign]
    return backend, seen


def ollama_backend(monkeypatch, script=None, *, gate=None, **kwargs):
    """An ``OllamaBackend`` over a double of ``httpx.AsyncClient.post``. Same ``script``/``gate``
    rules; an int in the script is an HTTP status the server answers with."""
    script = list(script or [])
    seen: list = []

    class _Resp:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "SERVER-ERROR-TEXT", request=httpx.Request("POST", "http://test"),
                    response=httpx.Response(self.status_code))

        def json(self):
            return {"response": "{}", "prompt_eval_count": 7, "eval_count": 3}

    async def post(self, url, json=None, **kw):
        index = len(seen)
        seen.append(copy.deepcopy(json))
        if gate and index in gate:
            await _released(gate[index])
        step = script.pop(0) if script else 200
        if isinstance(step, Exception):
            raise step
        return _Resp(step)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return OllamaBackend(model="local-model", **kwargs), seen


class PlainLink:
    """A link with NO schema support and a ``generate`` of exactly two arguments — the shape of
    every backend, wrapper and test double that existed before. A keyword it does not expect is
    a ``TypeError``, which is the dead turn."""

    def __init__(self, model="plain", exc=None):
        self.model = model
        self.exc = exc
        self.calls: list = []

    async def generate(self, system, prompt):
        self.calls.append((system, prompt))
        if self.exc:
            raise self.exc
        return ("plain", 1, 1)


class StructuredLink:
    """A leaf double that announces ``dialects`` (form → dialect) and records what it was handed."""

    def __init__(self, model="leaf", dialects=None, exc=None, receipt="same"):
        self.model = model
        self.dialects = {"open": "openai_json_schema", "strict": "openai_strict"} \
            if dialects is None else dialects
        self.exc = exc
        self.receipt = receipt
        self.plain: list = []
        self.structured: list = []
        self.last_schema_dialect = None

    def schema_dialect(self, form):
        return self.dialects.get(form)

    async def generate(self, system, prompt):
        self.plain.append((system, prompt))
        if self.exc:
            raise self.exc
        self.last_schema_dialect = None
        return ("leaf-plain", 1, 1)

    async def generate_structured(self, system, prompt, *, schema, name, form):
        self.structured.append({"system": system, "prompt": prompt, "schema": schema,
                                "name": name, "form": form})
        if self.exc:
            raise self.exc
        self.last_schema_dialect = self.dialects[form] if self.receipt == "same" else self.receipt
        return ("leaf-structured", 2, 2)


class Counting:
    def __init__(self):
        self.records: list = []

    def record(self, attempt):
        self.records.append((attempt.provider, attempt.ok))


# ══ the probe ═════════════════════════════════════════════════════════════════════════════════

def test_who_satisfies_the_protocol():
    """Run on every Python the CI matrix has: 3.12 resolves protocol members with
    ``inspect.getattr_static``, so the members must be on the CLASS."""
    for backend in (OpenAIBackend(api_key="k"), OllamaBackend("m"),
                    FallbackBackend([OllamaBackend("m")]), StructuredLink()):
        assert isinstance(backend, StructuredBackend), type(backend).__name__
        assert isinstance(backend, LLMBackend)
    for backend in (AnthropicBackend(api_key="k"), GeminiBackend(api_key="k"),
                    GroqBackend(api_key="k"), BedrockBackend("m"), PlainLink()):
        assert isinstance(backend, LLMBackend), type(backend).__name__      # control
        assert not isinstance(backend, StructuredBackend), type(backend).__name__


def test_the_four_without_a_structured_mode_answer_no_through_the_method():
    """The METHOD says no; there is no ``generate_structured`` beside it."""
    for cls in (AnthropicBackend, GeminiBackend, GroqBackend, BedrockBackend):
        backend = cls("m") if cls is BedrockBackend else cls(api_key="k")
        for form in SCHEMA_FORMS:
            assert backend.schema_dialect(form) is None, cls.__name__
            assert schema_dialect_of(backend, form) is None
        assert not hasattr(backend, "generate_structured"), cls.__name__


def test_what_each_leaf_announces_per_form():
    assert schema_dialect_of(OpenAIBackend(api_key="k"), "open") == "openai_json_schema"
    assert schema_dialect_of(OpenAIBackend(api_key="k"), "strict") == "openai_strict"
    assert schema_dialect_of(OllamaBackend("m"), "open") == "ollama_format"
    assert schema_dialect_of(OllamaBackend("m"), "strict") is None          # never demoted


def test_an_openai_compatible_endpoint_announces_nothing():
    compatible = OpenAIBackend(model="some-chat", api_key="k", base_url="https://compat.invalid/v1")
    for form in SCHEMA_FORMS:
        assert schema_dialect_of(compatible, form) is None
        assert schema_dialect_of(OpenAIBackend(model="some-chat", api_key="k"), form)   # control


def test_a_model_a_probe_showed_refuses_a_dialect_announces_none_for_it_only(monkeypatch):
    assert openai_module.SCHEMA_DIALECTS_REFUSED_BY_MODEL == {}, "a row needs its measurement"
    monkeypatch.setitem(openai_module.SCHEMA_DIALECTS_REFUSED_BY_MODEL, "Frugal-9",
                        frozenset({"openai_strict"}))
    refused = OpenAIBackend(model="frugal-9-mini", api_key="k")
    assert refused.schema_dialect("strict") is None
    assert refused.schema_dialect("open") == "openai_json_schema"       # the other form still goes
    assert OpenAIBackend(model="other-1", api_key="k").schema_dialect("strict") == "openai_strict"


@pytest.mark.asyncio
async def test_a_wrapper_that_forwards_through_getattr_is_never_half():
    """A wrapper that forwards every attribute through ``__getattr__`` reads differently on the
    Pythons this matrix runs, and BOTH readings are whole:

    * 3.12 and later resolve protocol members with ``inspect.getattr_static``, which never calls
      ``__getattr__``: the probe says no, and the call is today's ``generate``;
    * before 3.12 the probe is ``hasattr``: it says yes — and then the forwarded
      ``generate_structured`` really does carry the schema.

    What neither may be is a probe that says yes over a call that drops the schema or dies.
    """

    class Forwarding:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    leaf, seen = openai_backend()
    wrapper = Forwarding(leaf)
    assert wrapper.schema_dialect("open") == "openai_json_schema"       # the attribute resolves
    chain = FallbackBackend([wrapper])
    await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    receipt = schema_applied_of(chain)
    if sys.version_info >= (3, 12):
        assert not isinstance(wrapper, StructuredBackend)
        assert schema_dialect_of(wrapper, "open") is None and chain.schema_dialect("open") is None
        assert receipt is None and seen[0]["response_format"] == {"type": "json_object"}
    else:
        assert isinstance(wrapper, StructuredBackend)
        assert schema_dialect_of(wrapper, "open") == "openai_json_schema"
        assert receipt == "openai_json_schema"
        assert seen[0]["response_format"]["json_schema"]["schema"] == canonical()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_half_sink_is_a_plain_backend():
    """A class that ANNOUNCES a dialect and has no ``generate_structured`` is the half state the
    rule forbids. The probe is the protocol AND the answer, so it reads as plain — and a chain
    serves it through ``generate`` instead of calling a method that is not there (the dead turn).
    No ``__getattr__`` is involved, so this holds on every Python of the matrix."""

    class Half(PlainLink):
        def schema_dialect(self, form):
            return "openai_strict"

    half = Half("half")
    assert half.schema_dialect("strict") == "openai_strict"             # it does announce…
    assert not isinstance(half, StructuredBackend)
    assert schema_dialect_of(half, "strict") is None                    # …and the probe says no
    chain = FallbackBackend([half])
    assert chain.schema_dialect("strict") is None
    result = await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert result == ("plain", 1, 1) and schema_applied_of(chain) is None
    assert half.calls == [(SYSTEM, PROMPT)]
    # control: the same link WITH the method is probed as structured
    assert schema_dialect_of(StructuredLink(), "strict") == "openai_strict"


def test_the_probe_never_raises_and_never_passes_an_answer_outside_the_alphabet():
    for answer in ("OPENAI_STRICT", "json", "refused", "", 1, True, ["openai_strict"], object()):
        link = StructuredLink(dialects={"open": answer})
        assert schema_dialect_of(link, "open") is None, answer
    assert schema_dialect_of(StructuredLink(dialects={"open": "chain"}), "open") == "chain"

    class Raising(StructuredLink):
        def schema_dialect(self, form):
            raise RuntimeError("boom")

    assert schema_dialect_of(Raising(), "open") is None
    for junk in (None, object(), "backend", 3):
        assert schema_dialect_of(junk, "open") is None
    assert schema_dialect_of(StructuredLink(), "open") == "openai_json_schema"    # control


def test_the_receipt_reader_is_closed_too():
    link = StructuredLink()
    for value in ("openai_strict", "openai_json_schema", "ollama_format", "refused"):
        link.last_schema_dialect = value
        assert schema_applied_of(link) == value
    for value in ("chain", "OPEN", "", None, 1, True, ["refused"]):
        link.last_schema_dialect = value
        assert schema_applied_of(link) is None, value
    assert schema_applied_of(object()) is None


@pytest.mark.parametrize("cls", [OpenAIBackend, OllamaBackend, FallbackBackend])
def test_the_form_has_no_default_anywhere(cls):
    """Forgetting the form fails at the call. It never picks one quietly."""
    dialect = inspect.signature(cls.schema_dialect).parameters["form"]
    assert dialect.default is inspect.Parameter.empty
    params = inspect.signature(cls.generate_structured).parameters
    for name in ("schema", "name", "form"):
        assert params[name].default is inspect.Parameter.empty, name
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    # and `generate` is the two-argument call it always was
    assert list(inspect.signature(cls.generate).parameters) == ["self", "system", "prompt"]


@pytest.mark.asyncio
async def test_forgetting_the_form_raises():
    backend, seen = openai_backend()
    with pytest.raises(TypeError):
        backend.schema_dialect()                                    # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply")  # type: ignore[call-arg]
    assert seen == []


# ══ OpenAI ════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("form,dialect,strict", [("open", "openai_json_schema", False),
                                                 ("strict", "openai_strict", True)])
async def test_openai_sends_the_schema_in_the_dialect_of_the_form(form, dialect, strict):
    backend, seen = openai_backend([_response('{"verdict": true}')])
    text, tin, tout = await backend.generate_structured(
        SYSTEM, PROMPT, schema=canonical(), name="reply", form=form)
    receipt = schema_applied_of(backend)
    assert (text, tin, tout) == ('{"verdict": true}', 10, 5)
    assert receipt == dialect
    assert len(seen) == 1
    assert seen[0]["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "reply", "strict": strict, "schema": to_dialect(canonical(), dialect)}}


def test_the_parameter_is_built_in_one_place_and_refuses_before_any_request():
    """``response_format_for`` is what ``generate_structured`` sends — the test above pins the
    request against ``to_dialect``; this pins the builder against the same, and its refusals."""
    built = openai_module.response_format_for(canonical(), "reply", "openai_strict")
    assert built == {"type": "json_schema", "json_schema": {
        "name": "reply", "strict": True, "schema": to_dialect(canonical(), "openai_strict")}}
    assert openai_module.response_format_for(canonical(), "reply", "openai_json_schema")[
        "json_schema"] == {"name": "reply", "strict": False, "schema": canonical()}
    for dialect in ("ollama_format", "chain", "", None):
        with pytest.raises(ValueError, match="not an OpenAI schema dialect"):
            openai_module.response_format_for(canonical(), "reply", dialect)
    for name in ("", "has space", "x" * 65, None, 3):
        with pytest.raises(ValueError, match="name"):
            openai_module.response_format_for(canonical(), name, "openai_strict")
    spoiled = canonical()
    spoiled["properties"]["note"]["pattern"] = "^a"
    with pytest.raises(ValueError, match="outside dialect"):
        openai_module.response_format_for(spoiled, "reply", "openai_strict")


@pytest.mark.asyncio
async def test_the_two_forms_are_two_different_requests():
    """If they were one request, a comparison of the forms would compare a thing with itself."""
    one, seen_one = openai_backend()
    two, seen_two = openai_backend()
    await one.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    await two.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert seen_one[0]["response_format"] != seen_two[0]["response_format"]
    assert seen_two[0]["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert "additionalProperties" not in seen_one[0]["response_format"]["json_schema"]["schema"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.6-reasoner", "o3-mini"])
@pytest.mark.parametrize("system", [SYSTEM, "Answer in prose."])
@pytest.mark.parametrize("form", SCHEMA_FORMS)
async def test_asking_for_a_schema_changes_nothing_but_response_format(model, system, form):
    """The schema rides OUTSIDE the prompt: not a byte of ``system`` or ``prompt`` moves, and
    neither does any other parameter of the request."""
    plain, seen_plain = openai_backend(model=model, temperature=0.0)
    structured, seen_structured = openai_backend(model=model, temperature=0.0)
    await plain.generate(system, PROMPT)
    await structured.generate_structured(system, PROMPT, schema=canonical(), name="reply", form=form)
    without = {k: v for k, v in seen_plain[0].items() if k != "response_format"}
    with_schema = {k: v for k, v in seen_structured[0].items() if k != "response_format"}
    assert with_schema == without
    assert list(with_schema) == list(without)
    assert seen_structured[0]["messages"] == [{"role": "system", "content": system},
                                              {"role": "user", "content": PROMPT}]
    assert seen_structured[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_generate_is_the_request_it_always_was():
    """``generate`` byte for byte: the literal requests, written out. (The same four are compared
    against the tree before this change by digest, outside the suite.)"""
    cases = [
        (dict(model="gpt-4o-mini", temperature=0.0), SYSTEM,
         {"model": "gpt-4o-mini", "max_tokens": 4096, "temperature": 0.0,
          "response_format": {"type": "json_object"}}),
        (dict(model="gpt-4o-mini"), "Answer in prose.", {"model": "gpt-4o-mini", "max_tokens": 4096}),
        (dict(model="gpt-5.6-reasoner", temperature=0.0, reasoning_effort="low"), SYSTEM,
         {"model": "gpt-5.6-reasoner", "max_completion_tokens": 4096, "reasoning_effort": "low"}),
        (dict(model="some-chat", base_url="https://compat.invalid/v1"), SYSTEM,
         {"model": "some-chat", "max_tokens": 4096}),
    ]
    for kwargs, system, expected in cases:
        backend, seen = openai_backend(**kwargs)
        await backend.generate(system, PROMPT)
        assert schema_applied_of(backend) is None
        assert seen == [{**expected, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": PROMPT}]}], kwargs


@pytest.mark.asyncio
async def test_a_backend_with_no_dialect_for_the_form_serves_the_call_as_generate():
    compatible, seen = openai_backend(model="some-chat", base_url="https://compat.invalid/v1")
    plain, seen_plain = openai_backend(model="some-chat", base_url="https://compat.invalid/v1")
    await plain.generate(SYSTEM, PROMPT)
    result = await compatible.generate_structured(
        SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(compatible) is None
    assert result == ("{}", 10, 5)
    assert seen == seen_plain and "response_format" not in seen[0]


@pytest.mark.asyncio
async def test_a_refused_form_is_never_served_by_the_other_one(monkeypatch):
    """Asked for ``strict`` on a model that only takes the open dialect: applied ``None`` — never
    the open dialect. A strict arm must not be measured as an open one."""
    monkeypatch.setitem(openai_module.SCHEMA_DIALECTS_REFUSED_BY_MODEL, "frugal-9",
                        frozenset({"openai_strict"}))
    backend, seen = openai_backend(model="frugal-9")
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(backend) is None
    assert seen[0]["response_format"] == {"type": "json_object"}        # today's request
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(backend) == "openai_json_schema"           # control: open still goes


# ── a schema the provider refuses ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_400_is_one_repeat_on_the_same_backend_without_the_schema(caplog):
    backend, seen = openai_backend([_Status(400), _response('{"ok": 1}')])
    plain, seen_plain = openai_backend()
    await plain.generate(SYSTEM, PROMPT)
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.openai"):
        result = await backend.generate_structured(
            SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(backend) == "refused"
    assert result == ('{"ok": 1}', 10, 5)
    assert len(seen) == 2
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[1] == seen_plain[0]                                     # the repeat is `generate`
    sha = schema_digest(to_dialect(canonical(), "openai_strict"))
    lines = [r.getMessage() for r in caplog.records if "schema_refused" in r.getMessage()]
    assert lines == [f"event=schema_refused provider=openai model=gpt-4o-mini "
                     f"dialect=openai_strict schema_sha={sha}"]
    everything = " ".join(r.getMessage() for r in caplog.records)
    assert "PROVIDER-ERROR-TEXT" not in everything and "additionalProperties" not in everything


@pytest.mark.asyncio
async def test_the_instance_remembers_the_refusal_for_fifteen_minutes():
    now = [50.0]
    backend, seen = openai_backend([_Status(400)])
    backend._schema_refusals = SchemaRefusals(clock=lambda: now[0])
    call = dict(schema=canonical(), name="reply", form="strict")
    await backend.generate_structured(SYSTEM, PROMPT, **call)
    assert len(seen) == 2

    await backend.generate_structured(SYSTEM, PROMPT, **call)           # inside the window
    assert schema_applied_of(backend) == "refused"
    assert len(seen) == 3 and seen[2]["response_format"] == {"type": "json_object"}

    other = canonical()
    other["properties"]["kind"]["enum"].append("GAMMA")                 # another schema: it goes
    await backend.generate_structured(SYSTEM, PROMPT, schema=other, name="reply", form="strict")
    assert schema_applied_of(backend) == "openai_strict" and len(seen) == 4

    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(backend) == "openai_json_schema" and len(seen) == 5     # another dialect

    now[0] += SCHEMA_REFUSAL_TTL_S                                      # the window closes
    await backend.generate_structured(SYSTEM, PROMPT, **call)
    assert schema_applied_of(backend) == "openai_strict"
    assert len(seen) == 6 and seen[5]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_a_400_that_repeats_without_the_schema_was_never_about_the_schema():
    backend, seen = openai_backend([_Status(400), _Status(400)])
    with pytest.raises(_Status):
        await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert len(seen) == 2
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(backend) == "openai_json_schema"           # nothing was remembered
    assert seen[2]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [_Status(429), _Status(500), _Status(503),
                                     TimeoutError("slow"), ConnectionError("down")])
async def test_a_429_a_timeout_or_a_5xx_is_not_swallowed(failure):
    """The control of the 400: these belong to the failover loop, and reach it."""
    backend, seen = openai_backend([failure])
    with pytest.raises(type(failure)):
        await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert len(seen) == 1                                               # no repeat
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(backend) == "openai_strict"                # and nothing remembered


@pytest.mark.asyncio
async def test_a_rejected_key_is_still_a_rejected_key():
    from cogno_synapse import InvalidAPIKeyError
    backend, seen = openai_backend([_Status(401)])
    with pytest.raises(InvalidAPIKeyError):
        await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("spoil", ["keyword", "name", "blank-name"])
async def test_a_schema_that_cannot_be_said_goes_as_refused_before_any_request(spoil, caplog):
    schema, name = canonical(), "reply"
    if spoil == "keyword":
        schema["properties"]["note"]["maxLength"] = 9
    elif spoil == "name":
        name = "a name with spaces"
    else:
        name = ""
    backend, seen = openai_backend()
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.openai"):
        await backend.generate_structured(SYSTEM, PROMPT, schema=schema, name=name, form="strict")
    assert schema_applied_of(backend) == "refused"
    assert len(seen) == 1 and seen[0]["response_format"] == {"type": "json_object"}
    assert any("event=schema_unconvertible" in r.getMessage() for r in caplog.records)
    # control: the unspoiled call sends the schema
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(backend) == "openai_strict"


@pytest.mark.asyncio
async def test_the_models_own_refusal_is_an_empty_reply_and_a_countable_line(caplog):
    backend, seen = openai_backend([_response(content=None, refusal="I can't help with that — SECRET")])
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.openai"):
        text, _, _ = await backend.generate_structured(
            SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert text == ""
    assert schema_applied_of(backend) == "openai_strict"                # the schema DID travel
    lines = [r.getMessage() for r in caplog.records if "model_refusal" in r.getMessage()]
    assert lines == ["event=model_refusal provider=openai model=gpt-4o-mini"]
    assert "SECRET" not in " ".join(r.getMessage() for r in caplog.records)
    # control: an ordinary reply logs no such line
    caplog.clear()
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert not [r for r in caplog.records if "model_refusal" in r.getMessage()]


@pytest.mark.asyncio
async def test_a_plain_call_is_never_read_with_the_receipt_of_the_structured_call_before_it():
    backend, _ = openai_backend()
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(backend) == "openai_json_schema"           # the presence
    await backend.generate(SYSTEM, PROMPT)
    assert schema_applied_of(backend) is None
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    await backend.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert schema_applied_of(backend) is None


@pytest.mark.asyncio
async def test_two_concurrent_calls_on_one_openai_instance_never_swap_receipts():
    """The leaf is shared before any chain is. One call is held at the provider while the other
    starts and finishes on the same instance — in both orders. The client's ``close`` suspends
    (the double's does, as a real one does), so a receipt written before it would be overwritten."""
    for held_structured in (True, False):
        gate = {0: asyncio.Event()}
        backend, seen = openai_backend(gate=gate)
        receipts: dict = {}

        async def structured():
            await backend.generate_structured(SYSTEM, "S", schema=canonical(), name="reply", form="open")
            receipts["structured"] = schema_applied_of(backend)        # right after the await

        async def plain():
            await backend.generate(SYSTEM, "P")
            receipts["plain"] = schema_applied_of(backend)

        first, second = (structured, plain) if held_structured else (plain, structured)
        held = asyncio.ensure_future(first())
        await asyncio.sleep(0)
        await second()
        gate[0].set()
        await held
        assert receipts == {"structured": "openai_json_schema", "plain": None}, held_structured
        kinds = {kw["messages"][1]["content"]: kw["response_format"]["type"] for kw in seen}
        assert kinds == {"S": "json_schema", "P": "json_object"}, held_structured


# ══ Ollama ════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ollama_sends_the_schema_in_the_format_of_the_call_and_not_on_the_instance(monkeypatch):
    backend, seen = ollama_backend(monkeypatch, format="json", temperature=0.0)
    await backend.generate(SYSTEM, PROMPT)
    result = await backend.generate_structured(
        SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(backend) == "ollama_format"
    assert result == ("{}", 7, 3)
    assert backend.format == "json", "the instance's format was touched"
    assert seen[1]["format"] == canonical() == to_dialect(canonical(), "ollama_format")
    assert seen[0]["format"] == "json"
    assert {k: v for k, v in seen[1].items() if k != "format"} == \
           {k: v for k, v in seen[0].items() if k != "format"}
    assert list(seen[1]) == list(seen[0])
    await backend.generate(SYSTEM, PROMPT)                              # and the next plain call…
    assert seen[2] == seen[0]                                           # …is today's, byte for byte
    assert schema_applied_of(backend) is None


@pytest.mark.asyncio
async def test_ollama_generate_is_the_request_it_always_was(monkeypatch):
    backend, seen = ollama_backend(monkeypatch, format="json", temperature=0.0)
    await backend.generate(SYSTEM, PROMPT)
    assert seen == [{"model": "local-model", "system": SYSTEM, "prompt": PROMPT, "stream": False,
                     "format": "json", "think": False,
                     "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 4096}}]
    bare, seen_bare = ollama_backend(monkeypatch)
    await bare.generate(SYSTEM, PROMPT)
    assert seen_bare == [{"model": "local-model", "system": SYSTEM, "prompt": PROMPT,
                          "stream": False, "think": False,
                          "options": {"num_ctx": 8192, "num_predict": 4096}}]


@pytest.mark.asyncio
async def test_ollama_asked_for_strict_serves_the_call_as_generate(monkeypatch):
    """The twin of 'no demotion': applied ``None`` — never ``ollama_format``."""
    backend, seen = ollama_backend(monkeypatch, format="json")
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(backend) is None
    assert seen[0]["format"] == "json"
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(backend) == "ollama_format"                # control
    assert seen[1]["format"] == canonical()


@pytest.mark.asyncio
async def test_two_concurrent_calls_on_one_ollama_instance_each_carry_their_own_format(monkeypatch):
    """One instance serves several stages at once. The structured call is held open while a plain
    call starts AND finishes on the same instance; then the roles are swapped."""
    for held_structured in (True, False):
        gate = {0: asyncio.Event()}
        backend, seen = ollama_backend(monkeypatch, gate=gate, format="json")
        receipts: dict = {}

        async def structured():
            await backend.generate_structured(SYSTEM, "S", schema=canonical(), name="reply", form="open")
            receipts["structured"] = schema_applied_of(backend)        # right after the await

        async def plain():
            await backend.generate(SYSTEM, "P")
            receipts["plain"] = schema_applied_of(backend)

        first, second = (structured, plain) if held_structured else (plain, structured)
        held = asyncio.ensure_future(first())
        await asyncio.sleep(0)                                          # the held call is in flight
        await second()
        gate[0].set()
        await held
        by_prompt = {payload["prompt"]: payload["format"] for payload in seen}
        assert by_prompt == {"S": canonical(), "P": "json"}, held_structured
        assert receipts == {"structured": "ollama_format", "plain": None}, held_structured
        assert backend.format == "json"


@pytest.mark.asyncio
async def test_ollama_a_400_repeats_without_the_schema_and_a_5xx_does_not(monkeypatch, caplog):
    backend, seen = ollama_backend(monkeypatch, [400, 200], format="json")
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.ollama"):
        await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(backend) == "refused"
    assert [payload["format"] for payload in seen] == [canonical(), "json"]
    assert [r.getMessage() for r in caplog.records if "schema_refused" in r.getMessage()] == [
        f"event=schema_refused provider=ollama model=local-model dialect=ollama_format "
        f"schema_sha={schema_digest(canonical())}"]
    await backend.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert len(seen) == 3 and seen[2]["format"] == "json"               # remembered

    for status in (429, 500, 503):                                      # the control
        other, seen_other = ollama_backend(monkeypatch, [status], format="json")
        with pytest.raises(httpx.HTTPStatusError):
            await other.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
        assert len(seen_other) == 1


# ══ the chain ═════════════════════════════════════════════════════════════════════════════════

def test_a_chain_announces_chain_when_any_link_answers_that_form():
    openai, ollama, plain = OpenAIBackend(api_key="k"), OllamaBackend("m"), PlainLink()
    assert FallbackBackend([plain, openai]).schema_dialect("strict") == "chain"     # ANY, not all
    assert FallbackBackend([plain, ollama]).schema_dialect("open") == "chain"
    assert FallbackBackend([plain, ollama]).schema_dialect("strict") is None        # per FORM
    assert FallbackBackend([plain, AnthropicBackend(api_key="k")]).schema_dialect("open") is None
    assert FallbackBackend([FallbackBackend([plain]), plain]).schema_dialect("open") is None
    assert FallbackBackend([FallbackBackend([openai])]).schema_dialect("open") == "chain"
    for form in (None, "", "closed"):
        assert FallbackBackend([openai]).schema_dialect(form) is None


@pytest.mark.asyncio
async def test_a_link_that_takes_the_schema_gets_it_exactly_as_it_arrived():
    link = StructuredLink()
    chain = FallbackBackend([link])
    schema = canonical()
    result = await chain.generate_structured(SYSTEM, PROMPT, schema=schema, name="reply", form="strict")
    receipt = schema_applied_of(chain)
    assert result == ("leaf-structured", 2, 2) and receipt == "openai_strict"
    assert link.plain == []
    assert link.structured == [{"system": SYSTEM, "prompt": PROMPT, "schema": canonical(),
                                "name": "reply", "form": "strict"}]
    assert link.structured[0]["schema"] is schema                       # untouched: no conversion here


@pytest.mark.asyncio
async def test_mixed_chain_the_first_link_fails_and_a_link_without_support_serves():
    first = StructuredLink("first", exc=_Status(429))
    second = PlainLink("second")
    chain = FallbackBackend([first, second])
    result = await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    receipt = schema_applied_of(chain)
    assert result == ("plain", 1, 1)
    assert receipt is None                                              # nobody carried a schema
    assert second.calls == [(SYSTEM, PROMPT)]                           # two arguments, no keyword
    assert len(first.structured) == 1 and chain.model == "second"
    # control: when the first link serves, the receipt is its own
    healthy = FallbackBackend([StructuredLink("first"), PlainLink("second")])
    await healthy.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(healthy) == "openai_json_schema"


@pytest.mark.asyncio
async def test_a_link_with_no_dialect_for_the_form_is_called_as_plain_generate():
    """Strict asked of a chain whose first link only answers open: it serves through ``generate``
    and the receipt is ``None`` — never its open dialect."""
    open_only = StructuredLink("open-only", dialects={"open": "ollama_format"})
    chain = FallbackBackend([open_only, StructuredLink("full")])
    assert chain.schema_dialect("strict") == "chain"
    await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(chain) is None
    assert open_only.plain == [(SYSTEM, PROMPT)] and open_only.structured == []
    await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(chain) == "ollama_format"                  # control


@pytest.mark.asyncio
async def test_chain_inside_a_chain_only_the_leaf_converts_and_the_receipt_crosses_both(monkeypatch):
    conversions: list = []
    real = openai_module.to_dialect

    def counting(schema, dialect):
        conversions.append(dialect)
        return real(schema, dialect)

    monkeypatch.setattr(openai_module, "to_dialect", counting)
    leaf, seen = openai_backend()
    outer = FallbackBackend([FallbackBackend([leaf])])
    assert outer.schema_dialect("strict") == "chain"
    await outer.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    receipt = schema_applied_of(outer)
    assert receipt == "openai_strict"
    assert conversions == ["openai_strict"]                             # once, at the leaf
    assert seen[0]["response_format"]["json_schema"]["schema"] == real(canonical(), "openai_strict")


@pytest.mark.asyncio
async def test_a_nested_mixed_chain_hands_each_leaf_the_canonical_schema(monkeypatch):
    """Were a chain to convert for one leaf, the next leaf would be handed that leaf's shape. The
    OpenAI leaf fails over to an Ollama leaf, and the Ollama request carries the schema AS IT IS."""
    first, seen_first = openai_backend([_Status(503)])
    second, seen_second = ollama_backend(monkeypatch, format="json")
    outer = FallbackBackend([FallbackBackend([first]), FallbackBackend([second])])
    await outer.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(outer) == "ollama_format"
    assert seen_second[0]["format"] == canonical()
    assert seen_first[0]["response_format"]["json_schema"]["schema"] == canonical()


@pytest.mark.asyncio
async def test_a_400_inside_a_link_spends_no_retry_opens_no_breaker_and_uses_one_link():
    leaf, seen = openai_backend([_Status(400)])
    spare = PlainLink("spare")
    breaker, sink = CircuitBreaker(fail_threshold=1), Counting()
    chain = FallbackBackend([leaf, spare], breaker=breaker,
                            policy=RetryPolicy(max_retries=2, base_ms=0.0, jitter=False), metrics=sink)
    await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(chain) == "refused"
    assert len(seen) == 2 and spare.calls == []                         # one link, two requests
    assert sink.records == [("gpt-4o-mini", True)]                      # ONE attempt, a success
    assert breaker.store.get("gpt-4o-mini").failures == 0
    assert not breaker.is_open("gpt-4o-mini")
    assert chain.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_a_429_inside_a_link_is_ordinary_failover():
    """The control: same chain, a 429 instead of the 400."""
    leaf, seen = openai_backend([_Status(429)])
    spare = StructuredLink("spare", dialects={"strict": "openai_strict"})
    breaker, sink = CircuitBreaker(fail_threshold=1), Counting()
    chain = FallbackBackend([leaf, spare], breaker=breaker, metrics=sink)
    await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    assert schema_applied_of(chain) == "openai_strict"                  # the link that served
    assert len(seen) == 1 and len(spare.structured) == 1
    assert sink.records == [("gpt-4o-mini", False), ("spare", True)]
    assert breaker.is_open("gpt-4o-mini")
    assert chain.model == "spare"


@pytest.mark.asyncio
async def test_the_chains_generate_is_the_call_it_always_was_and_clears_the_receipt():
    link = StructuredLink()
    chain = FallbackBackend([link, PlainLink()])
    await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="open")
    assert schema_applied_of(chain) == "openai_json_schema"             # the presence
    assert await chain.generate(SYSTEM, PROMPT) == ("leaf-plain", 1, 1)
    assert schema_applied_of(chain) is None
    assert link.plain == [(SYSTEM, PROMPT)] and len(link.structured) == 1


@pytest.mark.asyncio
async def test_concurrent_calls_on_one_chain_and_one_leaf_never_swap_receipts():
    """A chain and its leaf are both shared. One call is held at the provider while another starts
    and finishes on the same objects; each receipt is read on the line after its own await."""
    for held_structured in (True, False):
        gate = {0: asyncio.Event()}
        leaf, seen = openai_backend(gate=gate)
        chain = FallbackBackend([FallbackBackend([leaf])])
        receipts: dict = {}

        async def structured():
            await chain.generate_structured("S " + SYSTEM, PROMPT, schema=canonical(),
                                            name="reply", form="strict")
            receipts["structured"] = schema_applied_of(chain)

        async def plain():
            await chain.generate("P " + SYSTEM, PROMPT)
            receipts["plain"] = schema_applied_of(chain)

        first, second = (structured, plain) if held_structured else (plain, structured)
        held = asyncio.ensure_future(first())
        await asyncio.sleep(0)
        await second()
        gate[0].set()
        await held
        assert receipts == {"structured": "openai_strict", "plain": None}, held_structured
        kinds = {kw["messages"][0]["content"][0]: kw["response_format"]["type"] for kw in seen}
        assert kinds == {"S": "json_schema", "P": "json_object"}, held_structured


@pytest.mark.asyncio
async def test_two_calls_that_finish_in_the_same_loop_turn_never_swap_receipts():
    """Both calls are held at the provider and released TOGETHER, so they walk back up the chain
    in lock-step, one statement of each per loop turn. A layer that put any suspension between
    its link's ``await`` and reading the link's receipt would read the OTHER call's here."""
    for structured_first in (True, False):
        gate = {0: asyncio.Event(), 1: asyncio.Event()}
        leaf, _ = openai_backend(gate=gate)
        chain = FallbackBackend([FallbackBackend([leaf])])
        receipts: dict = {}

        async def structured():
            await chain.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
            receipts["structured"] = schema_applied_of(chain)

        async def plain():
            await chain.generate(SYSTEM, PROMPT)
            receipts["plain"] = schema_applied_of(chain)

        order = (structured, plain) if structured_first else (plain, structured)
        tasks = [asyncio.ensure_future(call()) for call in order]
        await asyncio.sleep(0)                                          # both are at the provider
        gate[0].set()
        gate[1].set()
        await asyncio.gather(*tasks)
        assert receipts == {"structured": "openai_strict", "plain": None}, structured_first


@pytest.mark.asyncio
async def test_a_receipt_read_after_another_await_is_not_this_calls_receipt():
    """The limit, shown rather than asserted in prose: read late, the value is somebody else's.
    This is why every reader takes it on the line after its own await."""
    leaf, _ = openai_backend()
    await leaf.generate_structured(SYSTEM, PROMPT, schema=canonical(), name="reply", form="strict")
    await leaf.generate(SYSTEM, PROMPT)                                 # "another await"
    assert schema_applied_of(leaf) is None                              # not the structured call's
