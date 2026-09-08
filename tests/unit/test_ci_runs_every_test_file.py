"""Every versioned test file must sit UNDER a path that some CI job hands to pytest.

A test file no workflow reaches is not a weaker test — it is not a test. It was written,
reviewed, merged, and it has never executed anywhere automated; it is green only in the
reviewer's memory. That state is not hypothetical in this ecosystem: ``cogno-praxis`` carried a
bookkeeper suite in exactly it until its #90, and a sweep on 2026-09-02 found ``cogno-anima``
holding SIX files and 56 node ids the same way — every file of its ``tests/integration`` except
``test_noumeno.py``.

The mechanism is never exotic: CI enumerates the CHILDREN of ``tests/`` by name
(``pytest tests/unit``, ``pytest tests/integration``), so the day a third child appears the
workflow keeps passing and the new directory runs nowhere. This repo shipped exactly those two
lines until the commit that adds this file, and fifteen of the eighteen repos in the ecosystem
are one new folder away from the same hole. The change landing beside this guard removes the
class instead of watching it: the two steps became one ``pytest tests``, so CI names no child of
``tests/`` at all — the shape ``cogno-engram`` has been immune by all along. But immunity that
lives in the current text of a workflow is a rule someone has to remember. This is the rule that
runs.

**The invariant: the path CI invokes is an ANCESTOR of every test file it should collect.**

Skipping is not running, and in a log the two look almost alike — which is why this repo in
particular needs the structural answer. Its unit suite mocks every SDK and HTTP client; its
integration suite is gated on real provider keys and a reachable Ollama, so on CI it collects
and skips. "19 skipped" and "never invoked" print about the same amount of nothing. So the
question *is this file reached* is answered here from the workflow and ``git ls-files``, never
from a green tick.

Four things this had to get right, each of them a way the check could have been born useless:

* it enumerates with ``git ls-files``, not a walk of the disk. An uncommitted file is not a
  contract, and a scratch copy under ``tests/`` is not something CI ever promised to run;
* a CONDITIONAL job still counts as an invocation. A suite gated on ``schedule ||
  workflow_dispatch`` — because its assertions need a live model, and sampling a real model must
  not decide whether main is green — IS invoked; "only nightly" is a real answer, not a
  violation. That answer is empty in this repo today, and WHICH paths are conditional-only is a
  second, weaker, separate assertion at the bottom of this file rather than a silent default;
* the option table is the false-GREEN surface. ``pytest --rootdir tests`` collects nothing named
  ``tests``, but a parser that does not know ``--rootdir`` takes a value reads the next token as
  a path and pronounces the whole suite covered. Unknown flags are therefore assumed to take NO
  value (the ``-q``/``-x``/``-s`` case) and every flag that could swallow a path-shaped token is
  listed in ``_TAKES_VALUE`` and pinned one-by-one by a test below;
* ``testpaths`` is read from pytest itself, not re-parsed out of ``pyproject.toml``. A second
  parser is a second answer, and ``tomllib`` does not exist on the 3.10 leg of this matrix.

Parsed with a real YAML parser and a real shell lexer. A regex over the workflow text would have
to re-implement both, and every mistake it makes prints GREEN.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Paths whose ONLY invocation comes from a conditional job or step. Empty here, and that is a
# measurement rather than a default: the single workflow that runs pytest triggers on push +
# pull_request, its one `test` job carries no `if:`, and neither does any of its steps — so
# every versioned test file gates a pull request. A non-empty entry (the live-model canary other
# repos in the ecosystem run nightly) is a DECISION; making one here has to mean editing this
# line, which is the entire reason it is written down instead of inferred.
NIGHTLY_ONLY: "frozenset[str]" = frozenset()

# Options that consume the NEXT token. A missing entry here is the one way this guard goes
# falsely green: the swallowed token reads as a collection path, and `--rootdir tests` would
# then claim the whole suite is invoked. The `--flag=value` spellings need no entry.
_TAKES_VALUE = frozenset({
    "-k", "-m", "-p", "-n", "-c", "-o", "-W", "-r", "--maxfail", "--rootdir", "--deselect",
    "--ignore", "--ignore-glob", "--confcutdir", "--import-mode", "--basetemp", "--junitxml",
    "--junit-xml", "--log-file", "--numprocesses", "--dist", "--cov", "--cov-report",
    "--cov-config", "--timeout", "--durations", "--tb", "--color", "--capture",
})
# Shell tokens that end one command and begin another.
_SEPARATORS = frozenset({"|", "||", "&", "&&", ";", "(", ")", "<", ">", ">>", "|&", "&>"})
# Wrappers that may sit in front of `pytest` on the same command line.
_WRAPPERS = frozenset({"sudo", "env", "time", "nice", "xvfb-run", "poetry", "uv", "hatch",
                       "pdm", "rye", "run", "coverage", "python", "python3", "-m"})
_PYTEST = frozenset({"pytest", "py.test"})


# ── reading the workflows ─────────────────────────────────────────────────────────────
def _strip_heredocs(text: str) -> str:
    """Drop ``<<EOF ... EOF`` bodies. No workflow here embeds one today; the day a probe script
    arrives, its lines are not shell at all, and lexing them is noise at best and an unbalanced
    quote at worst — an exception that would take this guard down with it."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        if "<<" not in line:
            continue
        tail = line.split("<<", 1)[1].lstrip("-").strip()
        if not tail:
            continue
        marker = tail.split()[0].strip("\"'")
        while i < len(lines) and lines[i].strip() != marker:
            i += 1
        i += 1                                    # the terminator line itself
    return "\n".join(out)


def _commands(run_text: str) -> "list[list[str]]":
    """A ``run:`` block, split into argv lists. A newline separates commands as surely as a
    ``;`` does, so the ``pytest`` line cannot absorb the arguments of the
    ``ruff check cogno_synapse tests examples`` that sits two steps away."""
    text = _strip_heredocs(run_text).replace("\\\n", " ")
    cmds: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        try:
            tokens = list(lex)
        except ValueError:
            # Only silence a line that cannot be a pytest invocation at all — a guard that
            # quietly skips its own subject is the state it exists to end.
            assert "pytest" not in line, f"unparseable run: line that mentions pytest: {line!r}"
            continue
        current: list[str] = []
        for token in tokens:
            if token in _SEPARATORS:
                if current:
                    cmds.append(current)
                current = []
            else:
                current.append(token)
        if current:
            cmds.append(current)
    return cmds


def _pytest_argv(cmd: "list[str]") -> "list[str] | None":
    """``cmd``'s arguments if it invokes pytest, else None. Steps over ``VAR=value`` prefixes
    and the ``python -m`` / ``poetry run`` / ``coverage run -m`` wrappers."""
    i = 0
    while i < len(cmd) and cmd[i] not in _PYTEST and (
            cmd[i] in _WRAPPERS or ("=" in cmd[i] and not cmd[i].startswith("-"))):
        i += 1
    if i >= len(cmd) or cmd[i] not in _PYTEST:
        return None
    return cmd[i + 1:]


def _paths(argv: "list[str]") -> "list[str]":
    """The positional collection arguments, normalised repo-relative. An empty result means
    pytest was handed no path and falls back to ``testpaths``."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = token in _TAKES_VALUE
            continue
        if token.isdigit():
            continue                              # a file descriptor from `2>&1`, not a path
        out.append(_norm(token))
    return out


def _norm(raw: str) -> str:
    path = raw.split("::", 1)[0]                  # a node id points at its file
    path = path.removeprefix("./").rstrip("/")
    return str(PurePosixPath(path)) if path else "."


def _fallback_scope(pytestconfig) -> "list[str]":
    """What a bare ``pytest`` collects — asked of pytest, which owns the semantics, instead of
    re-parsed here. With no ``testpaths`` configured pytest collects from the rootdir, and a
    rootdir covers everything."""
    configured = [_norm(str(p)) for p in (pytestconfig.getini("testpaths") or [])]
    return configured or ["."]


def _triggers(doc: dict) -> "set[str]":
    """YAML 1.1 reads a bare ``on:`` as the boolean True, so both spellings must be tried or
    every workflow looks untriggered and every job looks conditional."""
    on = doc.get("on", doc.get(True))
    if isinstance(on, (dict, list)):
        return {str(k) for k in on}
    return {str(on)} if on else set()


def invocations(fallback: "list[str]") -> "list[tuple[str, str, bool]]":
    """``(path, "workflow:job", pr_gated)`` for every path any workflow hands to pytest.

    ``pr_gated`` is deliberately conservative: True only when the workflow triggers on
    ``pull_request`` AND neither the job nor the step carries an ``if:``. A condition this
    cannot evaluate is read as conditional, so the answer can only ever understate the gate —
    never invent one."""
    found: list[tuple[str, str, bool]] = []
    for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
        doc = yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
        on_pull_request = "pull_request" in _triggers(doc)
        for job_name, job in (doc.get("jobs") or {}).items():
            job_open = on_pull_request and not (job or {}).get("if")
            for step in ((job or {}).get("steps") or []):
                if not isinstance(step, dict) or not step.get("run"):
                    continue
                gated = job_open and not step.get("if")
                for cmd in _commands(step["run"]):
                    argv = _pytest_argv(cmd)
                    if argv is None:
                        continue
                    for path in (_paths(argv) or fallback):
                        found.append((path, f"{workflow.name}:{job_name}", gated))
    return found


# ── reading the repo ──────────────────────────────────────────────────────────────────
def versioned_test_files() -> "list[str]":
    """Committed test modules. ``git ls-files``, never a walk of the working tree: a file that
    is not committed is not a contract, and a leftover copy under ``tests/`` is not something
    anybody promised CI would run."""
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True)
    return sorted(p for p in proc.stdout.split("\0")
                  if p and _is_test_module(PurePosixPath(p).name))


def _is_test_module(name: str) -> bool:
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _covers(path: str, file: str) -> bool:
    return path in (".", "") or file == path or file.startswith(path + "/")


# ── guards on the guard ───────────────────────────────────────────────────────────────
def test_the_workflows_parse_and_invoke_pytest_at_all(pytestconfig):
    """Everything below is vacuous if the directory moved, the YAML changed shape, or the
    lexer quietly stopped recognising a pytest line."""
    assert WORKFLOWS.is_dir(), WORKFLOWS
    found = invocations(_fallback_scope(pytestconfig))
    assert found, "no workflow invokes pytest — this guard would pass over an empty set"
    assert versioned_test_files(), "git ls-files found no test modules — the guard has no subject"


def test_a_bare_pytest_means_testpaths_and_not_nothing(pytestconfig):
    """The ``cogno-engram`` shape, and the direction this repo just moved in. Reading
    ``pytest -q`` as invoking no path would make the only form that is immune by construction
    look exactly like the defect."""
    assert _paths(_pytest_argv(["pytest", "-q"])) == []
    assert _fallback_scope(pytestconfig) == ["tests"]


def test_the_lexer_reads_the_forms_this_repo_actually_uses():
    cases = {
        # the one pytest line ci.yml contains after the step collapse
        "pytest tests -q --cov=cogno_synapse --cov-report=term-missing": ["tests"],
        # the two lines it contained until then — the shape this guard exists to notice
        "pytest tests/unit -q --cov=cogno_synapse --cov-report=term-missing": ["tests/unit"],
        "pytest tests/integration -q": ["tests/integration"],
        # spellings a later edit could reach for
        "python -m pytest tests -q": ["tests"],
        "pytest ./tests/unit/ -q": ["tests/unit"],
        "pytest tests/unit/test_providers.py::test_openai_prefix": ["tests/unit/test_providers.py"],
        'pytest tests -q -m "not live"': ["tests"],
        "pytest tests/integration -q || [ $? -eq 5 ]": ["tests/integration"],
        "COGNO_TEST_OLLAMA_MODEL=qwen3:8b pytest tests -q": ["tests"],
        "pytest tests -q  # the pull-request gate": ["tests"],
    }
    for line, want in cases.items():
        cmds = [c for c in _commands(line) if _pytest_argv(c) is not None]
        assert len(cmds) == 1, (line, cmds)
        assert _paths(_pytest_argv(cmds[0])) == want, line


def test_an_option_value_is_never_mistaken_for_a_collection_path():
    """The false-GREEN surface, one line per flag that could swallow a path-shaped token.
    ``--rootdir tests`` collects nothing; a parser that thinks otherwise declares every file
    covered and this guard silently stops working."""
    for line in ("pytest --rootdir tests tests/unit",
                 "pytest -k tests tests/unit",
                 "pytest -m tests tests/unit",
                 "pytest -p no:cacheprovider tests/unit",
                 "pytest --ignore tests/integration tests/unit",
                 "pytest --deselect tests/integration tests/unit",
                 "pytest -c tests/pytest.ini tests/unit",
                 "pytest -o testpaths=tests tests/unit",
                 "pytest --cov cogno_synapse tests/unit",
                 "pytest -n 4 tests/unit"):
        assert _paths(_pytest_argv(_commands(line)[0])) == ["tests/unit"], line


def test_a_neighbouring_command_does_not_donate_its_arguments():
    """``ruff check cogno_synapse tests examples`` names ``tests``, and it is not a pytest path;
    ``[ $? -eq 5 ]`` after a ``||`` is not one either. Either would silently widen what looks
    invoked — and the ruff line is the more dangerous of the two, because the word it donates is
    exactly the word that would make this whole file pass."""
    block = ("pytest tests/unit -q --cov=cogno_synapse --cov-report=term-missing\n"
             "ruff check cogno_synapse tests examples\n"
             "mypy cogno_synapse")
    paths = [p for c in _commands(block)
             if (argv := _pytest_argv(c)) is not None for p in _paths(argv)]
    assert paths == ["tests/unit"], paths


def test_a_heredoc_body_is_not_shell():
    """No workflow here embeds a probe script today, so this pins the stripper against the day
    one arrives — apostrophes, path-shaped words and all."""
    block = ("python - <<'EOF'\n"
             "import os  # pytest tests, but this isn't a command\n"
             "print(\"the installed synapse doesn't honour the timeout\")\n"
             "EOF\n"
             "pytest tests/integration -q")
    cmds = [c for c in _commands(block) if _pytest_argv(c) is not None]
    assert len(cmds) == 1, cmds
    assert _paths(_pytest_argv(cmds[0])) == ["tests/integration"]


def test_the_invariant_would_notice_an_uninvoked_file(pytestconfig):
    """Mutation. Without this, a ``_covers`` that answered True unconditionally — or a
    ``versioned_test_files`` that returned nothing interesting — would satisfy every other
    assertion in the file. The orphan is deliberately ``examples/``: it is versioned, it is
    linted by the same ruff line, and it is the one place a test module can still land where
    even the collapsed ``pytest tests`` will not reach it."""
    invoked = {p for p, _, _ in invocations(_fallback_scope(pytestconfig))}
    orphan = "examples/test_never_run.py"
    assert not any(_covers(p, orphan) for p in invoked), sorted(invoked)
    assert _covers("tests", "tests/integration/test_ollama_live.py")
    assert _covers(".", "tests/integration/test_ollama_live.py")
    assert not _covers("tests/integration", "tests/integration_helpers/test_x.py")
    assert _is_test_module("test_x.py") and _is_test_module("x_test.py")
    assert not _is_test_module("conftest.py") and not _is_test_module("test_data.json")


# ── the invariant ─────────────────────────────────────────────────────────────────────
def test_every_versioned_test_file_is_under_a_path_ci_invokes(pytestconfig):
    invoked = {p for p, _, _ in invocations(_fallback_scope(pytestconfig))}
    orphans = [f for f in versioned_test_files() if not any(_covers(p, f) for p in invoked)]
    assert not orphans, (
        "these test files are committed but no CI job ever collects them. They pass in review "
        "and have never run anywhere automated. Widen an invoked path — or hand pytest the "
        "root and let `testpaths` do the scoping, which is the form that cannot regress — "
        "rather than deleting them.\n"
        f"  invoked: {sorted(invoked)}\n"
        f"  orphaned ({len(orphans)}): {orphans}")


def test_which_paths_run_only_on_a_conditional_job_is_a_decision_not_a_drift(pytestconfig):
    """The weaker, separate half, and the reason the assertion above does not simply demand a
    pull-request gate for everything. A path reached only by a nightly IS invoked, and that is
    right wherever the assertions depend on a live model: sampling a real model must not decide
    whether main is green. But it is a DECISION, so it is written down in ``NIGHTLY_ONLY`` and
    changing it has to mean changing that line. Here the set is empty — every test file this
    repo versions gates a pull request — and this is what would notice that stopping."""
    found = invocations(_fallback_scope(pytestconfig))
    gated = {p for p, _, is_pr in found if is_pr}
    conditional = {p for p, _, _ in found} - gated
    assert conditional == NIGHTLY_ONLY, (
        "the set of paths only a conditional job invokes has changed. A path that moved OUT of "
        "the pull-request gate is a suite quietly leaving CI; one that moved IN needs "
        f"NIGHTLY_ONLY updated.\n  now: {sorted(conditional)}\n"
        f"  declared: {sorted(NIGHTLY_ONLY)}")
