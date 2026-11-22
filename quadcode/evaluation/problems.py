"""Benchmark problems: a statement the agents see, and hidden tests they never see.

Layout of one problem directory::

    <problems_dir>/<id>/problem.md          statement given to the pipeline, nothing else
    <problems_dir>/<id>/hidden_tests.json   test cases used only to score finished runs
    <problems_dir>/<id>/meta.yaml           entry_point, language, optional mock_scenario
    <problems_dir>/<id>/reference.py        hand-written solution, used only by pytest to check the tests
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from quadcode.cache import sha256_text
from quadcode.schemas import TestCase
from quadcode.testing import load_test_cases

PROBLEM_FILE = "problem.md"
HIDDEN_FILE = "hidden_tests.json"
META_FILE = "meta.yaml"
REFERENCE_FILE = "reference.py"
_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class ProblemError(ValueError):
    """A benchmark problem directory is missing or inconsistent."""


class ProblemMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_point: str
    language: str = "python"
    mock_scenario: str | None = None
    source: str = ""


@dataclass(frozen=True)
class BenchProblem:
    id: str
    directory: Path
    statement: str
    meta: ProblemMeta
    hidden_tests: tuple[TestCase, ...]

    @property
    def entry_point(self) -> str:
        return self.meta.entry_point

    @property
    def statement_sha256(self) -> str:
        return sha256_text(self.statement)

    @property
    def hidden_sha256(self) -> str:
        return sha256_text((self.directory / HIDDEN_FILE).read_text(encoding="utf-8"))


def load_problem(directory: Path) -> BenchProblem:
    pid = directory.name
    if not _ID.match(pid):
        raise ProblemError(f"{directory}: problem ids are lower-case letters, digits and underscores")
    for name in (PROBLEM_FILE, HIDDEN_FILE, META_FILE):
        if not (directory / name).is_file():
            raise ProblemError(f"{pid}: missing {name}")
    statement = (directory / PROBLEM_FILE).read_text(encoding="utf-8")
    try:
        meta = ProblemMeta.model_validate(yaml.safe_load((directory / META_FILE).read_text(encoding="utf-8")) or {})
    except (ValidationError, yaml.YAMLError) as exc:
        raise ProblemError(f"{pid}: invalid {META_FILE}: {exc}") from exc
    if not re.search(rf"\b{re.escape(meta.entry_point)}\b", statement):
        raise ProblemError(f"{pid}: the statement never names the entry point '{meta.entry_point}', so agents "
                           "could not know which function the hidden tests call")
    try:
        tests = load_test_cases(directory / HIDDEN_FILE)
    except (ValidationError, ValueError) as exc:
        raise ProblemError(f"{pid}: invalid {HIDDEN_FILE}: {exc}") from exc
    if not tests:
        raise ProblemError(f"{pid}: {HIDDEN_FILE} holds no tests")
    ids = [t.id for t in tests]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ProblemError(f"{pid}: duplicate hidden test ids {dupes}")
    not_hidden = [t.id for t in tests if not t.hidden]
    if not_hidden:
        raise ProblemError(f"{pid}: tests not marked hidden: {not_hidden}")
    return BenchProblem(id=pid, directory=directory, statement=statement, meta=meta, hidden_tests=tuple(tests))


def available_ids(root: Path) -> list[str]:
    if not root.is_dir():
        raise ProblemError(f"no benchmark directory {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / PROBLEM_FILE).is_file())


def load_problems(root: Path, names: Sequence[str] | None = None, limit: int | None = None,
                  mock: bool = False) -> list[BenchProblem]:
    """Named problems in the given order, else the first ``limit`` in directory order.

    In mock mode only problems with a ``mock_scenario`` can run (mock answers are scripted per scenario
    and ignore the statement); naming one without a scenario is an error, and unnamed ones are skipped.
    """
    ids = available_ids(root)
    if names:
        unknown = [n for n in names if n not in ids]
        if unknown:
            raise ProblemError(f"unknown problems {unknown}; available: {ids}")
        if len(set(names)) != len(names):
            raise ProblemError("a problem is named twice")
        problems = [load_problem(root / n) for n in names]
        if mock:
            missing = [p.id for p in problems if p.meta.mock_scenario is None]
            if missing:
                raise ProblemError(f"no mock scenario for {missing}; in mock mode choose from "
                                   f"{[p for p in ids if load_problem(root / p).meta.mock_scenario]}")
    else:
        problems = [load_problem(root / n) for n in ids]
        if mock:
            problems = [p for p in problems if p.meta.mock_scenario is not None]
    if limit is not None:
        problems = problems[:limit]
    if not problems:
        raise ProblemError("no benchmark problems selected")
    return problems
