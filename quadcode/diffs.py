"""Parse, apply and generate single-file unified diffs.

Agents return modifications as unified diffs. Models get hunk header numbers wrong far
more often than they get the code wrong, so the header is only a hint: every hunk is
located by matching its old block (context plus removed lines) exactly against the base.
A hunk applies only where that match is unique, or where several matches exist and one of
them sits exactly at the stated line. There is no fuzzy matching of content; the one
relaxation is ignoring trailing whitespace, and it is reported.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

LineKind = Literal[" ", "-", "+"]

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_BARE_HUNK_RE = re.compile(r"^@@(?:\s*@@(?:\s.*)?)?$")
_NO_NEWLINE = "\\ No newline at end of file"


class DiffError(ValueError):
    """A diff could not be parsed or applied. ``hunk`` is the 0-based hunk index, if known."""

    def __init__(self, reason: str, hunk: int | None = None) -> None:
        self.reason = reason
        self.hunk = hunk
        where = f"hunk {hunk + 1}: " if hunk is not None else ""
        super().__init__(f"{where}{reason}")


class DiffParseError(DiffError):
    pass


class DiffApplyError(DiffError):
    pass


@dataclass(frozen=True)
class HunkLine:
    kind: LineKind
    text: str


@dataclass(frozen=True)
class Hunk:
    header: str
    old_start: int | None
    old_count: int | None
    new_start: int | None
    new_count: int | None
    lines: tuple[HunkLine, ...]

    @property
    def old_lines(self) -> list[str]:
        return [ln.text for ln in self.lines if ln.kind != "+"]

    @property
    def new_lines(self) -> list[str]:
        return [ln.text for ln in self.lines if ln.kind != "-"]

    @property
    def removed(self) -> list[str]:
        return [ln.text for ln in self.lines if ln.kind == "-"]

    @property
    def added(self) -> list[str]:
        return [ln.text for ln in self.lines if ln.kind == "+"]


@dataclass(frozen=True)
class ParsedDiff:
    old_path: str | None
    new_path: str | None
    hunks: tuple[Hunk, ...]
    old_missing_newline: bool = False
    new_missing_newline: bool = False


@dataclass(frozen=True)
class AppliedHunk:
    """Where a hunk landed. Line numbers are 1-based; ``core_*`` is the changed block with
    surrounding context trimmed. For a pure insertion ``core_old_lines`` is empty and
    ``core_old_start`` is the base line the new code was inserted before; for a pure
    deletion ``core_new_lines`` is empty and ``core_new_start`` is the line that now
    follows the removed block."""

    index: int
    stated_old_start: int | None
    old_start: int
    offset: int | None
    whitespace_relaxed: bool
    core_old_start: int
    core_old_lines: tuple[str, ...]
    core_new_start: int
    core_new_lines: tuple[str, ...]


@dataclass(frozen=True)
class ApplyResult:
    text: str
    hunks: tuple[AppliedHunk, ...]
    old_to_new: tuple[int | None, ...]
    """Entry i: the line in ``text`` that base line i+1 became, or None if it was removed."""
    new_to_old: tuple[int | None, ...]
    """Entry i: the base line that line i+1 of ``text`` came from, or None if it was added."""

    @property
    def whitespace_relaxed(self) -> bool:
        return any(h.whitespace_relaxed for h in self.hunks)


# ---------------------------------------------------------------- text helpers


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_lines(text: str) -> tuple[list[str], bool]:
    """Split into lines without terminators; also report whether the text ends with a newline."""
    text = normalize_newlines(text)
    if not text:
        return [], False
    ends = text.endswith("\n")
    lines = text.split("\n")
    if ends:
        lines.pop()
    return lines, ends


def join_lines(lines: Sequence[str], trailing_newline: bool) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + ("\n" if trailing_newline else "")


# ---------------------------------------------------------------- parsing


def _is_file_header(lines: Sequence[str], i: int) -> bool:
    return lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")


def _path(header_line: str) -> str:
    return header_line[4:].split("\t", 1)[0].strip()


def _hunk_end(lines: Sequence[str], i: int) -> bool:
    line = lines[i]
    return line.startswith("@@") or line.startswith("diff ") or _is_file_header(lines, i)


def _strip_trailing_blanks(body: list[tuple[str, bool]], old_count: int | None) -> list[tuple[str, bool]]:
    """Drop bare empty lines at the end of a hunk unless the stated old count needs them."""
    trailing = 0
    while trailing < len(body) and body[-1 - trailing] == ("", True):
        trailing += 1
    if not trailing:
        return body
    kept = body[: len(body) - trailing]
    have = sum(1 for text, _ in kept if not text.startswith(("+", "\\")))
    want_back = max(0, min(trailing, (old_count or 0) - have)) if old_count is not None else 0
    return kept + [("", True)] * want_back


def _build_hunk(header: str, body: list[tuple[str, bool]], index: int) -> tuple[Hunk, bool, bool]:
    m = _HUNK_RE.match(header)
    old_start = old_count = new_start = new_count = None
    if m:
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1
    body = _strip_trailing_blanks(body, old_count)
    out: list[HunkLine] = []
    old_nl_missing = new_nl_missing = False
    for raw, bare_blank in body:
        if raw == _NO_NEWLINE:
            if not out:
                raise DiffParseError("'No newline at end of file' marker before any line", index)
            last = out[-1]
            if last.kind in (" ", "-"):
                old_nl_missing = True
            if last.kind in (" ", "+"):
                new_nl_missing = True
            continue
        if bare_blank:
            out.append(HunkLine(" ", ""))
            continue
        kind = raw[0]
        if kind not in (" ", "-", "+"):
            raise DiffParseError(f"unexpected line in hunk body: {raw[:60]!r}", index)
        out.append(HunkLine(kind, raw[1:]))  # type: ignore[arg-type]
    if not any(ln.kind != " " for ln in out):
        raise DiffParseError("hunk adds and removes nothing", index)
    return Hunk(header, old_start, old_count, new_start, new_count, tuple(out)), old_nl_missing, new_nl_missing


def parse_diff(text: str) -> ParsedDiff:
    """Parse a single-file unified diff. Text before the first header (prose, ``diff --git``,
    ``index`` lines) is ignored. A diff that names a second file is rejected."""
    lines, _ = split_lines(text)
    old_path = new_path = None
    hunks: list[Hunk] = []
    old_nl_missing = new_nl_missing = False
    i = 0
    seen_header = False
    while i < len(lines):
        line = lines[i]
        if _is_file_header(lines, i):
            o, n = _path(line), _path(lines[i + 1])
            if seen_header and (o, n) != (old_path, new_path):
                raise DiffParseError(f"diff touches more than one file ({old_path!r} and {o!r}); "
                                     "only single-file diffs are supported")
            old_path, new_path, seen_header = o, n, True
            i += 2
            continue
        if line.startswith("@@"):
            if not (_HUNK_RE.match(line) or _BARE_HUNK_RE.match(line)):
                raise DiffParseError(f"malformed hunk header: {line[:60]!r}", len(hunks))
            header = line
            i += 1
            body: list[tuple[str, bool]] = []
            while i < len(lines) and not _hunk_end(lines, i):
                body.append((lines[i], lines[i] == ""))
                i += 1
            hunk, o_nl, n_nl = _build_hunk(header, body, len(hunks))
            hunks.append(hunk)
            old_nl_missing |= o_nl
            new_nl_missing |= n_nl
            continue
        if hunks and line.strip() and not line.startswith(("diff ", "index ")):
            raise DiffParseError(f"unexpected line after hunk: {line[:60]!r}", len(hunks) - 1)
        i += 1
    if not hunks:
        raise DiffParseError("no hunks found (expected lines starting with '@@')")
    return ParsedDiff(old_path, new_path, tuple(hunks), old_nl_missing, new_nl_missing)


# ---------------------------------------------------------------- applying


def _matches(base: Sequence[str], block: Sequence[str], pos: int, relaxed: bool) -> bool:
    if pos + len(block) > len(base):
        return False
    if relaxed:
        return all(base[pos + k].rstrip() == block[k].rstrip() for k in range(len(block)))
    return all(base[pos + k] == block[k] for k in range(len(block)))


def _locate(base: Sequence[str], block: Sequence[str], floor: int, hint: int | None,
            index: int) -> tuple[int, bool]:
    """Return the 0-based position of ``block`` in ``base`` at or after ``floor``, and whether
    trailing whitespace had to be ignored to find it."""
    for relaxed in (False, True):
        found = [p for p in range(floor, len(base) - len(block) + 1) if _matches(base, block, p, relaxed)]
        if len(found) == 1:
            return found[0], relaxed
        if len(found) > 1:
            if hint is not None and hint in found:
                return hint, relaxed
            lines = ", ".join(str(p + 1) for p in found[:6])
            raise DiffApplyError(f"ambiguous: the hunk's context matches at lines {lines}"
                                 + ("" if hint is None else f" but not at the stated line {hint + 1}"), index)
    if any(_matches(base, block, p, True) for p in range(0, min(floor, len(base)))):
        raise DiffApplyError("hunks overlap or are out of order", index)
    first = block[0] if block else ""
    raise DiffApplyError(f"context does not match the base; expected a block starting with {first.strip()[:60]!r}"
                         + ("" if hint is None else f" near line {hint + 1}"), index)


def _core_bounds(hunk: Hunk) -> tuple[int, int]:
    changed = [k for k, ln in enumerate(hunk.lines) if ln.kind != " "]
    return changed[0], changed[-1]


def apply_diff(base_text: str, diff: ParsedDiff | str,
               line_hint: Sequence[int | None] | None = None) -> ApplyResult:
    """Apply ``diff`` to ``base_text``.

    ``line_hint`` translates the line numbers stated in hunk headers (which refer to the
    version the diff was written against) into lines of ``base_text``; entry i is for stated
    line i+1. It is used when composing several diffs written against the same base.
    """
    parsed = parse_diff(diff) if isinstance(diff, str) else diff
    base, base_nl = split_lines(base_text)
    out: list[str] = []
    old_to_new: list[int | None] = [None] * len(base)
    new_to_old: list[int | None] = []
    applied: list[AppliedHunk] = []
    floor = 0
    touches_end = False

    def copy_until(stop: int) -> None:
        nonlocal floor
        for k in range(floor, stop):
            out.append(base[k])
            old_to_new[k] = len(out)
            new_to_old.append(k + 1)
        floor = stop

    for idx, hunk in enumerate(parsed.hunks):
        block = hunk.old_lines
        hint: int | None = None
        if hunk.old_start is not None and hunk.old_start >= 1:
            stated = hunk.old_start
            if line_hint is not None:
                mapped = line_hint[stated - 1] if stated - 1 < len(line_hint) else None
                hint = mapped - 1 if mapped is not None else None
            else:
                hint = stated - 1
        if not block:
            if base:
                raise DiffApplyError("hunk has no context or removed lines, so it cannot be located safely", idx)
            pos, relaxed = 0, False
        else:
            pos, relaxed = _locate(base, block, floor, hint, idx)
        copy_until(pos)
        first, last = _core_bounds(hunk)
        old_before = sum(1 for ln in hunk.lines[:first] if ln.kind != "+")
        new_before = sum(1 for ln in hunk.lines[:first] if ln.kind != "-")
        hunk_new_start = len(out) + 1
        k = pos
        for ln in hunk.lines:
            if ln.kind == " ":
                out.append(base[k])  # keep the base's own text when whitespace was relaxed
                old_to_new[k] = len(out)
                new_to_old.append(k + 1)
                k += 1
            elif ln.kind == "-":
                old_to_new[k] = None
                k += 1
            else:
                out.append(ln.text)
                new_to_old.append(None)
        core = hunk.lines[first: last + 1]
        core_old_start = pos + 1 + old_before
        core_new_start = hunk_new_start + new_before
        n_old = sum(1 for ln in core if ln.kind != "+")
        n_new = sum(1 for ln in core if ln.kind != "-")
        applied.append(AppliedHunk(
            index=idx,
            stated_old_start=hunk.old_start,
            old_start=pos + 1,
            offset=None if hunk.old_start is None or hint is None else pos - hint,
            whitespace_relaxed=relaxed,
            core_old_start=core_old_start,
            core_old_lines=tuple(base[core_old_start - 1: core_old_start - 1 + n_old]),
            core_new_start=core_new_start,
            core_new_lines=tuple(out[core_new_start - 1: core_new_start - 1 + n_new]),
        ))
        floor = k
        touches_end = k == len(base)
    copy_until(len(base))

    # Diffs from models rarely carry "No newline" markers, so without one the base's final
    # newline is kept. With a marker (or an empty base) the diff decides.
    new_nl = base_nl
    if (touches_end and (parsed.old_missing_newline or parsed.new_missing_newline)) or not base:
        new_nl = not parsed.new_missing_newline
    return ApplyResult(join_lines(out, new_nl), tuple(applied), tuple(old_to_new), tuple(new_to_old))


# ---------------------------------------------------------------- generating


def _keepends(text: str) -> list[str]:
    lines, ends = split_lines(text)
    out = [ln + "\n" for ln in lines]
    if out and not ends:
        out[-1] = out[-1][:-1]
    return out


def make_diff(a: str, b: str, label_a: str = "a/solution", label_b: str = "b/solution", context: int = 3) -> str:
    """Unified diff from ``a`` to ``b`` that ``apply_diff`` reproduces exactly, including a
    missing final newline (marked the way diff(1) marks it). Empty if the texts are equal."""
    out: list[str] = []
    for line in difflib.unified_diff(_keepends(a), _keepends(b), fromfile=label_a, tofile=label_b, n=context):
        if line.endswith("\n"):
            out.append(line)
        else:
            out.append(line + "\n")
            out.append(_NO_NEWLINE + "\n")
    return "".join(out)
