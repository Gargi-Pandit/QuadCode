"""Line mapping and per-line provenance across versions.

Every stored version carries one ``LineOrigin`` per line. Lines of an initial version belong
to the agent slot that wrote it. When a diff is applied, unchanged lines keep their origin
and lines the diff added are attributed to that modification and its proposer. Nothing else
ever assigns authorship, so a line is credited to an agent only when a record backs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quadcode.diffs import AppliedHunk, ApplyResult, DiffApplyError, ParsedDiff, apply_diff, parse_diff
from quadcode.schemas import LineOrigin, Role


def initial_origins(label: str, role: Role | None, line_count: int) -> list[LineOrigin]:
    return [LineOrigin(version=label, line=n, role=role) for n in range(1, line_count + 1)]


def propagate(origins: Sequence[LineOrigin], result: ApplyResult, new_label: str,
              modification_id: str, role: Role) -> list[LineOrigin]:
    """Origins for ``result.text``: kept lines keep theirs, added lines go to the modification."""
    if len(origins) != len(result.old_to_new):
        raise ValueError(f"origins cover {len(origins)} lines but the base has {len(result.old_to_new)}")
    out: list[LineOrigin] = []
    for n, old in enumerate(result.new_to_old, start=1):
        if old is None:
            out.append(LineOrigin(version=new_label, line=n, modification_id=modification_id, role=role))
        else:
            out.append(origins[old - 1])
    return out


def chain_maps(first: Sequence[int | None], second: Sequence[int | None]) -> list[int | None]:
    """Compose line maps A->B and B->C into A->C."""
    return [None if b is None else second[b - 1] for b in first]


def line_ranges(lines: Sequence[int]) -> list[tuple[int, int]]:
    """Group sorted line numbers into inclusive runs: [3, 4, 9] -> [(3, 4), (9, 9)]."""
    runs: list[tuple[int, int]] = []
    for n in sorted(set(lines)):
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs


def _fmt_range(start: int, end: int) -> str:
    return f"L{start}" if start == end else f"L{start}-L{end}"


def format_location(label: str, start: int, count: int) -> str:
    """'V0:L3-L4' for a block; 'V0:after L2' for an insertion point before line ``start``."""
    if count == 0:
        return f"{label}:start of file" if start <= 1 else f"{label}:after L{start - 1}"
    return f"{label}:{_fmt_range(start, start + count - 1)}"


def format_removed(label: str, next_line: int) -> str:
    """Location of a block that no longer exists in ``label``; ``next_line`` now follows it."""
    return f"{label}:removed, at start of file" if next_line <= 1 else f"{label}:removed, after L{next_line - 1}"


def format_lines(label: str, lines: Sequence[int]) -> str:
    return f"{label}:" + ", ".join(_fmt_range(a, b) for a, b in line_ranges(lines))


def old_location(label: str, hunk: AppliedHunk) -> str:
    return format_location(label, hunk.core_old_start, len(hunk.core_old_lines))


def new_location(label: str, hunk: AppliedHunk) -> str:
    if not hunk.core_new_lines:
        return format_removed(label, hunk.core_new_start)
    return format_location(label, hunk.core_new_start, len(hunk.core_new_lines))


# ---------------------------------------------------------------- composing several diffs


class ComposeConflict(ValueError):
    """A diff no longer applies after earlier diffs of the same round were applied."""

    def __init__(self, modification_id: str, error: DiffApplyError) -> None:
        self.modification_id = modification_id
        self.error = error
        super().__init__(f"{modification_id} conflicts with modifications applied before it: {error}")


@dataclass(frozen=True)
class ComposeStep:
    modification_id: str
    role: Role
    result: ApplyResult


@dataclass(frozen=True)
class ComposeResult:
    text: str
    origins: tuple[LineOrigin, ...]
    old_to_new: tuple[int | None, ...]
    """Base line i+1 -> line in the composed text."""
    steps: tuple[ComposeStep, ...]

    def location_of(self, modification_id: str, label: str) -> str:
        """Where a modification's change sits in the composed version, from provenance."""
        added = [n for n, o in enumerate(self.origins, start=1) if o.modification_id == modification_id]
        if added:
            return format_lines(label, added)
        for i, step in enumerate(self.steps):
            if step.modification_id != modification_id:
                continue
            spots: list[str] = []
            for hunk in step.result.hunks:
                anchor = hunk.core_new_start - 1  # line before the removed block, 0 = start of file
                for later in self.steps[i + 1:]:
                    mapping = later.result.old_to_new
                    while anchor > 0 and mapping[anchor - 1] is None:
                        anchor -= 1  # the line before was removed later; move up to one that survived
                    anchor = (mapping[anchor - 1] or 0) if anchor > 0 else 0
                spots.append(format_removed(label, anchor + 1))
            return "; ".join(spots)
        raise KeyError(modification_id)


def compose(base_text: str, base_origins: Sequence[LineOrigin], new_label: str,
            changes: Sequence[tuple[str, Role, str | ParsedDiff]]) -> ComposeResult:
    """Apply several diffs, all written against ``base_text``, one after another.

    Each diff is relocated by its context; the line numbers in its headers are translated
    through the changes applied before it. A diff that no longer applies raises
    ``ComposeConflict`` naming it, so the caller can decide what to drop.
    """
    text = base_text
    origins = list(base_origins)
    base_map: list[int | None] = list(range(1, len(origins) + 1))
    steps: list[ComposeStep] = []
    for mod_id, role, diff in changes:
        parsed = parse_diff(diff) if isinstance(diff, str) else diff
        try:
            result = apply_diff(text, parsed, line_hint=base_map)
        except DiffApplyError as exc:
            raise ComposeConflict(mod_id, exc) from exc
        origins = propagate(origins, result, new_label, mod_id, role)
        base_map = chain_maps(base_map, result.old_to_new)
        steps.append(ComposeStep(mod_id, role, result))
        text = result.text
    # Lines introduced in intermediate steps belong to the composed version, not to a
    # version that was never stored.
    origins = [
        LineOrigin(version=new_label, line=n, modification_id=o.modification_id, role=o.role)
        if o.version == new_label else o
        for n, o in enumerate(origins, start=1)
    ]
    return ComposeResult(text, tuple(origins), tuple(base_map), tuple(steps))
