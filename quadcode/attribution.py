"""Attribution comments in the final solution: ``# [role:M-xxx]`` above accepted changes.

A comment goes above the first line that an accepted modification inserted, with that line's
indentation. Attribution comes only from line provenance (``LineOrigin``), which exists only
when a modification record inserted the line. For Python, a comment is kept only if the
annotated code has exactly the same AST as before, so a comment can never end up inside a
string literal or change behaviour.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quadcode.diffs import join_lines, split_lines
from quadcode.schemas import LineOrigin, Role
from quadcode.testing import is_python

_COMMENT_PREFIX = {"python": "#", "python3": "#", "py": "#", "javascript": "//", "typescript": "//",
                   "java": "//", "c": "//", "cpp": "//", "c++": "//", "go": "//", "rust": "//", "csharp": "//",
                   "c#": "//", "kotlin": "//", "swift": "//", "ruby": "#", "bash": "#", "sh": "#"}


@dataclass(frozen=True)
class Annotation:
    modification_id: str
    role: Role
    line: int | None
    """Line of the comment in the annotated code; None if it was not placed."""
    skipped_reason: str | None = None


@dataclass(frozen=True)
class AnnotatedCode:
    text: str
    annotations: tuple[Annotation, ...]


def comment_text(role: Role, modification_id: str) -> str:
    return f"[{role.value}:{modification_id}]"


def _same_ast(a: str, b: str) -> bool:
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


def annotate(code: str, origins: Sequence[LineOrigin], accepted: Mapping[str, Role], language: str) -> AnnotatedCode:
    """Insert one attribution comment per accepted modification (in id order)."""
    lines, trailing = split_lines(code)
    if len(lines) != len(origins):
        raise ValueError(f"{len(lines)} lines but {len(origins)} line origins")
    prefix = _COMMENT_PREFIX.get(language.lower())
    planned: list[tuple[int, str, Role]] = []  # (0-based line index, id, role)
    skipped: list[Annotation] = []
    for mod_id in sorted(accepted, key=lambda m: int(m[2:])):
        role = accepted[mod_id]
        first = next((i for i, o in enumerate(origins) if o.modification_id == mod_id), None)
        if prefix is None:
            skipped.append(Annotation(mod_id, role, None, f"no comment syntax known for {language!r}"))
        elif first is None:
            skipped.append(Annotation(mod_id, role, None, "the modification inserted no line that is still "
                                                          "present (removal only or replaced later)"))
        else:
            planned.append((first, mod_id, role))

    current = list(lines)
    placed: dict[str, int] = {}
    # Bottom-up, so earlier indices stay valid while inserting.
    for index, mod_id, role in sorted(planned, key=lambda p: p[0], reverse=True):
        target = current[index] if index < len(current) else ""
        indent = target[: len(target) - len(target.lstrip())]
        trial = [*current[:index], f"{indent}{prefix} {comment_text(role, mod_id)}", *current[index:]]
        if is_python(language) and not _same_ast(join_lines(current, trailing), join_lines(trial, trailing)):
            skipped.append(Annotation(mod_id, role, None, "a comment at this position would change the code "
                                                          "(for example inside a string literal)"))
            continue
        current = trial
        placed = {m: (n + 1 if n >= index else n) for m, n in placed.items()}
        placed[mod_id] = index

    annotations = [Annotation(m, accepted[m], n + 1) for m, n in placed.items()] + skipped
    annotations.sort(key=lambda a: int(a.modification_id[2:]))
    return AnnotatedCode(join_lines(current, trailing), tuple(annotations))
