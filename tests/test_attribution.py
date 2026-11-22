from __future__ import annotations

import pytest

from quadcode.attribution import annotate
from quadcode.schemas import LineOrigin, Role

R = Role.REVIEWER
O = Role.OPTIMIZER  # noqa: E741


def _origins(spec: list[str | None]) -> list[LineOrigin]:
    """One origin per line; a string names the modification that inserted the line."""
    return [LineOrigin(version="V1", line=n, modification_id=m, role=R if m == "M-001" else O if m else None)
            for n, m in enumerate(spec, start=1)]


def test_comment_above_first_inserted_line_with_indent() -> None:
    code = "def f(x):\n    if x is None:\n        raise TypeError('x')\n    return x\n"
    out = annotate(code, _origins([None, "M-001", "M-001", None]), {"M-001": R}, "python")
    assert out.text == ("def f(x):\n    # [reviewer:M-001]\n    if x is None:\n        raise TypeError('x')\n"
                        "    return x\n")
    assert out.annotations[0].line == 2 and out.annotations[0].skipped_reason is None


def test_several_modifications_keep_correct_lines() -> None:
    code = "def f(x):\n    a = 1\n    b = 2\n    return a + b\n"
    out = annotate(code, _origins([None, "M-001", None, "M-002"]), {"M-002": O, "M-001": R}, "python")
    lines = out.text.splitlines()
    by_id = {a.modification_id: a.line for a in out.annotations}
    assert lines[by_id["M-001"] - 1] == "    # [reviewer:M-001]"
    assert lines[by_id["M-002"] - 1] == "    # [optimizer:M-002]"
    assert [a.modification_id for a in out.annotations] == ["M-001", "M-002"]


def test_comment_inside_string_literal_is_skipped() -> None:
    code = 'def f():\n    s = """\nline\n"""\n    return s\n'
    out = annotate(code, _origins([None, None, "M-001", None, None]), {"M-001": R}, "python")
    assert out.text == code
    assert out.annotations[0].line is None and "string literal" in (out.annotations[0].skipped_reason or "")


def test_removal_only_and_unknown_language() -> None:
    code = "x = 1\n"
    out = annotate(code, _origins([None]), {"M-001": R}, "python")
    assert out.text == code and "removal only" in (out.annotations[0].skipped_reason or "")
    out = annotate(code, _origins(["M-001"]), {"M-001": R}, "cobol")
    assert out.text == code and "no comment syntax" in (out.annotations[0].skipped_reason or "")
    out = annotate("int x = 1;\n", _origins(["M-001"]), {"M-001": R}, "c")
    assert out.text == "// [reviewer:M-001]\nint x = 1;\n"


def test_origins_must_match_lines() -> None:
    with pytest.raises(ValueError):
        annotate("a\nb\n", _origins([None]), {}, "python")
