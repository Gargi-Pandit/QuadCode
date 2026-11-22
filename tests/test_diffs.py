from __future__ import annotations

import random

import pytest

from quadcode.diffs import DiffApplyError, DiffParseError, apply_diff, make_diff, parse_diff, split_lines
from quadcode.providers.mock_scenarios import G1_CODE, M001_DIFF, M002_DIFF

BASE = "a\nb\nc\nd\ne\nf\ng\n"


def test_split_lines_tracks_final_newline() -> None:
    assert split_lines("a\nb\n") == (["a", "b"], True)
    assert split_lines("a\nb") == (["a", "b"], False)
    assert split_lines("") == ([], False)
    assert split_lines("a\r\nb\r\n") == (["a", "b"], True)
    assert split_lines("a\x0cb\n") == (["a\x0cb"], True)  # only \n splits lines


def test_parse_headers_and_counts() -> None:
    parsed = parse_diff(M001_DIFF)
    assert parsed.old_path == "a/solution.py" and parsed.new_path == "b/solution.py"
    (hunk,) = parsed.hunks
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (1, 5, 1, 7)
    assert len(hunk.old_lines) == 5 and len(hunk.new_lines) == 7
    assert hunk.removed == [] and len(hunk.added) == 2


def test_parse_ignores_leading_prose_and_git_lines() -> None:
    diff = "Here is the change:\ndiff --git a/x b/x\nindex 1..2 100644\n--- a/x\n+++ b/x\n@@ -2 +2 @@\n-b\n+B\n"
    assert apply_diff(BASE, diff).text == BASE.replace("b\n", "B\n")


def test_parse_rejects_multi_file_diff() -> None:
    diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+A\n--- a/y\n+++ b/y\n@@ -1 +1 @@\n-a\n+A\n"
    with pytest.raises(DiffParseError, match="more than one file"):
        parse_diff(diff)


def test_parse_allows_repeated_header_for_same_file() -> None:
    diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+A\n--- a/x\n+++ b/x\n@@ -7 +7 @@\n-g\n+G\n"
    assert apply_diff(BASE, diff).text == "A\nb\nc\nd\ne\nf\nG\n"


@pytest.mark.parametrize("diff, message", [
    ("", "no hunks"),
    ("just prose\n", "no hunks"),
    ("@@ -1 +1 @@\n a\n", "adds and removes nothing"),
    ("@@ -1 +1 @@\n-a\n+A\nnot a diff line\n", "unexpected line"),
    ("@@ -x +1 @@\n-a\n", "malformed hunk header"),
    ("@@ -1 +1 @@\n\\ No newline at end of file\n-a\n", "marker before any line"),
])
def test_parse_errors(diff: str, message: str) -> None:
    with pytest.raises(DiffParseError, match=message):
        parse_diff(diff)


def test_bare_blank_line_is_blank_context() -> None:
    base = "x = 1\n\ny = 2\n"
    diff = "@@ -1,3 +1,3 @@\n x = 1\n\n-y = 2\n+y = 3\n"  # the blank context line lost its leading space
    assert apply_diff(base, diff).text == "x = 1\n\ny = 3\n"


def test_trailing_blank_lines_after_last_hunk_are_dropped() -> None:
    diff = "@@ -2 +2 @@\n-b\n+B\n\n\n"
    assert apply_diff(BASE, diff).text == BASE.replace("b\n", "B\n")


def test_mock_diffs_apply_to_g1_at_stated_lines() -> None:
    r1 = apply_diff(G1_CODE, M001_DIFF)
    (h,) = r1.hunks
    assert h.offset == 0 and h.core_old_start == 3 and h.core_old_lines == ()
    assert h.core_new_start == 3 and h.core_new_lines[0] == "    if not isinstance(nums, list):"
    assert r1.old_to_new == (1, 2, 5, 6, 7, 8, 9, 10, 11)
    assert r1.new_to_old[:5] == (1, 2, None, None, 3)

    r2 = apply_diff(G1_CODE, M002_DIFF)
    (h2,) = r2.hunks
    assert h2.core_old_start == 8 and h2.core_old_lines == ("        seen[value] = i",)
    assert h2.core_new_lines == ("        if complement != value:", "            seen[value] = i")
    assert r2.old_to_new[7] is None


def test_wrong_header_numbers_are_only_hints() -> None:
    diff = "@@ -40,2 +40,9 @@\n c\n-d\n+D\n"
    r = apply_diff(BASE, diff)
    assert r.text == "a\nb\nc\nD\ne\nf\ng\n"
    assert r.hunks[0].offset == 2 - 39 and r.hunks[0].old_start == 3


def test_header_without_numbers() -> None:
    r = apply_diff(BASE, "@@ @@\n c\n-d\n+D\n")
    assert r.hunks[0].offset is None and r.hunks[0].stated_old_start is None


def test_ambiguous_context_is_rejected_unless_header_points_at_one() -> None:
    base = "x\ny\nz\nx\ny\nw\n"
    with pytest.raises(DiffApplyError, match="ambiguous.*lines 1, 4"):
        apply_diff(base, "@@ @@\n x\n-y\n+Y\n")
    with pytest.raises(DiffApplyError, match="not at the stated line 2"):
        apply_diff(base, "@@ -2,2 +2,2 @@\n x\n-y\n+Y\n")
    assert apply_diff(base, "@@ -4,2 +4,2 @@\n x\n-y\n+Y\n").text == "x\ny\nz\nx\nY\nw\n"


def test_context_mismatch() -> None:
    with pytest.raises(DiffApplyError, match="does not match") as info:
        apply_diff(BASE, "@@ -3,2 +3,2 @@\n c\n-q\n+Q\n")
    assert info.value.hunk == 0


def test_overlapping_or_out_of_order_hunks() -> None:
    diff = "@@ -5 +5 @@\n-e\n+E\n@@ -2 +2 @@\n-b\n+B\n"
    with pytest.raises(DiffApplyError, match="overlap or are out of order") as info:
        apply_diff(BASE, diff)
    assert info.value.hunk == 1


def test_trailing_whitespace_relaxed_but_reported() -> None:
    base = "def f():   \n    return 1\n"
    r = apply_diff(base, "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n")
    assert r.whitespace_relaxed and r.hunks[0].whitespace_relaxed
    assert r.text == "def f():   \n    return 2\n"  # context keeps the base's own text


def test_leading_whitespace_is_never_relaxed() -> None:
    base = "if x:\n    y = 1\n"
    with pytest.raises(DiffApplyError):
        apply_diff(base, "@@ -1,2 +1,2 @@\n if x:\n-  y = 1\n+  y = 2\n")


def test_hunk_without_context_on_nonempty_base_is_rejected() -> None:
    with pytest.raises(DiffApplyError, match="cannot be located"):
        apply_diff(BASE, "@@ -3,0 +4 @@\n+new\n")


def test_new_file_from_empty_base() -> None:
    r = apply_diff("", "--- /dev/null\n+++ b/x\n@@ -0,0 +1,2 @@\n+a\n+b\n")
    assert r.text == "a\nb\n" and r.old_to_new == () and r.new_to_old == (None, None)


def test_crlf_diff_and_base() -> None:
    r = apply_diff("a\r\nb\r\n", "@@ -1,2 +1,2 @@\r\n a\r\n-b\r\n+B\r\n")
    assert r.text == "a\nB\n"


def test_no_newline_markers() -> None:
    base = "a\nb"
    add_newline = "@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+b\n"
    assert apply_diff(base, add_newline).text == "a\nb\n"
    drop_newline = "@@ -1,2 +1,2 @@\n a\n-b\n+c\n\\ No newline at end of file\n"
    assert apply_diff("a\nb\n", drop_newline).text == "a\nc"
    # Without markers the base's final newline is kept.
    assert apply_diff(base, "@@ -1,2 +1,2 @@\n a\n-b\n+c\n").text == "a\nc"


def test_line_maps_are_consistent() -> None:
    diff = "@@ -2,3 +2,3 @@\n-b\n c\n+X\n d\n@@ -6,2 +6,1 @@\n-f\n g\n"
    r = apply_diff(BASE, diff)
    new_lines, _ = split_lines(r.text)
    base_lines, _ = split_lines(BASE)
    assert r.text == "a\nc\nX\nd\ne\ng\n"
    assert r.old_to_new == (1, None, 2, 4, 5, None, 6)
    for old, new in enumerate(r.old_to_new, start=1):
        if new is not None:
            assert base_lines[old - 1] == new_lines[new - 1]
            assert r.new_to_old[new - 1] == old


def test_line_hint_relocates_stated_lines() -> None:
    base = "x\ny\nz\nx\ny\nw\n"
    # Stated line 4 means "x" at line 4 of the original; in the current text it moved to 6.
    current = "p\nq\nx\ny\nz\nx\ny\nw\n"
    hint = [3, 4, 5, 6, 7, 8]
    r = apply_diff(current, "@@ -4,2 +4,2 @@\n x\n-y\n+Y\n", line_hint=hint)
    assert r.text == "p\nq\nx\ny\nz\nx\nY\nw\n" and r.hunks[0].offset == 0
    assert apply_diff(base, "@@ -4,2 +4,2 @@\n x\n-y\n+Y\n").text.endswith("Y\nw\n")


def test_make_diff_empty_for_equal_texts() -> None:
    assert make_diff(BASE, BASE) == ""


def _random_text(rng: random.Random) -> str:
    lines = [rng.choice(["", "a", "b", "x = 1", "    y", "return z"]) for _ in range(rng.randint(0, 14))]
    if not lines:
        return ""
    return "\n".join(lines) + ("\n" if rng.random() < 0.7 else "")


def _mutate(text: str, rng: random.Random) -> str:
    lines, nl = split_lines(text)
    for _ in range(rng.randint(0, 4)):
        op = rng.randint(0, 2)
        if op == 0 and lines:
            del lines[rng.randrange(len(lines))]
        elif op == 1:
            lines.insert(rng.randint(0, len(lines)), rng.choice(["new", "", "    deeper"]))
        elif lines:
            lines[rng.randrange(len(lines))] = rng.choice(["changed", "x = 2"])
    if not lines:
        return ""
    if rng.random() < 0.2:
        nl = not nl
    return "\n".join(lines) + ("\n" if nl else "")


def test_make_diff_round_trip_randomized() -> None:
    rng = random.Random(12345)
    for _ in range(1500):
        a = _random_text(rng)
        b = _mutate(a, rng)
        diff = make_diff(a, b)
        if a == b:
            assert diff == ""
            continue
        assert apply_diff(a, diff).text == b, diff


def test_apply_patch_wrapper_lines_are_ignored() -> None:
    base = "def f():\n    return 1\n"
    text = ("*** Begin Patch\n*** Update File: solution.py\n@@\n-def f():\n-    return 1\n+def f():\n"
            "+    return 2\n*** End Patch\n")
    assert apply_diff(base, parse_diff(text)).text == "def f():\n    return 2\n"


def test_other_text_after_a_hunk_is_still_rejected() -> None:
    with pytest.raises(DiffParseError, match="unexpected line"):
        parse_diff("@@\n-a\n+b\n*** something else\n")
