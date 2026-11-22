from __future__ import annotations

import pytest

from quadcode.diffs import apply_diff, split_lines
from quadcode.linemap import (
    ComposeConflict,
    chain_maps,
    compose,
    format_lines,
    format_location,
    format_removed,
    initial_origins,
    line_ranges,
    new_location,
    old_location,
    propagate,
)
from quadcode.providers.mock_scenarios import G1_CODE, M001_DIFF, M002_DIFF
from quadcode.schemas import LineOrigin, Role

G1_LINES = len(split_lines(G1_CODE)[0])


def test_format_helpers() -> None:
    assert format_location("V0", 3, 2) == "V0:L3-L4"
    assert format_location("V0", 3, 1) == "V0:L3"
    assert format_location("V0", 3, 0) == "V0:after L2"
    assert format_location("V0", 1, 0) == "V0:start of file"
    assert format_removed("V1", 5) == "V1:removed, after L4"
    assert format_removed("V1", 1) == "V1:removed, at start of file"
    assert line_ranges([9, 3, 4, 4]) == [(3, 4), (9, 9)]
    assert format_lines("V1", [3, 4, 9]) == "V1:L3-L4, L9"


def test_hunk_locations_for_mock_diffs() -> None:
    (h1,) = apply_diff(G1_CODE, M001_DIFF).hunks
    assert old_location("G1", h1) == "G1:after L2"
    assert new_location("G1+M-001", h1) == "G1+M-001:L3-L4"
    (h2,) = apply_diff(G1_CODE, M002_DIFF).hunks
    assert old_location("G1", h2) == "G1:L8"
    assert new_location("G1+M-002", h2) == "G1+M-002:L8-L9"


def test_pure_deletion_location() -> None:
    (h,) = apply_diff("a\nb\nc\n", "@@ -1,3 +1,2 @@\n a\n-b\n c\n").hunks
    assert old_location("V0", h) == "V0:L2"
    assert new_location("V1", h) == "V1:removed, after L1"


def test_propagate_attributes_only_added_lines() -> None:
    origins = initial_origins("G1", Role.GENERATOR, G1_LINES)
    result = apply_diff(G1_CODE, M001_DIFF)
    out = propagate(origins, result, "G1+M-001", "M-001", Role.REVIEWER)
    assert len(out) == G1_LINES + 2
    assert out[2] == LineOrigin(version="G1+M-001", line=3, modification_id="M-001", role=Role.REVIEWER)
    assert out[3].modification_id == "M-001"
    assert out[4] == LineOrigin(version="G1", line=3, role=Role.GENERATOR)
    assert all(o.modification_id is None for i, o in enumerate(out) if i not in (2, 3))


def test_propagate_rejects_mismatched_origins() -> None:
    with pytest.raises(ValueError, match="origins cover"):
        propagate(initial_origins("G1", Role.GENERATOR, 3), apply_diff(G1_CODE, M001_DIFF), "x", "M-001",
                  Role.REVIEWER)


def test_chain_maps() -> None:
    assert chain_maps([1, None, 2], [None, 5]) == [None, None, 5]


def test_compose_two_diffs_written_against_same_base() -> None:
    origins = initial_origins("G1", Role.GENERATOR, G1_LINES)
    composed = compose(G1_CODE, origins, "V1", [("M-001", Role.REVIEWER, M001_DIFF),
                                                  ("M-002", Role.OPTIMIZER, M002_DIFF)])
    lines, _ = split_lines(composed.text)
    assert lines[2] == "    if not isinstance(nums, list):"
    assert lines[9:11] == ["        if complement != value:", "            seen[value] = i"]
    assert composed.location_of("M-001", "V1") == "V1:L3-L4"
    assert composed.location_of("M-002", "V1") == "V1:L10-L11"
    # Base line 8 (seen[value] = i) was replaced; every other base line is mapped.
    assert composed.old_to_new == (1, 2, 5, 6, 7, 8, 9, None, 12)
    # Lines added in the intermediate step are renumbered to their place in V1.
    assert composed.origins[2] == LineOrigin(version="V1", line=3, modification_id="M-001", role=Role.REVIEWER)
    assert composed.origins[9] == LineOrigin(version="V1", line=10, modification_id="M-002", role=Role.OPTIMIZER)
    second_step_hunk = composed.steps[1].result.hunks[0]
    assert second_step_hunk.offset == 0  # stated line 6 was translated to line 8 of the intermediate text


def test_compose_conflict_names_the_modification() -> None:
    origins = initial_origins("V0", Role.GENERATOR, 3)
    first = "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    second = "@@ -1,3 +1,3 @@\n a\n-b\n+bb\n c\n"
    with pytest.raises(ComposeConflict) as info:
        compose("a\nb\nc\n", origins, "V1", [("M-001", Role.REVIEWER, first), ("M-002", Role.OPTIMIZER, second)])
    assert info.value.modification_id == "M-002"


def test_location_of_deletion_after_later_insertions() -> None:
    base = "a\nb\nc\nd\n"
    origins = initial_origins("V0", Role.GENERATOR, 4)
    delete_c = "@@ -2,3 +2,2 @@\n b\n-c\n d\n"
    insert_top = "@@ -1,2 +1,4 @@\n+x\n+y\n a\n b\n"
    composed = compose(base, origins, "V1", [("M-001", Role.VALIDATOR, delete_c),
                                             ("M-002", Role.GENERATOR, insert_top)])
    assert composed.text == "x\ny\na\nb\nd\n"
    assert composed.location_of("M-001", "V1") == "V1:removed, after L4"
    assert composed.location_of("M-002", "V1") == "V1:L1-L2"
    with pytest.raises(KeyError):
        composed.location_of("M-999", "V1")
