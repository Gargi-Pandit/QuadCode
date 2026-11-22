from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from quadcode.cache import sha256_text
from quadcode.diffs import apply_diff
from quadcode.linemap import compose
from quadcode.providers.mock_scenarios import C1_CODE, D1_CODE, G1_CODE, M001_DIFF, M002_DIFF
from quadcode.schemas import Role, VersionKind, VersionMeta
from quadcode.versions import (
    ImmutableVersionError,
    VersionError,
    VersionIntegrityError,
    VersionNotFound,
    VersionStore,
    candidate_label,
    extension_for,
    kind_for_label,
)


@pytest.fixture
def store(tmp_path: Path) -> VersionStore:
    return VersionStore(tmp_path / "run")


@pytest.mark.parametrize("label, kind", [
    ("G1", VersionKind.INITIAL), ("D1", VersionKind.INITIAL), ("V0", VersionKind.REFERENCE),
    ("V3", VersionKind.ROUND), ("V0+M-001", VersionKind.CANDIDATE), ("G1+M-1234", VersionKind.CANDIDATE),
])
def test_label_kinds(label: str, kind: VersionKind) -> None:
    assert kind_for_label(label) is kind


@pytest.mark.parametrize("label", ["G2", "v1", "V", "V0+M-1", "V0+M-001+M-002", "../G1", "G1.py", ""])
def test_invalid_labels(label: str) -> None:
    with pytest.raises(VersionError):
        kind_for_label(label)


def test_extension_for() -> None:
    assert extension_for("Python") == ".py" and extension_for("rust") == ".rs" and extension_for("cobol") == ".txt"


def test_save_initial_writes_readonly_files_with_meta(store: VersionStore) -> None:
    v = store.save_initial("G1", G1_CODE, "python")
    assert v.meta.kind is VersionKind.INITIAL and v.meta.created_by is Role.GENERATOR
    assert v.meta.sha256 == sha256_text(G1_CODE) and v.meta.line_count == 9
    code_path = store.root / "G1.py"
    assert code_path.read_text(encoding="utf-8") == G1_CODE
    for path in (code_path, store.root / "G1.json"):
        assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert [o.version for o in v.meta.line_origins] == ["G1"] * 9
    assert all(o.role is Role.GENERATOR and o.modification_id is None for o in v.meta.line_origins)
    assert not list(store.root.glob(".tmp-*"))


def test_initial_label_fixes_the_role(store: VersionStore) -> None:
    assert store.save_initial("C1", C1_CODE, "python").meta.created_by is Role.REVIEWER
    assert store.save_initial("D1", D1_CODE, "python").meta.created_by is Role.VALIDATOR
    with pytest.raises(VersionError):
        store.save_initial("V1", G1_CODE, "python")


def test_resave_identical_is_idempotent(store: VersionStore) -> None:
    first = store.save_initial("G1", G1_CODE, "python")
    again = store.save_initial("G1", G1_CODE, "python")
    assert again.meta == first.meta  # the stored copy, including its original timestamp


def test_never_overwrites_a_version(store: VersionStore) -> None:
    store.save_initial("G1", G1_CODE, "python")
    with pytest.raises(ImmutableVersionError):
        store.save_initial("G1", C1_CODE, "python")
    with pytest.raises(ImmutableVersionError):
        store.save_initial("G1", G1_CODE, "python", note="different metadata")
    assert store.load("G1").code == G1_CODE


def test_crlf_input_is_stored_normalized(store: VersionStore) -> None:
    v = store.save_initial("G1", G1_CODE.replace("\n", "\r\n"), "python")
    assert v.code == G1_CODE


def test_tampered_code_is_detected(store: VersionStore) -> None:
    store.save_initial("G1", G1_CODE, "python")
    path = store.root / "G1.py"
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    path.write_text(G1_CODE + "# edited\n", encoding="utf-8")
    with pytest.raises(VersionIntegrityError, match="sha256"):
        store.load("G1")


def test_missing_code_file_is_detected(store: VersionStore) -> None:
    store.save_initial("G1", G1_CODE, "python")
    (store.root / "G1.py").unlink()
    with pytest.raises(VersionIntegrityError, match="missing"):
        store.load("G1")


def test_crash_between_writes_is_completed_only_for_same_content(store: VersionStore) -> None:
    (store.root / "G1.py").write_text(G1_CODE, encoding="utf-8")  # code written, metadata never was
    v = store.save_initial("G1", G1_CODE, "python")
    assert store.load("G1").meta == v.meta

    (store.root / "C1.py").write_text("something else\n", encoding="utf-8")
    with pytest.raises(ImmutableVersionError, match="incomplete"):
        store.save_initial("C1", C1_CODE, "python")


def test_not_found(store: VersionStore) -> None:
    with pytest.raises(VersionNotFound):
        store.load("P1")
    assert not store.exists("P1")


def test_save_validates_meta_against_code(store: VersionStore) -> None:
    meta = VersionMeta(label="G1", kind=VersionKind.INITIAL, language="python", ext=".py",
                       sha256=sha256_text("x\n"), line_count=1, line_origins=[])
    with pytest.raises(VersionError, match="origins"):
        store.save(meta, "x\n")
    with pytest.raises(VersionError, match="sha256"):
        store.save(meta.model_copy(update={"sha256": "0" * 64}), "x\n")
    with pytest.raises(VersionError, match="does not fit kind"):
        store.save(meta.model_copy(update={"kind": VersionKind.ROUND}), "x\n")


def test_reference_copies_initial(store: VersionStore) -> None:
    store.save_initial("G1", G1_CODE, "python")
    v0 = store.save_reference("G1")
    assert v0.code == G1_CODE and v0.meta.parents == ["G1"] and v0.meta.kind is VersionKind.REFERENCE
    assert v0.meta.line_map_from_parent == list(range(1, 10))
    assert v0.meta.line_origins == store.load("G1").meta.line_origins
    with pytest.raises(VersionError, match="must copy an initial"):
        store.save_reference("V0", label="V0")


def test_parent_must_exist(store: VersionStore) -> None:
    with pytest.raises(VersionNotFound):
        store.save_reference("G1")


def test_candidate_and_blame(store: VersionStore) -> None:
    store.save_initial("G1", G1_CODE, "python")
    store.save_reference("G1")
    result = apply_diff(G1_CODE, M001_DIFF)
    cand = store.save_candidate("V0", "M-001", Role.REVIEWER, result)
    assert cand.label == "V0+M-001" and cand.meta.parents == ["V0"]
    assert cand.meta.created_by is Role.REVIEWER and cand.meta.modification_ids == ["M-001"]
    assert cand.meta.line_map_from_parent == [1, 2, 5, 6, 7, 8, 9, 10, 11]
    blame = store.blame("V0+M-001")
    assert [n for n, _, o in blame if o.modification_id == "M-001"] == [3, 4]
    assert blame[4][2].version == "G1" and blame[4][2].line == 3
    assert store.lineage("V0+M-001") == ["G1", "V0", "V0+M-001"]


def test_candidate_must_come_from_its_base(store: VersionStore) -> None:
    store.save_initial("G1", G1_CODE, "python")
    store.save_initial("D1", D1_CODE, "python")
    result = apply_diff(G1_CODE, M001_DIFF)
    with pytest.raises(VersionError, match="not computed from D1|does not match D1"):
        store.save_candidate("D1", "M-001", Role.REVIEWER, result)


def test_round_version_from_composition(store: VersionStore) -> None:
    store.save_initial("G1", G1_CODE, "python")
    v0 = store.save_reference("G1")
    composed = compose(v0.code, v0.meta.line_origins, "V1",
                       [("M-001", Role.REVIEWER, M001_DIFF), ("M-002", Role.OPTIMIZER, M002_DIFF)])
    v1 = store.save_round("V1", "V0", composed)
    assert v1.meta.modification_ids == ["M-001", "M-002"] and v1.meta.parents == ["V0"]
    assert v1.meta.line_map_from_parent == [1, 2, 5, 6, 7, 8, 9, None, 12]
    assert store.lineage("V1") == ["G1", "V0", "V1"]
    with pytest.raises(VersionError, match="not a round version label"):
        store.save_round("V0+M-003", "V0", composed)


def test_list_orders_by_lineage(store: VersionStore) -> None:
    store.save_initial("D1", D1_CODE, "python")
    store.save_initial("G1", G1_CODE, "python")
    store.save_initial("C1", C1_CODE, "python")
    v0 = store.save_reference("G1")
    composed = compose(v0.code, v0.meta.line_origins, "V1", [("M-002", Role.OPTIMIZER, M002_DIFF)])
    store.save_round("V1", "V0", composed)
    store.save_candidate("V0", "M-002", Role.OPTIMIZER, apply_diff(G1_CODE, M002_DIFF))
    store.save_candidate("V0", "M-001", Role.REVIEWER, apply_diff(G1_CODE, M001_DIFF))
    assert [m.label for m in store.list()] == ["G1", "C1", "D1", "V0", "V0+M-001", "V0+M-002", "V1"]


def test_candidate_label() -> None:
    assert candidate_label("V2", "M-010") == "V2+M-010"
    with pytest.raises(VersionError):
        candidate_label("V0+M-001", "M-002")
