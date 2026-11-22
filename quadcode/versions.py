"""Immutable on-disk version store: ``runs/<run_id>/versions/``.

Each version is two files: ``<label>.<ext>`` (the code) and ``<label>.json`` (``VersionMeta``
with its sha256, parents, modifications and per-line origins). Files are created with a hard
link from a temp file, which fails if the target exists, and are made read-only. Nothing in
this module overwrites or deletes a version. Saving the same version again (``--resume``)
returns the stored one; saving different content under an existing label raises.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from quadcode.cache import sha256_text
from quadcode.diffs import ApplyResult, normalize_newlines, split_lines
from quadcode.linemap import ComposeResult, initial_origins, propagate
from quadcode.schemas import ROLE_VERSION_LABEL, LineOrigin, Role, VersionKind, VersionMeta

VERSIONS_DIR = "versions"

_INITIAL = set(ROLE_VERSION_LABEL.values())
_LABEL_RE = re.compile(r"^(?P<base>G1|C1|P1|D1|V\d+)(?:\+(?P<mod>M-\d{3,}))?$")
_ROLE_FOR_LABEL = {label: role for role, label in ROLE_VERSION_LABEL.items()}

_EXTENSIONS = {
    "python": ".py", "javascript": ".js", "typescript": ".ts", "java": ".java", "c": ".c",
    "cpp": ".cpp", "c++": ".cpp", "go": ".go", "rust": ".rs", "ruby": ".rb", "csharp": ".cs",
    "c#": ".cs", "kotlin": ".kt", "swift": ".swift", "php": ".php",
}


class VersionError(Exception):
    pass


class ImmutableVersionError(VersionError):
    """A different version already exists under this label."""


class VersionIntegrityError(VersionError):
    """A stored file no longer matches its recorded hash or metadata."""


class VersionNotFound(VersionError, KeyError):
    pass


def extension_for(language: str) -> str:
    return _EXTENSIONS.get(language.strip().lower(), ".txt")


def kind_for_label(label: str) -> VersionKind:
    m = _LABEL_RE.match(label)
    if not m:
        raise VersionError(f"invalid version label {label!r}; expected G1, C1, P1, D1, V<n> or <base>+M-<nnn>")
    if m.group("mod"):
        return VersionKind.CANDIDATE
    base = m.group("base")
    if base in _INITIAL:
        return VersionKind.INITIAL
    return VersionKind.REFERENCE if base == "V0" else VersionKind.ROUND


def sort_key(label: str) -> tuple[int, int, int]:
    """Lineage order: initial versions in slot order, then V0, its candidates, V1, ..."""
    kind_for_label(label)
    m = _LABEL_RE.match(label)
    assert m is not None
    base, mod = m.group("base"), m.group("mod")
    mod_n = int(mod[2:]) if mod else 0
    if base in _INITIAL:
        return (-1, list(ROLE_VERSION_LABEL.values()).index(base), mod_n)
    return (int(base[1:]), 1 if mod else 0, mod_n)


def candidate_label(base: str, modification_id: str) -> str:
    label = f"{base}+{modification_id}"
    if kind_for_label(label) is not VersionKind.CANDIDATE:
        raise VersionError(f"cannot build a candidate label from {base!r}")
    return label


def role_for_initial(label: str) -> Role:
    try:
        return _ROLE_FOR_LABEL[label]
    except KeyError:
        raise VersionError(f"{label!r} is not an initial version label") from None


@dataclass(frozen=True)
class StoredVersion:
    meta: VersionMeta
    code: str

    @property
    def label(self) -> str:
        return self.meta.label


def _create_readonly(path: Path, data: str) -> None:
    """Create ``path`` with ``data``, read-only. Raises FileExistsError if it already exists."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _check_derived(label: str, base: StoredVersion, new_text: str, old_to_new: Sequence[int | None]) -> list[str]:
    """Make sure a diff or composition result really was computed from ``base``."""
    base_lines, _ = split_lines(base.code)
    new_lines, _ = split_lines(new_text)
    if len(old_to_new) != len(base_lines):
        raise VersionError(f"{label}: the change was not computed from {base.label}")
    for k, n in enumerate(old_to_new):
        if n is not None and (n > len(new_lines) or new_lines[n - 1] != base_lines[k]):
            raise VersionError(f"{label}: line map does not match {base.label} at line {k + 1}")
    return new_lines


def _same_definition(a: VersionMeta, b: VersionMeta) -> bool:
    skip = {"created_at"}
    return a.model_dump(exclude=skip) == b.model_dump(exclude=skip)


class VersionStore:
    def __init__(self, run_dir: Path) -> None:
        self.root = run_dir / VERSIONS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ paths and reading

    def _meta_path(self, label: str) -> Path:
        return self.root / f"{label}.json"

    def exists(self, label: str) -> bool:
        return self._meta_path(label).is_file()

    def code_path(self, label: str) -> Path:
        return self.root / f"{label}{self.load_meta(label).ext}"

    def load_meta(self, label: str) -> VersionMeta:
        kind_for_label(label)
        path = self._meta_path(label)
        if not path.is_file():
            raise VersionNotFound(label)
        meta = VersionMeta.model_validate_json(path.read_text(encoding="utf-8"))
        if meta.label != label:
            raise VersionIntegrityError(f"{path.name} describes {meta.label!r}, not {label!r}")
        return meta

    def load(self, label: str) -> StoredVersion:
        """Load a version and verify its code against the recorded sha256."""
        meta = self.load_meta(label)
        path = self.root / f"{label}{meta.ext}"
        try:
            code = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise VersionIntegrityError(f"{label}: metadata exists but {path.name} is missing") from None
        if sha256_text(code) != meta.sha256:
            raise VersionIntegrityError(f"{label}: {path.name} does not match its recorded sha256")
        return StoredVersion(meta, code)

    def list(self) -> list[VersionMeta]:
        """All stored versions in lineage order: G1, C1, P1, D1, V0, V0's candidates, V1, ..."""
        metas = [self.load_meta(p.stem) for p in self.root.glob("*.json") if _LABEL_RE.match(p.stem)]
        return sorted(metas, key=lambda m: sort_key(m.label))

    def lineage(self, label: str) -> list[str]:
        """``label`` and its first-parent ancestors, oldest first."""
        chain: list[str] = []
        current: str | None = label
        while current is not None:
            if current in chain:
                raise VersionIntegrityError(f"parent cycle through {current!r}")
            chain.append(current)
            parents = self.load_meta(current).parents
            current = parents[0] if parents else None
        return list(reversed(chain))

    def blame(self, label: str) -> list[tuple[int, str, LineOrigin]]:
        stored = self.load(label)
        lines, _ = split_lines(stored.code)
        return [(n, text, origin) for n, (text, origin) in enumerate(zip(lines, stored.meta.line_origins,
                                                                          strict=True), start=1)]

    # ------------------------------------------------------------ writing

    def save(self, meta: VersionMeta, code: str) -> StoredVersion:
        """Store a new immutable version. ``meta.sha256`` and ``line_count`` must match ``code``."""
        if kind_for_label(meta.label) is not meta.kind:
            raise VersionError(f"label {meta.label!r} does not fit kind {meta.kind.value!r}")
        if sha256_text(code) != meta.sha256:
            raise VersionError(f"{meta.label}: sha256 in metadata does not match the code")
        lines, _ = split_lines(code)
        if len(lines) != meta.line_count or len(meta.line_origins) != meta.line_count:
            raise VersionError(f"{meta.label}: {len(lines)} lines, line_count {meta.line_count}, "
                               f"{len(meta.line_origins)} origins")
        for parent in meta.parents:
            if not self.exists(parent):
                raise VersionError(f"{meta.label}: parent {parent!r} is not stored")

        code_path = self.root / f"{meta.label}{meta.ext}"
        meta_path = self._meta_path(meta.label)
        if meta_path.is_file():
            return self._existing(meta, code)
        if code_path.is_file():
            # A crash between the two writes left the code without its metadata.
            if sha256_text(code_path.read_text(encoding="utf-8")) != meta.sha256:
                raise ImmutableVersionError(f"{meta.label}: a different, incomplete version is already on disk")
        else:
            try:
                _create_readonly(code_path, code)
            except FileExistsError:
                return self._existing(meta, code)
        try:
            _create_readonly(meta_path, meta.model_dump_json(indent=2) + "\n")
        except FileExistsError:
            return self._existing(meta, code)
        return StoredVersion(meta, code)

    def _existing(self, meta: VersionMeta, code: str) -> StoredVersion:
        stored = self.load(meta.label)
        if stored.code != code or not _same_definition(stored.meta, meta):
            raise ImmutableVersionError(f"version {meta.label} already exists with different content; "
                                        "stored versions are never overwritten")
        return stored

    # ------------------------------------------------------------ typed helpers

    def save_initial(self, label: str, code: str, language: str, note: str = "") -> StoredVersion:
        """Store an agent's blind initial solution (G1, C1, P1 or D1)."""
        role = role_for_initial(label)
        code = normalize_newlines(code)
        n = len(split_lines(code)[0])
        meta = VersionMeta(label=label, kind=VersionKind.INITIAL, language=language, ext=extension_for(language),
                           sha256=sha256_text(code), line_count=n, created_by=role,
                           line_origins=initial_origins(label, role, n), note=note)
        return self.save(meta, code)

    def save_reference(self, source: str, label: str = "V0", note: str = "") -> StoredVersion:
        """Store V0 as an exact copy of the initial version chosen as the reference."""
        src = self.load(source)
        if src.meta.kind is not VersionKind.INITIAL:
            raise VersionError(f"the reference version must copy an initial version, not {source!r}")
        n = src.meta.line_count
        meta = VersionMeta(label=label, kind=VersionKind.REFERENCE, language=src.meta.language, ext=src.meta.ext,
                           sha256=src.meta.sha256, line_count=n, parents=[source],
                           line_origins=list(src.meta.line_origins),
                           line_map_from_parent=list(range(1, n + 1)), note=note)
        return self.save(meta, src.code)

    def save_candidate(self, base: str, modification_id: str, role: Role, result: ApplyResult,
                       note: str = "") -> StoredVersion:
        """Store ``base`` plus one modification, the version that modification's tests run on."""
        src = self.load(base)
        label = candidate_label(base, modification_id)
        lines = _check_derived(label, src, result.text, result.old_to_new)
        meta = VersionMeta(label=label, kind=VersionKind.CANDIDATE, language=src.meta.language, ext=src.meta.ext,
                           sha256=sha256_text(result.text), line_count=len(lines), parents=[base],
                           created_by=role, modification_ids=[modification_id],
                           line_origins=propagate(src.meta.line_origins, result, label, modification_id, role),
                           line_map_from_parent=list(result.old_to_new), note=note)
        return self.save(meta, result.text)

    def save_round(self, label: str, base: str, composed: ComposeResult, note: str = "") -> StoredVersion:
        """Store a round version: ``base`` with the round's accepted modifications integrated."""
        src = self.load(base)
        if kind_for_label(label) is not VersionKind.ROUND:
            raise VersionError(f"{label!r} is not a round version label (V1, V2, ...)")
        lines = _check_derived(label, src, composed.text, composed.old_to_new)
        mod_ids: Sequence[str] = [s.modification_id for s in composed.steps]
        meta = VersionMeta(label=label, kind=VersionKind.ROUND, language=src.meta.language, ext=src.meta.ext,
                           sha256=sha256_text(composed.text), line_count=len(lines), parents=[base],
                           modification_ids=list(mod_ids), line_origins=list(composed.origins),
                           line_map_from_parent=list(composed.old_to_new), note=note)
        return self.save(meta, composed.text)
