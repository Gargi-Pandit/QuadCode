"""A small format-neutral document tree. Report sections build it; renderers turn it into
Markdown and HTML, so both formats always carry the same content.

All text in the tree is plain text. Renderers escape it for their format, so agent output
(untrusted) can never inject markup.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal


def slugify(text: str) -> str:
    """Anchor slug compatible with GitHub's Markdown heading anchors for plain headings."""
    slug = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return slug.replace(" ", "-")


@dataclass(frozen=True)
class Heading:
    level: int
    text: str

    @property
    def anchor(self) -> str:
        return slugify(self.text)


@dataclass(frozen=True)
class Para:
    text: str


@dataclass(frozen=True)
class Bullets:
    items: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""


@dataclass(frozen=True)
class Code:
    text: str
    language: str = ""
    title: str = ""


@dataclass(frozen=True)
class Note:
    text: str
    kind: Literal["info", "warning"] = "info"


@dataclass(frozen=True)
class Source:
    """Where a section's content came from: file names and ``events:<line>`` citations."""

    refs: tuple[str, ...]


Block = Heading | Para | Bullets | Table | Code | Note | Source


@dataclass
class Document:
    title: str
    blocks: list[Block] = field(default_factory=list)

    def sections(self) -> list[Heading]:
        return [b for b in self.blocks if isinstance(b, Heading) and b.level == 2]


# ---------------------------------------------------------------- constructors


def bullets(items: Iterable[str]) -> Bullets:
    return Bullets(tuple(items))


def table(headers: Sequence[str], rows: Iterable[Sequence[object]], caption: str = "") -> Table:
    width = len(headers)
    out: list[tuple[str, ...]] = []
    for row in rows:
        cells = tuple("" if c is None else str(c) for c in row)
        if len(cells) != width:
            raise ValueError(f"table row has {len(cells)} cells, expected {width}: {cells!r}")
        out.append(cells)
    return Table(tuple(headers), tuple(out), caption)


def source(*refs: str | Iterable[str]) -> Source:
    flat: list[str] = []
    for ref in refs:
        if isinstance(ref, str):
            flat.append(ref)
        else:
            flat.extend(ref)
    return Source(tuple(dict.fromkeys(r for r in flat if r)))
