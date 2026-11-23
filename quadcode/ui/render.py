"""Show report document blocks and agent text in Streamlit without letting the text add markup.

Agent output is untrusted. Text goes through the report's Markdown escaping plus Streamlit's own
extensions (``$`` math, ``:emoji:`` / ``:color[...]`` directives), tables go to ``st.dataframe``
as plain cells, code to ``st.code``. HTML is never enabled.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import streamlit as st

from quadcode.report.doc import Block, Bullets, Code, Document, Heading, Note, Para, Source, Table
from quadcode.report.render_md import escape_inline, escape_text

_STREAMLIT_SPECIAL = re.compile(r"([$:_])")


def safe_inline(text: str) -> str:
    """One line of untrusted text as inert Markdown for ``st.markdown``."""
    return _STREAMLIT_SPECIAL.sub(r"\\\1", escape_inline(text))


def safe_text(text: str) -> str:
    """A paragraph of untrusted text (several lines) as inert Markdown for ``st.markdown``."""
    return _STREAMLIT_SPECIAL.sub(r"\\\1", escape_text(text))


def unique_headers(headers: Sequence[str]) -> list[str]:
    """Column names made unique, since a dataframe cannot hold two columns of the same name."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        seen[h] = seen.get(h, 0) + 1
        out.append(h if seen[h] == 1 else f"{h} ({seen[h]})")
    return out


def table_rows(t: Table) -> list[dict[str, str]]:
    headers = unique_headers(t.headers)
    return [dict(zip(headers, row, strict=True)) for row in t.rows]


def split_sections(doc: Document) -> list[tuple[str, list[Block]]]:
    """The document cut at its level-2 headings: (title, blocks including the heading)."""
    parts: list[tuple[str, list[Block]]] = []
    for block in doc.blocks:
        if isinstance(block, Heading) and block.level == 2:
            parts.append((block.text, [block]))
        elif parts:
            parts[-1][1].append(block)
        else:
            parts.append(("", [block]))
    return parts


def show_rows(rows: list[dict[str, Any]], empty: str = "Nothing recorded.", height: int | None = None) -> None:
    if not rows:
        st.caption(empty)
        return
    if height is None:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.dataframe(rows, hide_index=True, width="stretch", height=height)


def show_block(block: Block) -> None:
    match block:
        case Heading(level=level, text=text):
            if level <= 2:
                st.header(safe_inline(text), anchor=False)
            elif level == 3:
                st.subheader(safe_inline(text), anchor=False)
            else:
                st.markdown(f"**{safe_inline(text)}**")
        case Para(text=text):
            st.markdown(safe_text(text))
        case Bullets(items=items):
            st.markdown("\n".join(f"- {safe_text(i)}" for i in items) if items else "-")
        case Table():
            if block.caption:
                st.markdown(f"**{safe_inline(block.caption)}**")
            show_rows(table_rows(block), empty="(empty table)")
        case Code(text=text, language=language, title=title):
            if title:
                st.caption(safe_inline(title))
            st.code(text, language=language or None)
        case Note(text=text, kind=kind):
            (st.warning if kind == "warning" else st.info)(safe_text(text))
        case Source(refs=refs):
            st.caption("Source: " + safe_inline(", ".join(refs)))


def show_blocks(blocks: Sequence[Block]) -> None:
    for block in blocks:
        show_block(block)
