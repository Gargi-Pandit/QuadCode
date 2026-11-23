"""Escaping of untrusted text for st.markdown, and splitting the report into sections."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("streamlit")

from quadcode.report.doc import Document, Heading, Para, table  # noqa: E402
from quadcode.ui.render import safe_inline, safe_text, split_sections, table_rows, unique_headers  # noqa: E402


def test_streamlit_extensions_are_escaped() -> None:
    assert safe_inline("costs $x$") == "costs \\$x\\$"  # no LaTeX
    assert safe_inline(":red[y] :smile:") == "\\:red\\[y\\] \\:smile\\:"  # no colour directive or emoji
    assert safe_inline("_em_ *b*") == "\\_em\\_ \\*b\\*"


def test_markdown_markup_is_escaped() -> None:
    text = safe_text("<script>alert(1)</script>\n# heading\n- item\n[x](javascript:alert(3))")
    assert not re.search(r"(?<!\\)<", text)
    assert "\\# heading" in text and "\\- item" in text and "\\[x\\](javascript\\:alert(3))" in text


def test_sections_and_tables() -> None:
    doc = Document("t", [Para("intro"), Heading(2, "A"), Para("a"), Heading(3, "A.1"), Heading(2, "B")])
    assert [(t, len(b)) for t, b in split_sections(doc)] == [("", 1), ("A", 3), ("B", 1)]
    assert unique_headers(["x", "y", "x", "x"]) == ["x", "y", "x (2)", "x (3)"]
    assert table_rows(table(["k", "k"], [[1, 2]])) == [{"k": "1", "k (2)": "2"}]
