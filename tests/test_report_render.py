"""The document tree and its Markdown / HTML renderers: escaping, structure, determinism."""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

from quadcode.report.doc import Bullets, Code, Document, Heading, Note, Para, slugify, source, table
from quadcode.report.render_html import render_html
from quadcode.report.render_md import escape_inline, fence_for, render_markdown

HOSTILE = "<script>alert(1)</script> | [link](http://x) **bold** `tick` ~~s~~"
HOSTILE_CODE = "x = 1\n```\n</code></pre><script>alert(2)</script>\n````\n"


def _doc() -> Document:
    return Document("Report <title>", [
        Heading(2, "1. Problem analysis"),
        Para(HOSTILE + "\n# not a heading\n- not a list\n1. not ordered"),
        Bullets((HOSTILE, "second\nline")),
        table(["a|b", "c"], [[HOSTILE, "multi\nline"], ["x", None]], caption="cap <b>"),
        Code(HOSTILE_CODE, "python"),
        Note(HOSTILE, "warning"),
        source("events:3", HOSTILE),
        Heading(2, "10. Reproduction instructions"),
    ])


class _Checker(HTMLParser):
    VOID = {"meta", "br", "link", "img", "input", "hr"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.tags: list[str] = []
        self.attrs: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attrs.extend(attrs)
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        assert self.stack and self.stack[-1] == tag, f"unbalanced </{tag}> with open {self.stack}"
        self.stack.pop()


def test_table_rows_must_match_headers() -> None:
    with pytest.raises(ValueError):
        table(["a", "b"], [["only one"]])


def test_source_drops_empty_and_duplicate_refs() -> None:
    assert source("a", ["b", "", "a"], "c").refs == ("a", "b", "c")


def test_markdown_escapes_agent_text() -> None:
    md = render_markdown(_doc())
    outside_code = md.replace(HOSTILE_CODE, "")
    assert "<script>" not in outside_code
    assert "\\<script\\>" in md
    assert "\\[link\\](http://x)" in md and "\\*\\*bold\\*\\*" in md
    assert "\n\\# not a heading" in md and "\n\\- not a list" in md and "\n1\\. not ordered" in md


def test_markdown_table_cells_stay_on_one_row() -> None:
    md = render_markdown(_doc())
    rows = [line for line in md.splitlines() if line.startswith("| ")]
    assert rows[0] == "| a\\|b | c |"
    assert "multi<br>line" in rows[1] and rows[1].count(" | ") == 1
    assert rows[2] == "| x |  |"


def test_markdown_code_fence_is_longer_than_any_backtick_run() -> None:
    assert fence_for("no ticks") == "```"
    assert fence_for(HOSTILE_CODE) == "`````"
    md = render_markdown(_doc())
    assert f"`````python\n{HOSTILE_CODE}`````" in md


def test_markdown_toc_links_match_heading_anchors() -> None:
    md = render_markdown(_doc())
    assert "- [1. Problem analysis](#1-problem-analysis)" in md
    assert "- [10. Reproduction instructions](#10-reproduction-instructions)" in md
    assert slugify("3. Initial comparison") == "3-initial-comparison"


def test_html_escapes_everything_and_has_no_active_content() -> None:
    html = render_html(_doc())
    assert "<script" not in html.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;/code&gt;&lt;/pre&gt;&lt;script&gt;alert(2)" in html
    assert "<title>Report &lt;title&gt;</title>" in html
    checker = _Checker()
    checker.feed(html)
    checker.close()
    assert checker.stack == []
    assert not {"script", "iframe", "link", "img", "object", "embed", "form"} & set(checker.tags)
    hrefs = [v for k, v in checker.attrs if k == "href"]
    assert hrefs and all(v is not None and v.startswith("#") for v in hrefs)
    assert not [k for k, _ in checker.attrs if k == "src" or k.startswith("on")]


def test_html_heading_ids_match_toc() -> None:
    html = render_html(_doc())
    assert '<h2 id="1-problem-analysis">1. Problem analysis</h2>' in html
    assert '<a href="#1-problem-analysis">1. Problem analysis</a>' in html


def test_rendering_is_deterministic() -> None:
    assert render_markdown(_doc()) == render_markdown(_doc())
    assert render_html(_doc()) == render_html(_doc())


def test_escape_inline_leaves_plain_text_alone() -> None:
    assert escape_inline("two_sum returns O(n) results, 9/9 passed") == "two_sum returns O(n) results, 9/9 passed"
