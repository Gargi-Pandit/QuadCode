"""Render a report document as Markdown (CommonMark / GitHub flavored).

Plain text is backslash-escaped so agent output cannot add HTML, links, emphasis or block
structure. Code blocks use a fence longer than any backtick run inside the code.
"""

from __future__ import annotations

import re

from quadcode.report.doc import Block, Bullets, Code, Document, Heading, Note, Para, Source, Table

_INLINE_SPECIAL = re.compile(r"([\\`*\[\]<>|~])")
_BLOCK_START = re.compile(r"^(\s*)(?:([#>+=-])|(\d+)([.)]))")


def _escape_block_start(match: re.Match[str]) -> str:
    indent, marker, digits, delim = match.groups()
    return f"{indent}\\{marker}" if marker else f"{indent}{digits}\\{delim}"


def escape_inline(text: str) -> str:
    """Escape Markdown punctuation that could create markup inside a line."""
    return _INLINE_SPECIAL.sub(r"\\\1", text)


def escape_text(text: str) -> str:
    """Escape a multi-line paragraph: inline markup and block markers at line starts."""
    lines = [escape_inline(line) for line in text.split("\n")]
    lines = [_BLOCK_START.sub(_escape_block_start, line) for line in lines]
    return "  \n".join(lines)


def _cell(text: str) -> str:
    lines = [escape_inline(line) for line in text.split("\n")]
    return "<br>".join(lines) if any(lines) else ""


def fence_for(code: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


def _language_tag(language: str) -> str:
    return re.sub(r"[^a-z0-9+#-]", "", language.lower())


def _block(block: Block) -> str:
    match block:
        case Heading(level=level, text=text):
            return f"{'#' * level} {escape_inline(' '.join(text.split()))}"
        case Para(text=text):
            return escape_text(text)
        case Bullets(items=items):
            if not items:
                return "_(none)_"
            return "\n".join("- " + escape_text(item).replace("  \n", "  \n  ") for item in items)
        case Table(headers=headers, rows=rows, caption=caption):
            out: list[str] = []
            if caption:
                out.append(f"**{escape_inline(caption)}**\n")
            out.append("| " + " | ".join(_cell(h) for h in headers) + " |")
            out.append("|" + "|".join("---" for _ in headers) + "|")
            out.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
            if not rows:
                out.append("| " + " | ".join(["_(no rows)_"] + [""] * (len(headers) - 1)) + " |")
            return "\n".join(out)
        case Code(text=text, language=language, title=title):
            fence = fence_for(text)
            body = text if text.endswith("\n") or not text else text + "\n"
            head = f"*{escape_inline(title)}*\n\n" if title else ""
            return f"{head}{fence}{_language_tag(language)}\n{body}{fence}"
        case Note(text=text, kind=kind):
            label = "**Warning:**" if kind == "warning" else "**Note:**"
            return "\n".join(f"> {line}" if line else ">" for line in f"{label} {escape_text(text)}".split("\n"))
        case Source(refs=refs):
            return "*Source: " + ", ".join(escape_inline(r) for r in refs) + "*"
    raise TypeError(f"unknown block {block!r}")


def render_markdown(doc: Document) -> str:
    parts = [f"# {escape_inline(doc.title)}"]
    sections = doc.sections()
    if sections:
        parts.append("\n".join(f"- [{escape_inline(h.text)}](#{h.anchor})" for h in sections))
    parts.extend(_block(b) for b in doc.blocks)
    return "\n\n".join(parts) + "\n"
