"""Render a report document as one self-contained HTML page.

Every piece of text is escaped with ``html.escape``. The page has inline CSS only: no
scripts, no external resources, no links except in-page anchors.
"""

from __future__ import annotations

from html import escape

from quadcode.report.doc import Block, Bullets, Code, Document, Heading, Note, Para, Source, Table

_CSS = """
:root { --bg: #ffffff; --fg: #1d1f23; --muted: #5d6470; --line: #d9dde3; --code-bg: #f5f6f8;
  --warn-bg: #fff4e0; --warn-line: #d98a00; --info-bg: #eef4fb; --info-line: #4a7bb7; --head-bg: #f0f2f5; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #16181c; --fg: #e3e5e8; --muted: #9aa1ab; --line: #343941; --code-bg: #1f2227;
    --warn-bg: #3a2c12; --warn-line: #e0a030; --info-bg: #17263a; --info-line: #6c9bd6; --head-bg: #23262c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.7em; margin: 0 0 12px; }
h2 { font-size: 1.35em; margin: 40px 0 12px; padding-top: 12px; border-top: 1px solid var(--line); }
h3 { font-size: 1.12em; margin: 26px 0 8px; }
h4 { font-size: 1em; margin: 18px 0 6px; }
p { margin: 8px 0; white-space: pre-wrap; overflow-wrap: anywhere; }
ul { margin: 6px 0; padding-left: 22px; }
li { white-space: pre-wrap; overflow-wrap: anywhere; }
nav ul { columns: 2; }
.table-wrap { overflow-x: auto; margin: 10px 0; }
table { border-collapse: collapse; font-size: 0.92em; }
caption { text-align: left; font-weight: 600; padding: 4px 0; }
th, td { border: 1px solid var(--line); padding: 4px 8px; vertical-align: top; text-align: left;
  white-space: pre-wrap; overflow-wrap: anywhere; max-width: 420px; }
th { background: var(--head-bg); }
pre { background: var(--code-bg); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px;
  overflow-x: auto; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.code-title { font-style: italic; color: var(--muted); margin: 10px 0 2px; }
.note { border-left: 4px solid var(--info-line); background: var(--info-bg); padding: 6px 12px; margin: 10px 0;
  white-space: pre-wrap; overflow-wrap: anywhere; }
.note.warning { border-left-color: var(--warn-line); background: var(--warn-bg); }
.source { color: var(--muted); font-size: 0.86em; font-style: italic; }
"""


def _text(text: str) -> str:
    return escape(text, quote=True)


def _block(block: Block) -> str:
    match block:
        case Heading(level=level, text=text):
            return f'<h{level} id="{_text(block.anchor)}">{_text(text)}</h{level}>'
        case Para(text=text):
            return f"<p>{_text(text)}</p>"
        case Bullets(items=items):
            if not items:
                return "<p><em>(none)</em></p>"
            return "<ul>" + "".join(f"<li>{_text(i)}</li>" for i in items) + "</ul>"
        case Table(headers=headers, rows=rows, caption=caption):
            cap = f"<caption>{_text(caption)}</caption>" if caption else ""
            head = "<tr>" + "".join(f"<th>{_text(h)}</th>" for h in headers) + "</tr>"
            body = "".join("<tr>" + "".join(f"<td>{_text(c)}</td>" for c in row) + "</tr>" for row in rows)
            if not rows:
                body = f'<tr><td colspan="{len(headers)}"><em>(no rows)</em></td></tr>'
            return f'<div class="table-wrap"><table>{cap}<thead>{head}</thead><tbody>{body}</tbody></table></div>'
        case Code(text=text, language=language, title=title):
            head = f'<div class="code-title">{_text(title)}</div>' if title else ""
            lang = f' data-language="{_text(language)}"' if language else ""
            return f"{head}<pre{lang}><code>{_text(text)}</code></pre>"
        case Note(text=text, kind=kind):
            label = "Warning:" if kind == "warning" else "Note:"
            cls = "note warning" if kind == "warning" else "note"
            return f'<div class="{cls}"><strong>{label}</strong> {_text(text)}</div>'
        case Source(refs=refs):
            return f'<p class="source">Source: {_text(", ".join(refs))}</p>'
    raise TypeError(f"unknown block {block!r}")


def render_html(doc: Document) -> str:
    toc = "".join(f'<li><a href="#{_text(h.anchor)}">{_text(h.text)}</a></li>' for h in doc.sections())
    nav = f"<nav><ul>{toc}</ul></nav>" if toc else ""
    body = "\n".join(_block(b) for b in doc.blocks)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_text(doc.title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
        f"<h1>{_text(doc.title)}</h1>\n{nav}\n{body}\n</main>\n</body>\n</html>\n"
    )
