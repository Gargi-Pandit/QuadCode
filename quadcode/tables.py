"""Plain-text table formatting for CLI output."""

from __future__ import annotations

from collections.abc import Sequence


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Left-aligned columns with a dashed rule under the header."""
    table = [list(headers)] + [list(r) for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines: list[str] = []
    for i, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
