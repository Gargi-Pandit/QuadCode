"""Run report: ten sections plus availability, token/cost and contribution tables, built only
from the records in a run directory and written as ``report.md`` and ``report.html``."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quadcode.config import AppConfig
from quadcode.report.data import RunRecords, load_run
from quadcode.report.doc import Document
from quadcode.report.render_html import render_html
from quadcode.report.render_md import render_markdown
from quadcode.report.sections_eval import section_evaluation, section_reproduction
from quadcode.report.sections_log import section_communication
from quadcode.report.sections_problem import overview, section_comparison, section_initial, section_problem
from quadcode.report.sections_results import section_final, section_testing
from quadcode.report.sections_rounds import section_history, section_rounds

REPORT_MD = "report.md"
REPORT_HTML = "report.html"

ReportFormat = Literal["md", "html", "both"]


@dataclass(frozen=True)
class ReportPaths:
    markdown: Path | None
    html: Path | None


def build_document(rec: RunRecords, pricing: AppConfig | None = None) -> Document:
    """The report as a document tree. ``pricing`` supplies prices for the cost column only."""
    title = f"QuadCode run report: {rec.run_id}"
    doc = Document(title)
    for part in (overview(rec, pricing), section_problem(rec), section_initial(rec), section_comparison(rec),
                 section_communication(rec), section_rounds(rec), section_testing(rec), section_history(rec),
                 section_final(rec), section_evaluation(rec), section_reproduction(rec)):
        doc.blocks.extend(part)
    return doc


def _write(path: Path, text: str) -> Path:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-report-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_report(run_dir: Path, pricing: AppConfig | None = None, fmt: ReportFormat = "both") -> ReportPaths:
    """Build the report of ``run_dir`` from its records and write it there. Makes no API calls.

    Only ``report.md`` / ``report.html`` are written (replacing an earlier report, which is
    derived data). Versions, records and logs are never touched.
    """
    doc = build_document(load_run(run_dir), pricing)
    md = _write(run_dir / REPORT_MD, render_markdown(doc)) if fmt in ("md", "both") else None
    html = _write(run_dir / REPORT_HTML, render_html(doc)) if fmt in ("html", "both") else None
    return ReportPaths(md, html)
