"""The report of the full two_sum mock run, checked against the run's own records."""

from __future__ import annotations

import hashlib
import json
import re
from html import escape
from pathlib import Path

import pytest

from quadcode.providers.mock_scenarios import C1_CODE, D1_CODE, G1_CODE, P1_CODE
from quadcode.report import REPORT_HTML, REPORT_MD, build_document, write_report
from quadcode.report.data import load_run
from quadcode.report.fmt import seconds
from quadcode.report.render_md import render_markdown
from quadcode.schemas import BenchmarkReport, Role, RunSummary
from quadcode.versions import VersionStore

from .orch_helpers import Built, build, calls, make_config

VENDORS = re.compile(r"gemini|claude|gpt|deepseek|openai|anthropic", re.IGNORECASE)
SECTIONS = ["Run overview", "1. Problem analysis", "2. Four independent initial implementations",
            "3. Initial comparison", "4. Agent communication log", "5. Iterative optimization process",
            "6. Testing and benchmarking", "7. Complete modification history", "8. Final optimized implementation",
            "9. Final technical evaluation", "10. Reproduction instructions"]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Built, RunSummary, str, str]:
    built = build(tmp_path_factory.mktemp("report"))
    summary = built.orch.run()
    paths = write_report(built.run_dir)
    assert paths.markdown is not None and paths.html is not None
    return built, summary, paths.markdown.read_text(), paths.html.read_text()


def section(md: str, title: str) -> str:
    start = md.index(f"\n## {title}\n")
    nxt = md.find("\n## ", start + 1)
    return md[start: nxt if nxt != -1 else len(md)]


def captioned_tables(text: str) -> list[tuple[str, list[dict[str, str]]]]:
    """Every Markdown table in ``text`` as (caption, [{header: cell}]); caption is '' if none."""
    tables: list[tuple[str, list[dict[str, str]]]] = []
    block: list[str] = []
    previous: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2:
            cells = [[c.strip() for c in re.split(r"(?<!\\)\|", row)[1:-1]] for row in block]
            before = [p for p in previous if p.strip()]
            caption = before[-1].strip("*") if before and before[-1].startswith("**") else ""
            tables.append((caption, [dict(zip(cells[0], row, strict=True)) for row in cells[2:]]))
            previous = []
        block = []
        previous.append(line)
    return tables


def md_tables(text: str) -> list[list[dict[str, str]]]:
    return [rows for _, rows in captioned_tables(text)]


def test_every_section_in_both_formats(run: tuple[Built, RunSummary, str, str]) -> None:
    _, _, md, html = run
    for title in SECTIONS:
        assert f"\n## {title}\n" in md, title
        assert f">{title}</h2>" in html, title
    assert [h.text for h in build_document(load_run(run[0].run_dir)).sections()] == SECTIONS


def test_initial_and_final_code_appear_in_full(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, md, html = run
    final = (built.run_dir / "final_solution.py").read_text()
    for code in (G1_CODE, C1_CODE, P1_CODE, D1_CODE, final):
        assert f"```python\n{code}```" in md
        assert escape(code) in html
    assert "# [reviewer:M-001]" in section(md, "8. Final optimized implementation")


def test_modifications_and_disagreement(run: tuple[Built, RunSummary, str, str]) -> None:
    _, _, md, _ = run
    [history] = [t for t in md_tables(section(md, "7. Complete modification history")) if "status" in t[0]]
    by_id = {row["id"]: row for row in history}
    assert by_id["M-001"]["status"] == "ACCEPTED" and by_id["M-001"]["new version"] == "V1"
    assert by_id["M-002"]["status"] == "REJECTED" and "T03 fail" in by_id["M-002"]["reason"]
    log = section(md, "4. Agent communication log")
    [disagreements] = [t for t in md_tables(log) if "kind" in t[0] and "outcome" in t[0]]
    assert [r["modification"] for r in disagreements] == ["M-002"]
    assert disagreements[0]["kind"] == "split verdicts"
    assert "generator: accept" in disagreements[0]["verdicts and arguments"]
    assert "Disagree with the reviewer" in log  # free-text review disagreement kept verbatim


def test_unavailable_agent_is_reported_not_approved(run: tuple[Built, RunSummary, str, str]) -> None:
    _, _, md, _ = run
    overview = section(md, "Run overview")
    assert "RateLimitError after 4 attempt(s): mock 429 for validator/review/review" in overview
    comparison = section(md, "3. Initial comparison")
    assert "Review by validator (mock/mock-validator-v1) UNAVAILABLE" in comparison
    assert "not counted as approval" in comparison


def test_rounds_and_early_stop(run: tuple[Built, RunSummary, str, str]) -> None:
    _, _, md, _ = run
    rounds = section(md, "5. Iterative optimization process")
    assert "### Round 1" in rounds and "### Round 2" in rounds and "### Round 3" not in rounds
    assert "Resulting version: V1 = V0 with M-001." in rounds
    assert "Original code: none; this is a pure insertion (V0:after L2)." in rounds
    assert "Rounds stopped: round 2 accepted no modification (early stop)" in rounds


def test_event_citations_point_at_real_events(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, md, _ = run
    lines = (built.run_dir / "events.jsonl").read_text().splitlines()
    cited = {int(n) for n in re.findall(r"events:(\d+)", md)}
    assert cited and max(cited) <= len(lines)
    kinds = {json.loads(lines[n - 1])["kind"] for n in cited}
    assert {"agent_output", "agent_unavailable", "reference_selected", "final_benchmark", "final_written"} <= kinds


def test_benchmark_numbers_come_from_benchmark_json(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, md, _ = run
    bench = BenchmarkReport.model_validate_json((built.run_dir / "benchmark.json").read_text())
    stats = {(s.version, s.test_id): s for s in bench.stats}
    measured = [(cap, rows) for cap, rows in captioned_tables(section(md, "6. Testing and benchmarking"))
                if rows and "median s (measured)" in rows[0]]
    assert len(measured) == len(bench.suites) + 1  # one table per version plus the comparison table
    checked = 0
    for caption, rows in measured:
        for row in rows:
            version = row.get("version") or caption.split(" ")[0]
            st = stats.get((version, row["test"]))
            if st is None:
                assert row["median s (measured)"] == "not measured" and row["min s (measured)"] == "-"
                continue
            assert row["median s (measured)"] == seconds(st.median_s)
            assert row["min s (measured)"] == seconds(st.min_s) and row["max s (measured)"] == seconds(st.max_s)
            checked += 1
    perf = sum(1 for s in bench.stats if s.test_id == "T09")
    assert checked == len(bench.stats) + perf  # every stat once in its version table, T09 again in the comparison


def test_token_totals_equal_the_call_log(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, md, _ = run
    rows = calls(built.run_dir)
    [tokens] = [t for t in md_tables(section(md, "Run overview")) if "tokens in" in t[0]]
    total = tokens[-1]
    assert total["agent"] == "total"
    spent = [c for c in rows if not c["cache_hit"] and c["status"] != "budget_exhausted"]
    assert total["tokens in"] == f"{sum(c['tokens_in'] for c in spent)} (estimated)"
    assert total["tokens out"] == f"{sum(c['tokens_out'] for c in spent)} (estimated)"
    assert total["provider attempts"] == str(len(spent))
    assert total["provider errors"] == str(sum(c["status"] == "provider_error" for c in rows))
    validator = next(r for r in tokens if r["agent"] == "validator")
    assert validator["provider errors"] == "5"  # 4 failed review attempts + 1 retried 429 in round 1
    assert total["cost USD"] == "not configured"


def test_provider_and_model_come_from_the_config_only(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, md, html = run
    config = json.loads((built.run_dir / "config.json").read_text())
    for role in Role:
        agent = config["agents"][role.value]
        assert f"{role.value} ({agent['provider']}/{agent['model']})" in md
    assert not VENDORS.search(md) and not VENDORS.search(html)


def test_contributions_follow_line_origins(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, md, _ = run
    meta = VersionStore(built.run_dir).load_meta("V1")
    from_m001 = sum(1 for o in meta.line_origins if o.modification_id == "M-001")
    from_g1 = sum(1 for o in meta.line_origins if o.version == "G1")
    [contrib] = [t for t in md_tables(section(md, "9. Final technical evaluation")) if "analysis" in t[0]]
    by_agent = {r["agent"]: r for r in contrib}
    assert by_agent["reviewer"]["lines in final version"] == f"0 from its initial version, {from_m001} from accepted changes"
    assert by_agent["generator"]["lines in final version"] == f"{from_g1} from its initial version, 0 from accepted changes"
    assert by_agent["validator"]["review items"] == "UNAVAILABLE"
    assert by_agent["generator"]["analysis"] == "yes"


def test_labels_and_banners(run: tuple[Built, RunSummary, str, str]) -> None:
    _, _, md, _ = run
    assert "Mock mode: every agent slot used the scripted mock provider" in md
    assert "Weaker isolation" in md
    assert "time complexity (theoretical, as claimed by the agent)" in md
    assert "not re-derived after M-001" in md
    assert "quadcode solve runs/run/problem.md --rounds 3 --mock --scenario two_sum" in md


def test_report_is_deterministic_and_touches_nothing_else(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, md, html = run

    def digest() -> dict[str, str]:
        return {str(p.relative_to(built.run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(built.run_dir.rglob("*")) if p.is_file()}

    before = digest()
    write_report(built.run_dir)
    after = digest()
    assert after == before  # regenerated report is byte-identical, and nothing else changed
    assert (built.run_dir / REPORT_MD).read_text() == md and (built.run_dir / REPORT_HTML).read_text() == html


def test_cost_with_configured_prices(run: tuple[Built, RunSummary, str, str]) -> None:
    built, _, _, _ = run
    pricing = make_config(pricing={"mock/mock-generator-v1": {"input_per_mtok": 2.0, "output_per_mtok": 10.0}})
    md = render_markdown(build_document(load_run(built.run_dir), pricing))
    gen = [c for c in calls(built.run_dir, lambda c: c["role"] == "generator") if not c["cache_hit"]]
    expected = (sum(c["tokens_in"] for c in gen) * 2.0 + sum(c["tokens_out"] for c in gen) * 10.0) / 1e6
    [tokens] = [t for t in md_tables(section(md, "Run overview")) if "tokens in" in t[0]]
    by_agent = {r["agent"]: r for r in tokens}
    assert by_agent["generator"]["cost USD"] == f"{expected:.6f}"
    assert by_agent["reviewer"]["cost USD"] == "not configured"
    assert by_agent["total"]["cost USD"] == f"{expected:.6f} (models without a price not included)"
    assert "It is not a billed amount." in md


def test_summary_is_unchanged_by_reporting(run: tuple[Built, RunSummary, str, str]) -> None:
    built, summary, _, _ = run
    on_disk = RunSummary.model_validate_json((built.run_dir / "summary.json").read_text())
    assert on_disk.status is summary.status


def test_load_does_not_create_directories(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    rec = load_run(run_dir)
    assert rec.events == [] and rec.summary is None
    assert list(run_dir.iterdir()) == []
