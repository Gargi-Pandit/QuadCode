"""Pipeline ablations used by the evaluation harness, on the two_sum mock scenario."""

from __future__ import annotations

from pathlib import Path

import pytest

from quadcode.prompts import NO_REVIEW_CONTEXT, WITHHELD_REASON
from quadcode.report import write_report
from quadcode.runstate import load_state
from quadcode.schemas import Ablation, AgentStatus, CallType, Role, RunStatus, RunSummary

from .orch_helpers import Built, build, calls, events

RESULT_MARKERS = ("SANDBOX TEST RESULTS", "SANDBOX RESULTS OF", "RESULTS OF V", "passed |", "does not pass the suite",
                  "T03 fail")


def _run(root: Path, ablation: Ablation, **kw: object) -> tuple[Built, RunSummary]:
    built = build(root, ablation=ablation, **kw)  # type: ignore[arg-type]
    return built, built.orch.run()


def _all_prompts(built: Built) -> list[tuple[Role, CallType, str, str]]:
    return [(r, q.call_type, q.tag, q.prompt) for r in Role for q in built.provider(r).requests]


# ---------------------------------------------------------------- no review


@pytest.fixture(scope="module")
def no_review(tmp_path_factory: pytest.TempPathFactory) -> tuple[Built, RunSummary]:
    return _run(tmp_path_factory.mktemp("no_review"), Ablation.NO_REVIEW)


def test_no_review_makes_no_review_call(no_review: tuple[Built, RunSummary]) -> None:
    built, summary = no_review
    assert summary.status is RunStatus.COMPLETED
    assert summary.ablation is Ablation.NO_REVIEW
    assert not [p for p in _all_prompts(built) if p[1] is CallType.REVIEW]
    assert not calls(built.run_dir, lambda c: c["call_type"] == "review")
    skipped = events(built.run_dir, "phase_skipped")
    assert len(skipped) == 1 and skipped[0].phase.value == "3-review"
    assert load_state(built.run_dir).is_done("review")


def test_no_review_round_one_says_so(no_review: tuple[Built, RunSummary]) -> None:
    built, _ = no_review
    r1 = next(p for p in _all_prompts(built) if p[1] is CallType.MODIFY and p[2] == "round-1")[3]
    assert NO_REVIEW_CONTEXT in r1
    assert "Phase 3 review comments" not in r1


def test_no_review_report_names_the_ablation(no_review: tuple[Built, RunSummary]) -> None:
    built, _ = no_review
    md = write_report(built.run_dir).markdown.read_text(encoding="utf-8")
    assert "evaluation ablation (no\\_review)" in md or "evaluation ablation (no_review)" in md
    assert "Phase 3 cross review skipped" in md


# ---------------------------------------------------------------- no test feedback


@pytest.fixture(scope="module")
def no_feedback(tmp_path_factory: pytest.TempPathFactory) -> tuple[Built, RunSummary]:
    return _run(tmp_path_factory.mktemp("no_feedback"), Ablation.NO_TEST_FEEDBACK)


def test_no_feedback_prompts_carry_no_results(no_feedback: tuple[Built, RunSummary]) -> None:
    built, _ = no_feedback
    prompts = _all_prompts(built)
    assert {p[1] for p in prompts} >= {CallType.REVIEW, CallType.MODIFY, CallType.CHALLENGE}
    for role, call_type, tag, text in prompts:
        for marker in RESULT_MARKERS:
            assert marker not in text, (role, call_type, tag, marker)


def test_no_feedback_still_decides_on_tests(no_feedback: tuple[Built, RunSummary]) -> None:
    built, summary = no_feedback
    # Same decisions as the full pipeline: the orchestrator still runs and uses the tests.
    assert summary.accepted == ["M-001"] and "M-002" in summary.rejected
    decided = {e.payload["id"]: e.payload["reason"] for e in events(built.run_dir, "modification_decided")}
    assert "T03" in decided["M-002"]  # the logged reason keeps the evidence; only prompts omit it
    r2 = next(p for p in _all_prompts(built) if p[1] is CallType.MODIFY and p[2] == "round-2")[3]
    assert WITHHELD_REASON in r2


# ---------------------------------------------------------------- no optimizer


@pytest.fixture(scope="module")
def no_optimizer(tmp_path_factory: pytest.TempPathFactory) -> tuple[Built, RunSummary]:
    return _run(tmp_path_factory.mktemp("no_optimizer"), Ablation.NO_OPTIMIZER)


def test_no_optimizer_is_never_asked(no_optimizer: tuple[Built, RunSummary]) -> None:
    built, summary = no_optimizer
    assert built.provider(Role.OPTIMIZER).requests == []
    assert not calls(built.run_dir, lambda c: c["role"] == "optimizer")
    assert summary.status is RunStatus.COMPLETED
    assert not (built.run_dir / "versions" / "P1.py").exists()


def test_no_optimizer_is_excluded_not_unavailable(no_optimizer: tuple[Built, RunSummary]) -> None:
    built, summary = no_optimizer
    opt = [a for a in summary.availability if a.role is Role.OPTIMIZER]
    assert [a.status for a in opt] == [AgentStatus.EXCLUDED]
    started = events(built.run_dir, "run_started")[0]
    assert started.payload["ablation"] == "no_optimizer" and started.payload["excluded"] == ["optimizer"]
    review = next(q.prompt for q in built.provider(Role.GENERATOR).requests if q.call_type is CallType.REVIEW)
    assert review.startswith("Three agents") and "P1" not in review
    md = write_report(built.run_dir).markdown.read_text(encoding="utf-8")
    assert "excluded (ablation)" in md and "EXCLUDED (ablation)" in md


def test_no_optimizer_challengers_exclude_optimizer(no_optimizer: tuple[Built, RunSummary]) -> None:
    built, _ = no_optimizer
    for e in events(built.run_dir, "challenge_requested"):
        assert "optimizer" not in e.payload["challengers"]


# ---------------------------------------------------------------- resume and full pipeline


def test_full_pipeline_prompts_unchanged_by_ablation_support(tmp_path: Path) -> None:
    built, summary = _run(tmp_path, Ablation.NONE)
    assert summary.ablation is Ablation.NONE
    review = next(q.prompt for q in built.provider(Role.GENERATOR).requests if q.call_type is CallType.REVIEW)
    assert review.startswith("Four agents") and "SANDBOX TEST RESULTS (measured):" in review


def test_resume_keeps_the_ablation(tmp_path: Path) -> None:
    first = build(tmp_path, ablation=Ablation.NO_OPTIMIZER, budget=4)
    assert first.orch.run().status is RunStatus.ABORTED_BUDGET
    resumed = build(tmp_path, resume=True)
    assert resumed.orch.state.ablation is Ablation.NO_OPTIMIZER
    summary = resumed.orch.run()
    assert summary.status is RunStatus.COMPLETED and summary.ablation is Ablation.NO_OPTIMIZER
    assert resumed.provider(Role.OPTIMIZER).requests == []
