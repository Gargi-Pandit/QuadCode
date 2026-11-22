"""--budget-calls aborts gracefully; --resume continues without repeating completed calls."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from quadcode.modifications import ModificationLedger
from quadcode.runstate import ResumeError, load_state
from quadcode.schemas import RunStatus

from .orch_helpers import build, calls, events, make_config

# Provider attempts of a full two_sum run, in order: analysis 1, initial 4 (5), review 3 + 4
# validator attempts (12), round-1 modify 4 + 1 retry (17), challenges 3 (20), round-2 modify 4 (24).


def _ok_keys(run_dir: Path) -> Counter[tuple[str, str, str]]:
    return Counter((c["role"], c["call_type"], c["tag"]) for c in calls(run_dir, lambda c: c["status"] == "ok"))


@pytest.mark.parametrize("budget", [1, 3, 9, 14, 18, 22])
def test_budget_abort_then_resume_repeats_no_completed_call(tmp_path: Path, budget: int) -> None:
    first = build(tmp_path, budget=budget)
    summary = first.orch.run()
    assert summary.status is RunStatus.ABORTED_BUDGET
    assert "call budget of" in summary.stop_reason and "--resume run" in summary.stop_reason
    assert any(c["status"] == "budget_exhausted" for c in calls(first.run_dir))
    [aborted] = events(first.run_dir, "run_aborted")
    assert aborted.payload["step"] in summary.stop_reason
    assert (first.run_dir / "summary.json").is_file()
    done_before = list(load_state(first.run_dir).done)

    second = build(tmp_path, resume=True)
    assert second.orch.state.done == done_before
    final = second.orch.run()
    assert final.status is RunStatus.COMPLETED
    assert final.accepted == ["M-001"] and final.rejected == ["M-002"] and final.final_version == "V1"

    repeated = {k: n for k, n in _ok_keys(first.run_dir).items() if n > 1}
    assert repeated == {}, f"completed calls were repeated: {repeated}"
    # Every call the pipeline needs completed exactly once across both processes.
    assert len(_ok_keys(first.run_dir)) == 4 + 1 + 3 + 4 + 3 + 4  # one analysis, no validator review
    assert [e.payload["resumed"] for e in events(first.run_dir, "run_started")] == [False, True]


def test_resume_of_finished_run_makes_no_calls(tmp_path: Path) -> None:
    built = build(tmp_path)
    built.orch.run()
    n = len(calls(built.run_dir))
    again = build(tmp_path, resume=True).orch.run()
    assert again.status is RunStatus.COMPLETED and len(calls(built.run_dir)) == n
    assert len(ModificationLedger(built.run_dir).all()) == 2


def test_resume_refuses_changed_config_or_problem(tmp_path: Path) -> None:
    build(tmp_path, budget=1).orch.run()
    with pytest.raises(ResumeError, match="config"):
        build(tmp_path, resume=True, config=make_config(prompts={"max_error_chars": 50}))
    (tmp_path / "run" / "problem.md").write_text("another problem")
    with pytest.raises(ResumeError, match="problem"):
        build(tmp_path, resume=True)


def test_resume_allows_other_budget_and_backend_settings(tmp_path: Path) -> None:
    build(tmp_path, budget=1).orch.run()
    cfg = make_config(run={"rounds": 3, "budget_calls": 99, "cache_enabled": False, "parallel": False},
                      sandbox={"backend": "auto"})
    assert build(tmp_path, resume=True, config=cfg).orch.state.status is RunStatus.ABORTED_BUDGET


def test_cache_hits_are_logged_and_free(tmp_path: Path) -> None:
    first = build(tmp_path, cache_enabled=True, run_name="a")
    first.orch.run()
    second = build(tmp_path, cache_enabled=True, run_name="b", budget=4)
    summary = second.orch.run()
    rows = calls(second.run_dir)
    hits = [c for c in rows if c["cache_hit"]]
    assert hits and all(c["status"] == "ok" and c["latency_s"] == 0.0 for c in hits)
    # Only the never-cached validator review (always rate-limited) consumes budget.
    attempts = [c for c in rows if not c["cache_hit"] and c["status"] != "budget_exhausted"]
    assert {(c["role"], c["call_type"]) for c in attempts} == {("validator", "review")}
    assert summary.status is RunStatus.COMPLETED
