"""The conditions compared by the harness and how each one gets its code."""

from __future__ import annotations

from enum import StrEnum

from quadcode.schemas import ROLE_VERSION_LABEL, Ablation, Role


class Condition(StrEnum):
    SINGLE_GENERATOR = "single-generator"
    SINGLE_REVIEWER = "single-reviewer"
    SINGLE_OPTIMIZER = "single-optimizer"
    SINGLE_VALIDATOR = "single-validator"
    ORACLE = "oracle"
    SELECT_ONLY = "select-only"
    FULL = "full"
    NO_REVIEW = "no-review"
    NO_TEST_FEEDBACK = "no-test-feedback"
    NO_OPTIMIZER = "no-optimizer"


# Conditions that are a pipeline run of their own, with the ablation they run under.
RUN_ABLATION: dict[Condition, Ablation] = {
    Condition.FULL: Ablation.NONE,
    Condition.NO_REVIEW: Ablation.NO_REVIEW,
    Condition.NO_TEST_FEEDBACK: Ablation.NO_TEST_FEEDBACK,
    Condition.NO_OPTIMIZER: Ablation.NO_OPTIMIZER,
}

SINGLE_ROLE: dict[Condition, Role] = {
    Condition.SINGLE_GENERATOR: Role.GENERATOR,
    Condition.SINGLE_REVIEWER: Role.REVIEWER,
    Condition.SINGLE_OPTIMIZER: Role.OPTIMIZER,
    Condition.SINGLE_VALIDATOR: Role.VALIDATOR,
}

DERIVED = (*SINGLE_ROLE, Condition.ORACLE, Condition.SELECT_ONLY)

DESCRIPTION: dict[Condition, str] = {
    **{c: f"blind initial solution {ROLE_VERSION_LABEL[r]} of the {r.value} slot, taken from the full run "
          "(one agent alone, after the shared analysis call)" for c, r in SINGLE_ROLE.items()},
    Condition.ORACLE: "best of the four initial solutions by HIDDEN tests (ties: slot order). It looks at the "
                      "hidden tests, so it is an upper bound on picking one initial solution, not a usable system",
    Condition.SELECT_ONLY: "V0 of the full run: the initial solution the orchestrator picked on its visible tests, "
                           "before any optimization round",
    Condition.FULL: "final solution of the full pipeline",
    Condition.NO_REVIEW: "final solution of a run with the Phase 3 cross review skipped",
    Condition.NO_TEST_FEEDBACK: "final solution of a run whose agent prompts carry no sandbox results "
                                "(the orchestrator still decides on tests)",
    Condition.NO_OPTIMIZER: "final solution of a run without the optimizer slot (three agents)",
}


def run_conditions(names: list[str]) -> list[Condition]:
    """Pipeline-run conditions in canonical order, ``full`` always first."""
    wanted = {Condition(n) for n in names}
    return [c for c in RUN_ABLATION if c in wanted]


def all_conditions(runs: list[Condition]) -> list[Condition]:
    """Every condition reported for a bench that runs ``runs``: derived ones come from the full run."""
    return [*DERIVED, *[c for c in RUN_ABLATION if c in runs]]
