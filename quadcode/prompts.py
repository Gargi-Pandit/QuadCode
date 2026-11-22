"""Prompts for every agent call.

Agents are addressed by role only; provider and vendor names never appear in prompts.
Every prompt asks for one JSON object and lists the fields it must contain. Prompts are
built only from logged data (problem, spec, stored code, test results, records), and they
contain no timings, so the same state always produces the same prompt (cache friendly).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from quadcode.config import PromptConfig
from quadcode.schemas import ModificationRecord, ProblemSpec, Review, Role, SuiteRun, TestCase, TestStatus

JSON_ONLY = ("Reply with exactly one JSON object and nothing else: no prose, no markdown fences, "
             "no comments inside the JSON.")

ROLE_BRIEF: dict[Role, str] = {
    Role.GENERATOR: (
        "You are the Generator in a team of four independent coding agents. Your focus is a correct, "
        "clean and readable implementation with well-chosen algorithms and data structures, and an honest "
        "account of its weaknesses."),
    Role.REVIEWER: (
        "You are the Reviewer in a team of four independent coding agents. Your focus is logical "
        "correctness, compliance with every requirement, architecture, error handling, security and "
        "maintainability. Challenge wrong assumptions, including those of other agents."),
    Role.OPTIMIZER: (
        "You are the Optimizer in a team of four independent coding agents. Your focus is time and space "
        "complexity, bottlenecks and redundant work. Distinguish theoretical improvements from improvements "
        "that would show up in measurements, and never trade correctness for speed."),
    Role.VALIDATOR: (
        "You are the Validator in a team of four independent coding agents. Your focus is adversarial "
        "testing: edge cases, boundary values, invalid inputs, large inputs and hidden logical errors. Try "
        "to break every implementation and do not simply agree with others."),
}


def system_prompt(role: Role) -> str:
    return (f"{ROLE_BRIEF[role]}\nClaims are checked by executing code in a sandbox; decisions are made on "
            f"test evidence and measurements, not on opinions.\n{JSON_ONLY}")


REPAIR_PREVIOUS_CHARS = 12_000


def output_limit_note(max_tokens: int) -> str:
    """Appended to every request: the reply has a hard size limit, and fitting it matters more than detail."""
    return (f"\n\nOUTPUT LIMIT: your whole reply, including any reasoning before it, must fit in {max_tokens} "
            f"tokens (about {max_tokens * 3 // 4} words). Keep it short: brief text fields, no repetition, nothing "
            "the JSON fields do not ask for. A complete, valid JSON object that fits is required; extra detail is "
            "not.")


def repair_prompt(prompt: str, previous: str | None, problem: str, cut_off: bool) -> str:
    """The original request again, plus what was wrong with the answer the model gave to it."""
    parts = [prompt, "", "--- Your previous answer to this request could not be used. ---", f"Problem: {problem}"]
    if cut_off:
        parts.append("It was cut off at the output limit. Give a shorter answer: keep every required field, "
                     "but use brief text and fewer, smaller test cases.")
    elif previous:
        parts += ["Your previous answer was:", truncate(previous, REPAIR_PREVIOUS_CHARS)]
    parts.append("JSON allows only literal values: write 1000000000, not 10**9 or 1e9 arithmetic; null, true "
                 "and false, not None, True and False; no comments and no trailing commas. Inputs too large to "
                 "write out belong in a test's \"generator\" expression, not in \"args\".")
    parts.append("Reply again with the complete JSON object only, with that problem fixed.")
    return "\n".join(parts)


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _fields(spec: Mapping[str, str]) -> str:
    return "\n".join(f'- "{k}": {v}' for k, v in spec.items())


def _spec_block(spec: ProblemSpec) -> str:
    return json.dumps(spec.model_dump(exclude={"clarification_needed", "clarification_questions"}),
                      indent=1, ensure_ascii=False)


def _code_block(label: str, code: str, language: str) -> str:
    return f"### {label}\n```{language}\n{code.rstrip()}\n```"


# ---------------------------------------------------------------- phase 1

ANALYSIS_FIELDS = {
    "title": "short title",
    "language": "programming language of the solution, lower case (python if unspecified)",
    "summary": "one or two sentences",
    "entry_point": "name of the function the solution must define",
    "signature": "full function signature",
    "inputs": "input specification",
    "outputs": "output specification",
    "functional_requirements": "list of strings",
    "constraints": "list of strings",
    "performance_constraints": "list of strings",
    "edge_cases": "list of strings",
    "dependencies": "list of external packages (empty if none)",
    "assumptions": "list of strings",
    "clarification_needed": "true only if essential information is missing",
    "clarification_questions": "list of questions for the user (empty unless clarification_needed)",
}


def analysis_prompt(problem: str) -> str:
    return (f"Analyse this programming problem and produce a structured specification.\n\n"
            f"PROBLEM:\n{problem.strip()}\n\nReturn a JSON object with these fields:\n{_fields(ANALYSIS_FIELDS)}")


# ---------------------------------------------------------------- phase 2

INITIAL_FIELDS = {
    "language": "programming language",
    "code": "the complete implementation as one string, defining the entry point",
    "algorithm": "explanation of the algorithm",
    "data_structures": "list of strings",
    "time_complexity": "e.g. O(n log n)",
    "space_complexity": "e.g. O(n)",
    "pros": "list of advantages",
    "cons": "list of potential weaknesses",
    "assumptions": "list of assumptions",
}

TEST_FIELDS = {
    "id": "unique id such as T01",
    "category": "normal | boundary | adversarial | performance",
    "description": "what the test checks",
    "args": "list of positional arguments as literal JSON values: write 1000000000, never an expression such "
            "as 10**9; inputs that must be computed go in generator",
    "kwargs": "object of keyword arguments",
    "generator": "optional Python expression (not a lambda or function) that evaluates to a pair "
                 "(list_of_args, dict_of_kwargs), e.g. ([list(range(100000)), 199999], {}), for large inputs; "
                 "else null",
    "expected": "expected return value (JSON)",
    "expect_exception": "exception class name if the call must raise, else null",
    "compare": "exact | unordered | float",
}


def initial_prompt(problem: str, spec: ProblemSpec, with_tests: bool) -> str:
    """The same problem and spec for every agent; only the validator is also asked for tests."""
    text = (f"Solve this problem independently. You will not see other agents' solutions at this stage.\n\n"
            f"PROBLEM:\n{problem.strip()}\n\nSPECIFICATION:\n{_spec_block(spec)}\n\n"
            f"Return a JSON object with these fields:\n{_fields(INITIAL_FIELDS)}")
    if with_tests:
        text += (f'\n- "tests": a test suite as a list of objects with fields:\n{_fields(TEST_FIELDS)}\n'
                 "Cover normal inputs, boundaries (empty, single element, duplicates, negatives, zero), "
                 "adversarial inputs and at least one performance test with a large generated input. "
                 "Expected values must follow the specification exactly; they will be checked against every "
                 "implementation.")
    return text


# ---------------------------------------------------------------- phase 3

REVIEW_FIELDS = {
    "items": "list with one object per implementation you review, each with: "
             '"target" (version label), "strengths", "weaknesses", "bugs", "suggestions" (lists of strings), '
             '"justification" (string)',
}


def results_table(suites: Mapping[str, SuiteRun], cfg: PromptConfig) -> str:
    """Compact pass/fail table: no timings, failing tests truncated."""
    lines = ["version | passed | failing tests"]
    for label, suite in suites.items():
        if not suite.executed:
            lines.append(f"{label} | not executed | {truncate(suite.not_executed_reason or '', cfg.max_error_chars)}")
            continue
        failing = [r for r in suite.results if r.status is not TestStatus.PASS]
        shown = "; ".join(f"{r.test_id} {r.status.value}: {truncate(r.error or '', cfg.max_error_chars)}"
                          for r in failing[: cfg.max_failing_tests])
        if len(failing) > cfg.max_failing_tests:
            shown += f"; ... {len(failing) - cfg.max_failing_tests} more"
        lines.append(f"{label} | {suite.count(TestStatus.PASS)}/{len(suite.results)} | {shown or '-'}")
    return "\n".join(lines)


def suite_listing(tests: Sequence[TestCase], cfg: PromptConfig) -> str:
    rows = []
    for t in tests:
        call = t.generator if t.generator else json.dumps({"args": t.args, "kwargs": t.kwargs})
        want = f"raises {t.expect_exception}" if t.expect_exception else json.dumps(t.expected)
        rows.append(f"{t.id} [{t.category.value}] {truncate(t.description, 60)}: "
                    f"{truncate(call, cfg.max_error_chars)} -> {truncate(want, 80)}")
    return "\n".join(rows)


def review_prompt(spec: ProblemSpec, own_label: str, codes: Mapping[str, str | None], table: str | None) -> str:
    """``table`` None withholds the sandbox results (no-test-feedback ablation)."""
    blocks = [_code_block(label, code, spec.language) if code is not None
              else f"### {label}\nUNAVAILABLE: this agent produced no implementation."
              for label, code in codes.items()]
    results = f"\n\nSANDBOX TEST RESULTS (measured):\n{table}" if table is not None else ""
    return (f"{_count_word(len(codes))} agents solved the same problem independently. Your own implementation is "
            f"{own_label}. Review the other implementations (you may also comment on your own). You may disagree "
            f"with anyone.\n\nSPECIFICATION:\n{_spec_block(spec)}\n\nIMPLEMENTATIONS:\n" + "\n\n".join(blocks)
            + f"{results}\n\n"
            "Evaluate correctness, efficiency, time and space complexity, readability, maintainability, "
            "scalability, error handling, vulnerabilities and requirement compliance.\n"
            f"Return a JSON object with:\n{_fields(REVIEW_FIELDS)}")


def _count_word(n: int) -> str:
    return {2: "Two", 3: "Three", 4: "Four"}.get(n, str(n))


def review_digest(reviews: Mapping[Role, Review], target: str) -> str:
    """Bugs, weaknesses and suggestions that the Phase 3 reviews raised about ``target``."""
    lines: list[str] = []
    for role, review in reviews.items():
        for item in review.items:
            if item.target != target:
                continue
            for kind, entries in (("bug", item.bugs), ("weakness", item.weaknesses),
                                  ("suggestion", item.suggestions)):
                lines.extend(f"- {role.value} ({kind}): {e}" for e in entries)
    return "\n".join(lines) or "- (no bugs, weaknesses or suggestions were raised about it)"


# ---------------------------------------------------------------- rounds

MODIFY_FIELDS = {
    "proposals": "list (possibly empty) of objects with: "
                 '"diff" (unified diff against solution.py with ---/+++ headers, @@ hunks and 3 lines of '
                 'unchanged context), "justification", "expected_impact", "disadvantages" (strings)',
}


def modify_prompt(spec: ProblemSpec, round_index: int, label: str, code: str, tests: Sequence[TestCase],
                  current_results: str | None, context: str, cfg: PromptConfig) -> str:
    """``current_results`` None withholds the sandbox results (no-test-feedback ablation)."""
    results = (f"RESULTS OF {label} (measured):\n{current_results}\n\n" if current_results is not None else "")
    return (f"Optimization round {round_index}. The current version is {label}. Propose zero or more "
            "independent modifications as unified diffs against it. Every proposal is applied on its own, "
            "executed against the test suite and accepted only on evidence: all tests must pass with no measured "
            "regression, and formatting- or comment-only changes are rejected. Propose nothing if you see no "
            "real improvement; empty rounds end the optimization.\n\n"
            f"SPECIFICATION:\n{_spec_block(spec)}\n\n"
            f"{_code_block(f'{label} (solution.py)', code, spec.language)}\n\n"
            f"TEST SUITE:\n{suite_listing(tests, cfg)}\n\n{results}"
            f"{context}\n\nReturn a JSON object with:\n{_fields(MODIFY_FIELDS)}")


NO_REVIEW_CONTEXT = "No cross review was run for this version, so there are no review comments."


def round_one_context(reviews: Mapping[Role, Review], source_label: str) -> str:
    return (f"Phase 3 review comments about {source_label}, which this version was copied from:\n"
            f"{review_digest(reviews, source_label)}")


WITHHELD_REASON = "rejected on the orchestrator's evidence (the reason is withheld: no test feedback is shown)"


def later_round_context(prev_label: str, accepted_diff: str, rejected: Sequence[ModificationRecord],
                        cfg: PromptConfig, with_results: bool = True) -> str:
    """``with_results`` False hides why proposals were rejected, since the reasons quote test output."""
    parts = [f"Changes accepted into this version (diff from {prev_label}):\n```diff\n{accepted_diff.rstrip()}\n```"]
    if rejected:
        parts.append("Proposals rejected in the previous round:")
        parts.extend(f"- {r.id} by {r.proposed_by.value}: {truncate(r.justification, 120)} -> "
                     + (f"rejected: {truncate(r.decision_reason, cfg.max_error_chars)}" if with_results
                        else WITHHELD_REASON) for r in rejected)
    return "\n".join(parts)


CHALLENGE_FIELDS = {
    "verdict": "accept | reject | needs_changes",
    "comment": "your technical argument",
    "evidence": "concrete evidence: a failing input, a test id, or a complexity argument",
}


def challenge_prompt(spec: ProblemSpec, record: ModificationRecord, reason: str | None, results: str | None) -> str:
    """``reason``/``results`` None withhold the test evidence (no-test-feedback ablation)."""
    why = (f"It is under challenge because: {reason}" if reason is not None else
           "It is under challenge because the orchestrator's evidence speaks against it (details withheld: no "
           "test feedback is shown).")
    shown = (f"SANDBOX RESULTS OF {record.candidate_version} (measured):\n{results}\n\n" if results is not None
             else "")
    return (f"{record.proposed_by.value} proposed modification {record.id} to {record.base_version}. "
            f"{why}\n\nSPECIFICATION:\n{_spec_block(spec)}\n\n"
            f"DIFF:\n```diff\n{record.diff.rstrip()}\n```\nJustification: {record.justification}\n"
            f"Expected impact: {record.expected_impact}\nDisadvantages: {record.disadvantages or '-'}\n\n"
            f"{shown}"
            "Should this modification be accepted? Your verdict is recorded and shown, but the decision is "
            f"made on test evidence and measurements.\nReturn a JSON object with:\n{_fields(CHALLENGE_FIELDS)}")
