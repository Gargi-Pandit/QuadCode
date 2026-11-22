"""Build benchmark harnesses on mock agents, the subprocess sandbox and a temporary cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from quadcode.cli_solve import mock_config
from quadcode.config import AppConfig, Secrets
from quadcode.evaluation.conditions import Condition
from quadcode.evaluation.harness import BenchHarness, new_state
from quadcode.evaluation.problems import load_problems
from quadcode.evaluation.records import load_bench, save_bench
from quadcode.gateway import CallBudget, LLMGateway
from quadcode.sandbox import SubprocessSandbox

from .orch_helpers import make_config

PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "problems"


def bench_config(root: Path) -> AppConfig:
    cfg = make_config(run={"cache_dir": str(root / "cache")}, sandbox={
        "backend": "subprocess", "test_timeout_s": 1.0, "wall_timeout_s": 60.0, "test_repeats": 1,
        "benchmark_repeats": 1, "warmup": 1})
    return mock_config(cfg, None)


@dataclass
class Recorder:
    """Gateway factory that keeps every gateway, so tests can read what each mock agent was sent."""

    gateways: list[LLMGateway] = field(default_factory=list)

    def __call__(self, cfg: AppConfig, run_dir: Path, budget: CallBudget) -> LLMGateway:
        gw = LLMGateway.from_config(cfg, Secrets({}), run_dir, budget=budget)
        self.gateways.append(gw)
        return gw

    def prompts(self) -> list[str]:
        return [q.prompt for gw in self.gateways for chain in gw.clients.values() for c in chain
                for q in getattr(c, "requests", [])]


def make_harness(root: Path, bench_id: str, problems: list[str], conditions: list[Condition],
                 budget: int | None = None, resume: bool = False, rounds: int = 3,
                 recorder: Recorder | None = None) -> BenchHarness:
    cfg = bench_config(root)
    bench_dir = root / bench_id
    probs = load_problems(PROBLEMS_DIR, names=problems, mock=True)
    if resume:
        state = load_bench(bench_dir)
    else:
        bench_dir.mkdir(parents=True)
        state = new_state(bench_id, probs, conditions, rounds, True, cfg)
        save_bench(bench_dir, state)
    return BenchHarness(base_config=cfg, bench_dir=bench_dir, state=state, problems=probs,
                        sandbox=SubprocessSandbox(cfg.sandbox), make_gateway=recorder or Recorder(),
                        budget=CallBudget(budget), say=lambda _: None)
