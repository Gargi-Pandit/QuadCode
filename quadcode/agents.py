"""The four agent slots and a pool that runs their calls in parallel or in sequence.

Each ``Agent`` is one role bound to the gateway, which routes it to that slot's own,
independently configured client. Agents never share state: everything an agent sees is
in the prompt it is given.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from quadcode.gateway import BudgetExhausted, CallOutcome, LLMGateway
from quadcode.prompts import system_prompt
from quadcode.schemas import CallType, Role


class Agent:
    def __init__(self, role: Role, gateway: LLMGateway) -> None:
        self.role = role
        self.gateway = gateway

    @property
    def provider(self) -> str:
        return self.gateway.config.agent(self.role).provider

    @property
    def model(self) -> str:
        return self.gateway.config.agent(self.role).model

    def ask[T: BaseModel](self, call_type: CallType, prompt: str, schema: type[T], tag: str) -> CallOutcome[T]:
        return self.gateway.call(self.role, call_type, system_prompt(self.role), prompt, schema, tag=tag)


Task = Callable[[Agent], CallOutcome[Any]]


@dataclass
class PoolResult:
    outcomes: dict[Role, CallOutcome[Any]] = field(default_factory=dict)
    budget_exhausted: bool = False
    """At least one call hit the call budget; the outcomes hold only calls that finished."""


class AgentPool:
    def __init__(self, gateway: LLMGateway, parallel: bool) -> None:
        self.agents = {role: Agent(role, gateway) for role in Role}
        self.parallel = parallel

    def run(self, tasks: Mapping[Role, Task]) -> PoolResult:
        """Run one call per listed role. Results come back in role order whatever order they finish."""
        result = PoolResult()
        roles = [r for r in Role if r in tasks]
        if self.parallel and len(roles) > 1:
            with ThreadPoolExecutor(max_workers=len(roles), thread_name_prefix="agent") as pool:
                futures = {r: pool.submit(tasks[r], self.agents[r]) for r in roles}
                for role in roles:
                    try:
                        result.outcomes[role] = futures[role].result()
                    except BudgetExhausted:
                        result.budget_exhausted = True
        else:
            for role in roles:
                if result.budget_exhausted:
                    break
                try:
                    result.outcomes[role] = tasks[role](self.agents[role])
                except BudgetExhausted:
                    result.budget_exhausted = True
        return result
