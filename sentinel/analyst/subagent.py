"""
sentinel.analyst.subagent
=========================

Phase 4: subagent spawning -- the Task-tool analog. The Analyst delegates a
*bounded* objective to a named specialist (agent_def spec: profiler / sampling /
comparison / feature_engineering / typology_comparison / a user agent). A
subagent runs in **isolated context** (its own FindingsLog + output dir) and
returns a *compact* result -- so its verbose trace never pollutes the main
loop's context. This is the context-economy tool, distinct from the fluid
in-context analysis the main Analyst does itself.

Reuse: construction goes through the existing ``agents.factory.load_agent`` /
``agent_from_spec``; a subagent is just a spec-built EDAAgent. The heavy build
is injected (``agent_builder``) so this module is testable without an LLM.

Parallelism: ``spawn_many`` runs independent subagents concurrently (threads --
the work is IO/LLM-bound), each fully isolated. Big-data constraint holds: the
subagent computes in its own sandbox; only its compact summary returns.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sentinel.analyst.analyst import Analyst

# agent_builder(role, analyst, out_dir) -> agent with .ask(question)->AgentResult
AgentBuilder = Callable[[str, Analyst, str], Any]


@dataclass
class SubagentResult:
    role: str
    objective: str
    answer: str = ""
    finding_ids: list[str] = field(default_factory=list)
    out_dir: str = ""
    ok: bool = True
    error: str = ""

    def summary(self) -> str:
        """The compact line that returns to the main context (not the trace)."""
        if not self.ok:
            return f"[{self.role}] FAILED: {self.error}"
        fc = f"{len(self.finding_ids)} finding(s)" if self.finding_ids else "no findings"
        head = self.answer.strip().splitlines()[0] if self.answer.strip() else ""
        return f"[{self.role}] {head} ({fc})"


def _default_builder(role: str, analyst: Analyst, out_dir: str):
    """Production builder: a spec-built specialist over the grounded data, with
    an ISOLATED FindingsLog + output dir. Raises a friendly error if the model
    or data is missing."""
    data = analyst.grounding.data.strip()
    if not data or not Path(data).exists():
        raise RuntimeError(f"data path not usable: {data!r}")
    cfg = analyst.project.config() if analyst.project is not None else None
    model_ref = getattr(cfg, "model_config_ref", "") if cfg else ""
    if not model_ref:
        raise RuntimeError("no model configured -- set one with /model")

    from sentinel.agents.factory import load_agent
    from sentinel.core import Sentinel
    from sentinel.findings import FindingsLog
    from sentinel.llm.client import LLMClient

    llm = LLMClient.from_config(model_ref)
    eng = Sentinel(data=data)
    findings_p = str(Path(out_dir) / "findings.jsonl")
    agent = load_agent(role, llm=llm, backend=eng.backend, knowledge=eng.knowledge,
                       findings_log=FindingsLog(findings_p), output_dir=out_dir)
    agent.set_digest(eng.profile())
    return agent


def spawn_subagent(analyst: Analyst, role: str, objective: str, *,
                   out_dir: str | Path = "outputs/shell/subagents",
                   agent_builder: AgentBuilder = _default_builder,
                   inject_preamble: bool = False) -> SubagentResult:
    """Run one specialist on a bounded objective in isolated context."""
    sub_dir = str(Path(out_dir) / _slug(role))
    try:
        agent = agent_builder(role, analyst, sub_dir)
    except Exception as e:  # noqa: BLE001
        return SubagentResult(role, objective, ok=False, error=str(e))
    # Subagents usually get a clean objective, not the whole main context.
    question = objective
    if inject_preamble:
        pre = analyst.context_preamble()
        question = f"{pre}\n\n# Objective\n{objective}" if pre else objective
    try:
        result = agent.ask(question)
    except Exception as e:  # noqa: BLE001
        return SubagentResult(role, objective, ok=False, error=str(e),
                              out_dir=sub_dir)
    return SubagentResult(
        role=role, objective=objective,
        answer=getattr(result, "answer", str(result)),
        finding_ids=list(getattr(result, "finding_ids", []) or []),
        out_dir=sub_dir)


def spawn_many(analyst: Analyst, jobs: list[tuple[str, str]], *,
               out_dir: str | Path = "outputs/shell/subagents",
               agent_builder: AgentBuilder = _default_builder,
               max_workers: int = 3) -> list[SubagentResult]:
    """Run several (role, objective) subagents concurrently, each isolated."""
    if not jobs:
        return []

    def _one(job: tuple[str, str]) -> SubagentResult:
        role, objective = job
        return spawn_subagent(analyst, role, objective, out_dir=out_dir,
                              agent_builder=agent_builder)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as ex:
        return list(ex.map(_one, jobs))


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s.lower()).strip("_") or "sub"
