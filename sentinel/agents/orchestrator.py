"""
sentinel.agents.orchestrator
============================

The Orchestrator + playbooks-as-code (rank 12 -- the capstone).

An analyst study is a **DAG of agent tasks**: each task names a specialist
agent (profiler / sampling / comparison / feature_engineering /
typology_comparison / eda / a user agent) and an objective, and may depend on
earlier tasks. The Orchestrator validates the DAG, runs tasks in dependency
order (grouping independent tasks into levels that *may* run in parallel), and
-- because every task shares one FindingsLog -- lets downstream agents build on
upstream findings via ``list_findings``. It then synthesises one
``study_report/v1``.

Playbooks are data (``study/v1`` YAML), so an analyst can version, share, and
re-run a whole multi-agent study without code. Agent construction is injected
via an ``agent_factory`` so the orchestration logic is testable without an LLM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

STUDY_SCHEMA = "study/v1"


# -- plan model --------------------------------------------------------------


@dataclass
class StudyTask:
    id: str
    agent: str
    objective: str
    depends_on: list[str] = field(default_factory=list)
    data: Optional[str] = None      # per-task data override

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StudyPlan:
    name: str
    tasks: list[StudyTask] = field(default_factory=list)
    schema: str = STUDY_SCHEMA

    # -- validation / ordering ----------------------------------------------

    def validate(self) -> "StudyPlan":
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("study task ids must be unique")
        idset = set(ids)
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in idset:
                    raise ValueError(f"task {t.id!r} depends on unknown {dep!r}")
        self.levels()  # raises on cycle
        return self

    def levels(self) -> list[list[StudyTask]]:
        """Topological levels; tasks in the same level are independent."""
        by_id = {t.id: t for t in self.tasks}
        indeg = {t.id: len(t.depends_on) for t in self.tasks}
        remaining = set(by_id)
        levels: list[list[StudyTask]] = []
        while remaining:
            ready = sorted(i for i in remaining if indeg[i] == 0)
            if not ready:
                raise ValueError("cycle detected in study plan dependencies")
            levels.append([by_id[i] for i in ready])
            for i in ready:
                remaining.discard(i)
                for t in self.tasks:
                    if i in t.depends_on and t.id in indeg:
                        indeg[t.id] -= 1
        return levels

    def order(self) -> list[StudyTask]:
        return [t for level in self.levels() for t in level]

    # -- (de)serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {"schema": self.schema, "name": self.name,
                "tasks": [t.to_dict() for t in self.tasks]}

    @classmethod
    def from_dict(cls, d: dict) -> "StudyPlan":
        tasks = [StudyTask(**{k: v for k, v in t.items()
                              if k in StudyTask.__dataclass_fields__})
                 for t in d.get("tasks", [])]
        return cls(name=d.get("name", "study"), tasks=tasks)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StudyPlan":
        import yaml
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})


# -- report model ------------------------------------------------------------


@dataclass
class StudyTaskResult:
    id: str
    agent: str
    objective: str
    answer: str = ""
    finding_ids: list[str] = field(default_factory=list)
    status: str = "success"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StudyReport:
    plan_name: str
    schema: str = "study_report/v1"
    results: list[StudyTaskResult] = field(default_factory=list)
    summary: str = ""
    elapsed_ms: float = 0.0

    @property
    def total_findings(self) -> int:
        return sum(len(r.finding_ids) for r in self.results)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status != "success")

    def to_dict(self) -> dict:
        return {"schema": self.schema, "plan_name": self.plan_name,
                "summary": self.summary, "elapsed_ms": self.elapsed_ms,
                "results": [r.to_dict() for r in self.results]}

    def to_markdown(self) -> str:
        lines = [f"# Study report -- {self.plan_name}", "",
                 f"*{len(self.results)} tasks, {self.total_findings} findings, "
                 f"{self.failed} failed, {self.elapsed_ms/1000:.1f}s*"]
        for r in self.results:
            lines.append(f"\n## [{r.agent}] {r.id}")
            lines.append(f"*objective:* {r.objective}")
            lines.append(r.answer if r.status == "success"
                         else f"**FAILED:** {r.error}")
        if self.summary:
            lines.append("\n---\n## Executive summary\n")
            lines.append(self.summary)
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        return p


# -- orchestrator -------------------------------------------------------------


class Orchestrator:
    """Run a :class:`StudyPlan` DAG over the specialist agents.

    Parameters
    ----------
    agent_factory
        ``callable(task: StudyTask) -> agent`` returning an object with an
        ``ask(objective, interactive=...) -> result`` method (an EDAAgent, or a
        fake in tests). Injecting this keeps orchestration testable without an
        LLM. The default (see :func:`default_agent_factory`) builds agents from
        the registry over each task's data.
    interactive
        Stream progress to the console.
    """

    def __init__(self, agent_factory: Callable[[StudyTask], Any], *,
                 interactive: bool = True):
        self.agent_factory = agent_factory
        self.interactive = interactive

    def run(self, plan: StudyPlan) -> StudyReport:
        plan.validate()
        report = StudyReport(plan_name=plan.name)
        t0 = time.perf_counter()
        for level in plan.levels():
            # Tasks in a level are independent (parallel-safe by dependency);
            # executed sequentially here for agent/LLM-state safety.
            for task in level:
                report.results.append(self._run_task(task))
        report.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return report

    def _run_task(self, task: StudyTask) -> StudyTaskResult:
        if self.interactive:
            logger.info("Orchestrator: task %s -> agent %s", task.id, task.agent)
        try:
            agent = self.agent_factory(task)
            ar = agent.ask(task.objective, interactive=self.interactive)
            return StudyTaskResult(
                id=task.id, agent=task.agent, objective=task.objective,
                answer=getattr(ar, "answer", ""),
                finding_ids=list(getattr(ar, "finding_ids", [])),
                status="success")
        except Exception as exc:  # noqa: BLE001
            logger.warning("task %s failed: %s", task.id, exc)
            return StudyTaskResult(
                id=task.id, agent=task.agent, objective=task.objective,
                status="failed", error=str(exc))


def default_agent_factory(
    llm: Any, *, findings_log: Any, default_data: Optional[str] = None,
    registry: Any = None, output_dir: str = "outputs/study",
) -> Callable[[StudyTask], Any]:
    """Build a factory that materialises agents from the registry per task.

    One backend/digest is built per distinct data path (cached). Every agent
    shares ``findings_log`` so the study compounds.
    """
    from sentinel.agents.factory import load_agent
    from sentinel.core import Sentinel

    if registry is None:
        from sentinel.agents.registry import AgentRegistry
        registry = AgentRegistry().load()
    _engine_cache: dict[str, Any] = {}

    def _engine(data: str):
        if data not in _engine_cache:
            eng = Sentinel(data=data)
            eng._digest = eng.profile()
            _engine_cache[data] = eng
        return _engine_cache[data]

    def factory(task: StudyTask):
        data = task.data or default_data
        if not data:
            raise ValueError(f"task {task.id!r} has no data (set task.data or "
                             "pass default_data)")
        eng = _engine(data)
        agent = load_agent(task.agent, llm=llm, backend=eng.backend,
                           knowledge=eng.knowledge, findings_log=findings_log,
                           output_dir=output_dir)
        agent.set_digest(eng._digest)
        return agent

    return factory
