"""
sentinel.memory.skill_exec
==========================

V5-006. A learned skill that is runnable code, and is only trusted once it has
been made to run.

``memory/skill_memory.py`` already has the feedback half of this: skills carry
a success rate, `reinforce()` updates it, `prune()` drops what stops working,
and confidence decays with age. What it does not have is *executability* --
`Skill.steps` is `list[str]`, prose for a model to read. Prose cannot be
validated, so a skill that quietly stopped being correct looks exactly like one
that still works until somebody notices the outputs are wrong.

EvoDS (arXiv 2606.03841, KDD 2026) calls the missing piece Autonomous Skill
Acquisition: synthesise, **validate**, reuse. The validate step is the whole
point. This module is that step.

The contract
------------
An `ExecutableSkill` carries a Python function body, the names it expects, and
at least one **self-check** -- an assertion over the skill's own output that
must hold. A skill is `VALIDATED` only after every check has actually passed on
real input. Until then it is `UNVALIDATED` and callers are told so.

Why a self-check rather than a test suite: a skill is learned mid-session from
work that just happened, and there is no opportunity to author tests for it. A
check the skill states about its own output ("the result has one row per
account", "no value is negative") is cheap to write at the moment of learning
and is exactly what catches the skill going stale later.

Safety
------
Skill code runs in the same restricted shape as an analyst cell -- executed
against a namespace the caller supplies, with the same import discipline. This
is not a sandbox and does not pretend to be one: a skill is code this project
generated and recorded, carrying the same trust as the cell it was distilled
from. What it adds over `exec()` is that the code is *named, versioned,
checked, and revocable*.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

UNVALIDATED = "unvalidated"
VALIDATED = "validated"
FAILED = "failed"           # ran, but a self-check did not hold
BROKEN = "broken"           # raised


@dataclass
class SelfCheck:
    """One assertion the skill makes about its own output."""
    description: str
    expression: str          # evaluated with `result` (and the namespace) bound

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationOutcome:
    status: str
    checks_passed: int = 0
    checks_total: int = 0
    error: str = ""
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == VALIDATED

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutableSkill:
    """A learned procedure as runnable, checkable code."""
    name: str
    description: str
    code: str                                    # defines `def run(...)`
    expects: list[str] = field(default_factory=list)   # namespace names needed
    checks: list[SelfCheck] = field(default_factory=list)
    status: str = UNVALIDATED
    n_validations: int = 0
    n_failures: int = 0
    last_error: str = ""
    learned_at: float = field(default_factory=time.time)
    last_validated_at: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [c.to_dict() if isinstance(c, SelfCheck) else c
                       for c in self.checks]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutableSkill":
        d = dict(d)
        d["checks"] = [SelfCheck(**c) if isinstance(c, dict) else c
                       for c in d.get("checks", [])]
        return cls(**d)

    @property
    def trustworthy(self) -> bool:
        """Validated at least once and not broken since."""
        return self.status == VALIDATED and self.n_validations > 0


class SkillValidationError(RuntimeError):
    pass


def _compile(skill: ExecutableSkill, namespace: dict) -> Any:
    local: dict = {}
    exec(compile(skill.code, f"<skill:{skill.name}>", "exec"), namespace, local)
    fn = local.get("run") or namespace.get("run")
    if fn is None or not callable(fn):
        raise SkillValidationError(
            f"skill {skill.name!r} defines no callable `run(...)`")
    return fn


def validate(skill: ExecutableSkill, namespace: dict, **kwargs
             ) -> ValidationOutcome:
    """Run the skill against a real namespace and hold it to its own checks.

    Mutates `skill` in place with the outcome -- the record of whether a
    skill still works belongs on the skill, not in the caller's head.
    """
    started = time.perf_counter()
    ns = dict(namespace)

    missing = [n for n in skill.expects if n not in ns]
    if missing:
        skill.status = BROKEN
        skill.n_failures += 1
        skill.last_error = f"namespace is missing {missing}"
        return ValidationOutcome(status=BROKEN, error=skill.last_error,
                                 checks_total=len(skill.checks))

    try:
        fn = _compile(skill, ns)
        result = fn(**kwargs) if kwargs else fn()
    except Exception as e:  # noqa: BLE001
        skill.status = BROKEN
        skill.n_failures += 1
        skill.last_error = f"{type(e).__name__}: {e}"
        return ValidationOutcome(
            status=BROKEN, checks_total=len(skill.checks),
            error=traceback.format_exc(limit=3)[-1500:],
            duration_ms=(time.perf_counter() - started) * 1000)

    if not skill.checks:
        # A skill with nothing to check cannot be validated, only run. Saying
        # so is the point -- silently promoting it would make `validated`
        # meaningless for every other skill.
        skill.status = UNVALIDATED
        skill.last_error = "no self-checks declared; cannot be validated"
        return ValidationOutcome(status=UNVALIDATED, error=skill.last_error,
                                 duration_ms=(time.perf_counter() - started) * 1000)

    ns["result"] = result
    passed = 0
    for check in skill.checks:
        try:
            if bool(eval(check.expression, ns)):  # noqa: S307
                passed += 1
            else:
                skill.status = FAILED
                skill.n_failures += 1
                skill.last_error = (
                    f"self-check failed: {check.description} "
                    f"({check.expression})")
                return ValidationOutcome(
                    status=FAILED, checks_passed=passed,
                    checks_total=len(skill.checks), error=skill.last_error,
                    duration_ms=(time.perf_counter() - started) * 1000)
        except Exception as e:  # noqa: BLE001
            skill.status = BROKEN
            skill.n_failures += 1
            skill.last_error = (f"self-check raised: {check.description} "
                                f"-- {type(e).__name__}: {e}")
            return ValidationOutcome(
                status=BROKEN, checks_passed=passed,
                checks_total=len(skill.checks), error=skill.last_error,
                duration_ms=(time.perf_counter() - started) * 1000)

    skill.status = VALIDATED
    skill.n_validations += 1
    skill.last_error = ""
    skill.last_validated_at = time.time()
    return ValidationOutcome(status=VALIDATED, checks_passed=passed,
                             checks_total=len(skill.checks),
                             duration_ms=(time.perf_counter() - started) * 1000)


def run(skill: ExecutableSkill, namespace: dict, **kwargs) -> Any:
    """Execute a skill that has been validated. Refuses one that has not.

    The refusal is the feature. An unvalidated skill is a guess with a name,
    and a guess that runs silently is worse than no skill at all.
    """
    if not skill.trustworthy:
        raise SkillValidationError(
            f"skill {skill.name!r} is {skill.status} and has not been "
            f"validated -- call validate() against real input first. "
            f"{('Last error: ' + skill.last_error) if skill.last_error else ''}")
    ns = dict(namespace)
    fn = _compile(skill, ns)
    return fn(**kwargs) if kwargs else fn()


class ExecutableSkillStore:
    """JSONL-backed store. Never raises on load: a corrupt skill file must
    not cost the analyst their session."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.skills: dict[str, ExecutableSkill] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                s = ExecutableSkill.from_dict(json.loads(line))
                self.skills[s.name] = s
            except Exception:  # noqa: BLE001
                continue

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                "\n".join(json.dumps(s.to_dict(), default=str)
                          for s in self.skills.values()) + "\n",
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def add(self, skill: ExecutableSkill) -> None:
        self.skills[skill.name] = skill
        self.save()

    def get(self, name: str) -> ExecutableSkill | None:
        return self.skills.get(name)

    def trustworthy(self) -> list[ExecutableSkill]:
        return [s for s in self.skills.values() if s.trustworthy]

    def revoke(self, name: str) -> bool:
        """Drop a skill outright. The counterpart to `prune()` in
        skill_memory: a skill that broke should stop being offered."""
        if name in self.skills:
            del self.skills[name]
            self.save()
            return True
        return False

    def summary(self) -> str:
        if not self.skills:
            return "No executable skills learned."
        lines = ["Executable skills:"]
        for s in sorted(self.skills.values(), key=lambda x: x.name):
            lines.append(
                f"  {s.name}: {s.status} "
                f"({s.n_validations} validations, {s.n_failures} failures)"
                + (f" -- {s.last_error}" if s.last_error else ""))
        return "\n".join(lines)
