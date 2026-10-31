"""
sentinel.memory.skill_memory
============================

Self-evolving skills (rank 11) -- MemSkill / ProcMEM-style procedural memory
built on top of :class:`~sentinel.memory.lesson_memory.LessonMemory`.

Where a *lesson* is a single "this worked / this bit us" note, a *skill* is a
distilled, reusable **procedure**: a named sequence of steps with a trigger
(when to apply it), plus a track record. Skills are the agent improving its
own playbooks:

- **distil**: repeated lessons that co-occur under a tag get promoted into a
  skill (the recurring procedure behind them).
- **recall**: given a new task, surface the applicable skills (trigger/step
  match, ranked by effective confidence).
- **reinforce**: each application records success/failure -- confidence rises
  with corroborated successes and *decays* when a skill goes stale, so the
  playbook self-corrects instead of ossifying.

Confidence = Laplace-smoothed success rate x a recency decay, so a skill that
stops working (or stops being used) fades and can be pruned. Pure-Python and
dependency-free (mirrors LessonMemory), so it runs anywhere.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HALF_LIFE_DAYS = 30.0  # confidence halves after this long unused


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


@dataclass
class Skill:
    """A distilled, reusable procedure with a track record."""

    id: str
    name: str
    trigger: str                       # when to apply (free text / keywords)
    steps: list[str] = field(default_factory=list)
    dataset: str = "*"                 # "*" = general
    source_lesson_ids: list[str] = field(default_factory=list)
    uses: int = 0
    successes: int = 0
    created_at: str = field(default_factory=_now)
    last_used: str = ""

    def success_rate(self) -> float:
        # Laplace smoothing: unproven skills start near 0.5, not 1.0
        return (self.successes + 1) / (self.uses + 2)

    def _age_days(self, now: Optional[datetime] = None) -> float:
        stamp = self.last_used or self.created_at
        try:
            t = datetime.fromisoformat(stamp)
        except Exception:
            return 0.0
        now = now or datetime.now(timezone.utc)
        return max((now - t).total_seconds() / 86400.0, 0.0)

    def effective_confidence(self, now: Optional[datetime] = None) -> float:
        """Success rate discounted by a recency half-life decay."""
        decay = math.pow(0.5, self._age_days(now) / _HALF_LIFE_DAYS)
        return round(self.success_rate() * decay, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = round(self.success_rate(), 4)
        d["effective_confidence"] = self.effective_confidence()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


class SkillMemory:
    """Append-only, name-keyed store of self-evolving skills (JSONL)."""

    def __init__(self, store_path: str | Path):
        self.path = Path(store_path)
        self._skills: dict[str, Skill] = {}
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s = Skill.from_dict(json.loads(line))
                self._skills[s.name] = s
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping bad skill line: %s", exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for s in self._skills.values():
                f.write(json.dumps({k: v for k, v in asdict(s).items()},
                                   default=str) + "\n")

    # -- CRUD ----------------------------------------------------------------

    def learn(self, name: str, trigger: str, steps: list[str], *,
              dataset: str = "*", source_lesson_ids: Optional[list[str]] = None
              ) -> Skill:
        """Add or update a skill by name (idempotent). Persists."""
        existing = self._skills.get(name)
        if existing:
            existing.trigger = trigger or existing.trigger
            if steps:
                existing.steps = steps
            if source_lesson_ids:
                existing.source_lesson_ids = sorted(
                    set(existing.source_lesson_ids) | set(source_lesson_ids))
            skill = existing
        else:
            skill = Skill(id=f"sk_{uuid.uuid4().hex[:8]}", name=name,
                          trigger=trigger, steps=list(steps), dataset=dataset,
                          source_lesson_ids=list(source_lesson_ids or []))
            self._skills[name] = skill
        self.save()
        return skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    # -- self-evolution ------------------------------------------------------

    def reinforce(self, name: str, success: bool) -> Optional[Skill]:
        """Record an application outcome; confidence adapts. Persists."""
        s = self._skills.get(name)
        if s is None:
            return None
        s.uses += 1
        if success:
            s.successes += 1
        s.last_used = _now()
        self.save()
        return s

    def prune(self, *, min_confidence: float = 0.15) -> list[str]:
        """Drop skills whose effective confidence has decayed below floor.

        Only skills that have actually been tried (uses > 0) are eligible --
        a brand-new unproven skill is never pruned for being new.
        """
        now = datetime.now(timezone.utc)
        dropped = [n for n, s in self._skills.items()
                   if s.uses > 0 and s.effective_confidence(now) < min_confidence]
        for n in dropped:
            del self._skills[n]
        if dropped:
            self.save()
        return dropped

    # -- distillation from lessons --------------------------------------------

    def distil(self, lessons: list, *, min_support: int = 2) -> list[Skill]:
        """Promote recurring lesson clusters into skills.

        Lessons sharing a tag with at least ``min_support`` distinct members
        become a skill named for that tag, whose steps are the lesson texts.
        Idempotent: re-running updates the same skills rather than duplicating.
        """
        by_tag: dict[str, list] = {}
        for les in lessons:
            for tag in getattr(les, "tags", []) or []:
                by_tag.setdefault(tag, []).append(les)
        learned: list[Skill] = []
        for tag, group in by_tag.items():
            if len(group) < min_support:
                continue
            name = f"skill_{tag}"
            steps = [getattr(l, "text", "") for l in group]
            ids = [getattr(l, "id", "") for l in group]
            datasets = {getattr(l, "dataset", "*") for l in group}
            ds = datasets.pop() if len(datasets) == 1 else "*"
            learned.append(self.learn(
                name, trigger=f"tasks tagged '{tag}'", steps=steps,
                dataset=ds, source_lesson_ids=ids))
        return learned

    # -- recall ---------------------------------------------------------------

    def recall(self, query: str, *, dataset: Optional[str] = None,
               k: int = 5) -> list[Skill]:
        """Return applicable skills ranked by match x effective confidence."""
        q = _tokens(query)
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, Skill]] = []
        for s in self._skills.values():
            if dataset and s.dataset not in ("*", dataset):
                continue
            hay = _tokens(s.trigger + " " + s.name + " " + " ".join(s.steps))
            overlap = len(q & hay)
            if q and overlap == 0:
                continue
            match = overlap / len(q) if q else 1.0
            score = (0.5 + 0.5 * match) * s.effective_confidence(now)
            scored.append((score, s))
        scored.sort(key=lambda t: -t[0])
        return [s for _, s in scored[:k]]

    def block(self, query: str = "", *, dataset: Optional[str] = None,
              max_items: int = 5) -> str:
        """Prompt-injection block of applicable skills (empty if none)."""
        hits = self.recall(query, dataset=dataset, k=max_items)
        if not hits:
            return ""
        lines = ["LEARNED SKILLS (apply if relevant):"]
        for s in hits:
            steps = "; ".join(s.steps[:4])
            lines.append(f"- {s.name} [{s.effective_confidence():.2f}] "
                         f"when {s.trigger}: {steps}")
        lines.append("END LEARNED SKILLS")
        return "\n".join(lines)

    def stats(self) -> dict:
        proven = [s for s in self._skills.values() if s.uses > 0]
        return {
            "skills": len(self._skills),
            "proven": len(proven),
            "avg_confidence": round(
                sum(s.effective_confidence() for s in self._skills.values())
                / len(self._skills), 3) if self._skills else 0.0,
        }
