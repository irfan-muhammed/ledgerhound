"""Gates checked against a run_python cell BEFORE it executes."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol


class CodeGate(Protocol):
    """A pre-execution check on one cell. `kind` names the refusal in the
    event log (``goal_gate`` events)."""

    kind: str

    def check(self, code: str) -> str: ...


class ModelingGate:
    """Model training is refused until a human authorises it, or until the
    active methodology reaches a phase that produces a model. The rule
    itself lives on SessionGoal.check_modeling."""

    kind = "modeling_unauthorised"

    def __init__(self, goal: Callable[[], Any], methodology: Callable[[], Any]):
        self._goal = goal
        self._methodology = methodology

    def check(self, code: str) -> str:
        return self._goal().check_modeling(code, methodology=self._methodology())


class ExportGrainGate:
    """With an account-level goal, exporting a frame that has more than one
    row per entity key is refused BEFORE the write runs, so a wrong-grain
    deliverable is never produced. Only frames that already exist can be
    inspected (one built and exported in the same cell is caught by the
    post-execution grain check instead).

    `frames` returns ``{var_name: {"n_rows": int, "dup_keys": {col: n}}}``
    for the frames currently in the execution namespace. Best effort: any
    failure allows the cell -- a broken gate must not block legitimate work.
    """

    kind = "grain_violation"

    def __init__(self, goal: Callable[[], Any], methodology: Callable[[], Any],
                 frames: Callable[[], dict]):
        self._goal = goal
        self._methodology = methodology
        self._frames = frames

    def check(self, code: str) -> str:
        goal = self._goal()
        if not goal.is_set() or goal.unit_of_analysis != "account":
            return ""
        try:
            from ledgerhound.shell.session_goal import _EXPORT_RE
            if not _EXPORT_RE.search(code):
                return ""
            return goal.check_export(code, self._frames(),
                                     methodology=self._methodology())
        except Exception:  # noqa: BLE001 -- a broken gate must not block work
            return ""


class StatisticalAssumptionsGate:
    """V5-003: refuses a statistical test that cannot be valid as written;
    for one that merely needs care it allows the cell and leaves the caveat
    in `advisory`, which the caller appends to the cell's output so the
    numbers and their caveat travel together.

    `n_rows` gives the true frame size, or None when only a sample is held.
    Each issue is reported once per session (`seen`)."""

    kind = "invalid_test"

    def __init__(self, n_rows: Callable[[], Optional[int]],
                 on_issue: Callable[..., None] = lambda **kw: None):
        self._n_rows = n_rows
        self._on_issue = on_issue
        self.seen: set = set()
        self.advisory: str = ""

    def check(self, code: str) -> str:
        self.advisory = ""
        try:
            from ledgerhound.shell.stat_checks import (
                check_statistical_assumptions,
                is_blocking,
                render,
            )
            issues = check_statistical_assumptions(
                code, n_rows=self._n_rows(), seen=self.seen)
            if not issues:
                return ""
            block = render(issues)
            for i in issues:
                self._on_issue(test=i.test, kind=i.kind, blocking=i.blocking,
                               detail=i.message[:1000])
            if is_blocking(issues):
                return block
            self.advisory = block
            return ""
        except Exception:  # noqa: BLE001 -- a check must never break a cell
            return ""


class ExecBlocklist:
    """Exec mode only (no kernel): a substring blocklist on the cell.

    This is NOT a security boundary -- substring matching is trivially
    bypassed (e.g. ``getattr(__builtins__, ...)``). It stops accidental
    file-system and process access by a well-behaved model; real isolation
    is the kernel subprocess. Kept as-is during the Phase 1 refactor; the
    design doc removes exec mode in favour of an always-isolated kernel.
    """

    kind = "exec_blocklist"
    BLOCKED = ("import os", "import sys", "import subprocess", "import shutil",
               "open(", "__import__", "eval(", "exec(", "requests.", "socket.")

    def check(self, code: str) -> str:
        low = code.lower()
        for bad in self.BLOCKED:
            if bad in low:
                return (f"blocked: '{bad}' is not allowed in the sandbox "
                        "(df/pd/np are in scope).")
        return ""
