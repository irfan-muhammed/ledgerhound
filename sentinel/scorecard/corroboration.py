"""
sentinel.scorecard.corroboration
================================

Build a cross-tool corroboration matrix from a workspace.

The corroboration score asks: *for accounts an analysis tool flagged as
suspicious, do other tools -- independently -- flag the same accounts?*
Two tools agreeing on an account is much stronger evidence than one tool
calling it alone. The score captures that agreement rate.

Sources we read from disk
-------------------------
* `reports/graph/motifs_*.json`        -- motif matches with severity field
* `reports/graph/mule_rings_*.json`    -- community detection alerts
* `reports/graph/fingerprints_*.json`  -- 14-signal fingerprint alerts
* `reports/graph/centrality_*.json`    -- pass-through / hub heuristic flags
* `findings/*.jsonl`                    -- any finding tagged to a specific account
* `reports/playbook/run_*.json`         -- typology + signal-decay step alerts
                                          carrying account_id lists in detail

The output is a CorroborationMatrix: account_id -> list of TooledAlert
entries (tool name, severity, optional score). The scorecard component
consumes this matrix to compute its score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

log = logging.getLogger(__name__)


# -- Records ----------------------------------------------------------------


@dataclass
class TooledAlert:
    """One tool's claim that a specific account is suspicious."""
    tool: str
    severity: str          # "ALERT" | "WARNING" | "INFO"
    score: Optional[float] = None   # tool-specific intensity (mule_likelihood, etc.)
    source_path: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorroborationMatrix:
    """All tools' alerts grouped by account."""
    alerts_by_account: dict[str, list[TooledAlert]] = field(default_factory=dict)
    alerts_by_tool: dict[str, set[str]] = field(default_factory=dict)
    sources_read: list[str] = field(default_factory=list)

    # -- Derived metrics --------------------------------------------------

    @property
    def total_alerted_accounts(self) -> int:
        return len(self.alerts_by_account)

    @property
    def total_tools_alerting(self) -> int:
        return sum(1 for v in self.alerts_by_tool.values() if v)

    @property
    def corroborated_accounts(self) -> list[str]:
        """Accounts flagged by 2+ tools."""
        return [
            acc for acc, alerts in self.alerts_by_account.items()
            if len({a.tool for a in alerts}) >= 2
        ]

    @property
    def corroboration_rate(self) -> Optional[float]:
        """Fraction of alerted accounts that have 2+ tool agreement.
        Returns None when no alerts exist at all (the vacuous case)."""
        total = self.total_alerted_accounts
        if total == 0:
            return None
        return len(self.corroborated_accounts) / total

    @property
    def is_vacuous(self) -> bool:
        """No tool flagged any account -- we cannot judge corroboration."""
        return self.total_alerted_accounts == 0

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_alerted_accounts":     self.total_alerted_accounts,
            "n_tools_alerting":       self.total_tools_alerting,
            "n_corroborated":         len(self.corroborated_accounts),
            "corroboration_rate":     self.corroboration_rate,
            "is_vacuous":             self.is_vacuous,
            "alerts_by_account":      {
                acc: [a.to_dict() for a in alerts]
                for acc, alerts in self.alerts_by_account.items()
            },
            "alerts_by_tool":         {k: sorted(v) for k, v in self.alerts_by_tool.items()},
            "sources_read":           self.sources_read,
        }

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p

    # -- For the existing scorecard component contract --------------------

    def to_alerts_by_tool(self) -> dict[str, set[str]]:
        """Adapter for sentinel.scorecard.components.corroboration_score(),
        which expects `dict[tool, set[account_id]]`."""
        return {k: set(v) for k, v in self.alerts_by_tool.items()}


# -- Builder ----------------------------------------------------------------


def build_corroboration_matrix(
    workspace,
    *,
    include_warnings: bool = True,
    include_findings: bool = True,
    include_playbook: bool = True,
) -> CorroborationMatrix:
    """Walk the workspace and assemble a CorroborationMatrix.

    Parameters
    ----------
    workspace : WorkspaceLayout-like with .reports_dir and .root
    include_warnings : also count WARNING severity (in addition to ALERT)
    include_findings : also consume findings/*.jsonl entries tied to accounts
    include_playbook : also consume reports/playbook/run_*.json step detail
    """
    matrix = CorroborationMatrix()
    severities = {"ALERT", "WARNING"} if include_warnings else {"ALERT"}

    # -- Graph reports ----------------------------------------------------
    gdir = workspace.reports_dir / "graph"
    if gdir.exists():
        for path in sorted(gdir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("could not read %s: %s", path, exc)
                continue
            matrix.sources_read.append(str(path))
            _consume_graph_report(matrix, data, str(path), severities)

    # -- Findings JSONL ---------------------------------------------------
    if include_findings:
        for fpath in _iter_findings_files(workspace):
            try:
                _consume_findings_file(matrix, fpath)
            except Exception as exc:
                log.warning("could not read findings %s: %s", fpath, exc)

    # -- Playbook step detail ---------------------------------------------
    if include_playbook:
        pdir = workspace.reports_dir / "playbook"
        if pdir.exists():
            for path in sorted(pdir.glob("run_*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                matrix.sources_read.append(str(path))
                _consume_playbook_report(matrix, data, str(path), severities)

    return matrix


# -- Per-source consumers ---------------------------------------------------


def _add(matrix: CorroborationMatrix, account: str, alert: TooledAlert) -> None:
    matrix.alerts_by_account.setdefault(account, []).append(alert)
    matrix.alerts_by_tool.setdefault(alert.tool, set()).add(account)


def _consume_graph_report(
    matrix: CorroborationMatrix, data: dict, src: str, severities: set[str],
) -> None:
    # Motif report
    if "matches" in data and "matches_by_pattern" in data:
        for m in data.get("matches", []):
            if m.get("severity") in severities:
                for n in m.get("nodes", []):
                    _add(matrix, str(n), TooledAlert(
                        tool="detect_mule_motifs",
                        severity=m.get("severity", "ALERT"),
                        score=m.get("score"),
                        source_path=src,
                        detail={"pattern": m.get("pattern")},
                    ))

    # Communities / mule rings
    if "communities" in data:
        for com in data.get("communities", []):
            if com.get("severity") in severities:
                for n in com.get("nodes", []):
                    _add(matrix, str(n), TooledAlert(
                        tool="find_mule_rings",
                        severity=com.get("severity", "ALERT"),
                        score=com.get("mule_score"),
                        source_path=src,
                        detail={"community_id": com.get("community_id")},
                    ))

    # Fingerprints
    if "fingerprints" in data:
        for fp in data.get("fingerprints", []):
            if fp.get("severity") in severities:
                _add(matrix, str(fp.get("account_id")), TooledAlert(
                    tool="fingerprint_accounts",
                    severity=fp.get("severity", "ALERT"),
                    score=fp.get("mule_likelihood"),
                    source_path=src,
                ))

    # Centrality -- pass-through heuristic (flow_balance <= 0.10, in-degree >= 3,
    # non-zero amounts both sides)
    if "accounts" in data and "measures_computed" in data:
        for a in data.get("accounts", []):
            fb = a.get("flow_balance")
            in_deg = int(a.get("in_degree", 0))
            if (
                fb is not None and fb <= 0.10 and in_deg >= 3
                and float(a.get("in_amount", 0)) > 0
                and float(a.get("out_amount", 0)) > 0
            ):
                _add(matrix, str(a.get("account_id")), TooledAlert(
                    tool="compute_account_centrality",
                    severity="ALERT",
                    score=fb,
                    source_path=src,
                    detail={"in_degree": in_deg, "flow_balance": fb},
                ))


def _iter_findings_files(workspace) -> Iterable[Path]:
    # Findings can live in a few places depending on workspace version.
    candidates = [
        workspace.root / "findings",
        getattr(workspace, "findings_dir", None),
    ]
    for c in candidates:
        if c is None:
            continue
        c = Path(c)
        if c.exists() and c.is_dir():
            yield from sorted(c.glob("*.jsonl"))


def _consume_findings_file(matrix: CorroborationMatrix, path: Path) -> None:
    """Extract account-tied alerts from findings JSONL. We look for
    `account_id` in the finding's `detail` dict or `evidence` list."""
    matrix.sources_read.append(str(path))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            confidence = float(d.get("confidence", 0.0) or 0.0)
            if confidence < 0.5:
                continue
            severity = "ALERT" if confidence >= 0.75 else "WARNING"
            accounts: list[str] = []
            detail = d.get("detail") or {}
            if isinstance(detail, dict):
                if isinstance(detail.get("account_id"), (str, int)):
                    accounts.append(str(detail["account_id"]))
                if isinstance(detail.get("account_ids"), list):
                    accounts.extend(str(x) for x in detail["account_ids"])
            for acc in accounts:
                _add(matrix, acc, TooledAlert(
                    tool="finding",
                    severity=severity,
                    score=confidence,
                    source_path=str(path),
                    detail={"finding_id": d.get("id"), "claim": (d.get("claim") or "")[:120]},
                ))


def _consume_playbook_report(
    matrix: CorroborationMatrix, data: dict, src: str, severities: set[str],
) -> None:
    """Extract account-level alerts from playbook step detail."""
    for step in data.get("steps", []):
        if step.get("status") not in {"PASS", "WARNING", "FAIL"}:
            continue
        detail = step.get("detail") or {}

        # Hard negatives step often returns account_id list
        for key in ("hard_negative_account_ids", "alerted_account_ids",
                    "account_ids"):
            ids = detail.get(key)
            if isinstance(ids, list):
                sev = "ALERT" if step.get("status") == "FAIL" else "WARNING"
                if sev not in severities:
                    continue
                for acc in ids:
                    _add(matrix, str(acc), TooledAlert(
                        tool=f"playbook.{step.get('step_name', 'unknown')}",
                        severity=sev,
                        source_path=src,
                    ))
