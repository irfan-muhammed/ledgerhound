"""
ledgerhound.scorecard.scorecard
============================

The Scorecard aggregator.

Builds a five-component scorecard from a workspace's persisted artifacts.
Reconstructs:
  - playbook reports (for data quality + step status)
  - findings (for discipline)
  - knowledge entries (for process telemetry)
  - graph reports (for corroboration)
  - coverage report (from the live CoverageTracker singleton OR from a
    saved coverage.json if the scorecard is computed in a separate
    process)

Outputs
-------
``Scorecard`` carries five ``ComponentScore`` objects, an ``overall_score``,
and a JSON-serialisable ``to_dict()`` representation.  ``display()``
renders a Rich panel + table.

Usage
-----
    from ledgerhound.scorecard import Scorecard
    sc = Scorecard.from_workspace(workspace, dataset_name="upi_fraud")
    sc.display()
    sc.save(workspace.reports_dir / "scorecard.json")
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ledgerhound.scorecard.components import (
    ComponentScore,
    completeness_score,
    discipline_score,
    corroboration_score,
    data_quality_score,
    process_score,
)

log = logging.getLogger(__name__)


def _ledgerhound_version() -> str:
    """Read the canonical version at instantiation time.

    These records stamp WHICH Ledgerhound produced a saved artifact, so a value
    that silently lags a release mislabels provenance. It was hardcoded in
    three separate dataclasses and had to be hand-edited every bump; imported
    lazily here because `ledgerhound/__init__` imports `ledgerhound.core`, so a
    module-level import would risk a cycle.
    """
    from ledgerhound import __version__
    return __version__


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Scorecard:
    """One Ledgerhound EDA Scorecard."""

    dataset_name: str
    period: str
    generated_at: str = field(default_factory=_now)
    ledgerhound_version: str = field(default_factory=_ledgerhound_version)

    components: list[ComponentScore] = field(default_factory=list)
    overall_score: float = 0.0

    # Source artifacts referenced (paths) -- for reproducibility
    sources: dict[str, str | list[str]] = field(default_factory=dict)
    notes: str = ""

    # -- Construction ------------------------------------------------------

    @classmethod
    def from_workspace(
        cls,
        workspace,
        dataset_name: str = "default",
        period: Optional[str] = None,
        playbook_report: Optional[dict] = None,
        coverage_report: Optional[dict] = None,
        findings: Optional[list] = None,
        alerts_by_tool: Optional[dict[str, set[str]]] = None,
    ) -> "Scorecard":
        """Build a scorecard from artifacts persisted under ``workspace``.

        All inputs except ``workspace`` are optional -- the method auto-loads
        what it can from the workspace's reports directory.
        """
        sources: dict[str, str | list[str]] = {"workspace": str(workspace.root)}

        # 0. Reproducibility manifest -- pick the most recent if multiple
        try:
            manifests = sorted(Path(workspace.root).glob("MANIFEST*.json"))
            if manifests:
                sources["manifest"] = str(manifests[-1])
        except Exception as exc:
            log.debug("Could not locate manifest: %s", exc)

        # 1. Coverage -- prefer in-process tracker, fall back to file
        if coverage_report is None:
            cov_path = workspace.reports_dir / "coverage.json"
            if cov_path.exists():
                try:
                    coverage_report = json.loads(cov_path.read_text(encoding="utf-8"))
                    sources["coverage"] = str(cov_path)
                except Exception as exc:
                    log.warning("Could not load coverage.json: %s", exc)
            else:
                # Use live singleton
                try:
                    from ledgerhound.instrumentation import coverage as _cov
                    coverage_report = _cov().report()
                    sources["coverage"] = "live_tracker"
                except Exception as exc:
                    log.warning("No live coverage tracker: %s", exc)
                    coverage_report = {}
        coverage_report = coverage_report or {}

        # 2. Playbook (most recent run by glob -- caller can override)
        if playbook_report is None:
            pb_dir = workspace.reports_dir / "playbook"
            candidates = sorted(pb_dir.glob("run_*.json")) if pb_dir.exists() else []
            if candidates:
                pb_path = candidates[-1]
                try:
                    playbook_report = json.loads(pb_path.read_text(encoding="utf-8"))
                    sources["playbook"] = str(pb_path)
                    if period is None:
                        period = playbook_report.get("period", "current")
                except Exception as exc:
                    log.warning("Could not load %s: %s", pb_path, exc)

        # 3. Findings -- load from .ledgerhound/findings/ if present
        if findings is None:
            try:
                from ledgerhound.findings import FindingsLog
                fdir = Path(".ledgerhound/findings")
                if fdir.exists():
                    candidates = list(fdir.glob(f"{dataset_name}*.jsonl"))
                    if candidates:
                        fl = FindingsLog(candidates[0])
                        findings = list(fl.iter())
                        sources["findings"] = str(candidates[0])
            except Exception as exc:
                log.debug("Could not auto-load findings: %s", exc)
            findings = findings or []

        # 4. Alerts by tool -- build the richer CorroborationMatrix
        #    (graph + findings + playbook) and emit reports/corroboration.json
        corroboration_matrix = None
        if alerts_by_tool is None:
            from ledgerhound.scorecard.corroboration import build_corroboration_matrix
            corroboration_matrix = build_corroboration_matrix(workspace)
            alerts_by_tool = corroboration_matrix.to_alerts_by_tool()
            if corroboration_matrix.sources_read:
                sources["corroboration_sources"] = corroboration_matrix.sources_read
            # Persist the matrix for downstream inspection
            try:
                out = workspace.reports_dir / "corroboration.json"
                corroboration_matrix.save(out)
                sources["corroboration_matrix"] = str(out)
            except Exception as exc:
                log.warning("Could not save corroboration.json: %s", exc)

        # 5. Extract per-step details from the playbook report
        consistency_report = None
        psi_report = None
        signal_health_report = None
        n_knowledge = 0
        if playbook_report:
            for s in playbook_report.get("steps", []):
                name = s.get("step_name")
                detail = s.get("detail", {})
                if name == "consistency":
                    consistency_report = detail
                elif name == "psi_certification":
                    psi_report = detail
                elif name == "signal_health":
                    signal_health_report = detail
            n_knowledge = int(playbook_report.get("knowledge_entries_deposited") or 0)

        # 6. Knowledge entries (DatasetKnowledge live)
        try:
            from ledgerhound.memory.knowledge import DatasetKnowledge
            dk = DatasetKnowledge.load(workspace, dataset_name)
            n_knowledge = max(n_knowledge, len(dk))
            sources["knowledge"] = str(dk.entries_path)
        except Exception:
            pass

        # 7. Artifacts saved -- quick glob
        n_artifacts = 0
        for sub in ("playbook", "graph", "typology", "consistency"):
            d = workspace.reports_dir / sub
            if d.exists():
                n_artifacts += sum(1 for _ in d.glob("*.json"))

        # 8. Tool calls / elapsed -- from playbook only (agent telemetry is
        #    available too but isn't always on)
        n_tool_calls = None
        elapsed_ms = None
        if playbook_report:
            n_tool_calls = len(playbook_report.get("steps") or [])
            elapsed_ms = playbook_report.get("elapsed_ms")

        n_tools_used = (
            len(coverage_report.get("columns_by_tool") or {})
            if coverage_report else 0
        )

        # -- Score components ---------------------------------------------
        comps: list[ComponentScore] = [
            completeness_score(coverage_report, weight=0.25),
            discipline_score(findings, weight=0.30),
            corroboration_score(alerts_by_tool, weight=0.15),
            data_quality_score(
                consistency_report=consistency_report,
                psi_report=psi_report,
                signal_health_report=signal_health_report,
                weight=0.15,
            ),
            process_score(
                n_tool_calls=n_tool_calls,
                elapsed_ms=elapsed_ms,
                n_tools_used=n_tools_used,
                n_artifacts_saved=n_artifacts,
                n_knowledge_entries_deposited=n_knowledge,
                weight=0.15,
            ),
        ]

        overall = sum(c.score * c.weight for c in comps) / max(
            1e-9, sum(c.weight for c in comps)
        )

        return cls(
            dataset_name=dataset_name,
            period=period or "current",
            components=comps,
            overall_score=round(overall, 4),
            sources=sources,
        )

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _collect_alerts_by_tool(workspace) -> dict[str, set[str]]:
        """Read every *.json under reports/graph and build alerts_by_tool."""
        out: dict[str, set[str]] = {}
        gdir = workspace.reports_dir / "graph"
        if not gdir.exists():
            return out
        for path in gdir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Motif report
            if "matches" in data and "matches_by_pattern" in data:
                for m in data.get("matches", []):
                    if m.get("severity") == "ALERT":
                        for n in m.get("nodes", []):
                            out.setdefault("detect_mule_motifs", set()).add(str(n))

            # Community / mule_rings report
            if "communities" in data:
                for com in data.get("communities", []):
                    if com.get("severity") == "ALERT":
                        for n in com.get("nodes", []):
                            out.setdefault("find_mule_rings", set()).add(str(n))

            # Fingerprints
            if "fingerprints" in data:
                for fp in data.get("fingerprints", []):
                    if fp.get("severity") == "ALERT":
                        out.setdefault(
                            "fingerprint_accounts", set(),
                        ).add(str(fp.get("account_id")))

            # Centrality (passthrough candidates count as alerts)
            if "accounts" in data and "measures_computed" in data:
                for a in data.get("accounts", []):
                    fb = a.get("flow_balance")
                    in_deg = int(a.get("in_degree", 0))
                    if (
                        fb is not None and fb <= 0.10 and in_deg >= 3
                        and float(a.get("in_amount", 0)) > 0
                        and float(a.get("out_amount", 0)) > 0
                    ):
                        out.setdefault("compute_account_centrality", set()).add(
                            str(a.get("account_id"))
                        )
        return out

    # -- Output -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "dataset_name":     self.dataset_name,
            "period":           self.period,
            "generated_at":     self.generated_at,
            "ledgerhound_version": self.ledgerhound_version,
            "overall_score":    round(self.overall_score, 4),
            "components":       [c.to_dict() for c in self.components],
            "sources":          self.sources,
            "notes":            self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8",
        )
        log.info("Scorecard saved: %s", path)

    def display(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.rule import Rule

        c = Console()
        # Pick an overall-score colour band
        if self.overall_score >= 0.85:
            colour, badge = "green", "EXCELLENT"
        elif self.overall_score >= 0.70:
            colour, badge = "cyan", "GOOD"
        elif self.overall_score >= 0.50:
            colour, badge = "yellow", "ADEQUATE"
        else:
            colour, badge = "red", "NEEDS WORK"

        # Headline panel
        head = (
            f"[bold]Ledgerhound EDA Scorecard -- {self.dataset_name} / {self.period}[/bold]\n"
            f"[dim]generated {self.generated_at}[/dim]\n\n"
            f"  Overall:  [{colour}][bold]{self.overall_score:.2f}[/bold] {badge}[/{colour}]"
        )
        c.print(Panel(head, border_style=colour))

        # Components table
        t = Table(
            "Component", "Weight", "Score", "Sub-metrics",
            title="Component scores", show_header=True, header_style="bold dim",
        )
        for comp in self.components:
            sm_summary = ", ".join(
                f"{k}={v}" for k, v in list(comp.sub_metrics.items())[:3]
            )
            sm_summary = sm_summary[:80] + ("..." if len(sm_summary) > 80 else "")
            comp_colour = (
                "green" if comp.score >= 0.85 else
                "cyan"  if comp.score >= 0.70 else
                "yellow" if comp.score >= 0.50 else
                "red"
            )
            t.add_row(
                comp.name,
                f"{comp.weight:.2f}",
                f"[{comp_colour}]{comp.score:.3f}[/{comp_colour}]",
                sm_summary or "[dim](no data)[/dim]",
            )
        c.print(t)

        # Per-component detail
        for comp in self.components:
            comp_colour = (
                "green" if comp.score >= 0.85 else
                "cyan"  if comp.score >= 0.70 else
                "yellow" if comp.score >= 0.50 else
                "red"
            )
            c.print(Rule(
                f"[{comp_colour}]{comp.name}[/{comp_colour}] -- {comp.score:.3f}",
                style=comp_colour,
            ))
            sm_lines = []
            for k, v in comp.sub_metrics.items():
                if isinstance(v, float):
                    sm_lines.append(f"  {k}: {v:.4f}")
                elif v is None:
                    sm_lines.append(f"  {k}: --")
                else:
                    val_str = str(v)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    sm_lines.append(f"  {k}: {val_str}")
            if sm_lines:
                c.print("\n".join(sm_lines))
            if comp.suggestions:
                c.print("[bold]Suggestions:[/bold]")
                for s in comp.suggestions:
                    c.print(f"  [yellow]->[/yellow] {s}")
            if comp.notes:
                c.print(f"[dim]{comp.notes}[/dim]")
        c.print()
