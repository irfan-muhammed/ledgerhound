"""
sentinel.agents.two_dataset
===========================

Three-phase fraud x payee x merged-verdict playbook orchestrator (P1.F).

Why three phases?
-----------------
A flat list of questions on a joined table loses the analytical
structure: an analyst first understands fraud-side dynamics, then
payee-side dynamics, and only then asks the cross-cutting questions
that depend on the first two.  Findings from earlier phases become
priors for later phases -- and the shared FindingsLog is what makes that
compounding *real* rather than implicit.

Architecture
------------

::

    fraud_agent -+
                 +-   shared FindingsLog
    payee_agent -+        |
                 |        +- merged_agent reads via list_findings
    merged_agent +

* **Phase A -- Fraud Evolution**:  drift, clustering, account-age x fraud
  lifts, purpose-code skew.  Runs against ``fraud_agent`` (transaction
  data only).
* **Phase B -- Payee Evolution**:  high-risk payee discovery, complaint
  spikes, anomalous totals.  Runs against ``payee_agent`` (payee dimension).
* **Phase C -- Merged Verdict**:  cross-references A's fraud signals with
  B's risk-payee list; identifies pre-fraud signatures; held-out
  validation.  Runs against ``merged_agent`` (left-join of fraud + payee
  on the join key) and uses ``list_findings`` to consume the priors set
  by phases A and B.

The orchestrator is intentionally thin: each phase calls
``EDAAgent.ask()`` per question and accumulates findings into a single
:class:`TwoDatasetPlaybookResult`.  All sample-frame discipline,
KB-first checks, and structured-finding emission come "for free" from
P1.A / P1.B / P1.E.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from sentinel.agents.eda_agent import EDAAgent

if TYPE_CHECKING:
    from sentinel.findings.log import FindingsLog
    from sentinel.memory.models import DatasetMemory

log = logging.getLogger(__name__)
console = Console()


# -- Default playbook dict -------------------------------------------------


DEFAULT_PLAYBOOK: dict = {
    "name": "Fraud Evolution x Payee Evolution x Merged Verdict",
    "description": (
        "Three-phase EDA: fraud-side dynamics, payee-side dynamics, then "
        "cross-cutting verdict using priors from both."
    ),
    "phases": {
        "A_fraud_evolution": {
            "name": "Phase A -- Fraud Evolution",
            "agent": "fraud",
            "questions": [
                "What is the overall fraud rate in this sample, and how does "
                "it vary across the time window?  Tag your finding with the "
                "appropriate sample_frame.",
                "Bucket transactions by account_age_days.  Is there a "
                "fraud-rate lift in young accounts (<30 days)?  Quantify "
                "the lift and record it as a finding.",
                "Among purpose_code values, are any over-represented in "
                "fraud transactions?  Decode the codes via dict_decode_code "
                "before reporting them.",
                "Look at the upper amount tail (top 1%).  Is there elevated "
                "fraud risk?  Use a chart to show the distribution.",
            ],
        },
        "B_payee_evolution": {
            "name": "Phase B -- Payee Evolution",
            "agent": "payee",
            "questions": [
                "Identify the top-N payees by complaint count and by "
                "anomalous total_amount_30d.  Are these the same group?",
                "What is the distribution of n_complaints across payees, "
                "and what cutoff would flag the top decile of risk?",
                "Cluster payees by their behavioral features "
                "(n_txns_30d, total_amount_30d, avg_amount, n_complaints).  "
                "Describe the resulting clusters.",
            ],
        },
        "C_merged_verdict": {
            "name": "Phase C -- Merged Verdict",
            "agent": "merged",
            "questions": [
                "Use list_findings to read what phases A and B found.  "
                "Cross-reference: do high-risk payees from B account for a "
                "disproportionate share of fraud txns from A?",
                "Identify pre-fraud signatures: which payee-level features "
                "(n_complaints, n_txns_30d, etc.) correlated with the "
                "txn-level fraud label, controlling for account age?",
                "Hold out 20% of payees, derive a fraud-likeness score on "
                "the rest using the signals you found, and validate it on "
                "the held-out group.  Report KS / lift on the validation set.",
            ],
        },
    },
}


# -- Result types ----------------------------------------------------------


@dataclass
class PhaseResult:
    """One phase's contribution to the full playbook run."""

    name: str
    agent_name: str   # "fraud" | "payee" | "merged"
    findings: list[dict] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    finding_warnings: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    journal_paths: list[str] = field(default_factory=list)
    questions_run: int = 0
    questions_succeeded: int = 0
    questions_failed: int = 0
    elapsed_ms: float = 0.0


@dataclass
class TwoDatasetPlaybookResult:
    """Full result of a 3-phase run."""

    playbook_name: str = ""
    phases: list[PhaseResult] = field(default_factory=list)
    summary: str = ""
    elapsed_ms: float = 0.0

    # Aggregates (computed)
    @property
    def total_findings(self) -> int:
        return sum(len(p.finding_ids) for p in self.phases)

    @property
    def total_charts(self) -> int:
        return sum(len(p.charts) for p in self.phases)

    @property
    def total_warnings(self) -> int:
        return sum(len(p.finding_warnings) for p in self.phases)

    def to_markdown(self) -> str:
        lines = [f"# {self.playbook_name}\n"]
        lines.append(
            f"*{self.total_findings} findings . "
            f"{self.total_charts} charts . "
            f"{self.elapsed_ms / 1000:.1f}s . "
            f"{self.total_warnings} warnings*\n"
        )
        for phase in self.phases:
            lines.append(f"\n## {phase.name}")
            lines.append(
                f"*{phase.questions_succeeded}/{phase.questions_run} questions, "
                f"{len(phase.finding_ids)} findings, "
                f"{phase.elapsed_ms / 1000:.1f}s*\n"
            )
            for f in phase.findings:
                q = f.get("question", "?")
                a = f.get("answer", "")
                fids = f.get("finding_ids", [])
                lines.append(f"\n### Q: {q}\n")
                lines.append(a if a else "_(no answer)_")
                if fids:
                    lines.append(
                        f"\n*Findings recorded: {', '.join('`' + i + '`' for i in fids)}*"
                    )
            if phase.finding_warnings:
                lines.append("\n**Sample-frame / evidence warnings:**\n")
                for w in phase.finding_warnings:
                    lines.append(f"- {w}")
        if self.summary:
            lines.append("\n---\n## Executive Summary\n")
            lines.append(self.summary)
        return "\n".join(lines)

    def save(self, path: str | Path, *, write_pdf: bool = True) -> Path:
        """Write the report to Markdown (and PDF, if fpdf2 is available).

        Markdown is always written.  PDF is written next to it (same stem,
        ``.pdf`` extension) when ``write_pdf=True`` and ``fpdf2`` is
        installed.  PDF failures are non-fatal -- a warning is logged and
        the markdown path is returned.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")

        if write_pdf:
            try:
                from sentinel.output.pdf_report import SentinelPDF, save_pdf_report

                # The PDF backend (fpdf2 + Helvetica) doesn't grok unicode
                # punctuation like em-dashes -- sanitise everything first.
                _s = SentinelPDF._safe

                # Flatten the per-phase findings into the format save_pdf_report
                # expects (a flat list with a ``phase`` key per finding).
                flat: list[dict] = []
                all_charts: list[str] = []
                for phase in self.phases:
                    for f in phase.findings:
                        flat.append({
                            "phase":    _s(phase.name),
                            "question": _s(f.get("question", "")),
                            "finding":  _s(
                                f.get("answer", "") or f.get("error", "Failed")
                            ),
                            "status":   f.get("status", "failed"),
                            "charts":   f.get("charts", []),
                        })
                        all_charts.extend(f.get("charts", []))

                pdf_path = p.with_suffix(".pdf")
                save_pdf_report(
                    str(pdf_path),
                    findings=flat,
                    summary=_s(self.summary or ""),
                    digest=None,  # TwoDataset playbook isn't tied to one digest
                    charts=all_charts,
                    title=_s(self.playbook_name),
                    elapsed_ms=self.elapsed_ms,
                    analyses_run=sum(p.questions_run for p in self.phases),
                    analyses_succeeded=sum(
                        p.questions_succeeded for p in self.phases
                    ),
                )
                log.debug("Two-dataset PDF written to %s", pdf_path)
            except Exception as exc:
                log.warning("PDF report generation failed (markdown OK): %s", exc)

        return p

    def display(self):
        console.print(Panel(Markdown(self.to_markdown()),
                            title="Two-Dataset Playbook", border_style="cyan"))


# -- Orchestrator ----------------------------------------------------------


class TwoDatasetPlaybook:
    """Run Phase A -> Phase B -> Phase C against three EDA agents.

    Parameters
    ----------
    fraud_agent, payee_agent, merged_agent:
        Pre-constructed :class:`EDAAgent` instances.  All three should
        share the same :class:`FindingsLog` so phases compound -- pass
        the log to each via ``EDAAgent(findings_log=...)`` or
        ``set_findings_log(...)`` before constructing the playbook.
    playbook:
        Override the default playbook structure.  Must have the same
        shape as :data:`DEFAULT_PLAYBOOK`.
    interactive:
        Stream Think/Act/Observe to the console (recommended for live
        runs; turn off in batch / CI / tests).
    synthesise:
        After all phases complete, call the LLM once more to write an
        executive summary.  Default: True; set False for tests.
    """

    AGENT_KEYS = ("fraud", "payee", "merged")

    def __init__(
        self,
        fraud_agent: EDAAgent,
        payee_agent: EDAAgent,
        merged_agent: EDAAgent,
        *,
        playbook: Optional[dict] = None,
        interactive: bool = True,
        synthesise: bool = True,
    ):
        self.agents: dict[str, EDAAgent] = {
            "fraud":  fraud_agent,
            "payee":  payee_agent,
            "merged": merged_agent,
        }
        self.playbook = playbook or DEFAULT_PLAYBOOK
        self.interactive = interactive
        self.synthesise = synthesise

    # -- Public API ---------------------------------------------------------

    def run(self, report_path: Optional[str] = None) -> TwoDatasetPlaybookResult:
        """Execute every phase in order and return the aggregate result."""
        result = TwoDatasetPlaybookResult(
            playbook_name=self.playbook.get("name", "Two-Dataset Playbook"),
        )
        t0 = time.perf_counter()

        phases = self.playbook.get("phases", {})
        if not phases:
            raise ValueError("Playbook has no phases")

        # Phases run in dict insertion order (Python 3.7+)
        for key, phase_def in phases.items():
            phase_result = self._run_phase(key, phase_def)
            result.phases.append(phase_result)

            # Incremental save after each phase, so a crash mid-Phase-C
            # still leaves Phase-A and Phase-B on disk.
            result.elapsed_ms = self._ms(t0)
            if report_path:
                try:
                    result.save(report_path)
                    if self.interactive:
                        console.print(
                            f"[dim] Report updated: {report_path}[/dim]"
                        )
                except Exception as exc:  # pragma: no cover
                    log.warning("Incremental save failed: %s", exc)

        # Final synthesis (one extra LLM call) -- uses the fraud agent's LLM
        if self.synthesise:
            try:
                result.summary = self._synthesise(result)
            except Exception as exc:
                log.warning("Synthesis failed: %s", exc)
                result.summary = f"_(synthesis failed: {exc})_"

        result.elapsed_ms = self._ms(t0)
        if report_path:
            try:
                result.save(report_path)
            except Exception:
                pass

        if self.interactive:
            console.print(Rule(
                f"[bold green]Two-dataset playbook complete.[/bold green] "
                f"{result.total_findings} findings, "
                f"{result.total_charts} charts, "
                f"{result.elapsed_ms / 1000:.1f}s"
            ))
            if result.total_warnings:
                console.print(
                    f"[yellow]! {result.total_warnings} sample-frame / evidence "
                    f"warnings -- review the report.[/yellow]"
                )

        return result

    # -- Internals ----------------------------------------------------------

    def _run_phase(self, key: str, phase_def: dict) -> PhaseResult:
        name = phase_def.get("name", key)
        agent_name = phase_def.get("agent", "fraud")
        if agent_name not in self.agents:
            raise ValueError(
                f"Phase {key!r} requests agent {agent_name!r} but only "
                f"{list(self.agents)} are configured."
            )
        agent = self.agents[agent_name]

        if self.interactive:
            console.print(Rule(f"[bold cyan]{name}[/bold cyan] "
                                f"[dim](agent={agent_name})[/dim]"))

        phase = PhaseResult(name=name, agent_name=agent_name)
        questions = phase_def.get("questions", [])
        t_phase = time.perf_counter()

        for q in questions:
            phase.questions_run += 1
            if self.interactive:
                console.print(f"\n  [bold]> {q}[/bold]")
            try:
                ar = agent.ask(q, interactive=self.interactive)
                phase.findings.append({
                    "question": q,
                    "answer": ar.answer,
                    "finding_ids": list(ar.finding_ids),
                    "charts": list(ar.charts),
                    "journal_path": ar.journal_path,
                    "status": "success",
                })
                phase.finding_ids.extend(ar.finding_ids)
                phase.finding_warnings.extend(ar.finding_warnings)
                phase.charts.extend(ar.charts)
                if ar.journal_path:
                    phase.journal_paths.append(ar.journal_path)
                phase.questions_succeeded += 1
                if self.interactive:
                    console.print("  [green]OK Done[/green]")
            except Exception as exc:
                phase.findings.append({
                    "question": q, "error": str(exc), "status": "failed",
                })
                phase.questions_failed += 1
                if self.interactive:
                    console.print(f"  [red]X {str(exc)[:80]}[/red]")

        phase.elapsed_ms = self._ms(t_phase)
        return phase

    def _synthesise(self, result: TwoDatasetPlaybookResult) -> str:
        """Ask the fraud agent's LLM for an executive summary."""
        text_chunks = []
        for phase in result.phases:
            text_chunks.append(f"\n## {phase.name}")
            for f in phase.findings:
                q = f.get("question", "?")
                a = f.get("answer", "") or f.get("error", "")
                fids = f.get("finding_ids", [])
                text_chunks.append(f"\n### {q}\n{a}")
                if fids:
                    text_chunks.append(f"_Findings: {', '.join(fids)}_")
        prompt = "\n".join(text_chunks)

        # Use whichever agent has an LLM
        llm = self.agents["fraud"].llm
        try:
            r = llm.generate(
                [
                    {"role": "system", "content": (
                        "You are synthesising a 3-phase fraud x payee x "
                        "merged-verdict EDA. Write an executive summary "
                        "(top 5 findings with magnitudes, recommended "
                        "features for the model, sample-frame caveats, "
                        "data-quality blockers, next-step queries). Be "
                        "concrete. Reference finding IDs."
                    )},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
            )
            return r.content
        except Exception as exc:
            return f"_(synthesis failed: {exc})_"

    @staticmethod
    def _ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 1)


# -- High-level factory ----------------------------------------------------


def build_merged_dataframe(
    fraud_data: str | Path,
    payee_data: str | Path,
    join_key: str,
):
    """Read both parquet/CSV files and left-join on ``join_key``.

    Returns the merged DataFrame.  Left-join (keep all txns, even those
    with no matching payee record) is the default for fraud analysis
    because un-matched payees themselves are a signal.
    """
    import pandas as pd

    def _read(p: str | Path):
        path = Path(p)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() in (".csv", ".tsv"):
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            return pd.read_csv(path, sep=sep)
        raise ValueError(f"Unsupported data file: {path}")

    fraud_df = _read(fraud_data)
    payee_df = _read(payee_data)

    if join_key not in fraud_df.columns:
        raise KeyError(
            f"join_key {join_key!r} not in fraud_data columns "
            f"({list(fraud_df.columns)[:10]}...)"
        )
    if join_key not in payee_df.columns:
        raise KeyError(
            f"join_key {join_key!r} not in payee_data columns "
            f"({list(payee_df.columns)[:10]}...)"
        )

    return fraud_df.merge(payee_df, on=join_key, how="left",
                          suffixes=("", "_payee"))


def build_two_dataset_playbook(
    *,
    fraud_data: str | Path,
    payee_data: str | Path,
    join_key: str,
    llm,
    fraud_memory: "DatasetMemory | None" = None,
    payee_memory: "DatasetMemory | None" = None,
    findings_log: "FindingsLog | None" = None,
    schema_kb=None,
    output_dir: str = "outputs",
    journal_dir: str | None = None,
    write_journal: bool = True,
    interactive: bool = True,
    playbook: Optional[dict] = None,
) -> TwoDatasetPlaybook:
    """One-shot factory: data + LLM -> fully-wired :class:`TwoDatasetPlaybook`.

    Builds three :class:`Sentinel` engines (one per phase), shares the
    findings log + KB across them, and returns a ready-to-run playbook.

    The merged DataFrame is computed in-memory via a left-join; for very
    large datasets, prefer constructing engines manually with a backend
    that supports lazy I/O.
    """
    import tempfile

    from sentinel.core import Sentinel
    from sentinel.findings.log import FindingsLog as _FindingsLog

    # -- Validate the join key BEFORE constructing engines, so a bad key
    # surfaces a KeyError without leaving half-built state behind.
    merged_df = build_merged_dataframe(fraud_data, payee_data, join_key)

    # Resolve a shared findings log
    if findings_log is None:
        ds_name = Path(fraud_data).stem
        findings_log = _FindingsLog(
            Path(".sentinel/findings") / f"{ds_name}.jsonl"
        )

    def _make_engine_and_agent(data, memory, ds_name):
        """Build a Sentinel engine + a directly-wired EDAAgent.

        We intentionally construct ``EDAAgent`` ourselves rather than
        going through ``engine.eda`` -- the lazy property requires an
        LLM to be set on the engine, and we'd rather inject ours.
        """
        engine = Sentinel(data=data)
        digest = engine.profile()

        agent = EDAAgent(
            llm=llm,
            backend=engine.backend,
            knowledge=engine.knowledge,
            schema_kb=schema_kb,
            dataset_memory=memory,
            findings_log=findings_log,
            journal_dir=journal_dir or str(Path(output_dir) / "journals"),
            write_journal=write_journal,
            output_dir=output_dir,
        )
        agent.set_digest(digest)
        # Replace the engine's lazy slot so callers who later touch
        # ``engine.eda`` get the same configured agent.
        engine._eda = agent
        return engine, agent

    fraud_engine, fraud_agent = _make_engine_and_agent(
        str(fraud_data), fraud_memory, Path(fraud_data).stem,
    )
    payee_engine, payee_agent = _make_engine_and_agent(
        str(payee_data), payee_memory, Path(payee_data).stem,
    )

    # Persist the merged DF to a temp parquet so the third engine can
    # ingest it via the same backend pipeline as the others.
    tmp = (Path(tempfile.gettempdir())
           / f"sentinel_merged_{Path(fraud_data).stem}.parquet")
    merged_df.to_parquet(tmp, index=False)

    # Carry the fraud-side memory into the merged engine so sample-frame
    # priors travel through Phase C.
    merged_engine, merged_agent = _make_engine_and_agent(
        str(tmp), fraud_memory,
        f"{Path(fraud_data).stem}__merged__{Path(payee_data).stem}",
    )

    return TwoDatasetPlaybook(
        fraud_agent=fraud_agent,
        payee_agent=payee_agent,
        merged_agent=merged_agent,
        playbook=playbook,
        interactive=interactive,
    )
