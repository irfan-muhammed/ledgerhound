"""
ledgerhound.sampling.certificate
==============================

SamplingCertificate -- the auditable proof that a non-fraud sample
faithfully represents the true population on the features that matter.

A certificate answers two questions:
  1. Is this sample representative of the true population?
     -> Per-feature PSI with pass/fail, weighted by discriminative importance.
  2. Is this sample better than a declared baseline (e.g. a naive random
     draw)?
     -> Delta PSI vs baseline, with % improvement per feature.

The certificate never asserts what ratio or strategy that baseline was
drawn at -- it only knows what the caller declares via
``baseline_description`` on :meth:`CertificateGenerator.generate`. A
caller that wants "beats the current 1:1000 random baseline" printed on
the certificate must say so explicitly by passing that string; the
certificate does not assume it, because no code path in this repo (as
of 2026-07) actually measures what ratio the incumbent baseline uses.

The certificate is written to the workspace as certificate.json and
displayed as a rich table. It is the artifact the FE team receives
alongside the sample data -- the proof that the sample is trustworthy.

Certificate status rules:
  PASS    -- weighted PSI < 0.10 AND no individual feature FAIL
  WARNING -- weighted PSI 0.10-0.20 OR >=1 WARNING feature, no FAIL
  FAIL    -- weighted PSI > 0.20 OR any individual feature FAIL
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ledgerhound.sampling.psi import PSIResult, PSIStatus, FeaturePSI

console = Console()


class CertStatus(str):
    PASS    = "PASS"
    WARNING = "WARNING"
    FAIL    = "FAIL"


def _cert_status(psi_result: PSIResult) -> str:
    if psi_result.n_fail > 0 or psi_result.weighted_psi > 0.20:
        return CertStatus.FAIL
    if psi_result.n_warning > 0 or psi_result.weighted_psi >= 0.10:
        return CertStatus.WARNING
    return CertStatus.PASS


@dataclass
class FeatureCertResult:
    """Per-feature certification result for the certificate."""
    feature: str
    feature_type: str
    psi: float
    psi_status: str
    ks_statistic: Optional[float]
    ks_status: Optional[str]
    weight: float              # importance weight used in weighted PSI
    weighted_psi: float        # this feature's contribution
    n_sample: int
    baseline_psi: Optional[float] = None    # this feature's PSI in the declared baseline
    improvement_pct: Optional[float] = None # % PSI reduction vs baseline
    psi_ci_lower: Optional[float] = None
    psi_ci_upper: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SamplingCertificate:
    """
    The complete sampling certification for a non-fraud sample.

    Produced by CertificateGenerator.generate() and persisted to
    workspace/samples/{name}/certificate.json.
    """
    sample_name: str
    sample_version: str
    generated_at: str
    fraud_dataset_key: str
    nonfr_dataset_key: str
    overall_status: str
    weighted_psi: float
    features: list[FeatureCertResult]
    fraud_rows: int
    nonfr_rows: int
    ratio: float                              # nonfr / fraud
    sampling_strategy: str
    # Baseline comparison. baseline_description is free text the CALLER
    # declares (e.g. "1:1000 random draw") -- the certificate does not
    # infer or assume a ratio it did not measure.
    baseline_weighted_psi: Optional[float] = None
    baseline_sample_name: Optional[str] = None
    baseline_description: str = ""
    improvement_over_baseline_pct: Optional[float] = None
    # Hard negative stats
    hard_negative_inclusion_rate: Optional[float] = None
    hard_negative_count: Optional[int] = None
    # Summary
    n_pass: int = 0
    n_warning: int = 0
    n_fail: int = 0
    notes: str = ""

    def is_approved(self) -> bool:
        return self.overall_status == CertStatus.PASS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = [f.to_dict() if hasattr(f, "to_dict") else f for f in self.features]
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SamplingCertificate":
        raw = json.loads(path.read_text(encoding="utf-8"))
        features = [FeatureCertResult(**f) for f in raw.pop("features", [])]
        return cls(features=features, **raw)

    def display(self) -> None:
        """Rich table display of the certificate."""
        status_color = {"PASS": "green", "WARNING": "yellow", "FAIL": "red"}.get(
            self.overall_status, "white"
        )
        console.print(Panel(
            f"[bold {status_color}]{self.overall_status}[/bold {status_color}]  "
            f"Weighted PSI: {self.weighted_psi:.4f}  "
            f"Pass: {self.n_pass}  Warn: {self.n_warning}  Fail: {self.n_fail}\n"
            f"Sample: {self.nonfr_rows:,} non-fraud + {self.fraud_rows:,} fraud  "
            f"(ratio 1:{self.ratio:.0f})  Strategy: {self.sampling_strategy}",
            title=f"Sampling Certificate -- {self.sample_name} v{self.sample_version}",
            border_style=status_color,
        ))

        if self.baseline_weighted_psi is not None:
            impr = self.improvement_over_baseline_pct or 0
            # Never assert a ratio/strategy we did not measure -- print
            # exactly what the caller declared, or an honest "declared
            # baseline" placeholder if they declared nothing.
            baseline_label = self.baseline_description or self.baseline_sample_name \
                or "declared baseline"
            console.print(
                f"[dim]Baseline ({baseline_label}) PSI: {self.baseline_weighted_psi:.4f}  "
                f"Improvement: [bold]{impr:+.1f}%[/bold][/dim]"
            )

        t = Table(
            "Feature", "Type", "PSI", "Status", "Weight", "Baseline PSI", "%",
            show_header=True,
            header_style="bold dim",
        )
        status_symbols = {"pass": "OK", "warning": "!", "fail": "X"}
        for f in sorted(self.features, key=lambda x: -x.psi):
            sym = status_symbols.get(f.psi_status, "?")
            color = {"pass": "green", "warning": "yellow", "fail": "red"}.get(f.psi_status, "")
            baseline_str = f"{f.baseline_psi:.4f}" if f.baseline_psi is not None else "--"
            impr_str = (
                f"{f.improvement_pct:+.1f}%" if f.improvement_pct is not None else "--"
            )
            t.add_row(
                f.feature[:38],
                f.feature_type[:4],
                f"{f.psi:.4f}",
                f"[{color}]{sym} {f.psi_status.upper()}[/{color}]",
                f"{f.weight:.2f}",
                baseline_str,
                impr_str,
            )
        console.print(t)


class CertificateGenerator:
    """
    Generates a SamplingCertificate from PSI results.

    Parameters
    ----------
    weights : {feature: importance_weight}
              If provided, PSI is weighted by feature importance.
              Use model feature importance from a prior run, or leave
              None for equal weighting (first certification).
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or {}

    def generate(
        self,
        psi_result: PSIResult,
        sample_name: str,
        sample_version: str,
        fraud_dataset_key: str,
        nonfr_dataset_key: str,
        fraud_rows: int,
        nonfr_rows: int,
        sampling_strategy: str,
        baseline_psi_result: Optional[PSIResult] = None,
        baseline_sample_name: Optional[str] = None,
        baseline_description: str = "",
        hard_negative_count: Optional[int] = None,
        notes: str = "",
    ) -> SamplingCertificate:
        """
        Build the certificate from a PSIResult.

        baseline_psi_result : PSIResult for the sample being compared against
                              (e.g. a naive random draw). If provided, the
                              certificate shows % improvement over baseline
                              for each feature.
        baseline_sample_name, baseline_description : caller-declared identity
                              and description of that baseline (e.g. name
                              "nonfr_random_v0", description "1:1000 random
                              draw"). Neither is inferred from the PSI data --
                              the certificate has no way to know what ratio or
                              strategy produced a baseline sample, only what
                              its distributions look like. Passing
                              baseline_psi_result without either of these
                              still works; the certificate will just print a
                              generic "declared baseline" label rather than
                              assert a ratio nobody told it.
        """
        # Build per-feature results with weights and baseline comparison
        baseline_by_feature: dict[str, FeaturePSI] = {}
        if baseline_psi_result:
            baseline_by_feature = {f.feature: f for f in baseline_psi_result.features}

        feature_results = []
        for fp in psi_result.features:
            w = self.weights.get(fp.feature, 1.0)
            total_w = sum(self.weights.get(f.feature, 1.0) for f in psi_result.features) or 1.0
            weighted_psi = fp.psi * w / total_w

            baseline_fp = baseline_by_feature.get(fp.feature)
            baseline_psi = baseline_fp.psi if baseline_fp else None
            improvement_pct = (
                ((baseline_psi - fp.psi) / baseline_psi * 100)
                if baseline_psi and baseline_psi > 0
                else None
            )

            feature_results.append(FeatureCertResult(
                feature=fp.feature,
                feature_type=fp.feature_type,
                psi=round(fp.psi, 6),
                psi_status=fp.status.value,
                ks_statistic=fp.ks_statistic,
                ks_status=fp.ks_status.value if fp.ks_status else None,
                weight=round(w, 4),
                weighted_psi=round(weighted_psi, 6),
                n_sample=fp.n_sample,
                baseline_psi=round(baseline_psi, 6) if baseline_psi else None,
                improvement_pct=round(improvement_pct, 2) if improvement_pct else None,
                psi_ci_lower=fp.psi_ci_lower,
                psi_ci_upper=fp.psi_ci_upper,
            ))

        # Baseline weighted PSI
        baseline_wpsi = None
        improvement_over_baseline = None
        if baseline_psi_result:
            baseline_wpsi = baseline_psi_result.weighted_psi
            if baseline_wpsi and baseline_wpsi > 0:
                improvement_over_baseline = round(
                    (baseline_wpsi - psi_result.weighted_psi) / baseline_wpsi * 100, 2
                )

        ratio = nonfr_rows / max(fraud_rows, 1)
        hn_rate = None
        if hard_negative_count and nonfr_rows:
            hn_rate = round(hard_negative_count / nonfr_rows, 4)

        return SamplingCertificate(
            sample_name=sample_name,
            sample_version=sample_version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            fraud_dataset_key=fraud_dataset_key,
            nonfr_dataset_key=nonfr_dataset_key,
            overall_status=_cert_status(psi_result),
            weighted_psi=round(psi_result.weighted_psi, 6),
            features=feature_results,
            fraud_rows=fraud_rows,
            nonfr_rows=nonfr_rows,
            ratio=round(ratio, 2),
            sampling_strategy=sampling_strategy,
            baseline_weighted_psi=baseline_wpsi,
            baseline_sample_name=baseline_sample_name if baseline_psi_result else None,
            baseline_description=baseline_description if baseline_psi_result else "",
            improvement_over_baseline_pct=improvement_over_baseline,
            hard_negative_inclusion_rate=hn_rate,
            hard_negative_count=hard_negative_count,
            n_pass=psi_result.n_pass,
            n_warning=psi_result.n_warning,
            n_fail=psi_result.n_fail,
            notes=notes,
        )
