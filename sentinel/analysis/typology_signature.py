"""
sentinel.analysis.typology_signature
====================================

Cross-dataset typology signatures (rank 10 -- the flagship).

A *typology signature* characterises one scam type (new-user fraud, stolen-
phone, low-awareness scam, ...) as a per-feature statistical profile. Where
``TypologyAnalyzer`` clusters *within* one dataset and ``typology_evolution``
tracks a typology *over time*, this module compares typologies *across
datasets*: given several signatures it identifies which dimensions are
**shared** across scam types and which **discriminate** one from another.

The comparison is an ANOVA-style decomposition per feature: between-typology
variance vs pooled within-typology variance (an F-ratio). A high F means the
feature separates the typologies (a discriminating / unique dimension); a low
F means the typologies look alike on it (a shared dimension). Signatures carry
only summary stats (mean/std/n per feature), so they compare cheaply and can
be shipped to the drift monitor without moving raw data.

Pure numpy/pandas; no LLM. Contract: ``typology_signature/v1``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA = "typology_signature/v1"


@dataclass
class TypologySignature:
    """A scam type's per-feature statistical profile."""

    name: str
    dataset: str = ""
    owner: str = ""
    n: int = 0
    # feature -> {"mean": float, "std": float, "n": int}
    profile: dict[str, dict[str, float]] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema: str = SCHEMA

    def features(self) -> list[str]:
        return sorted(self.profile)

    def mean(self, feature: str) -> Optional[float]:
        f = self.profile.get(feature)
        return f["mean"] if f else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TypologySignature":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


def build_signature(
    df: Any,
    features: list[str],
    *,
    name: str,
    dataset: str = "",
    owner: str = "",
    subset_col: Optional[str] = None,
    subset_value: Any = 1,
) -> TypologySignature:
    """Build a signature from ``df`` over ``features``.

    ``subset_col``/``subset_value`` restrict to a scam-type slice (e.g. the
    fraud rows, ``is_fraud == 1``, or a specific ``typology`` label).
    """
    import pandas as pd

    frame = df
    if subset_col is not None and subset_col in df.columns:
        frame = df[df[subset_col] == subset_value]
    profile: dict[str, dict[str, float]] = {}
    for f in features:
        if f not in frame.columns or not pd.api.types.is_numeric_dtype(frame[f]):
            continue
        s = frame[f].dropna().astype(float)
        if s.empty:
            continue
        profile[f] = {"mean": float(s.mean()),
                      "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                      "n": int(len(s))}
    return TypologySignature(name=name, dataset=dataset, owner=owner,
                             n=int(len(frame)), profile=profile)


@dataclass
class DimensionVerdict:
    feature: str
    f_ratio: float
    kind: str           # "shared" | "discriminating"
    # per-typology deviation from the grand mean, in pooled-std units
    deviations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "f_ratio": round(self.f_ratio, 4),
                "kind": self.kind,
                "deviations": {k: round(v, 3) for k, v in self.deviations.items()}}


@dataclass
class TypologyComparison:
    signatures: list[str] = field(default_factory=list)
    dimensions: list[DimensionVerdict] = field(default_factory=list)

    def shared(self) -> list[DimensionVerdict]:
        return [d for d in self.dimensions if d.kind == "shared"]

    def discriminating(self) -> list[DimensionVerdict]:
        return [d for d in self.dimensions if d.kind == "discriminating"]

    def unique_to(self, typology: str, *, top: int = 5) -> list[str]:
        """Features where ``typology`` deviates most from the others."""
        ranked = sorted(
            (d for d in self.discriminating() if typology in d.deviations),
            key=lambda d: -abs(d.deviations[typology]))
        return [d.feature for d in ranked[:top]]

    def to_dict(self) -> dict[str, Any]:
        return {"signatures": list(self.signatures),
                "dimensions": [d.to_dict() for d in self.dimensions]}

    def to_markdown(self) -> str:
        lines = ["## Typology comparison", "",
                 f"*Signatures: {', '.join(self.signatures)}*", "",
                 "| Feature | F-ratio | Dimension |", "|---|---|---|"]
        for d in sorted(self.dimensions, key=lambda x: -x.f_ratio):
            lines.append(f"| `{d.feature}` | {d.f_ratio:.3f} | {d.kind} |")
        return "\n".join(lines)


def compare_signatures(
    signatures: list[TypologySignature],
    *,
    f_threshold: float = 2.0,
) -> TypologyComparison:
    """Classify each shared feature as shared vs discriminating (F-ratio).

    Only features present in >= 2 signatures are compared. ``f_threshold`` is
    the between/within variance ratio above which a feature is deemed to
    discriminate the typologies.
    """
    import math

    if len(signatures) < 2:
        raise ValueError("need at least 2 signatures to compare")

    # features present in at least two signatures
    counts: dict[str, int] = {}
    for sig in signatures:
        for f in sig.features():
            counts[f] = counts.get(f, 0) + 1
    common = [f for f, c in counts.items() if c >= 2]

    dims: list[DimensionVerdict] = []
    for f in sorted(common):
        groups = [(s.name, s.profile[f]) for s in signatures if f in s.profile]
        total_n = sum(g["n"] for _, g in groups)
        k = len(groups)
        if k < 2 or total_n <= k:
            continue
        grand = sum(g["n"] * g["mean"] for _, g in groups) / total_n
        ss_between = sum(g["n"] * (g["mean"] - grand) ** 2 for _, g in groups)
        ss_within = sum(max(g["n"] - 1, 0) * g["std"] ** 2 for _, g in groups)
        df_between = k - 1
        df_within = total_n - k
        ms_within = ss_within / df_within if df_within > 0 else 0.0
        ms_between = ss_between / df_between if df_between > 0 else 0.0
        f_ratio = (ms_between / ms_within) if ms_within > 1e-12 else (
            math.inf if ms_between > 0 else 0.0)
        pooled_std = math.sqrt(ms_within) if ms_within > 0 else 0.0
        devs = {
            name: ((g["mean"] - grand) / pooled_std) if pooled_std > 0 else 0.0
            for name, g in groups
        }
        kind = "discriminating" if f_ratio >= f_threshold else "shared"
        f_val = 1e9 if f_ratio == math.inf else f_ratio
        dims.append(DimensionVerdict(feature=f, f_ratio=f_val, kind=kind,
                                     deviations=devs))
    return TypologyComparison(signatures=[s.name for s in signatures],
                              dimensions=dims)


class SignatureLibrary:
    """Append-only, name-keyed store of typology signatures (JSONL)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._sigs: dict[str, TypologySignature] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s = TypologySignature.from_dict(json.loads(line))
                self._sigs[s.name] = s
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping bad signature line: %s", exc)

    def deposit(self, sig: TypologySignature) -> TypologySignature:
        self._sigs[sig.name] = sig
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for s in self._sigs.values():
                f.write(json.dumps(s.to_dict(), default=str) + "\n")
        return sig

    def get(self, name: str) -> Optional[TypologySignature]:
        return self._sigs.get(name)

    def all(self) -> list[TypologySignature]:
        return list(self._sigs.values())
