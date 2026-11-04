"""
ledgerhound.context.domain_profiles
===================================

Declarative "what kind of dataset is this?" profiles, matched on COLUMN
NAMES ONLY (no data is read) the moment ``load_data`` resolves a file.

Why (Kubeflow run 2026-08-06)
-------------------------------
The analyst spent 7.8 hours on a UPI mule table, correctly worked out on
its own that "the flag is effectively an account-level mule label" -- and
then shipped a transaction-level model anyway. The knowledge that this
schema means an account-level mule problem was available from the first
second, in the column names. Nothing captured it, so it had to be
rediscovered, and rediscovering it late did not change what got built.

A profile PROPOSES; it never silently overrides a goal the user already
declared. Each profile is a YAML file in this directory -- adding support
for a new table shape is a text file, not a code change.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_PROFILE_DIR = Path(__file__).parent


def _norm(name: str) -> str:
    """Lowercase and strip non-alphanumerics so payer_ifsc, payerIFSC and
    'Payer IFSC' all compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _profile_dirs() -> list[Path]:
    """Packaged (generic, public) profiles first, then an optional
    institutional override directory.

    This package must never carry real customer/production data or
    proprietary business rules -- it is shipped in the public repo. Real,
    sensitive profiles (an organisation's own fraud findings, real column
    samples, production business rules) belong OUTSIDE the repo entirely,
    in a directory named by main_config.yaml's knowledge.profiles_dir (or
    the older LEDGERHOUND_PROFILES_DIR env var). Both are scanned; the
    override directory can add new profiles or, by matching a packaged
    profile's filename stem, extend/replace one without touching this
    package's tracked files."""
    from ledgerhound.config import get_str
    dirs = [_PROFILE_DIR]
    extra = get_str("knowledge", "profiles_dir", env="LEDGERHOUND_PROFILES_DIR")
    if extra:
        p = Path(extra)
        if p.is_dir():
            dirs.append(p)
        else:
            logger.warning("knowledge.profiles_dir=%s is not a directory -- ignored", extra)
    return dirs


def load_profiles() -> list[dict]:
    """Every profile YAML across the packaged dir and (if set)
    LEDGERHOUND_PROFILES_DIR. Best-effort: a malformed file is skipped with a
    warning rather than breaking data loading. A later directory's profile
    with the same filename stem REPLACES an earlier one -- so an
    institutional override directory can supersede a packaged default."""
    import yaml

    by_stem: dict[str, dict] = {}
    for profile_dir in _profile_dirs():
        for path in sorted(profile_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("match"):
                    data.setdefault("name", path.stem)
                    by_stem[path.stem] = data
            except Exception as e:  # noqa: BLE001
                logger.warning("domain profile %s failed to load: %s", path.name, e)
    return list(by_stem.values())


def match_profile(columns) -> dict | None:
    """Return the best-matching profile for `columns`, or None.

    Both bars must be cleared: every `requires_all` column present, AND at
    least `min_optional` of the `signals`. That second bar is what stops a
    generic payments table matching on two coincidental column names.
    """
    have = {_norm(c) for c in (columns or [])}
    if not have:
        return None
    best, best_score = None, -1
    for prof in load_profiles():
        m = prof.get("match") or {}
        required = [_norm(c) for c in (m.get("requires_all") or [])]
        if not required or not all(r in have for r in required):
            continue
        signals = [_norm(c) for c in (m.get("signals") or [])]
        hits = sum(1 for s in signals if s in have)
        if hits < int(m.get("min_optional") or 0):
            continue
        score = len(required) + hits
        if score > best_score:
            best, best_score = prof, score
    return best


def resolve_label_column(profile: dict, columns) -> str:
    """First label candidate from the profile that actually exists."""
    have = {_norm(c): str(c) for c in (columns or [])}
    for cand in ((profile.get("goal") or {}).get("label_column_candidates") or []):
        if _norm(cand) in have:
            return have[_norm(cand)]
    return ""


def resolve_entity_key(profile: dict, columns) -> str:
    have = {_norm(c): str(c) for c in (columns or [])}
    declared = (profile.get("goal") or {}).get("entity_key", "")
    if declared and _norm(declared) in have:
        return have[_norm(declared)]
    for cand in (profile.get("suggested_entity_keys") or []):
        if _norm(cand) in have:
            return have[_norm(cand)]
    return ""


def describe(profile: dict, columns) -> str:
    """The block shown to the analyst when a profile matches."""
    goal = profile.get("goal") or {}
    lines = [f"[DATASET RECOGNISED] {profile.get('title', profile.get('name'))}"]
    lines.append(f"proposed objective: {(goal.get('objective') or '').strip()}")
    ek = resolve_entity_key(profile, columns)
    lab = resolve_label_column(profile, columns)
    lines.append(f"proposed unit of analysis: {goal.get('unit_of_analysis', '?')}"
                 + (f"   entity key: {ek}" if ek else ""))
    if lab:
        lines.append(f"detected label column: {lab}")
    for n in (profile.get("notes") or []):
        lines.append(f"  - {' '.join(str(n).split())}")

    # V3-048: reviewer rulings and priors. Rendered here or they are inert
    # -- describe() previously emitted `notes` only, so any other key in the
    # profile yaml was silently ignored.
    excl = profile.get("exclude_from_features") or []
    if excl:
        lines.append("")
        lines.append("DO NOT USE AS FEATURES (domain ruling, not a guess):")
        for e in excl:
            lines.append(f"  - {e.get('column', '?')}: "
                         f"{' '.join(str(e.get('reason', '')).split())}")
            if e.get("also_avoid"):
                lines.append(f"    {' '.join(str(e['also_avoid']).split())}")

    avoid = profile.get("avoid_analyses") or []
    if avoid:
        lines.append("")
        lines.append("ANALYSES JUDGED UNPRODUCTIVE ON THIS TABLE:")
        for a in avoid:
            lines.append(f"  - {a.get('analysis', '?')}: "
                         f"{' '.join(str(a.get('reason', '')).split())}")

    known = profile.get("established_findings") or []
    if known:
        lines.append("")
        lines.append(
            "ALREADY ESTABLISHED -- DO NOT RE-REPORT THESE AS FINDINGS.")
        lines.append(
            "  Re-deriving them is exactly what made four review iterations "
            "produce no new insight. Start here; report what lies BEYOND "
            "them. Magnitudes are from one extract and window -- re-measure "
            "before asserting them elsewhere.")
        for f in known:
            lines.append(f"  - [{f.get('id', '?')}] "
                         f"{' '.join(str(f.get('claim', '')).split())}")
            if f.get("beyond"):
                lines.append(f"      -> go further: "
                             f"{' '.join(str(f['beyond']).split())}")

    directions = profile.get("priority_directions") or []
    if directions:
        lines.append("")
        lines.append("WHERE THE REVIEWER ASKED THE NEXT ITERATION TO GO:")
        for d in directions:
            lines.append(f"  - {' '.join(str(d).split())}")

    lines.append("")
    lines.append("This is a PROPOSAL from a column-name profile, not a "
                 "measurement. Confirm it against the data (and with the "
                 "user via ask_user if it conflicts with what they asked "
                 "for), then call set_goal to lock it in.")
    return "\n".join(lines)


def established_claims(profile: dict) -> list[tuple[str, str]]:
    """``(id, claim)`` for every already-established finding in a profile.

    Used by record_finding to notice when the analyst is about to report
    something the reviewer already accepted (V3-048).
    """
    return [(str(f.get("id", "")), str(f.get("claim", "")))
            for f in (profile.get("established_findings") or [])
            if f.get("claim")]
