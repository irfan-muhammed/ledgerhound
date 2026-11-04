"""Gates on record_finding: restatements of known findings, and findings
that compare the same quantity under different predicates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

# Words that carry no subject in a fraud claim. Deliberately includes the
# domain's own vocabulary ("fraud", "transaction", "account", "rate"):
# every claim contains them, so they say nothing about WHICH claim it is.
CLAIM_STOPWORDS = frozenset("""
    a an and are as at be been by for from has have in is it its of on or
    that the this to with which while when where than then these those
    show shows showed shown indicate indicates higher lower high low more
    less most least very rate rates fraud fraudulent non nonfraud data
    dataset transaction transactions txn account accounts analysis
    observed observation appears appear seem seems significant
    significantly increase increases increased decrease decreases across
    between compared comparison overall total count counts value values
    """.split())


def claim_tokens(s: str) -> set:
    """Distinctive tokens of a claim (lower-cased, stopwords dropped)."""
    raw = re.findall(r"[a-z0-9]+", (s or "").lower())
    return {t for t in raw if len(t) > 1 and t not in CLAIM_STOPWORDS}


def claims_match(mine: set, theirs: set) -> bool:
    """Both bars: enough absolute overlap to be the same subject, and enough
    of the known claim covered that it is a restatement rather than a
    passing mention. Deliberately crude -- distinctive-token overlap, no
    semantics. A false positive costs one sentence of pushback; the
    alternative cost four review iterations that produced "no substantially
    new fraud insights"."""
    if not mine or not theirs:
        return False
    shared = mine & theirs
    return len(shared) >= 3 and len(shared) / len(theirs) >= 0.5


class RestatementGuard:
    """V3-061: record_finding is checked BEFORE anything is written against
    three sources -- the domain profile's established findings, this
    book's own cross-session history, and the team knowledge base. A match
    without a stated `delta` is REFUSED; the model must say what the claim
    adds. Also counts consecutive refusals to suggest (never force) that a
    chapter has run its course."""

    STALL_THRESHOLD = 3

    def __init__(self):
        self.consecutive_refusals = 0

    # -- matching ---------------------------------------------------------
    @staticmethod
    def established_match(claim: str, domain_profile: Optional[dict]) -> tuple[str, str]:
        """``(finding_id, established_claim)`` if `claim` restates a domain-
        profile established finding, else ``("", "")``."""
        if not domain_profile or not claim:
            return ("", "")
        try:
            from ledgerhound.context.domain_profiles import established_claims
        except Exception:  # noqa: BLE001
            return ("", "")
        mine = claim_tokens(claim)
        if not mine:
            return ("", "")
        for fid, known in established_claims(domain_profile):
            if claims_match(mine, claim_tokens(known)):
                return (fid, known)
        return ("", "")

    @staticmethod
    def source_matches(claim: str, book_claims: Iterable[tuple[str, str]],
                       team_claims: Iterable[tuple[str, str]]) -> list[tuple[str, str, str]]:
        """``(source, ref, matched_text)`` for every book-history or team-KB
        claim that `claim` looks like a restatement of."""
        mine = claim_tokens(claim)
        if not mine:
            return []
        matches = []
        for ref, known in book_claims:
            if claims_match(mine, claim_tokens(known)):
                matches.append(("this book's history", ref, known))
        for name, known in team_claims:
            if claims_match(mine, claim_tokens(known)):
                matches.append(("the team knowledge base", name, known))
        return matches

    # -- the gate ---------------------------------------------------------
    def check(self, claim: str, delta: str, *, domain_profile: Optional[dict],
              book_claims: Iterable[tuple[str, str]],
              team_claims: Iterable[tuple[str, str]]) -> tuple[str, dict]:
        """Returns ``(refusal, detail)``. `refusal` is "" when the finding
        may be recorded (no match, or a non-blank delta). `detail` carries
        the matches for the event log either way."""
        fid, known = self.established_match(claim, domain_profile)
        others = self.source_matches(claim, book_claims, team_claims)
        detail = {"domain_profile_match": fid or "",
                  "other_matches": [m[0] for m in others]}
        if not (fid or others) or delta.strip():
            return "", detail

        self.consecutive_refusals += 1
        stall = self.stall_note()
        if fid:
            msg = (
                f"REFUSED (not recorded) -- this looks like established "
                f"finding [{fid}], which the reviewer has already "
                f"accepted:\n  \"{known}\"\n"
                "Re-deriving accepted results is what made four review "
                "iterations produce no new insight. Call record_finding "
                "again with the SAME claim plus delta=\"...\" stating "
                "what this specifically ADDS (a mechanism, an "
                "interaction, account-level behaviour, a feature it "
                "justifies) -- or drop it and move on.")
            return (f"{msg}\n{stall}" if stall else msg), detail
        lines = ["REFUSED (not recorded) -- this looks like a restatement of:"]
        for source, ref, known_claim in others:
            lines.append(f"  [{source}] {ref}: \"{known_claim}\"")
        lines.append(
            "Call record_finding again with the SAME claim plus "
            "delta=\"...\" stating what this specifically ADDS beyond "
            "what's already known -- or drop it and move on.")
        if stall:
            lines.append(stall)
        return "\n".join(lines), detail

    def note_recorded(self) -> None:
        self.consecutive_refusals = 0

    def stall_note(self) -> str:
        n = self.consecutive_refusals
        if n < self.STALL_THRESHOLD:
            return ""
        return (
            f"NOTE: the last {n} finding attempts in a row were restatements "
            "of something already known, with no new delta stated. Consider "
            "whether this chapter/phase has run its course -- wrap up and "
            "move on, or open_chapter on a genuinely different approach. "
            "Not forced; your call.")


class PredicateConsistencyCheck:
    """V3-052 (advisory): if an earlier finding in this run shares the
    claim's distinctive subject but was recorded under a DIFFERENT
    predicate, say so -- the two numbers are not comparable. Reproduces the
    7.8%-vs-31.9% weekend-share contradiction that two windows produced."""

    @staticmethod
    def check(claim: str, predicate: str, findings_path: Path) -> str:
        if not predicate or not claim or not findings_path.exists():
            return ""
        mine = claim_tokens(claim)
        if not mine:
            return ""
        try:
            lines = findings_path.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: BLE001
            return ""
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            other_claim = rec.get("claim", "")
            other_pred = rec.get("predicate", "")
            if not other_pred or other_pred == predicate:
                continue
            if claims_match(mine, claim_tokens(other_claim)):
                return (
                    "NOTE: this looks like it measures the same thing as an "
                    f"earlier finding (\"{other_claim}\"), but under a "
                    f"DIFFERENT predicate -- this one: \"{predicate}\", the "
                    f"earlier one: \"{other_pred}\". Do not compare the two "
                    "numbers directly until you've confirmed they use the "
                    "same condition/window.")
        return ""
