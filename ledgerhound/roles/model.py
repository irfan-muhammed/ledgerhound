"""
ledgerhound.roles.model
=======================

Column roles: what each column IS in a fraud dataset, independent of what it
is called. Guardrails key off roles, never names, so any schema works once
its roles are confirmed (design doc, "Column roles").

A RoleMap is one dataset's assignment. Each column has a role, a confidence
and a status saying who settled it:

    proposed   inferred, not settled yet
    auto       inferred with high confidence and settled without asking
               (never allowed for label, label_time, label_source)
    person     confirmed by a person (terminal prompt or MCP elicitation)
    agent      relayed by the host agent; reported as unverified

The label roles (label, label_time, label_source) drive the leakage and
label-maturity guards, so they are always put to the person, including the
answer "this dataset has no such column".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# role -> one-line meaning (shown to the person and the agent)
ROLES: dict[str, str] = {
    "payer": "who sends the money (sender account, card holder, originator)",
    "payee": "who receives the money (beneficiary, merchant account, destination)",
    "account": "an account or customer identifier that is neither side of a payment",
    "card": "a card identifier",
    "device": "a device identifier or fingerprint",
    "merchant": "a merchant identifier or merchant category",
    "transaction_id": "a unique row or transaction identifier",
    "event_time": "when the transaction happened",
    "amount": "the transaction amount",
    "channel": "the payment rail, product or transaction type",
    "label": "the fraud outcome being predicted",
    "label_time": "when the label was assigned (report or chargeback date)",
    "label_source": "where the label came from (chargeback, complaint, rule, investigation)",
    "rule_output": "the output of a live rule or alert (downstream of the label)",
    "score": "a model or vendor risk score (downstream of the label)",
    "pii": "personal data (name, phone, email, address)",
    "feature": "an ordinary input with no domain role",
}

# Never settled without the person (Phase 3 decision, 2026-11-05).
PERSON_ONLY = ("label", "label_time", "label_source")
ENTITY_ROLES = ("payer", "payee", "account", "card", "device", "merchant")
# Roles a model must never use as an input: the label itself, facts that only
# exist after the label, and outputs computed downstream of it.
NOT_A_FEATURE = ("label", "label_time", "label_source", "rule_output", "score",
                 "transaction_id")
# One column at most per role.
SINGLE = ("label", "label_time", "label_source", "event_time", "amount")

SETTLED = ("auto", "person", "agent")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Assignment:
    role: str
    confidence: str = "low"          # high | medium | low
    reason: str = ""
    status: str = "proposed"         # proposed | auto | person | agent

    @property
    def settled(self) -> bool:
        return self.status in SETTLED


@dataclass
class RoleMap:
    columns: dict[str, Assignment] = field(default_factory=dict)
    # Person-only roles the person said the dataset does NOT have:
    # role -> status ("person" | "agent"); proposed absences are "proposed".
    absent: dict[str, str] = field(default_factory=dict)
    data_sha256: str = ""
    confirmed_by: str = ""           # "" | person | agent
    confirmed_at: str = ""
    history: list[dict] = field(default_factory=list)

    # -- queries -------------------------------------------------------------
    def role_of(self, column: str) -> str:
        a = self.columns.get(column)
        return a.role if a else ""

    def columns_with(self, role: str) -> list[str]:
        return [c for c, a in self.columns.items() if a.role == role]

    def first(self, role: str) -> str:
        cols = self.columns_with(role)
        return cols[0] if cols else ""

    @property
    def confirmed(self) -> bool:
        """Every column settled, and every person-only role either assigned
        or declared absent -- by a person or the agent."""
        if not self.confirmed_by:
            return False
        for a in self.columns.values():
            if not a.settled:
                return False
        for r in PERSON_ONLY:
            if not self.columns_with(r) and self.absent.get(r) not in ("person", "agent"):
                return False
        return True

    def needs_asking(self) -> list[str]:
        """Columns the person must settle: person-only roles, and anything
        not inferred with high confidence."""
        return [c for c, a in self.columns.items()
                if a.role in PERSON_ONLY or a.confidence != "high"]

    # -- mutation ------------------------------------------------------------
    def settle(self, by: str) -> None:
        """Record the person's (or agent's) yes to the whole map."""
        for a in self.columns.values():
            if a.status == "proposed" or a.role in PERSON_ONLY:
                a.status = by
        for r in PERSON_ONLY:
            if not self.columns_with(r):
                self.absent[r] = by
        self.confirmed_by = by
        self.confirmed_at = _now()
        self.history.append({"ts": self.confirmed_at, "by": by,
                             "roles": {c: a.role for c, a in self.columns.items()
                                       if a.role != "feature"}})

    def auto_settle(self) -> None:
        for a in self.columns.values():
            if a.status == "proposed" and a.confidence == "high" \
                    and a.role not in PERSON_ONLY:
                a.status = "auto"

    # -- display -------------------------------------------------------------
    def table(self, *, include_features: bool = False, only: list[str] | None = None) -> str:
        rows = []
        for c, a in self.columns.items():
            if only is not None and c not in only:
                continue
            if a.role == "feature" and not include_features and only is None:
                continue
            rows.append(f"  {c} -> {a.role} ({a.confidence}; {a.reason})")
        return "\n".join(rows)

    def summary(self) -> str:
        """Compact: non-feature roles, then counts."""
        parts = [f"{c}={a.role}" for c, a in self.columns.items() if a.role != "feature"]
        n_feat = sum(1 for a in self.columns.values() if a.role == "feature")
        missing = [r for r in PERSON_ONLY if not self.columns_with(r)]
        s = ", ".join(parts) or "(no domain roles)"
        s += f"; {n_feat} feature column(s)"
        if missing:
            s += "; not in the data: " + ", ".join(missing)
        return s

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["columns"] = {c: asdict(a) for c, a in self.columns.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RoleMap":
        cols = {c: Assignment(**{k: v for k, v in a.items()
                                 if k in Assignment.__dataclass_fields__})
                for c, a in (d.get("columns") or {}).items()}
        return cls(columns=cols, absent=dict(d.get("absent") or {}),
                   data_sha256=str(d.get("data_sha256") or ""),
                   confirmed_by=str(d.get("confirmed_by") or ""),
                   confirmed_at=str(d.get("confirmed_at") or ""),
                   history=list(d.get("history") or []))

    def copy(self) -> "RoleMap":
        return RoleMap.from_dict(self.to_dict())
