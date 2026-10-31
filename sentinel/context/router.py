"""
sentinel.context.router
=======================

Selects which parts of the digest and knowledge base to load into the
LLM prompt, based on the user's question topic.

With 70+ columns, we can't load everything.  The router matches
keywords in the question to column groups and knowledge tags, then
builds a ``PromptContext`` with three tiers:

* **Tier 0** -- always present (overview, priority, quality).
* **Tier 1** -- on-demand column profiles for the relevant topic.
* **Knowledge** -- matched FAQs, patterns, column semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sentinel.profiler.digest import Digest
from sentinel.knowledge.loader import KnowledgeBase


@dataclass
class PromptContext:
    """Assembled context ready to be inserted into the system prompt."""
    tier0: str = ""
    tier1: str = ""
    knowledge: str = ""
    total_tokens_est: int = 0


class ContextRouter:
    """Map question topics -> relevant digest sections + knowledge."""

    # question keywords -> column names to load into tier-1 context
    _COL_MAP: dict[str, list[str]] = {
        r"amount|spend|value|revenue|atv|gmv": ["txnamount", "initamount"],
        r"decline|fail|error|reject|success|status": ["currstatusdesc", "errorcode", "finalrespcode", "pyrrespcode", "remrespcode", "pyerespcode", "benrespcode"],
        r"merchant|mcc|p2m|shop|category|store": ["pycode", "prcode", "merchantgen", "merchanttype", "ownertype", "payee_verified_name"],
        r"device|mobile|app|os|prdid|fingerprint": ["prdid", "prdtype", "prdos"],
        r"bank|psp|ifsc|remitter|beneficiary": ["prnfsparticipantid", "drnfsparticipantid", "crnfsparticipantid", "pynfsparticipantid"],
        r"time|hour|day|temporal|trend|night|date|month": ["trndate", "crtdate", "asdt"],
        r"account|savings|credit|wallet|acctype|card": ["pracctype", "pyacctype", "cardvariant"],
        r"collect|pay|txntype|p2p|p2m|transaction": ["txntype", "txnsubtype", "purposecode", "initmode"],
        r"fraud|suspicious|risk|pattern|signal|anomal": [],  # -> all high-priority columns
        r"vpa|sender|receiver|payer|payee|address": ["prfvaddr", "pyfvaddr"],
        r"auth|credential|pin|otp|mpin": ["cred_type", "cred_subtype"],
        r"lite|autopay|mandate|nfc|tap|circle|ipo": ["initmode", "purposecode", "pracctype"],
        # -- Behavioral feature dataset (bank_fraud) columns --
        r"payee|incoming|receive": ["py_cnt_paid", "py_cnt_txn", "py_tot_amt", "py_avg_amt", "py_cnt_devid", "py_cnt_vpa"],
        r"payer|outgoing|send": ["pr_cnt_paid", "pr_cnt_txn", "pr_tot_amt", "pr_avg_amt", "pr_cnt_devid", "pr_cnt_vpa"],
        r"device|devid|os|switch": ["py_cnt_devid", "pr_cnt_devid", "py_cnt_os", "pr_cnt_os"],
        r"vpa|switch|address": ["py_cnt_vpa", "pr_cnt_vpa", "vpa_switch_count_payee", "vpa_switch_rate_payee", "vpa_switch_count_payer"],
        r"velocity|count|volume|txn": ["py_cnt_txn", "pr_cnt_txn", "in_avg_txns_per_day", "out_avg_txns_per_day"],
        r"amount|value|spend|total": ["py_tot_amt", "pr_tot_amt", "py_avg_amt", "pr_avg_amt", "py_max_amt", "pr_max_amt"],
        r"time|night|morning|hour|slot|temporal": ["in_night_txns", "in_morning_txns", "in_afternoon_txns", "in_evening_txns", "py_07_12_cnt", "py_17_22_cnt"],
        r"week|weekly|period": ["in_week1", "in_week2", "in_week3", "in_week4", "out_week1", "out_week2"],
        r"fail|decline|error|insufficient|pin|z9|zm|z6": ["cnt_z9_py", "amt_z9_py", "cnt_zm_pr", "amt_zm_pr", "cnt_z6_py"],
        r"high.?value|large|big": ["high_value_in_txns", "high_value_in_txns_1l", "high_value_out_txns", "high_value_out_txns_1l"],
        r"collect|pull|request": ["py_cnt_p2p_collect", "pr_cnt_p2p_collect", "ratio_collect_amt", "py_p2p_collect_amt"],
        r"p2m|merchant": ["out_gng_p2m", "p2m_amount_out", "in_gng_p2m", "p2m_amount_in"],
        r"active|dormant|age|days": ["in_active_days", "out_active_days", "py_cnt_days", "pr_cnt_days", "in_txn_days_ratio"],
        r"surname|same.?name": ["count_sur_name", "same_sur_name_amount"],
        r"target|fraud|label|flag": ["bank_frd_flag"],
    }

    # question keywords -> knowledge tags for FAQ/pattern retrieval
    _TAG_MAP: dict[str, list[str]] = {
        r"fraud|risk|pattern|suspicious": ["fraud_vector", "mule", "account_takeover", "collect"],
        r"amount|spend|value|round": ["amount", "pattern"],
        r"time|hour|night|temporal|trend": ["temporal", "seasonality"],
        r"device|sim|fingerprint": ["device", "fraud_ring", "sim_swap"],
        r"merchant|mcc|shop|category": ["mcc", "merchant"],
        r"collect|request": ["collect", "fraud_vector"],
        r"velocity|burst|speed|rate": ["velocity", "mule"],
        r"decline|fail|error|reject": ["decline", "data_quality"],
        r"account.*age|new.*account|onboard": ["account_age", "onboarding"],
        r"graph|network|ring|chain|mule": ["graph", "mule", "fraud_ring"],
        r"psp|phonepe|gpay|paytm|bhim": ["psp", "descriptive"],
        r"quality|empty|null|hash|broken": ["data_quality", "empty_columns"],
        r"auth|pin|credential": ["authentication", "credential"],
        # -- Behavioral feature tags --
        r"payee|incoming|receive": ["payee", "fraud_signal"],
        r"payer|outgoing|send": ["payer", "fraud_signal"],
        r"switch|vpa.?switch": ["vpa_switching", "fraud_signal"],
        r"dormant|active|age": ["account_age", "fraud_signal"],
        r"surname|same.?name": ["mule", "fraud_signal"],
        r"high.?value|large": ["high_value", "velocity"],
        r"response.?code|z9|zm|z6|u30|u16": ["decline_pattern", "response_code"],
        r"week|weekly": ["temporal"],
    }

    def __init__(self, digest: Digest, knowledge: KnowledgeBase):
        self.digest = digest
        self.kb = knowledge

    def build_context(self, question: str) -> PromptContext:
        ctx = PromptContext()
        ctx.tier0 = self._tier0()
        cols = self._match_cols(question)
        ctx.tier1 = self._tier1(cols)
        tags = self._match_tags(question)
        real_cols = set(self.digest.column_profiles.keys())
        items = self.kb.retrieve(topic_tags=tags, columns=cols, max_items=5,
                                 dataset_columns=real_cols)
        ctx.knowledge = self.kb.format_for_prompt(items)
        ctx.total_tokens_est = (len(ctx.tier0) + len(ctx.tier1) + len(ctx.knowledge)) // 4
        return ctx

    # -- builders -------------------------------------------

    def _tier0(self) -> str:
        o, dq, cp = self.digest.overview, self.digest.data_quality, self.digest.column_priority
        lines = [
            "=== DATASET OVERVIEW ===",
            f"Source: {o.get('source','?')}  |  Rows: {o.get('rows',0):,}  |  "
            f"Columns: {o.get('columns',0)}  |  Target: {o.get('target','none')}",
            f"Quality: {dq.get('score','?')}/100",
        ]
        for issue in dq.get("critical", []):
            lines.append(f"  ! {issue.get('column','?')}: {issue.get('issue','')}")
        lines.append(f"High-signal: {', '.join(cp.get('high',[])[:10])}")
        if cp.get("drop_candidates"):
            lines.append(f"Drop candidates: {', '.join(cp['drop_candidates'])}")
        # The FULL, authoritative column list -- so the agent uses real column
        # names instead of guessing plausible ones (e.g. `account_age_days`
        # that does not exist). This is the source of truth for `df` columns.
        all_cols = list(self.digest.column_profiles.keys())
        if all_cols:
            lines.append(f"ALL COLUMNS ({len(all_cols)}) -- use ONLY these "
                         f"names, do not invent any: {', '.join(all_cols)}")
        return "\n".join(lines)

    def _tier1(self, columns: list[str]) -> str:
        if not columns:
            columns = self.digest.column_priority.get("high", [])[:8]
        profiles = self.digest.column_profiles
        parts = ["=== COLUMN PROFILES ==="]
        for col in columns:
            p = profiles.get(col)
            if not p:
                continue
            parts.append(f"\n--- {col} ---")
            if p.get("description"):
                parts.append(f"  Desc: {p['description']}")
            parts.append(f"  Type: {p.get('semantic_type','?')}")
            st = p.get("stats", {})
            if "mean" in st:
                parts.append(f"  Stats: mean={st['mean']}, median={st['median']}, std={st['std']}")
                if st.get("percentiles"):
                    q = st["percentiles"]
                    parts.append(f"  Pctls: p5={q.get('p5','?')}, p50={q.get('p50','?')}, p95={q.get('p95','?')}")
            if "top_values" in st:
                top5 = list(st["top_values"].items())[:5]
                parts.append(f"  Top: {', '.join(f'{k}:{v:,}' for k,v in top5)}")
            if p.get("ks_test"):
                parts.append(f"  KS: stat={p['ks_test'].get('statistic','?')}, p={p['ks_test'].get('p_value','?')}")
            if p.get("by_target"):
                parts.append(f"  By target: {p['by_target']}")
        return "\n".join(parts)

    def _match_cols(self, q: str) -> list[str]:
        q_low = q.lower()
        matched = []
        for pattern, cols in self._COL_MAP.items():
            if re.search(pattern, q_low):
                matched.extend(cols or self.digest.column_priority.get("high", []))
        # deduplicate preserving order
        seen: set[str] = set()
        return [c for c in matched if not (c in seen or seen.add(c))] or \
               self.digest.column_priority.get("high", [])[:8]

    def _match_tags(self, q: str) -> list[str]:
        q_low = q.lower()
        tags: list[str] = []
        for pattern, t in self._TAG_MAP.items():
            if re.search(pattern, q_low):
                tags.extend(t)
        return list(set(tags))
