"""
ledgerhound.mcp.prompts
=======================

The MCP prompts (design doc, "MCP extras"): /fraud-eda, /feature-review and
/sample-audit. Each is a short workflow that names the Ledgerhound tools in
the order the server's rules expect, so a host agent starts on the right
path instead of discovering the refusals one by one. They are text only: no
state, and nothing here bypasses a server-side rule.

Payment-generic by design (Phase 3 gate): nothing assumes a scheme,
currency or column naming.
"""

from __future__ import annotations

_RULES = (
    "Rules that hold throughout: raw data stays in the Ledgerhound kernel, so "
    "compute with run_analysis or sql, not your own pandas. When a tool returns "
    "NEEDS USER DECISION, ask the user exactly that question and never answer it "
    "yourself. Column roles (label, label_time, label_source above all) are the "
    "user's call. Only findings recorded with record_finding reach the report; "
    "anything you quote from kb_search is the team's knowledge, unverified on "
    "this data, unless you check it and record it.")


def fraud_eda(path: str, goal: str = "") -> str:
    goal_line = (f"The user's goal: {goal}. Pass it to set_goal."
                 if goal.strip() else
                 "Ask the user what decision this analysis feeds, then set_goal.")
    return f"""Run a fraud exploratory analysis of {path} with Ledgerhound.

1. load_dataset(path="{path}"), then profile. Ask the user every open data
   question profile lists (record answers with record_data_answer).
2. confirm_roles: show the user the proposed roles and get their answer. If
   there is no label-time column, say so: label maturity will be an
   assumption.
3. {goal_line}
4. kb_search for the typologies, rules and known findings that apply, so you
   know what the team already believes and what is already established.
5. Explore every column, not only the ones that predict the label:
   distributions, missingness, cardinality, time patterns, entity-level
   behaviour (velocity, distinct counterparties, shared devices or payment
   identifiers). Use run_analysis or sql.
6. For each claim worth keeping: falsify(claim), run the routes it lists,
   then record_finding with the sample frame and evidence. If it contradicts
   a Knowledge Pack entry, name it in kb_contradicts.
7. report, with a short narrative that cites recorded findings only.

{_RULES}"""


def feature_review(features: str, label: str = "") -> str:
    lab = f' with label="{label}"' if label.strip() else ""
    return f"""Review these candidate fraud features with Ledgerhound: {features}.

1. If no data is loaded or the roles are not confirmed, do that first
   (load_dataset, profile, confirm_roles with the user).
2. kb_search for each feature's name and for the rules in force: a feature
   that is a rule's output, or computed after the label, is leakage.
3. check_feature(features=[...]{lab}). Treat a REFUSED or leakage verdict as
   final for this sample; explain it to the user instead of working around it.
4. For each feature that survives, state why it should generalise: which
   behaviour it captures, at which entity grain, over which time window. Test
   that claim with falsify, run the routes, and record_finding.
5. List the features you rejected and why, then report.

{_RULES}"""


def sample_audit(design_prevalence: str = "", label_lag: str = "") -> str:
    prev = (f'design_prevalence="{design_prevalence}"' if design_prevalence.strip()
            else "design_prevalence (ask the user: a fraction, or 'natural')")
    lag = (f', label_lag="{label_lag}"' if label_lag.strip() else "")
    return f"""Build and audit a modelling sample with Ledgerhound.

1. Make sure roles are confirmed (confirm_roles with the user). The sample is
   built from them: label, event_time, label_time and the entity roles.
2. build_sample({prev}{lag}). If it refuses, follow the fix it names. With no
   label-time column it needs label_lag: ask the user how long fraud labels
   take to arrive (or whether they are final). Their answer is recorded as an
   assumption, not a check.
3. Read ledgerhound://certificate. Go through every section with the user:
   its status, each finding, and every NOT CHECKED line. A PASS never covers
   what is listed as not checked.
4. Verify the sample in the kernel (run_analysis on `train` and `test`):
   prevalence per split, entity overlap, time ranges against the maturity
   horizon.
5. Record what you verified with record_finding (sample frame:
   the train/test sample), then report. The report carries the certificate's
   hash and whether label maturity was measured or assumed.

{_RULES}"""


PROMPTS = {
    "fraud-eda": (fraud_eda, "Fraud EDA: load, confirm roles, explore every "
                  "column, falsify and record findings, report."),
    "feature-review": (feature_review, "Review candidate features for leakage and "
                       "generalisation, against the roles and the Knowledge Pack."),
    "sample-audit": (sample_audit, "Build a roles-driven sample and audit its "
                     "construction certificate with the user."),
}
