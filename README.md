# Ledgerhound — Agentic EDA for UPI Fraud & Mule Modeling

> **Renamed from Sentinel.** The package, CLI and config are now
> `ledgerhound`. Old `SENTINEL_*` environment variables are still read (with
> a deprecation warning), and existing `.sentinel/` folders, `SENTINEL.md`
> and `sentinel_workspace/` are renamed automatically on first run.

> Autonomous evidence generation for fraud feature engineering teams.
> Not a BI tool. Not a dashboard. A statistically rigorous, compute-aware
> EDA agent that produces auditable, falsifiable findings at UPI scale.

**Just want to run it?** → [How to run — the short version](#how-to-run--the-short-version)
below, or the full checklist in [`docs/USER_GUIDE.md` §0](docs/USER_GUIDE.md#0-how-to-run----start-to-finish).

---

## Why This Exists

Every fraud modeling team runs EDA. Almost none of it is reproducible, statistically certified,
or institutionally remembered. A data scientist spends 3 weeks on EDA, produces a notebook
nobody revisits, the findings live in their head, and 6 months later a new team member starts
from zero on the same dataset.

Meanwhile the real problem compounds: the true fraud-to-non-fraud ratio in UPI transactions
is not 1:1000 — it is closer to **1:100,000**. The 1:1000 training ratio every team uses is
a convention nobody has ever validated against the true population. The non-fraud sample the
model trains on might be systematically biased in ways nobody has measured. Feature signal
decays as fraudsters adapt, but nobody tracks this until the model degrades in production.

Ledgerhound exists to fix all of this — with numbers, not intuition.

---

## Current State — 2026-09-19

Most of this README describes the original design intent. This section says
what is **actually built**, so the two are never confused.

[![CI](https://github.com/irfan-muhammed/ledgerhound/actions/workflows/ci.yml/badge.svg)](https://github.com/irfan-muhammed/ledgerhound/actions/workflows/ci.yml)

Version `0.7.9`. **151 tests** (`pytest`), run by CI on Python 3.10, 3.11
and 3.12. They cover the HITL gate, guardrail refusals, the restatement
guard, tool-call parsing (native, text and SSE), the hash-chained event log,
the tool layer without an LLM, and a full scripted-LLM session. That is a safety net for the core loop, not
coverage of every module: most of the ~80k lines still have no tests.
(Earlier drafts of this README claimed ~2,700 tests; they were never
committed.)

### The live path: `ledgerhound shell` *(the actual product today)*

Everything below in this subsection is the CLI shell (`ConversationalAnalyst`,
`ledgerhound/shell/conversation.py`) — the engine a user actually talks to. The
v4 pipeline described next has **no production caller yet**; this does.

It is three layers: `shell/loop.py` (the LLM turn loop), `shell/tools/` (session
state and every tool, usable without an LLM) and `shell/guardrails/` (one class
per guardrail, enforced inside the tools).

- **GPU-first, not GPU-optional.** Kernel mode eagerly initialises RMM
  (managed memory, so it can oversubscribe past one card), installs
  `cudf.pandas`, and imports cuDF before the model's first cell runs — a
  stray pandas frame is pulled back to GPU after every cell. A GPU detected
  but stranded behind missing kernel deps now prints a loud warning instead
  of silently degrading to CPU.
- **Outputs follow the data.** Feature tables go under `features_dir`: the
  run folder for local data; for `/s3` data, next to the RAM copy and
  uploaded to the bucket after every step. Writes aimed anywhere else (the
  source-data folder, a bare `features/` path, pyarrow, plain files) are
  redirected there, never lost or overwritten. `/paths` shows where.
- **Team knowledge, shared automatically.** Dataset cards, answers to data
  questions, findings and books sync through `s3://sentinel-data/kb/v1/` --
  append-only, outbox-backed, customer identifiers removed before upload.
- **Asks instead of assuming.** After profiling, the agent asks what the
  label's classes mean, which entity and time column, units and code
  meanings -- once per dataset for the whole team.
- **The Librarian — book / story / chapter / page.** One *book* per
  dataset (content-hash keyed), a new *story* opens automatically when the
  data changes, and re-opening a *chapter* (a declared approach, from a
  fixed taxonomy) that already has recorded pages on the *same* data is
  **refused**, showing what was already found — this is what stops a second
  session from silently redoing the first session's work. `/library`,
  `/book`, `/story`, `/chapter` in the shell. Cross-period stability
  (`record_percentile_snapshot` / `check_cross_period_stability`) diffs a
  finding across two data versions and classifies it stable vs. shifted,
  mechanically — not an LLM judgment call.
- **Feature-tiering catches what IV alone hides.** A feature can score a
  healthy Information Value while its AUC sits at or below 0.5 — no real
  separation, or the direction is inverted. `tier_features` now catches
  this instead of shipping the feature as `core`. A percentile-table +
  separation-callout check (crossing tails, where two distributions
  actually diverge) is available as a sandbox helper and wireable into the
  exported feature catalog.
- **Dual-node aggregation.** Features can be generated against *both* sides
  of a two-party relationship (e.g. sender and receiver) in one pass,
  tagged so a reader can tell which side a feature came from — aggregation
  is no longer implicitly single-sided.
- **Findings can't be silently lost.** A save-report failure used to mark
  findings "consumed" before the write succeeded, permanently dropping
  them; that ordering bug is fixed. A `predicate` field on findings lets a
  cross-finding check catch two claims that sound comparable but were
  computed under different conditions/windows before they get compared as
  if equivalent.
- **No real institutional data ships in this package.** Domain profiles and
  knowledge-base content (real column semantics, business rules, prior
  findings) load from `main_config.yaml`'s `knowledge.profiles_dir` /
  `knowledge.knowledge_dir` if set — nothing customer- or
  production-specific is packaged by default. See `main_config.example.yaml`
  for every setting the project reads.

### v4 — sample construction and certification *(new, `ledgerhound/pipeline/`)*

The v4 work answers a specific question: *was this training sample built the
way it claims to have been?* Twelve modules; per-module rationale lives in
each module's docstring.

| Module | What it does |
|---|---|
| `model_story` | Declarative spec for a model story. **Label grain is a parameter** — account-grain (mule) and transaction-grain (ATO/phishing) ship together. |
| `history_fetch` | Raw fraud reports → entity keys → Trino SQL for each entity's N-month history. Generates SQL; never executes it. Injection-safe. |
| `typology` | Reconciles the typologies present in a dataset against the rules that exist, **with volumes**, before labelling runs. Portable scope map + an explicitly-declared default for the long tail. |
| `labeller` | Applies each typology's rule to fetched history. **Tri-state labels** — `1` / `0` / `<NA>`. |
| `aggregate` | Trino push-down or chunked local. **Refuses** aggregations that cannot be composed from chunks rather than approximating them. |
| `prevalence` | Measures the real fraud rate, checks it against the declared design, and corrects precision and alert volume to deployment conditions. |
| `appropriateness` | For a sample somebody hands you: does it match the construction it claims? |
| `splits` | Point-in-time correct **and** entity-disjoint **and** label-maturity aware — three properties that conflict. |
| `cluster_psi` | PSI at the *effective* sample size, because payee-drawn samples are cluster samples. |
| `pu` | The negative class is **"not-reported"**, never "non-fraud", plus contamination sensitivity. |
| `certificate` | Assembles all of the above onto one face — **and gates on it**. |

### The three ideas that matter

**1. The negative class is "not-reported", not "non-fraud".** Fraud is only
ever observed through detection and complaint. An unflagged transaction is
either genuinely legitimate or fraud nobody caught, and we cannot tell which
— so labels are tri-state and a `0` is only produced where a rule
*positively establishes* legitimacy.

**2. Prevalence has a grain, and the grain changes the number.**
Account-grain and transaction-grain prevalence are different figures for the
same data. Comparing across them is refused rather than reported as a
divergence. (This README's own "1:100,000" above is a *population* estimate
at unstated grain — exactly the kind of bare ratio `prevalence.py` now
declines to check against a measurement.)

**3. The gate is the product.** `require_certified()` **raises**. Every
comparable tool emits an advisory warning a user clicks through; feature
engineering on an uncertified sample optimises against a sampling artifact,
so the refusal is the contract.

### What is NOT yet true

Stated plainly because a reader should not have to infer it:

- **Nothing has run against real data.** Modules were exercised only on
  synthetic fixtures during development, and those tests were never
  committed to this repository.
- **No Trino query has ever executed.** SQL is generated, never run.
- **Scale is untested** — largest frame tried was ~200k rows against an
  8-crore target.
- **The gate has no production caller.** Nothing in `cli/` or `shell/`
  imports `ledgerhound.pipeline` yet. Intended for Kubeflow components, which
  import libraries directly; no Kubeflow deployment has been validated.
- **Label rules are undecided**, so the pipeline cannot run end to end until
  the fraud team confirms which typologies apply at which scope.
- **cuDF/GPU: smoke-tested on real hardware, not this codebase's actual
  workload.** 2026-08-04, on a live GPU box: `cudf-cu12` installed and ran
  (`from_pandas` → `groupby` → `to_pandas`, correct results). That test also
  found and fixed a real bug — the router sized GPU capacity off *total*
  card memory only, so on a *shared* GPU (another process already holding
  most of it) it would have greenlit cuDF into headroom that didn't exist;
  it now checks free memory too. `ledgerhound shell` auto-detects a GPU and
  defaults to kernel mode (the only execution path with a cuDF branch) so
  acceleration is reachable without a flag. Still not done: this repo's own
  pipeline/aggregation code has never run on a GPU (only the CLI/shell
  backend routing was tested — a hand-built frame, not this project's data
  or code path); nothing has run at 8-crore scale; and there is no
  multi-GPU sharding (dask-cudf) — a single card tops out well under the
  8-crore target, so full-population EDA stays chunked/pushed to Trino.

---

## The Scale Problem

| Data Object | Grain | Monthly Scale | Compute Tier |
|---|---|---|---|
| UPI Raw Transactions | 1 row = 1 transaction | Billions | TrinoSQL / KubeFlow |
| UPI Summary Table | 1 row = 1 account aggregate | Hundreds of millions | Local after Trino pull |
| Historical Confirmed Fraud | 1 row = 1 confirmed fraud case | Thousands to lakhs | Local H100 / cuDF |

The confirmed fraud data is small enough to run entirely on a local H100 GPU with cuDF.
The non-fraud population is not. No amount of hardware makes it feasible to bring billions
of rows to the GPU for EDA.

Ledgerhound's answer: **never bring the population to the compute — bring the questions to the
population as push-down aggregation queries, then bring only the statistically necessary
sample back locally.**

A Trino aggregation query (`SELECT feature, COUNT(*), AVG(), PERCENTILE_CONT(...)`) over
billions of rows returns kilobytes of statistics in minutes. Only the final targeted sample
— a few million rows — ever moves to local storage. This is the architectural spine that
makes the entire system feasible.

---

## The Class Imbalance Reality

The real fraud rate in UPI is **1:100,000 or worse** — not 1:1000. The current industry
convention of training at 1:1000 is an undersampled non-fraud set, and nobody has proven
it is a *good* undersampled set.

Ledgerhound's position: **any change to the sampling ratio requires statistical evidence, not
opinion.** The EDA produces a three-way comparison:

```
Fraud distribution              (local, cuDF, small — fast)
    vs
Current 1:1000 non-fraud sample (local — already in training data)
    vs
True non-fraud population       (Trino aggregation — statistics only, fast)
```

If the current sample diverges significantly from the true population on the top discriminative
features, that is the evidence required to change the strategy — quantified by PSI
(Population Stability Index) and KS tests per feature, not opinion.

---

## What Ledgerhound Delivers

> **This section is the original phased design, not a status report.** Much
> of Phases 1–3 is built; Phase 4 is partly built. For what actually exists
> today — and what explicitly does not — read *Current State* above.

### Phase 1 — Foundation *(build first, everything depends on this)*

| Module | Delivers |
|---|---|
| **Workspace & Artifact Store** | Structured, versioned file system for every dataset, sample, report, chart, finding, and Trino job. Registry-indexed. Nothing escapes the workspace. |
| **Async Trino Job Manager** | Submit → checkpoint → wait hours → resume without losing state. HITL required before every query. SQL shown in full. COUNT estimation before execution. `use_trino` toggle for local-only mode. |
| **Compute Tier Classifier** | Routes every operation to local H100/cuDF or Trino before the LLM attempts it. Prevents wasted compute and unintended large pulls. |
| **Dataset Knowledge Base** | Living, append-only memory per dataset. Every session deposits insights. Accumulated understanding survives team turnover. Injected into agent context at session start. |
| **Session Log** | Every session recorded: datasets used, findings produced, Trino jobs submitted, summary. Searchable history. |
| **Dual-Track Architecture** | Fraud detection track and mule detection track are first-class from day one. Architecture is neutral — neither track is privileged. Mule track ships in Phase 3. |

### Phase 2 — Core EDA Capabilities

| Deliverable | What It Produces |
|---|---|
| **Evidence-Based Sampling Certification** | PSI-weighted comparison of current sample vs true population. Pass/fail per feature. Weighted by discriminative power — certifies hardest on the features that matter most. |
| **Hard Negative Boundary Identification** | Non-fraud cases closest to the fraud cluster. Inclusion rate vs current random sample. The cases where the current model fails. |
| **Cross-Dataset Consistency Validation** | Summary table vs raw transaction agreement checks. Catches silent aggregation bugs that corrupt model training. |
| **Fraud Typology Clustering** | Cluster confirmed fraud cases by feature similarity. Distinct fraud types have different feature signatures and may need separate models. |
| **Population Profiler** | Push-down Trino aggregation. Returns statistics, not rows. The foundation for all sample certification and bias detection. |

### Phase 3 — Advanced Intelligence

| Deliverable | What It Produces |
|---|---|
| **Feature Signal Decay Analysis** | Monthly univariate AUC per feature over 12–18 month window. Dying features = fraudsters have adapted. Rising features = new patterns emerging. |
| **Temporal Drift Detection** | Data drift vs concept drift, separated. Rolling PSI per feature across time windows. |
| **Survivor Bias Quantification** | Estimated contamination rate in non-fraud labels from undetected fraud. Gives the modeling team a realistic ceiling on achievable precision/recall. |
| **Mule Detection Track** | Account-level behavioral EDA on the summary table. Velocity, dispersion, lightweight graph features from cuDF groupby. Separate sampling strategy from fraud track. |

### Phase 4 — Validation Harness

| Deliverable | What It Proves |
|---|---|
| **Retrospective Claim Validator** | Runs EDA on historical data. Scores every agent claim against what actually happened. Precision and recall for the EDA tool itself. |
| **Targeted Holdout Test Sets** | Four dimensions: hard negative set, rare typology set, temporal holdout, label-contaminated set. Replaces "overall AUC" with four surgical benchmarks. |
| **Drift Detection Lead Time** | How many weeks before model performance degrades does Ledgerhound detect the underlying drift? This is the number that goes in front of leadership. |

---

## Where Ledgerhound Beats Humans

These are not areas where Ledgerhound is *better* — they are areas where the task is
**humanly infeasible at this scale** and Ledgerhound makes it feasible:

**PSI across all features against true population.** A data scientist can check 10–20
features in a day. Ledgerhound checks all 200+ in minutes, against a billion-row population,
every time, with consistent methodology.

**Feature signal decay tracking.** Monthly univariate AUC across all features over
18 months. No human does this as part of regular workflow — it is too tedious.
Ledgerhound does it automatically every session.

**Institutional memory.** A human analyst's findings live in a notebook nobody revisits
after they leave the team. Ledgerhound's dataset knowledge base accumulates across every
session and every analyst. The 6-month-old finding about the null spike in column X is
still there when the new team member joins.

**Consistent methodology.** Humans cut corners on their 50th EDA iteration. Ledgerhound
applies the same statistical tests every time with no fatigue.

**Reproducibility.** Every finding has code, every sample has provenance, every chart
has the query that generated it. Any result can be regenerated from scratch.

---

## Where Ledgerhound Does Not Try to Compete

These are deliberate non-goals. Building them would dilute the above:

| Out of Scope | Why |
|---|---|
| Real-time fraud scoring | Different product. Requires low-latency serving infrastructure. |
| Model training | Ledgerhound feeds the FE team. Half-baked model training produces metrics people quote incorrectly. |
| Full graph analysis at population scale | Requires Neo4j/GraphX. Not feasible on this stack. Will consume months with nothing to show. |
| SMOTE at billion-row scale | Pairwise distance at this scale is infeasible. Hard negative mining (filtering, not generation) achieves more with less. |
| Causal inference / counterfactuals | Fraud behavior is adversarial and non-stationary. Causal claims in this domain are unreliable. High effort, low trust. |
| PII detection or anonymization | Compliance problem. Ledgerhound assumes data arrives clean. |
| Overall model AUC as primary benchmark | Too indirect, too many confounders, takes months. Wrong metric for an EDA tool. |

---

## Benchmarks

### Tier 1 — Measurable Immediately (Before Any Model Training)

**Sample representativeness score:** PSI of current 1:1000 sample vs true population,
vs PSI of Ledgerhound-certified sample vs true population. Baseline is the current sample.
Ledgerhound wins if its certified sample has lower PSI on the top-K discriminative features.

**Hard negative inclusion rate:** What percentage of the agent-identified boundary region
(non-fraud closest to fraud cluster) is present in the current training sample vs
the Ledgerhound-certified sample. Current random sample: near zero. Target: 5–10x more.

**Contamination rate estimation accuracy:** Validate the survivor bias estimate against
accounts in the non-fraud set that were later confirmed as mule/fraud cases.

### Tier 2 — Measurable After Model Training (Surgical, Not Overall AUC)

Four targeted holdout sets instead of one:

| Test Set | What Improves |
|---|---|
| Hard negative holdout | Precision — fewer false positives on legitimate near-fraud transactions |
| Rare typology holdout | Recall on specific fraud clusters |
| Temporal holdout (last 3 months) | Drift robustness |
| Label-contaminated holdout | Model behavior on cleaner labels |

### Tier 3 — The Number That Goes to Leadership

**Drift detection lead time:** When does model performance visibly degrade in production
vs when does Ledgerhound first detect the underlying drift? If Ledgerhound detects drift
6 weeks before performance degrades, that 6 weeks is the value. Fraud victims not
caught in those 6 weeks is the cost of not having Ledgerhound.

### Tier 4 — The EDA Tool's Own Score (Retrospective Claim Validation)

Every EDA claim Ledgerhound produces is a testable hypothesis. Running the agent on
historical data and scoring its claims against what actually happened gives the tool
its own precision/recall metric — independent of the FE team's work.

---

## Architecture

Grouped by role rather than listed exhaustively — 43 packages, so an
alphabetical dump would obscure more than it shows.

```
ledgerhound/
│
│  ── SAMPLE CONSTRUCTION & CERTIFICATION (v4) ──────────────────────────
├── pipeline/           # Build the sample, then certify HOW it was built
│   ├── model_story.py      # the declarative spec everything else reads
│   ├── history_fetch.py    # entity-grain Trino SQL generation
│   ├── typology.py         # reconcile typologies -> scopes; portable map
│   ├── labeller.py         # per-typology rules -> TRI-STATE labels
│   ├── aggregate.py        # push-down or chunked; refuses non-composable
│   ├── prevalence.py       # measure, check, correct to population
│   ├── appropriateness.py  # is a handed-over sample what it claims?
│   ├── splits.py           # point-in-time + entity-disjoint + maturity
│   ├── cluster_psi.py      # PSI at the EFFECTIVE sample size
│   ├── pu.py               # "not-reported", + contamination sensitivity
│   └── certificate.py      # assembles it all -- and GATES on it
│
│  ── DISCIPLINE: what a claim is allowed to say ────────────────────────
├── discipline/         # sample-frame algebra + label provenance
├── sampling/           # non-fraud design, PSI, certificate, hard negatives
├── findings/           # append-only, structured, auditable findings
├── validation/         # claim validation
│
│  ── AGENTS & EXECUTION ────────────────────────────────────────────────
├── shell/              # conversational analyst (data_checks, verification)
│   └── prompts/        # SYSTEM_PROMPT etc. as editable .md files (code keeps a fallback copy)
├── adversarial/        # /adversarial: analyst-vs-adversary debate + judge verdict per fraud case
├── agents/             # ReAct loop, playbooks, tools, critic, orchestrator
├── auto/               # autonomous runner
├── plan/ graph/        # re-plannable DAG; transaction-graph motifs
├── skills/ tools/      # skill memory; tool registry
│
│  ── DATA & COMPUTE ────────────────────────────────────────────────────
├── trino/              # config, client (HITL), async job registry, SQL
├── compute/ backends/  # routing; pandas / cuDF backends
├── profiler/           # deterministic profiling (no LLM)
├── analysis/           # feature eval, recipes, typology, drift, selection
├── modeling/           # GBM + model registry
│
│  ── MEMORY & KNOWLEDGE ────────────────────────────────────────────────
├── memory/             # dataset memory, knowledge, session log
├── knowledge/ dictionary/ schema/   # accumulated insight; column semantics
│
│  ── GOVERNANCE & OUTPUT ───────────────────────────────────────────────
├── runtime/            # hash-chained event log, replay
├── reproducibility/    # manifests
├── monitor/            # drift scheduler, Trino source, auto-investigate
├── eval/ benchmarks/   # fluidity metrics, judge config, golden scoring
├── reporting/ narrative/ notebook/ output/  # reports, journals, charts
├── scorecard/ project/ workspace/ session/  # scorecards; project + artifacts
│
│  ── INTERFACES ────────────────────────────────────────────────────────
├── cli/                # `ledgerhound` CLI (~100 commands)
├── llm/                # LLM client (vLLM / OpenAI-compatible); model discovery/selection, ask-once API keys
└── integrations/       # Jira, git, MinIO
```

This repository ships the CLI only. There is no web front-end in it.

### Workspace on Disk

```
ledgerhound_workspace/
├── datasets/
│   ├── registry.json
│   └── upi_summary_v1/
│       ├── metadata.json        # source, row count, schema hash, pull date
│       ├── profile.json         # statistical fingerprint
│       ├── quality.json         # null rates, anomalies
│       └── raw/                 # immutable — never modified after registration
│
├── samples/
│   ├── registry.json
│   └── fraud_nonfr_v1/
│       ├── metadata.json        # parent datasets, sampling strategy
│       ├── certificate.json     # PSI scores, KS results, pass/fail per feature
│       ├── data/
│       └── validation/
│
├── reports/
│   ├── registry.json
│   └── 2026_05_fraud_eda/
│       ├── manifest.json        # links every artifact to its source
│       ├── findings/findings.jsonl
│       ├── charts/
│       ├── notebooks/
│       └── certificates/
│
├── trino_jobs/
│   ├── registry.json
│   └── job_a3f7/
│       ├── query.sql            # exact SQL shown to user and approved
│       ├── metadata.json        # submitted_at, estimated_rows, actual_runtime
│       └── result/
│
└── memory/
    ├── index.json               # searchable index across all dataset knowledge
    ├── datasets/
    │   ├── upi_summary.md       # living document — grows with every session
    │   └── fraud_labels.md
    ├── knowledge/
    │   ├── feature_signal_history.json   # signal decay timeline
    │   ├── drift_events.json             # every drift detection, ever
    │   └── typology_evolution.json       # fraud cluster changes over time
    └── sessions/
        ├── sessions.jsonl
        └── [session_id].json
```

---

## Hard Boundaries — What Does Not Change

1. **Trino queries require explicit HITL approval** — no query ever executes automatically.
   The SQL is shown in full. The estimated row count and runtime are shown. The user approves.

2. **Original data is immutable** — registered datasets are never modified.
   All transformations produce a new named version. The `df_original` is always there.

3. **Every finding is a falsifiable hypothesis** — not an observation. Every claim the agent
   makes has a proposed validation method and a confidence interval. Findings with
   insufficient sample size are flagged as inconclusive rather than reported.

4. **Architecture is not biased toward fraud** — mule detection is a first-class track.
   Every data model, sampling design, and compute tier decision is built to support both.

5. **Trino pull is always statistics-first** — population profiling (aggregates only)
   runs before any row-level sample pull. No large query is submitted without first
   understanding what it will return.

---

## Compute Requirements

| Workload | Hardware | Notes |
|---|---|---|
| Fraud data EDA, local profiling | H100 GPU, cuDF | Full EDA on confirmed fraud sample |
| Summary table EDA (post-pull) | H100 GPU or CPU | After targeted Trino pull |
| Population profiling queries | TrinoSQL / KubeFlow | Aggregation only — returns stats not rows |
| Non-fraud sample pulls | TrinoSQL / KubeFlow | Async, hours, checkpoint/resume |

---

## How to run — the short version

Full checklist with explanations: [`docs/USER_GUIDE.md` §0](docs/USER_GUIDE.md#0-how-to-run----start-to-finish).

```bash
# 1. install (on a pod: boto3 for /s3, exact tokenizers, the analysis kernel)
pip install -e ".[s3,tokenizers,test]"

# 2. configure -- gitignored; fill llm.gateway and minio (access_key + secret_key)
cp main_config.example.yaml main_config.yaml

# 3. check the machine
ledgerhound doctor
cat /sys/fs/cgroup/memory.max && df -h /dev/shm      # memory limit + RAM disk for /s3 data

# 4. start (any folder; the project folder is created if absent)
ledgerhound shell --project ~/work/july-mules --use-kernel
```

Inside the shell:

```
you > /minio                                         # checks both object-store keys (or /minio --key if not in the file)
you > /s3 --path s3://sentinel-data/data/july/          # data -> RAM, loaded (or: /local --path /data/july.parquet)
you > find the top mule signals at account level     # then just talk
you > /paths                                          # where the data came from, where outputs are going
you > exit
```

Everything is saved in ONE place, one level above the launch folder:
`../Agentic_eda_results/<username>/` with `run_outputs/`, `data_cache/`
and `library/` -- nowhere else by default. First start asks your
`work` email, then the LLM key if `main_config.yaml` has none.

| After the run | Where |
|---|---|
| Reports, findings, audit log, `paths.json` | `Agentic_eda_results/<you>/run_outputs/<run>/` |
| Feature tables (local data) | `Agentic_eda_results/<you>/run_outputs/<run>/features/` |
| Feature tables (`/s3` data) | `s3://sentinel-data/outputs/<you>/<run>/features/` (uploaded after every step) |
| What was learned about each dataset | `Agentic_eda_results/<you>/library/`, and shared with the team in `s3://sentinel-data/kb/v1/` |
| `/s3` data copies, tokenizers | `Agentic_eda_results/<you>/data_cache/` |

After profiling, the agent **asks** what it must not guess -- which column
is the label and what its classes mean, the entity, the event time, units,
code meanings -- before any analysis. Answers, findings and dataset cards
are shared with the whole team automatically (nothing to run, nothing ever
overwritten), so each question is asked once per dataset Customer identifiers (VPAs, accounts,
mobiles, PAN) are removed before anything is shared.

First run on a new pod: `python scripts/prod_checks.py` checks the gateway,
object store, tokenizer, GPU and one `/adversarial` case end to end.

## Install

```bash
git clone https://github.com/irfan-muhammed/ledgerhound && cd ledgerhound
pip install -e .                 # core (incl. the v4 pipeline package)
pip install -e ".[s3]"           # + boto3: /s3 data from the object store
pip install -e ".[tokenizers]"   # + exact token counting
pip install -e ".[test]"         # + the persistent analysis kernel (jupyter_client, ipykernel)
pip install -e ".[wheelhouse]"   # + modelling: xgboost, lightgbm, optuna, feature selection
pip install -e ".[sql]"          # + sqlalchemy
pip install --extra-index-url=https://pypi.nvidia.com ".[gpu]"       # + cuDF (CUDA 13 only -- the only toolkit this project supports)
pip install -e ".[dev,test,wheelhouse]"   # + pytest, ruff
```

Two notes worth having before an image build:

- `pyproject.toml` declares `typer[all]`, and typer ≥0.20 no longer provides
  that extra. It is a **warning, not an error** — `rich` is declared
  separately and the extra only added shell-completion helpers — but it will
  appear in every build log.
- The `test` extra installs the persistent analysis kernel
  (`jupyter_client`, `ipykernel`); without it, kernel mode is unavailable.

### Configuration — `main_config.yaml`, not environment variables

**Everything the project reads from the environment lives in one
gitignored file, not a dozen exported variables.** Copy the template and
fill in what you need:

```bash
cp main_config.example.yaml main_config.yaml   # gitignored, never pushed
```

Covers LLM/judge access (`llm.method: self|gateway|api`, then that
method's `base_url`/`model`/`api_key` — `gateway` is the org's internal
gateway, one shape for any model behind it; the judge is independent and
can run a different method/model, or be left blank to self-judge), real
institutional knowledge kept **outside** this repo
(`knowledge.profiles_dir` / `knowledge.knowledge_dir`), GPU tuning, Trino,
identity, and integrations. Nothing to `export` before every
session — `ledgerhound shell` reads the file directly.

Every setting still has an environment-variable fallback (named in
`main_config.example.yaml`) for production paths that inject env vars
directly — e.g. Kubeflow's PodDefault for `LEDGERHOUND_USER` — but
`main_config.yaml` wins when both are present.

### Team object store (MinIO) — `/minio --key`

Shared files (tokenizers today; the team knowledge base next) live in
the org's S3-compatible store. `main_config.yaml` says where:

```yaml
minio:
  endpoint: "objectstore.example.com:9000"
  bucket: "sentinel-data"
  access_key: "your-access-key-id"   # the access key you were issued
  secret_key: "<its secret key>"      # main_config.yaml is gitignored
  verify_ssl: false
```

Both keys can live in `main_config.yaml` (gitignored, never pushed) --
the simplest setup. Or leave `secret_key` blank and set it inside the
shell:

```
ledgerhound shell
you > /minio --key          # hidden prompt; checked against the store before it is saved
you > /minio                # status + connection test
you > /minio --clear        # forget saved keys
```

Load data from it, or from disk, with the source stated explicitly:

```
you > /s3 --path s3://sentinel-data/data/july/     # copied once into RAM (/dev/shm), then loaded
you > /local --path /data/upi/july.parquet
you > /paths                                     # where data came from, where outputs go
```

With `/s3` data, feature tables are uploaded back to
`s3://<bucket>/outputs/<user>/<run>/features/` after every step (boto3 if
installed: `pip install ".[s3]"`). See `docs/USER_GUIDE.md`, "Loading
data".

The secret is saved to `~/.ledgerhound/credentials.json` (owner-only), never sent to
the model and never logged. A secret the store rejects is not saved.
`MINIO_SECRET_KEY` in the environment overrides it for pods.
`ledgerhound doctor` checks the connection; `python scripts/prod_checks.py`
checks the whole deployment. Details: `docs/USER_GUIDE.md`, "The team
object store".

### Trino Configuration

Fill in the `trino:` block in `main_config.yaml` (never hardcode
credentials):

```yaml
trino:
  host: your-trino-host
  port: 8080
  user: your-user
  password: your-password
  catalog: hive
  schema: upi_fraud
  use: true
```

Environment variables (`LEDGERHOUND_TRINO_HOST`, `_PORT`, `_USER`,
`_PASSWORD`, `_CATALOG`, `_SCHEMA`, `_HTTP_SCHEME`, `_USE`) still work as a
fallback for each field individually if `main_config.yaml` doesn't set it.

To run in local-only mode (no Trino):

```bash
ledgerhound eda --no-trino "What are the top fraud features?"
```

---

## Quick Start

The primary interface is the **fluid analyst shell** (`ledgerhound shell`) — one
command, no fixed pipeline. A short grounding on-ramp (goal -> data ->
output-format), then the analyst reasons over its tools to reach your goal,
human-in-the-loop on the big calls. It also handles the chores — opt-in **git**
and **Jira** tools (push your work, pull/post tasks) that switch on only when
configured. **Full manual: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).**

```bash
pip install -e ".[s3,tokenizers,test]"
cp main_config.example.yaml main_config.yaml   # fill llm.gateway + minio; the LLM key can
                                               # stay blank -- the shell asks once

ledgerhound shell --project ./run1 --use-kernel
#   (first run on a machine: mandatory work login, asked once)
#   model discovery/selection happens automatically (best available picked,
#   or a pick-list if you're at a terminal) -- --model <id> to pin one
#   /minio --key                              # once: object-store secret
#   /s3 --path s3://sentinel-data/data/july/     # or /local --path /path/to/data.parquet
#   goal:  find fraud signals in this data    # real scale -> auto cuDF/GPU
#   /paths                                    # where data came from, where outputs go
#   /model <id>                               # switch model live, no restart
#   /model-judge                              # pick the eval/adversarial judge model
#   /adversarial --cases 25                   # adversarial debate over fraud cases
#   <ask anything>  ->  /eval  ->  /book save  ->  /exit
```

### Try it without an LLM

```bash
python -c "from pathlib import Path; from ledgerhound.data.synthetic import UpiSyntheticGenerator as G; G(n_accounts=5000).generate().save(Path('sample_run'))"
ledgerhound profile --data sample_run/upi_summary.parquet --target is_fraud --output sample_run/digest.yaml
```

Artifacts land per project under `.ledgerhound/` (books, sessions, findings,
journals). Skills inject institutional method; books make understanding compound
across sessions.

### Library / scripted use

```python
from ledgerhound.workspace.layout import WorkspaceLayout
from ledgerhound.trino.config import TrinoConfig
from ledgerhound.trino.client import TrinoClient
from ledgerhound.memory.knowledge import DatasetKnowledge

# Initialize workspace
ws = WorkspaceLayout.from_cwd()
ws.initialize()

# Load dataset knowledge (empty on first run, accumulates over sessions)
fraud_knowledge = DatasetKnowledge.load(ws, "fraud_labels")
print(fraud_knowledge.for_session_context())

# Trino client (use_trino=False for local-only dev)
cfg = TrinoConfig.from_env()
client = TrinoClient(cfg, ws)

# Population profiling — returns statistics, not rows
stats = client.execute_stats_only("""
    SELECT
        amount_bucket,
        COUNT(*) as n,
        AVG(txn_amount) as avg_amount,
        STDDEV(txn_amount) as std_amount
    FROM upi_fraud.transactions
    WHERE txn_date >= DATE '2025-01-01'
    GROUP BY amount_bucket
""")
```

---

## Telemetry & Observability — what is written, and where

Ledgerhound writes two kinds of record. They serve different purposes and live in
different places; **neither is sent anywhere.** Everything below is a local
file on the machine (or the mounted volume) that ran the session.

There is no analytics SDK, no phone-home and no third-party collector in the
package — verified by grep, not by assertion. The only outbound calls Ledgerhound
ever makes are ones you configure yourself: the LLM endpoint, and the opt-in
integrations (a drift-alert webhook, Jira, git). None of them carry the logs
described here.

### 1. Deployment-wide session-quality log

| | |
|---|---|
| **Path** | `~/.ledgerhound/sessions.jsonl` |
| **Override** | `observability.sessions_log` in `main_config.yaml`, or `$LEDGERHOUND_SESSIONS_LOG` |
| **Grain** | One compact line per **turn**, from every user, every project |
| **Contents** | Counts only — tool calls, errors, refusals, HITL pauses. **Not** their bodies, not code, not data |
| **Size control** | Rotating, 20 MB x 10 backups. The one-line-per-turn grain is the real control; rotation is the safety net |

This is the file to point at shared storage when Ledgerhound runs as a service
for other people, so every pod and user appends to one place:

```yaml
# main_config.yaml
observability:
  sessions_log: /mnt/shared/ledgerhound/sessions.jsonl
```

Because a line lands on **every turn** rather than at clean exit, a crashed or
killed session still leaves its last-known scorecard behind. Take the **last**
line for a given `run_id` as that session's current state; earlier lines for
the same `run_id` are its history turn over turn.

Read it with:

```sh
ledgerhound sessions collect       # gather records
ledgerhound sessions report        # summarise across sessions
```

### 2. Per-run forensic record

Written under `<project>/run_outputs/<YYYY-MM-DD_HH-MM-SS>/`. Full fidelity,
unbounded, never rotated — this is the audit trail.

| File | What it is |
|---|---|
| `events.jsonl` | **The canonical record.** Typed, append-only, and **hash-chained** — each event carries the previous event's hash, so tampering with any stored line is detectable and `EventStream.verify()` reports the exact `seq` where the chain breaks |
| `findings.jsonl` | Recorded findings — a projection of `events.jsonl` |
| `session_code.py` | Every executed cell — also a projection of `events.jsonl` |
| `transcript.jsonl` | Flat per-line log, written in parallel for backward compatibility |
| `search_effort.json` | Per-chapter test families: attempts, refusals, cells run, declared p-values (see the user guide, SS14) |
| `plan.json`, `methodology.json`, `session_goal.json` | Declared plan, phase sequence, and goal |
| `digest.json` | Dataset profile digest |
| `results/` | Full text of any tool output too large to keep inline |
| `scratch/`, `features/` | Work products — plots, exports, feature tables |

`findings.jsonl` and `session_code.py` are generated from `events.jsonl` by the
same projection functions used live and in bulk, so the live files and a
regenerated set cannot drift. Regenerate with `ledgerhound run-views <run_dir>`.

### 3. Project state

`<project>/.ledgerhound/` holds durable state rather than logs — `project.json`,
`dataset_cards/` (one card per dataset, content-hash keyed), `books/` (the
Librarian), `findings/`, and `sessions/`.

### What is NOT recorded

The session-quality log stores **counts, never content** — no code, no query
results, no column values, no claim text. Row-level data never leaves the
machine at any point: `events.jsonl` stores code and output previews with a
hash of the full text, and the profiler reports statistics rather than rows.

---

## The One-Line Promise

Ledgerhound gives the fraud modeling team **statistically audited, falsifiable, institutionally
remembered evidence** — the kind that justifies changing a sampling ratio, retiring a dying
feature, or telling leadership exactly how many weeks of early warning the system provides.
