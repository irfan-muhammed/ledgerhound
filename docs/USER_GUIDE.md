# Ledgerhound -- User Guide

**Version 0.7.9** (2026-10-05)

A practical, task-oriented manual for the `ledgerhound shell` -- the
Claude-Code-style conversational analyst.

> **New in 0.7.0 -- the inference gates.** Earlier versions gated what gets
> *computed* (grain, modelling authorisation, chapter redo). 0.7.0 gates what
> gets *concluded*: a finding now has to survive the statistics before it is
> recorded as fact. See [SS14](#14-what-070-added--the-inference-gates).

---

## 0. How to run -- start to finish

The whole path on an internal Kubeflow pod, in order. Each step is explained in
more detail further down; this is the checklist.

**1. Get the code.** Clone the repository (`main` branch) and open a
terminal in it.

**2. Install.**

```bash
pip install -e ".[s3,tokenizers,test]"
#   s3         -> boto3, for loading data from the object store (/s3)
#   tokenizers -> exact token counting
#   test       -> jupyter_client + ipykernel: the persistent analysis kernel
pip install -e ".[wheelhouse]"                     # optional: modelling / feature selection
pip install --extra-index-url=https://pypi.nvidia.com ".[gpu]"       # optional: GPU (CUDA 13 only)
```

On a machine without public PyPI, set up the internal mirror first
(section 1, step 2).

**3. Write `main_config.yaml`** (gitignored -- your keys never reach git):

```bash
cp main_config.example.yaml main_config.yaml
```

The parts that matter for a first run:

```yaml
llm:
  method: gateway
  gateway:
    base_url: "https://llm-gateway.example.com/v1"
    model: ""            # blank = pick from what the gateway serves
    api_key: ""          # blank = the shell asks once and remembers it
minio:
  endpoint: "objectstore.example.com:9000"
  bucket: "sentinel-data"
  access_key: "your-access-key-id"   # the access key you were issued
  secret_key: "<its secret key>"      # this file is gitignored -- it never reaches git
  verify_ssl: false
```

Both object-store keys can go here. `main_config.yaml` is gitignored, so
nothing in it is ever pushed. (If you would rather not keep the secret in
a file, leave it blank and use `/minio --key` in step 6.) The shell finds
`main_config.yaml` in the folder you launch from, the repo, or
`$LEDGERHOUND_MAIN_CONFIG` -- and you can launch from any folder.

**4. Check the environment.**

```bash
ledgerhound doctor                      # your login, results folder, config, keys, object store, tokenizer
cat /sys/fs/cgroup/memory.max        # the pod's real memory limit ("max" = none)
df -h /dev/shm                       # the RAM disk /s3 copies data into
```

`/dev/shm` must be bigger than the data you will load (Parquet size, e.g.
5-15 GB for 10 crore transactions) and the pod's memory limit must cover
the data *in memory* as well (roughly 3-10x the Parquet size in pandas).

**5. Start the shell** on a project folder (created if absent):

```bash
ledgerhound shell --project ~/work/july-mules --use-kernel
```

- First run on a machine: it asks for your `work` email once.
- Right after that, if `llm.gateway.api_key` is blank in `main_config.yaml`,
  it asks for the LLM API key (hidden input) and saves it -- asked once.
- It prints where everything will be saved:
  `results: ../Agentic_eda_results/<you>  (run_outputs, data_cache, library)`.
- `--use-kernel` runs analysis in a persistent Python kernel (needed for
  GPU, recommended for big data). It is automatic when a GPU is present.
- It picks the best model the gateway serves; `--model <id>` pins one.
- On the first data load it asks which methodology to follow
  (`--methodology` sets it up front).

**6. Object-store secret -- only if it is NOT in `main_config.yaml`.**

```
you > /minio --key          # hidden prompt, checked against the store, then saved
you > /minio                # status + connection test
```

**7. Load the data.**

```
you > /s3 --path s3://sentinel-data/data/july/        # from the object store
you > /local --path /data/upi/july.parquet         # or from this machine
```

`/s3` copies the data once, in parallel, into RAM (`/dev/shm`), then
loads it. A folder of Parquet parts loads as one dataset (`--as-one` for a
folder of plain files that really is one dataset).

**8. Answer the data questions.** After profiling, the agent asks what it
must not guess: which column is the label and what its classes mean, the
entity, the event time, units, what code columns mean. Answers are saved
and shared with the team, so they are asked once per dataset.

**9. Work.** Just say what you want: "find the top mule signals at account
level", "build account features for July". The agent analyses, records
findings and saves feature tables under `features_dir`. Findings, answers
and the dataset's card are shared with the team automatically.

**10. See where everything is.**

```
you > /paths
data source  : s3  s3://sentinel-data/data/july/
local copy   : ../Agentic_eda_results/<you>/data_cache/sentinel-data/data/july
run folder   : ../Agentic_eda_results/<you>/run_outputs/2026-09-21_10-02-21
features     : ../Agentic_eda_results/<you>/run_outputs/2026-09-21_10-02-21/features
  uploaded to: s3://sentinel-data/outputs/<you>/2026-09-21_10-02-21/features/
scratch      : ../Agentic_eda_results/<you>/run_outputs/2026-09-21_10-02-21/scratch
```

Ask for a report ("save a report") and it is written into the run folder.
`exit` leaves; everything is already on disk and, for `/s3` data, the
feature tables are already in the bucket.

**Where everything is saved -- one folder, strictly.** By default
Ledgerhound writes only here, one level above the folder you launch from:

```
../Agentic_eda_results/<username>/
    run_outputs/    one dated folder per session: reports, findings.jsonl,
                    events.jsonl (audit trail), transcript.jsonl, paths.json,
                    feature_catalog.*, features/ (feature tables), scratch/
    data_cache/     local copies of /s3 data, downloaded tokenizers
    library/        the project: dataset cards, books, settings, knowledge
```

All three folders are created at startup. Nothing is written in the launch
folder or next to your data. The only other places are:

| Where | What |
|---|---|
| `s3://sentinel-data/outputs/<you>/<run>/features/` (`/s3` data) | feature tables, uploaded after every step |
| `s3://sentinel-data/kb/v1/` | the team knowledge base: every card, answer, finding and book, shared |
| `~/.ledgerhound/kb_outbox/` | team-knowledge updates not uploaded yet (retried automatically) |
| `~/.ledgerhound/` | machine-level state: saved keys (owner-only), your login, the team-knowledge upload queue |

Any of the three folders can be moved in `main_config.yaml`
(`paths.results_root`, `paths.run_outputs_dir`, `paths.data_cache_dir`,
`paths.workspace_dir`), or per session with `--project` / `--out`. On a
pod with a small PVC and lots of RAM, `paths.data_cache_dir:
/dev/shm/sentinel-data` holds `/s3` data in RAM; that session's feature
tables are then written next to it and uploaded after every step.

**Before a first real run on a new pod**, run the deployment checks from
the repo -- gateway, object store, team knowledge base, tokenizer, GPU and
one `/adversarial` case:

```bash
python scripts/prod_checks.py
```

---

## 1. Install

**Prerequisites:** Python 3.10+ and `pip`. Nothing else — no conda, no admin
rights needed.

**Step 1 — get the code.** Either clone the repo or unzip the ship artifact,
then open a terminal in that folder.

**Step 2 — (internal servers only) point pip at the internal mirror.** On a
machine that can't reach public PyPI, copy the template and fill in the real
URL + access token:

```bash
cp configs/pip_index.conf.example configs/pip_index.conf   # then edit it
export PIP_CONFIG_FILE=configs/pip_index.conf              # PowerShell: $env:PIP_CONFIG_FILE="configs/pip_index.conf"
```

The same config works unchanged on a laptop with internet: if the internal
host doesn't resolve, pip falls back to public PyPI automatically. Skip this
step entirely on a normal internet-connected machine.

**Step 3 — install.**

```bash
pip install -e .                        # core: shell, profiler, CLI
pip install -e ".[s3]"                  # + boto3: load data from the object store (/s3)
pip install -e ".[tokenizers]"          # + exact token counting
pip install -e ".[test]"                # + the persistent analysis kernel (jupyter_client, ipykernel)
pip install -e ".[wheelhouse]"          # + modeling: xgboost, lightgbm, optuna, feature selection
pip install -e ".[dev,test]"            # + dev: pytest, ruff, notebook-execution deps
```

Core alone is enough to use the shell. On a pod, install
`.[s3,tokenizers,test]` (section 0). Add `wheelhouse` for the
modeling/feature-selection stack.

**Step 4 — configure and check the environment.** Copy
`main_config.example.yaml` to `main_config.yaml` and fill in the `llm:`
block (section 0, step 3 shows the minimum), then:

```bash
ledgerhound doctor                             # verifies version, config, keys, object store, tokenizer
```

The LLM key can be left blank: `ledgerhound shell` prompts for it the first
time it's actually needed and remembers it (see "API keys, asked once"
below). A literal key in `main_config.yaml` or an env var
(`LEDGERHOUND_GATEWAY_API_KEY` for the organization gateway) takes priority over that
prompt.

**Step 5 — first launch.** The very first time `ledgerhound shell` runs on a
machine, it asks for your `work` email (mandatory, validated) and
derives your username from it -- stored locally, never asked again. Then
point the shell at a folder for your project (it creates `.ledgerhound/` there):

```bash
ledgerhound shell --project ./my-analysis
```

It can be launched from any folder: the LLM config named by `llm.method`
(`configs/gateway.yaml` for the organization gateway) is found in the Ledgerhound
checkout even when you are elsewhere. `--llm-config <yaml>` picks a
different one.

When the shell opens it prints a banner showing the model, project, run
folder, and which **integrations** are switched on (see section 5).

**Model selection.** On startup, Ledgerhound queries the configured endpoint for
what it actually serves (`GET /models`) and picks the best one automatically
(thinking/pro variants preferred over flash/mini/distilled) -- at an
interactive terminal with more than one model available, it shows a pick-list
(Enter = recommended); in an unattended/`--autonomous` run it just picks and
prints which one, never blocking. Pass `--model <id>` to skip discovery and
pin one directly. Mid-session, `/model` re-lists and switches live (no
restart); `/model <id>` sets one directly; `/model configs/<other>.yaml`
switches provider (restart required). The model's context window is also
auto-sized: the gateway's own number wins when its `/models` listing reports
one (`max_model_len`, as vLLM-style servers do), then a small known-models
table (Kimi-K2/K2.5, DeepSeek, GLM, Qwen3/3.6, GPT-4o). An unknown model
keeps the default and the startup line says `ASSUMED` so you know to check.
A `/model <id>` switch mid-session re-sizes it too.

**Token counting.** The shell compacts its history before the model's
window fills, so it needs to know how many tokens it is sending. It learns
the real number from the gateway: every reply reports the prompt's exact
token count, and the shell calibrates from that after the first call. With
a tokenizer available, counts are exact from the very first request.
Tokenizers come from the team's object store (the `minio:` block in
`main_config.yaml`, with the secret set by `/minio --key` -- see "The team
object store" below) or a mounted folder (`tokenizers.local_dir`). Install
the readers with `pip install ".[tokenizers]"`. At startup you see one line:

```
tokenizer: exact token counts for qwen-3.6-35B-A3B (using the Qwen3.6-35B-A3B tokenizer, from object_store)
```

or, if anything is missing, one yellow block naming the cause, the fix, and
what the session uses meanwhile, for example:

```
tokenizer NOT available for model 'qwen-3.6-35B-A3B' [auth_failed]: the object store is configured (...) but its secret key is not set, and nothing is cached from an earlier session.
  fix: run /minio --key in the shell (or export MINIO_SECRET_KEY), and set minio.access_key in main_config.yaml or with /minio --access-key.
  meanwhile: token counts are estimated until the gateway's first reply, then calibrated from the usage it reports -- the session is not affected beyond that.
```

The codes are `not_configured`, `store_unreachable`, `auth_failed`,
`manifest_missing`, `model_not_found`, `download_failed`,
`checksum_mismatch`, `library_missing` and `load_failed`. The session never
stops over any of them. If a loaded tokenizer disagrees with the gateway's
own count twice in a row, the shell prints `[count_mismatch]`, drops that
tokenizer, and keeps calibrating from the gateway.

**The team object store (`/minio`).** Ledgerhound keeps shared files -- the
tokenizers today, the team knowledge base next -- in the org's S3-compatible
store. Where it is lives in `main_config.yaml`:

```yaml
minio:
  endpoint: "objectstore.example.com:9000"
  bucket: "sentinel-data"
  access_key: "your-access-key-id"   # the access key you were issued
  secret_key: "<its secret key>"      # main_config.yaml is gitignored -- never pushed
  verify_ssl: false                   # internal certificate
```

The store needs **both** keys. The simplest setup is to put both in
`main_config.yaml`, which is gitignored and never reaches git. A value in
the file wins over one saved from the shell. To keep the secret out of
files, leave `secret_key` blank and set it from inside the shell:

| Command | Does |
|---|---|
| `/minio --key` | asks for the secret with hidden input, **checks it against the store**, then saves it |
| `/minio --key <secret>` | same, inline (the value stays visible in your terminal scrollback -- the prompt form is safer) |
| `/minio --access-key <key>` | saves the access key, if it isn't in `main_config.yaml` |
| `/minio` | shows endpoint, bucket, whether each key is set and where from, and tests the connection |
| `/minio --clear` | forgets the keys `/minio` saved |

```
you > /minio --key
MinIO secret key (input hidden):
secret verified -- objectstore.example.com:9000 reachable, bucket 'sentinel-data' visible with these credentials. Saved to ~/.ledgerhound/credentials.json (owner-only); it is not shown or logged anywhere.
```

How it behaves:
- A secret the store **rejects is not saved** -- a typo can't replace a
  working key. If the store can't be reached (for example off the org's
  network), the key is saved and the reply says it is **unverified**.
- Keys are stored in `~/.ledgerhound/credentials.json`, readable only by you.
  `MINIO_SECRET_KEY` / `MINIO_ACCESS_KEY` in the environment take priority
  (pod secrets).
- A new key takes effect immediately: the tokenizer is re-loaded in the
  running session.
- `/minio` commands never reach the model or the session logs. A mistyped
  command that carries a secret -- `/mino --key abc` -- is **not** sent to
  the model either; you get "unknown command ... NOT sent to the model".
- `/config` shows the store's endpoint and whether each key is set;
  `ledgerhound doctor` checks the connection and the tokenizer.

**Loading data: `/local` and `/s3`.** Say where the data is, and where it
comes from decides where this run's outputs go.

| Command | Does |
|---|---|
| `/local --path /data/upi/july.parquet` | loads a file on this machine |
| `/local --path /data/upi/july/` | loads a folder of Parquet parts as one dataset |
| `/s3 --path s3://sentinel-data/data/july/` | downloads from the object store into RAM, then loads it |
| `/s3 --path data/july/txn.parquet` | same, in the default bucket (`minio.bucket`) |
| `... --as-one` | read a folder of plain Parquet files as ONE dataset (see below) |
| `... --name trino:hive.payments.txn_sample` | give the data its shared name -- e.g. the Trino table an export came from (see "Team knowledge") |
| `/paths` | where the data came from, the local copy, where outputs go, the dataset's shared name, team-knowledge status |

You can still just tell the agent "load s3://sentinel-data/data/july/" -- it
uses the same loader.

How `/s3` works:
- The data is copied **once**, in parallel, to
  `Agentic_eda_results/<you>/data_cache/`, then loaded into memory; later
  passes run at memory speed. A second load of unchanged data reuses the
  copy ("already cached"). The load stops with a clear message if there is
  not enough space. On a pod with a small PVC, set `paths.data_cache_dir:
  /dev/shm/sentinel-data` to keep the copy in RAM instead.
- boto3 is used when installed (`pip install ".[s3]"`), else the `minio`
  package. Keys are the `/minio` ones.
- A folder is read as one dataset only when it clearly is one: `col=value`
  partition folders, a `_SUCCESS` file, or `part-*` files. A folder of
  `july.parquet`, `aug.parquet` is refused with the list of files -- point
  at one, or add `--as-one` if they really are one dataset split up.

**Where outputs go.**

| Data from | Feature tables | Reports, findings, logs |
|---|---|---|
| `/local` | `run_outputs/<run>/features/` | `run_outputs/<run>/` |
| `/s3` | `run_outputs/<run>/features/`, and **uploaded after every step** to `s3://<bucket>/outputs/<user>/<run>/features/` (with `data_cache_dir` on `/dev/shm`: next to the RAM copy) | `run_outputs/<run>/` |

`run_outputs/<run>/paths.json` records all of it; `/paths` prints it. The
agent is told to save every feature table under `features_dir`, and saving
is forgiving: `features/x.parquet`, subfolders, pyarrow and plain file
writes all land there, and a write aimed at the source-data folder is
redirected into `features_dir` with a `WRITE REDIRECTED` line. Two files
with the same name from different folders never overwrite each other.

**Questions about the data -- asked, never assumed.** Once the data is
profiled, Ledgerhound lists what it must not guess and the agent asks you
before any analysis:

```
QUESTIONS TO ASK THE USER NOW -- before any analysis. Do NOT assume ...
  1. [label] Several label-like columns: fraud_flag, is_mule. Which one is the target ...?
  2. [label_classes:fraud_flag] `fraud_flag` has 4 classes: 0 (2,834), 2 (63), 1 (61), 3 (42).
     What does each class mean, and which of them count as fraud?
  3. [entity] Several identifier columns: payer_vpa, payee_vpa. Which entity is this about ...?
  4. [event_time] Several time columns: txn_ts, settle_dt. Which one is the event time ...?
  5. [units:txn_amt] Is `txn_amt` in rupees or paise?
  6. [codes:resp_cd] What do the values of `resp_cd` mean (5 classes: Z9, XY, B3, 00, U30)?
```

Your answers are saved on the dataset's card (`record_data_answer`) and
shared with the team: next time -- yours or a colleague's -- they appear as
`KNOWN ABOUT THIS DATA` and are not asked again. If the agent starts
computing while questions are still open, the step's output reminds it.
In an unattended `--autonomous` run nobody can answer, so its choices are
kept for that session only, marked unconfirmed, and never written to the
shared card.

**Team knowledge -- shared automatically.** Everything learned about a
dataset is shared with the whole team through the object store, with no
command to run:

| Shared | When |
|---|---|
| the dataset card: profile, stories/chapters, decisions, quirks, answers to data questions | pulled when the data is loaded; pushed on every change |
| findings (claim, evidence, sample frame) | pushed when recorded; a teammate's finding is shown as already known |
| books (`save_understanding`) | pushed when saved; recalled on similar data |

Never shared: raw data, run logs, transcripts, anything typed in chat, keys
-- and no customer identifiers. Before anything leaves the machine,
identifier-like columns (VPA, account, mobile, card, customer, device ...)
lose their example values, and any VPA, e-mail, mobile number,
account/card/Aadhaar-length number or PAN anywhere in the text becomes
`[redacted]`. Business codes and label classes (`U30`, `P2M`, `5411`,
`fraud_flag` 0-3) are kept -- that is what the team needs. Your local card
keeps everything.

A dataset has one name for everyone:

| Data | Shared name |
|---|---|
| `/s3 --path s3://sentinel-data/data/july/` | `s3://sentinel-data/data/july` |
| an export from Trino | the table: `--name trino:hive.payments.txn_sample` |
| any other local file | its content hash -- the same bytes are the same data on any machine |

When the data under a name changes (a new month under the same prefix),
the card opens a new *story* for it automatically; a teammate's profile is
reused only if it was computed on exactly the same content.

Nothing is ever lost: every save writes a NEW object in
`s3://<bucket>/kb/v1/` (nothing is overwritten, so two people saving at
once both land), and every save first goes to `~/.ledgerhound/kb_outbox/` on
disk and leaves it only once the store has it -- a network outage, a
missing secret or a crash just delays the upload to the next save or
start. `/paths` shows the status (`N update(s) waiting to upload`). Turn
it off for a sandbox with `minio.team_kb: false`.

**The organization gateway is `gateway`.** `llm.method` (and `judge.method`) is one
of `self`, `gateway` or `api`, with a block of the same name holding
`base_url`, `model` and `api_key`; the key env var is
`LEDGERHOUND_GATEWAY_API_KEY`. Any other method name stops startup with a
message saying exactly that. A project that remembers an LLM config file
which no longer exists falls back to the one `llm.method` names
(`configs/gateway.yaml`) and saves it.

**API keys, asked once.** If a `main_config.yaml`-based config genuinely has
no key configured anywhere, `ledgerhound shell` prompts for one the first time
it's needed and stores it at `~/.ledgerhound/credentials.json` -- never asked
again after that. If the gateway later rejects a request as an auth failure
(expired/revoked key), Ledgerhound identifies that specifically (not a generic
error), clears the stored key, and prompts once more -- your next message
just works with the new key, no restart needed.

---

## 2. The core loop -- data science by conversation

Just talk to it. The model drives, calls tools, and grounds every number.

```
you > I've got fraud data in ../fraud_july.parquet, find the top signals
```

(Or load it first with `/local --path` / `/s3 --path` -- see "Loading data"
in section 1 -- and then say what you want.)

It will resolve the path, load the data, profile it, run analysis in a
persistent Python sandbox, and record evidence-backed findings. Nothing is
railroaded -- "EDA on fraud data" gives you real exploratory analysis, not a
forced template.

Key behaviours:

- **Findings are grounded.** Every quantitative claim is logged with an honest
  sample-frame tag (`within_stratum`, `cross_stratum`, `population_estimate`).
  A rate from a filtered subset is never reported as a population rate.
- **The sandbox persists.** Variables you (or the model) define in one step
  survive into the next, like notebook cells. `df`, `pd`, `np` are in scope.
- **Understanding compounds.** Ask it to save what you learned and it writes a
  "book"; next time you load similar data, prior understanding is recalled.
- **Reports** land in a dated run folder (`run_outputs/<timestamp>/`).
- **Feature tables** go under `features_dir` -- the run folder for local
  data, uploaded to the object store for `/s3` data. `/paths` shows both.

---

## 3. Planning on big tasks

For multi-step work the analyst lays out a visible checklist and ticks it off
as it goes:

```
you > /plan compare fraud vs non-fraud on the top 10 features and report
```

You will see:

```
plan:
[x] 1. profile both datasets
[>] 2. compute PSI + KS per feature
[ ] 3. write the comparison report
```

You never have to use `/plan` explicitly -- the model proposes a plan itself
whenever a task has several steps.

---

## 4. Slash commands

Slash commands are thin -- most things you just ask for in plain language.

| Command | Does |
|---|---|
| `/help` | list commands |
| `/config` | show the integration config (secret values masked) |
| `/model` | re-list + pick a model on the CURRENT endpoint, applies immediately |
| `/model <id>` | set a model id directly on the current endpoint, applies immediately |
| `/model configs/x.yaml` | switch the LLM config/provider (restart to apply) |
| `/model-judge` | same as `/model`, but for the eval/adversarial judge model |
| `/adversarial --cases 25` | deep-research debate over N random fraud cases -- see section 8b |
| `/local --path <file or folder>` | load local data |
| `/s3 --path s3://bucket/key` | load object-store data into RAM; features upload back |
| `/paths` | where this run's data came from and where outputs go |
| `/minio` | show the team object store and test the connection |
| `/minio --key` | set the store's secret (hidden prompt, checked before saving) -- see "The team object store" |
| `/jira` | shortcut for "show my pending Jira issues" |
| `/git` | shortcut for "show git status of this project" |
| `/plan <task>` | ask for an explicit plan first |
| `/digest` | summarise today's work (findings + actions + commits) |
| `/library` | the whole library: every book, story and chapter recorded |
| `/book` | the current dataset's book (one book per dataset, keyed by content hash) |
| `/story` | the current story -- a new one opens automatically when the data changes |
| `/chapter` | the open chapter, plus the taxonomy of approaches you can open |
| `exit` / `quit` | leave |

The four Librarian commands are how you find out whether a question has
already been answered on this data. A chapter that already has recorded pages
against the *same* data cannot be silently re-opened -- see SS14.

---

## 5. Integrations are opt-in sliders

Integrations are **off by default** and the analyst works fully without any of
them. A family's tools only appear when that integration is switched on:

- **git** switches on automatically when the project folder is a git repo.
- **jira** switches on once you give it a base URL + login email.
- **minio / trino** are configured for the big-data / memory layers (not shell
  tools).

You never edit config files by hand. Paste a value in the chat and the agent
files it for you -- secrets go to a gitignored `.ledgerhound/secrets.env`, and the
config only ever stores the *name* of the environment variable, never the
secret itself. The agent's own record of the value is redacted before it is
logged.

One honest caveat: the line **you type** is sent to the model and kept in
that session's transcript, so a token pasted into chat is in that run
folder's logs. For the object store always use `/minio --key` (hidden
prompt, never logged). For a Jira token you'd rather not paste, put
`LEDGERHOUND_JIRA_TOKEN=<token>` on a line in `.ledgerhound/secrets.env` (or
export it) and just give the agent the URL and email.

```
you > here's my Jira: https://myorg.atlassian.net, email me@myorg.com,
      token ATATT3xFfGF0...
ledgerhound > (confirms, stores URL+email in config.yaml, token in secrets.env)
           jira integration switched on
```

From the next turn, Jira tools are live.

### git -- push and pull your work

- `git_status`, `git_log`, `git_diff`, `git_today` -- free, read-only.
- `git_pull` -- asks yes/no first.
- `git_commit_push` -- stages your changes, commits, pushes. The commit
  **message is auto-drafted from what you actually did this session** (your
  findings and actions). You see every file and the message, then confirm.
  - It pushes the branch you are on, not a fixed `main`.
  - It **refuses** to commit anything that looks like data or Ledgerhound
    output -- `.parquet`/`.csv`/`.jsonl` and similar, anything under
    `run_outputs/`, `features/`, `scratch/`, `.ledgerhound/`, or any file over
    5 MB -- and lists them. Customer data does not leave through git. Add
    them to `.gitignore`, or commit the files you mean by hand.
  - git tools act on the project's own repo, or on the repo you launched
    from **only if the project is inside it** -- never on an unrelated
    checkout.

```
you > push my work
ledgerhound > confirm  Commit & push 2 file(s) to origin/feature/july?
           message: Analysis: fraud rate baseline on July data
           files:
             analysis.py
             notes.md
(yes/no) > yes
```

### Jira -- pull tasks and post updates

- `jira_pending_tasks` -- your open, not-Done issues. Free.
- `jira_post_update <ISSUE>` -- posts a work update as a comment or worklog.
  Leave the text out and it **auto-drafts today's work** from the session
  digest. Confirmed before posting.
- `jira_create_issue`, `jira_transition` -- create / move issues. Confirmed.
- **the org's own Jira (Server / Data Center):** set `jira.api_version: "2"`,
  `jira.auth: bearer` for a personal access token, and
  `jira.verify_ssl: "false"` if its certificate comes from an internal CA
  (the error message says so when that is the problem). Jira Cloud keeps the
  defaults: `api_version "3"`, `auth basic` (email + API token).

```
you > post today's update to AUD-42
ledgerhound > confirm  Post this comment to AUD-42?
           ## Work update -- 2026-07-09
           - fraud rate is 4.9% [within_stratum]
           - ...
(yes/no) > yes
```

---

## 6. Safety model (human-in-the-loop)

- **Reads never ask** (status, log, pending tasks, profiling).
- **Every write asks** -- git push/pull, any Jira write, any config change --
  with a preview of exactly what will happen. Answer `no` and the analyst just
  continues the conversation; it won't retry behind your back.
- **Secrets are never shown** -- not in `/config`, not in confirm prompts, not
  in the committed config file.
- **Ctrl+C stops the turn, not the session.** Interrupting a long turn keeps
  the loaded data, kernel and history; the shell repairs the conversation and
  waits for your next message. `exit` (or Ctrl+D at the prompt) leaves.
- **`--autonomous` and `ask_user`.** In an unattended run nobody can answer an
  `ask_user` question, so the analyst takes the FIRST option it offered (its
  own recommendation), says so in the observation, and logs it. Modelling
  authorisation and external writes are never auto-approved.

---

## 7. Automatic data-comparability checks

Some data problems make every downstream number meaningless, and the worst
of them are invisible once the data is combined. The shell checks for these
**mechanically, on the data itself** — not by asking the model to remember
to look.

After any `run_python` cell, the frames that cell created are inspected. If
two datasets (or the two classes inside one labelled dataset) cover very
different time windows, an unmissable `[DATA CHECK]` observation is
injected into the analyst's context:

```
[DATA CHECK] Time-window mismatch detected:
  - 'fraud_df' spans 31d [2026-03-01 -> 2026-03-31] but 'nonfraud_df'
    spans only 24h [2026-03-15 -> 2026-03-15] -- 31x narrower.

This is a SAMPLING DESIGN problem, not a modelling one...
```

What it catches today (deliberately narrow — a check that cries wolf gets
ignored):

- **Time-window asymmetry** — one sample spans 3× longer than the other.
  Every velocity/recency/day-of-week feature then separates the classes
  *by construction*, and the class ratio is an artifact of sampling, not a
  fraud rate.
- **Non-overlapping windows** — the samples come from different periods, so
  any comparison confounds fraud with whatever else changed.
- **The concat case** — the mismatch survives `pd.concat` and stops being
  visible as two frames. The label's classes are compared inside the single
  frame too.

It fires **once per real issue** (not on every cell), is recorded as a
`data_check` event in the hash-chained audit log, and **warns rather than
blocks** — you can proceed knowingly, but not unknowingly.

**Why mechanical:** on this exact scenario, a live DeepSeek run *did* notice
the mismatch by itself — but reported it as a closing caveat *after* a full
production recommendation, and nothing guaranteed it would notice at all.
The check makes it deterministic and puts it in front of the work.

**Not covered:** this is not general data-quality validation, and it does
not yet design a sampling strategy for you (the existing
`NonFraudSamplingDesigner`/`SamplingCertificate` are wired into the
scripted playbook stack, not the shell).

---

## 8. Setting the judge model

Benchmarks and evals (`ledgerhound/eval/`) use a second model as a **reference-guided judge** to grade the analyst's
claims. The `/adversarial` deep-research loop (section 8b) also uses it, both
as the adversary's own model and for the final verdict. If you don't
configure one, the actor grades *itself* — that still runs, but it's not
independent evidence, so every record is stamped `self_judge: true`. Set a
different model to fix that.

**Option A — `/model-judge` in the shell (easiest):** run `/model-judge` with
no arguments to list what the judge's currently-resolved endpoint actually
serves and pick one (same UX as `/model`); `/model-judge <id>` sets a model
override directly without touching the rest of the judge config;
`/model-judge configs/x.yaml` points the judge at a whole different provider.
All of these persist to `.ledgerhound/config.yaml` (Option B below), so they
also apply outside the shell (benchmarks, `/adversarial`).

**Option B — in the project config (persists):**

```yaml
# .ledgerhound/config.yaml
judge:
  config_ref: "configs/gateway.yaml"   # any LLM config
  api_key_env: "LEDGERHOUND_JUDGE_API_KEY"  # optional; else the yaml's own key
```

**Option C — environment variables (per-run override, wins over the config):**

```bash
export LEDGERHOUND_JUDGE_API_KEY="..."
export LEDGERHOUND_JUDGE_BASE_URL="https://api.moonshot.cn/v1"
export LEDGERHOUND_JUDGE_MODEL="kimi-k2.5"
```

**All three env vars are required together.** Setting only some of them
logs a warning and falls through — it does not half-configure a judge.

Resolution order is: **env vars → `judge.config_ref` → self-judge**. The
scripts consult `./.ledgerhound/config.yaml` automatically when run from the
repo root, so setting the judge with `/model-judge` and then running a
benchmark just works. Check which one was used in the run's record:

```json
"judge": {"model": "kimi-k2.5", "endpoint": "...", "self_judge": false, "source": "project_config"}
```

A misconfigured judge (missing key, bad path) never crashes a run — it
logs why and falls back to a disclosed self-judge.

---

## 8b. `/adversarial` — deep-research debate over fraud cases

Motivated by a real production incident: a run that flipped fraud and
non-fraud labels wholesale. A single analyst pass tends to accept its own
first plausible story; this puts adversarial pressure on it instead.

**What it does.** For a random sample of fraud cases (default 25), two full
agents — each with the same tool access as a normal shell session
(`run_python`, `record_finding`, `load_data`, GPU/kernel mode, everything) —
debate whether the fraud label is actually correct, against that ONE case's
own data (the flagged entity plus every account it transacted with, not the
whole dataset):

- the **analyst** (your regular/base model) has to build a concrete,
  evidence-based case for *why* the label is right — specific transactions,
  timing, amounts, network structure — every round.
- the **adversary** (the judge model — see section 8) is never convinced by
  assertion alone: demands evidence, proposes innocent explanations, and
  runs its own tool calls to check the analyst's numbers independently.

Each case runs **100 rounds on the payer side** (50 analyst + 50 adversary,
alternating — the payer is the fraud side) and **50 rounds on the payee
side** (25 + 25 — the counterparty gets scrutiny too, just less). A third
pass, also on the judge model, then reads the *entire* transcript (both
sides, every round) and delivers one careful verdict — explicitly instructed
not to rubber-stamp the original label.

**Usage** (after `set_goal(label_column=..., entity_key=...)` — it needs to
know which column marks fraud and which is the payer account):

```
/adversarial --cases 25
/adversarial --cases 10 --seed 42     # reproducible sample
/adversarial --mode off               # no-op reminder: it only runs when called
```

This is a long, unattended, expensive run by design — up to 150 LLM turns
per case, each with unbounded tool calls (the deliberate budget is the round
count, not tool-call count). Results land in
`<run_dir>/adversarial/report.json`, with cases the judge flagged as
`LABEL QUESTIONABLE` or `LIKELY MISLABELLED` surfaced first — that's the
short list worth a human looking at, not all 25 cases' full transcripts.

If no separate judge is configured (section 8), the adversary and the
verdict both run on the actor's own model — Ledgerhound warns about this
explicitly (self-judging your own fraud calls is weaker evidence than an
independent model). `/model-judge` fixes it in one command.

---

## 9. Configuration reference

Everything lives in one file per project: `.ledgerhound/config.yaml` (heavily
commented). Secret *values* live in `.ledgerhound/secrets.env` (auto-gitignored);
the yaml only names which env var to read.

Sections: `llm`, `jira`, `git`, `minio`, `trino`. You rarely touch this by hand
-- use `configure_integration` (via chat) or `/config` to inspect it.

```yaml
jira:
  base_url: "https://myorg.atlassian.net"
  email: "me@myorg.com"
  project_key: "AUD"
  token_env: "LEDGERHOUND_JIRA_TOKEN"   # value is in secrets.env
  api_version: "3"                    # "3" = Cloud, "2" = Server
git:
  remote: "origin"
  default_branch: "main"
```

Machine-wide settings (LLM/judge endpoints, GPU tuning, paths) live in
`main_config.yaml` -- see `main_config.example.yaml`. The ones you are most
likely to touch:

| Setting | What it does |
|---|---|
| `llm.method` + `llm.gateway.*` | which gateway, model and key the analyst uses |
| `judge.*` | a separate model for `/eval` and `/adversarial` (blank = self-judge) |
| `identity.user` | your `work` identity (runs and uploads are filed under it) |
| `minio.endpoint` / `bucket` / `access_key` / `verify_ssl` | the team object store (secret via `/minio --key`) |
| `minio.outputs_prefix` | where `/s3` sessions upload features (default `outputs`) |
| `minio.client` | force `boto3` or `minio` as the S3 library (blank = boto3 if installed) |
| `paths.run_outputs_dir` | put every run folder in one place |
| `paths.data_cache_dir` | where `/s3` data is copied (blank = `/dev/shm/sentinel-data`) |
| `kernel.cell_timeout_seconds` | how long one analysis step may run (default 1800) |
| `tokenizers.local_dir` | a mounted folder of tokenizers instead of the object store |

One worth knowing in kernel mode: `kernel.cell_timeout_seconds` (env `LEDGERHOUND_CELL_TIMEOUT`,
default 1800) is how long one `run_python` cell may run before it is
interrupted. Raise it if a single groupby on crore-scale data on the CPU
fallback legitimately takes longer.

---

## 10. CLI quick reference

`ledgerhound shell` is the front door — everything below is the scripted /
power-user surface the shell composes for you. Run any command with
`--help` for its full options.

### Daily use

| Command | Does |
|---|---|
| `ledgerhound shell` | The conversational analyst. **Start here.** |
| `ledgerhound shell --model <id>` | Pin a model, skipping discovery/selection. |
| `ledgerhound shell --max-steps N --step-call-budget N --turn-budget-seconds N` | Raise the tool-call/wall-clock ceilings for a deliberately deep, cost-no-object exploration run (defaults: 400 / 60 / 14400s -- safety nets, not a pace-setter). |
| `ledgerhound sessions collect` / `ledgerhound sessions report` | Pull + summarise session-quality logs collected from separately-installed Kubeflow instances (no shared filesystem needed) — see the module docstring in `ledgerhound/runtime/session_log.py`. |
| `ledgerhound doctor` | Environment check: version, identity, keys, object store connection, tokenizer. |
| `python scripts/prod_checks.py [gateway store kb tokenizer gpu adversarial]` | Deployment check on a pod: PASS/WARN/FAIL per check, JSON result saved. |
| `ledgerhound profile <data>` | Profile a dataset — no LLM needed. |
| `ledgerhound health` | Ping the configured LLM endpoint. |
| `ledgerhound version` / `ledgerhound stats` / `ledgerhound whoami` | Install info, run/memory summary, current identity. |

### Scripted analysis (no conversation)

| Command | Does |
|---|---|
| `ledgerhound eda --digest <yaml>` | Agentic EDA with findings + notebook journal. |
| `ledgerhound eda-multi` | Three-phase fraud × payee × merged-verdict playbook. |
| `ledgerhound playbook` | The deterministic Fraud EDA Playbook. |
| `ledgerhound auto run "<goal>" --data <file>` | Self-healing autonomous loop toward a natural-language goal. |
| `ledgerhound scorecard` / `ledgerhound diff` | EDA scorecard for a run; compare two digests. |

### Fraud-domain tools

| Command | Does |
|---|---|
| `ledgerhound signal` | Feature discriminative power + decay over time. |
| `ledgerhound typology` | Cluster fraud typologies, track their evolution. |
| `ledgerhound graph` | Account graphs + mule-detection EDA. |
| `ledgerhound monitor` | Scheduled, self-investigating drift monitoring — see §9 below. |

### Audit & reproducibility

| Command | Does |
|---|---|
| `ledgerhound replay <run>` | Re-execute a recorded session against the same data — PASS/DIFF verdict. |
| `ledgerhound run-views` | Regenerate `session_code.py` + `findings.jsonl` from a run's event log. |
| `ledgerhound manifest` | Reproducibility manifest (hashes of code, data, packages) for a workspace. |
| `ledgerhound eval` | Golden-dataset evaluation (groundedness via recomputation). |

### Workspace & memory

| Command | Does |
|---|---|
| `ledgerhound project` / `ledgerhound workspace` | Create/inspect projects (`.ledgerhound/`) and workspaces. |
| `ledgerhound start` | Start a new experiment or resume a previous session. |
| `ledgerhound memory` / `ledgerhound knowledge` / `ledgerhound dictionary` | Dataset memory (HITL bootstrap), institutional knowledge store, schema KB. |
| `ledgerhound upgrade` | Upgrade ledgerhound from the internal wheel index (`$LEDGERHOUND_INDEX_URL`). |

---

## 11. GPU acceleration (cuDF)

Ledgerhound uses your GPU for **analysis** when three things are true:

1. **A GPU is actually available** and cuDF (RAPIDS) is installed.
2. **Kernel mode is on** — `ledgerhound shell --use-kernel`. This is the
   opt-in real-IPython-kernel mode (needs `jupyter_client` + `ipykernel`,
   i.e. `pip install -e ".[test]"`). The default in-process sandbox is
   pandas-only.
3. **The data fits.** The compute router routes to CPU when the file is
   large relative to GPU memory (>70%), because spilling is slower than
   just using pandas.

```bash
pip install -e ".[test]"                       # kernel deps
ledgerhound shell --use-kernel --llm-config configs/deepseek.yaml
```

On startup the shell tells you which engine it actually got:

```
(real Python kernel started -- full language, no restrictions -- data engine: cuDF (GPU))
```

**If it says `pandas (CPU)` on a machine with a GPU, run the diagnostic —
it tells you exactly which cause it is:**

```bash
python scripts/diagnose_gpu.py --data yourfile.parquet
```

It prints what each GPU probe sees (torch / cupy / nvidia-smi), whether
cuDF imports, the router's decision and its reason, and the engine the
kernel would request. The usual causes:

- **No GPU visible to this process** — all three probes report 0. Check
  `nvidia-smi` runs and that `import cudf` works in the same environment.
- **File too big for the card** — the router routes to CPU above 70% of
  GPU memory (parquet is estimated at 4× its on-disk size).
- **A local LLM took the only GPU** — point `--llm-config` at a remote API
  (DeepSeek, the organization gateway) and the GPU is freed for data.
- **Kernel mode off** — GPU needs `--use-kernel`.

GPU detection tries **torch, then cupy, then nvidia-smi** and takes the
first that reports a GPU — so a RAPIDS install without torch (torch is not
a dependency of this project) is detected correctly. The profile records
which probe found them: `GPUs: 2 [via cupy]`.

**Fallback is automatic and safe.** If cuDF fails to initialise, the
kernel silently downgrades to pandas rather than failing the session — so
the same command works on a laptop and a GPU box.

### What is NOT GPU-accelerated (be aware)

- **The default (non-kernel) sandbox** is always pandas/CPU. GPU
  acceleration requires `--use-kernel`.
- **Profiling / the digest** is computed on CPU, even on a GPU box. The
  compute router's cuDF branch reads the file with cuDF and then converts
  straight back to pandas (`.to_pandas()`) before profiling, so only the
  file *read* touches the GPU. Making the profiler natively cuDF is real,
  unbuilt work — not a claim we make today.

---

## 12. Drift Monitor — scheduled, self-investigating drift detection

`ledgerhound monitor` watches a dataset over time (signal decay + fraud
typology evolution), alerts on WARNING/FAIL severity via webhook, and
can optionally investigate what it found automatically instead of just
paging a human.

### Quick start (files, on an interval)

```bash
ledgerhound monitor init --dataset upi_fraud \
  --fraud data/fraud.parquet --nonfr data/nonfr.parquet \
  --interval 3600 --webhook https://hooks.slack.com/services/...

ledgerhound monitor run --once           # one cycle now (cron-friendly)
ledgerhound monitor run                  # daemon loop, Ctrl+C to stop
ledgerhound monitor status               # config + latest cycle
ledgerhound monitor history              # past cycles
```

### Choosing when it runs

Two modes, set at `init` time:

- **Interval** (default): `--interval 3600` — every N seconds.
- **Calendar**: `--days mon,thu --time 06:00 --timezone Asia/Kolkata` —
  specific days of the week at a specific time, in a specific timezone.
  Omitting `--days` keeps interval mode; passing it switches the config
  to calendar mode. `ledgerhound monitor status` shows which mode is active
  and the next-run description.

### Pulling data straight from Trino (no manual export step)

Instead of `--fraud`/`--nonfr` file paths, give the monitor SQL:

```bash
ledgerhound monitor init --dataset upi_fraud \
  --trino-fraud-sql "SELECT * FROM fraud_txns WHERE ..." \
  --trino-nonfr-sql "SELECT * FROM nonfraud_txns WHERE ..." \
  --trino-max-rows 2000000 \
  --days mon,thu --time 06:00 --timezone Asia/Kolkata
```

**Trino SQL must be approved before it will run — this is mandatory,
not optional:**

```bash
ledgerhound monitor approve-sql --by "your.name"
```

This prints the exact SQL, asks you to confirm, then records a sha256
hash of it. Every unattended cycle re-checks that hash — if the SQL is
ever edited afterward, the monitor refuses to run until you approve
again. A scheduled job has nobody present to click "yes" the way the
interactive chat shell's Trino tools do, so this hash-pinned,
approved-once model is how it stays safe unattended: connection
credentials come from `LEDGERHOUND_TRINO_*` environment variables (same as
everywhere else Trino is used), and a hard row cap always applies
regardless of what the approved SQL itself contains.

### Auto-investigation (the part that makes this more than a scheduler)

By default a WARNING/FAIL alert just fires a webhook — a human still has
to go look. With `--auto-investigate`, the monitor instead spins up a
real autonomous analyst session against the exact data slice that
triggered the alert, asks it to explain the drift, and lets it record a
grounded, evidence-backed finding — landing in the same audit trail
(hash-chained event log, replayable) as any human-driven session.

```bash
ledgerhound monitor init --dataset upi_fraud --fraud f.parquet --nonfr n.parquet \
  --auto-investigate

ledgerhound monitor run --once --llm-config configs/deepseek.yaml
```

`--llm-config` is required whenever `auto_investigate` is on — the run
fails fast with a clear error if you forget it.

**Important if you're also using `--trino-fraud-sql`/`--trino-nonfr-sql`:**
auto-investigation sends the drifted data slice to whichever LLM
`--llm-config` points at. If that data came from Trino (i.e. it's real,
possibly regulated data) and your LLM config points at a public API
instead of your organization's internal gateway, this crosses the zone
boundary real transaction data is never supposed to cross. The CLI prints a warning every time this
combination is active — read it, and point `--llm-config` at your
internal gateway (e.g. `configs/gateway.yaml`) for any Trino-sourced
monitor with auto-investigation on.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `env var is empty` on launch | The LLM key isn't set. Leave `llm.gateway.api_key` blank and run `ledgerhound shell` at a terminal to be asked once, or set it in `main_config.yaml` / `LEDGERHOUND_GATEWAY_API_KEY`. |
| `LLM config not found` | Run from the Ledgerhound checkout's install, or pass `--llm-config <yaml>`; the message lists where it looked. |
| `main_config.yaml not found` | Copy `main_config.example.yaml` to `main_config.yaml` in the repo or the folder you launch from, or set `$LEDGERHOUND_MAIN_CONFIG`. |
| Jira tools "not available" | Jira is off. Give the agent your base URL + email + token; it switches on next turn. |
| git tools missing | The project folder isn't a git repo. `git init` (or open the shell inside your repo). |
| A write went through without asking | It shouldn't -- writes are always gated. Check you didn't run headless (`auto_yes`). |
| Unicode errors on a legacy console | None expected -- the shell forces utf-8. If you see one, report it. |
| `llm.method='...' is not one of self\|gateway\|api` | An old `main_config.yaml`. Set `method: gateway` and name its block `gateway:` (the org's gateway). |
| `/adversarial` says a judge IS configured but could not be used | The judge block has no key (or a bad config). Fix it, or `/model-judge`; until then it self-judges. |
| `context window: ... (ASSUMED ...)` | The model is unknown to Ledgerhound and the gateway did not report a window. Fine if it really is 256k; otherwise set a smaller one. |
| A kernel cell stops with `step exceeded ...s` | Raise `kernel.cell_timeout_seconds` in `main_config.yaml`. |
| `tokenizer: estimated -- ... secret key is not set` | Run `/minio --key` in the shell. The tokenizer re-loads immediately. |
| `/minio --key` says `the store REJECTED this secret -- NOT saved` | The store refused that secret. Check the value with whoever issued it; your previous key is still in place. |
| `/minio` says `connection : FAILED [store_unreachable]` | You are off the org's network or the endpoint is wrong. Check `minio.endpoint`; a saved key waits until the store is reachable. |
| `/s3` says `do not look like parts of one dataset` | The folder holds separate files. Point at one, or add `--as-one` if they are one dataset split into files. |
| `/s3` says `not enough space` | The RAM disk (or cache folder) is too small for the data. Delete old datasets under it, or set `paths.data_cache_dir`. |
| `upload FAILED -- files are kept locally` | The store was unreachable or refused the key. Files stay in `features_dir` and are retried after the next step; `/minio` checks the connection. |
| `WRITE REDIRECTED` in a step's output | The code saved outside the workspace; the file is in `features_dir` under the name printed. Nothing to fix. |
| `git_commit_push` says `REFUSED` | It found data or Ledgerhound output among your changes. Add them to `.gitignore`, or commit the files you mean by hand. |
| Jira `HTTP 401` | Wrong token -- or a Server/Data Center Jira that wants `jira.auth: bearer` and `jira.api_version: "2"`. |
| Jira `certificate verify failed` | Internal CA. Set `jira.verify_ssl: "false"` for this project. |

**Checking a deployment end to end.** On a prod pod, run from the repo:

```
python scripts/prod_checks.py                 # gateway, store, team kb, tokenizer, GPU, one /adversarial case
python scripts/prod_checks.py gateway tokenizer   # just these (gateway store kb tokenizer gpu adversarial)
```

Each check prints PASS / WARN / FAIL / SKIP with a reason, and the full
result lands in `prod_checks_<timestamp>.json`. `ledgerhound doctor` is the
quick version: config, keys, the object store connection and the tokenizer.

---

## 14. What 0.7.0 added — the inference gates

Up to 0.6.0 Ledgerhound gated the **process**: the right grain, modelling only
once authorised, no silently re-running a finished chapter. Those checks stop
you computing the wrong thing. They say nothing about whether the number you
got supports the claim you are about to make.

0.7.0 closes that. Four things are new in the shell, and you will notice them
because they change what `record_finding` says back to you.

### Findings now carry a status

`record_finding` takes an optional `pvalue`. When you supply one, it is
corrected across **every test declared in the current chapter**, and the
finding comes back marked:

| Status | Meaning |
|---|---|
| `ESTABLISHED` | survived correction across the chapter's other tests |
| `EXPLORATORY` | did not — recorded, but as a lead, not a fact |
| `UNQUANTIFIED` | no p-value given; the search effort is still recorded |

The point is that the same p-value means different things depending on how
hard you looked. `p = 0.01` as the first thing you tried is a finding. The
same `p = 0.01` as the sixteenth is usually a coincidence, and it is now
marked as one.

Crucially the correction is **adaptive**: forty tests that all come back
significant are collective evidence and are *not* downgraded. Only a lone hit
among misses is. A gate that punished agreement would just get switched off.

A later test can also **demote an earlier finding**. If you record something
on your third comparison and then run thirty more, the ledger will tell you
the earlier claim no longer survives and must be restated wherever it has
already been reported.

### Search effort is recorded whether or not you quantify it

Every `record_finding` attempt counts toward the chapter's family — including
the ones refused as restatements. Analysis cells are counted too, so a chapter
with thirty cells and one recorded finding says so plainly: most of what was
computed never reached the ledger, and no correction can account for tests it
was never told about.

### Statistical assumptions are checked before a cell runs

Cells calling `scipy.stats` are checked against what the test assumes. Most
cases annotate and let the cell run. Two refuse, because they are wrong
whatever the data looks like:

- `ttest_rel` on two separate cohorts — a paired test needs the same entity
  observed twice
- `shapiro` above a few thousand rows — it rejects normality for every real
  dataset at that size, so the result carries no information

### A chapter for attacking your own findings

`falsification` is a new chapter type. `propose_falsification(claim)` returns
the specific ways a claim of that shape has turned out wrong before —
subgroup reversal, period holdout, prevalence shift, grain restatement,
leakage origin, effect size, sample frame. Every other chapter looks for
support; this one is for trying to kill the claim, which is the only test
that tells you anything.

### Two new evaluation commands

```sh
ledgerhound eval null-data      # false-discovery rate under known ground truth
ledgerhound eval blade          # external benchmark of analysis decisions
```

`null-data` is the one to run. It generates data with **no signal by
construction**, so every finding established on it is false and the answer
key cannot be argued with — then does the mirror with known planted effects,
because a gate that refuses everything would score perfectly on the first half
and be worthless. Measured at the default size: **2.83 false findings per run
ungated versus 0.03 gated, with power unchanged at 100%.**

It is deterministic and runs in seconds, so it can gate CI.
