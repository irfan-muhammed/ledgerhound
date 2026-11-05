"""
ledgerhound.shell.tools.specs
=============================

TOOL_SPECS: the OpenAI-format schema of every analyst tool. The handler for
tool `x` is the `_tool_x` method on AnalystTools.
"""

from __future__ import annotations







# ---- native tool specs (OpenAI function-calling) ----------------------------
TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and subfolders at a path so you can see what data is available.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Folder path (default current dir)."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "load_data",
        "description": "Resolve a data path (tries as-given, ./, ../, cwd, project root, and globs a folder) and load it as the session dataframe `df`. Returns shape + columns, or a folder listing to choose from.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File or folder path, may be relative."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "profile_data",
        "description": "Deterministically profile the loaded dataframe into a compact digest (columns, dtypes, missingness, target).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Execute Python in a persistent sandbox. `df`, `pd`, `np` are in scope; variables persist across calls. Print results you want to see. Save feature tables and other data files under `features_dir` (a Path in scope), e.g. `feat.to_parquet(features_dir / 'account_features.parquet')`.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "propose_falsification",
        "description": "Given a claim you have recorded (or are about to), "
                       "return the specific ways a claim of that shape has "
                       "actually turned out wrong before, as an attack plan. "
                       "Use inside the 'falsification' chapter. Every other "
                       "chapter looks for support; this one tries to kill the "
                       "claim, which is the only test that tells you anything.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string",
                      "description": "The claim to attack, stated as recorded."}},
            "required": ["claim"]}}},
    {"type": "function", "function": {
        "name": "record_finding",
        "description": "Log an evidence-backed finding with an honest sample-frame tag. Checked BEFORE recording against the domain profile, this book's own history, and the team knowledge base -- a claim that matches one of these is REFUSED (not recorded) unless you also pass delta explaining what it adds.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "evidence": {"type": "string"},
            "sample_frame": {"type": "string",
                             "enum": ["within_stratum", "cross_stratum",
                                      "population_estimate", "unknown"]},
            "predicate": {"type": "string",
                         "description": "The exact condition/window the "
                         "claim's numbers were computed under, e.g. "
                         "'hour>=22 OR hour<6'. Optional, but if you record "
                         "a claim that will later be COMPARED against "
                         "another finding about the same quantity (e.g. "
                         "fraud vs non-fraud night-share), give both the "
                         "SAME predicate wording or they cannot be checked "
                         "for actually meaning the same thing."},
            "pvalue": {"type": "number",
                      "description": "The raw p-value of the statistical test "
                      "behind this claim, if there was one. Supply it whenever "
                      "a test was actually run: it is corrected across every "
                      "test declared in this chapter, and only a claim that "
                      "survives that correction is recorded as ESTABLISHED "
                      "rather than EXPLORATORY. Omitting it does not make a "
                      "claim stronger -- it makes it unquantified."},
            "delta": {"type": "string",
                     "description": "Required ONLY if this claim is refused"
                     "as a restatement of something already known "
                     "(established finding, this book's history, or the "
                     "team KB). State specifically what this version ADDS "
                     "-- a mechanism, an interaction, account-level "
                     "behaviour, a feature it justifies -- then call "
                     "record_finding again with the same claim plus this."}},
            "required": ["claim", "sample_frame"]}}},
    {"type": "function", "function": {
        "name": "save_report",
        "description": "Save a markdown report into the session's dated run folder. Each save writes a NEW numbered file rather than overwriting, so a first broad pass and a later deeper pass both survive. The filename carries the current phase and the sequence number automatically -- you do not supply either. Findings you recorded since your last save are attached to the report automatically, so record_finding as you work. EVERY PHASE MUST HAVE AT LEAST ONE SAVED REPORT: advance_phase refuses while the current phase has none. Saving itself is not gated -- just call it.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "markdown": {"type": "string", "description": "Full report body in markdown. Include the actual numbers -- rates, counts, thresholds, and the n they were computed on."}},
            "required": ["title", "markdown"]}}},
    {"type": "function", "function": {
        "name": "consult_skill",
        "description": "Optionally look up the team's proven procedure for a task (e.g. 'fraud vs non-fraud', 'mule graph', 'signal decay'). Reference only -- adapt it, don't follow blindly.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What you're trying to do."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": "Pause and ask the analyst a concrete multiple-choice (single answer) or multi-select (several answers) question, then WAIT for their real answer before continuing -- for genuine judgment calls a mechanical check flagged as ambiguous (e.g. 'is this column operational metadata or real behavioural signal?'), not for things you can decide yourself. Give concrete options, not an open-ended question -- the analyst picks from what you offer.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The concrete question, with enough context to answer it."},
            "options": {"type": "array", "items": {"type": "string"},
                       "description": "2-6 concrete options to choose from."},
            "multi": {"type": "boolean",
                     "description": "true if more than one option may apply (multi-select); false (default) for pick-exactly-one."}},
            "required": ["question", "options"]}}},
    {"type": "function", "function": {
        "name": "set_methodology",
        "description": "Adopt a declared analysis methodology: an ordered sequence of phases, each with its own data scope, allowed grains and a checklist of concrete EDA steps. Call this at the start. Templates: 'topdown_7phase' (fraud-only EDA -> non-fraud -> contrast -> aggregation plan -> features at BOTH grains -> selection -> modelling), 'sample_first_7phase' (phase 1 DECIDES how non-fraud is sampled), 'bottom_up_7phase' (label withheld until phase 4 -- profile the whole population's structure first, then test what survives contact with the fraud label), 'deep_eda_only' (bottom-up exploration only, refuses to reach feature construction), 'fe_only' (skips exploration, starts straight at feature construction off already-known signal), 'eda_fe_only' (exploration + features, refuses to reach modelling). Call with no name to list what is available. The methodology is pinned every turn and drives the grain/phase gates -- transaction-level work is legitimate wherever a phase declares it.",
        "parameters": {"type": "object", "properties": {
            "template": {"type": "string", "description": "Template name, e.g. 'topdown_7phase', 'deep_eda_only'. Omit to list the available templates."},
            "mode": {"type": "string", "enum": ["", "eda_only", "fe_only", "eda_fe_only"],
                    "description": "A SEPARATE axis from which template -- controls how far the pipeline goes, not the reasoning order. '' (default) = full sequence. 'eda_only' = stop before feature construction. 'fe_only' = skip exploration, jump straight to building features off already-known/stated signal. 'eda_fe_only' = go through feature construction but stop before modelling authorisation."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "complete_step",
        "description": "Mark a step of the current phase done (or skipped, with a reason) and record what it found. Keeps 'this phase is complete' checkable rather than asserted.",
        "parameters": {"type": "object", "properties": {
            "step_id": {"type": "string", "description": "Step id, e.g. 'p1.04'."},
            "note": {"type": "string", "description": "What this step established -- the finding, not just 'done'."},
            "skipped": {"type": "boolean", "description": "true if the step was deliberately skipped; put the reason in note."}},
            "required": ["step_id"]}}},
    {"type": "function", "function": {
        "name": "add_step",
        "description": "Append a step the data actually demands to the current phase. The shipped checklist is a floor, not a ceiling -- when the analysis surfaces something worth chasing, add it here so it is tracked rather than done ad hoc and forgotten.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to investigate, concretely."},
            "grain": {"type": "string", "description": "'transaction', 'account' or 'any'."}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "advance_phase",
        "description": "Finish the current phase and move to the next. Refuses while steps are pending unless force=true; a forced skip marks the remaining steps skipped and is recorded, so the audit trail stays honest.",
        "parameters": {"type": "object", "properties": {
            "force": {"type": "boolean", "description": "Move on with steps still pending (they are recorded as skipped)."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "phase_status",
        "description": "Where the methodology stands: current phase, per-phase step progress, and what remains in this phase.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "plan_features",
        "description": "Enumerate MANY candidate entity-level features (hundreds; 1000 is fine) from the loaded schema, and get runnable code to compute them in batches. Use this after the EDA iterations, when you move into feature engineering. These are CANDIDATES, not recommendations -- you must then compute them and run evaluate_features(...) to rank/prune, and only hand over what survives. Breadth here is good; breadth in the deliverable is what the pilot rejected.",
        "parameters": {"type": "object", "properties": {
            "entity_key": {"type": "string", "description": "Column to aggregate to, e.g. payer_account_number. Defaults to the session goal's entity_key."},
            "time_column": {"type": "string", "description": "Timestamp column, to enable velocity/rolling-window and temporal-shape features."},
            "max_features": {"type": "integer", "description": "Cap on candidates generated (default 1000)."},
            "windows": {"type": "array", "items": {"type": "integer"}, "description": "Rolling windows in days (default [1,7,30])."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "export_feature_catalog",
        "description": "Export the documented feature catalog: every candidate with its description, the exact reproducing code, its measured quality (IV/KS/AUC), and its TIER. Features are tiered (core / supporting / review / dropped), not pruned to a shortlist -- a modest univariate IV can still earn its place in a multivariate model, so only leakage-vetoed, redundant, or genuinely signal-free features are dropped. Thresholds are adjustable. Call after computing the features and running evaluate_features. Writes feature_catalog.csv + .md. When the goal declares entity_key_secondary, every feature appears for BOTH sides (`<name>__remitter`, `<name>__beneficiary`): an evaluation keyed by the plain name counts for the primary (remitter) side only -- compute the beneficiary-side feature and pass its evaluation as `<name>__beneficiary`, or that side stays unmeasured.",
        "parameters": {"type": "object", "properties": {
            "evaluations": {"type": "array", "items": {"type": "object"}, "description": "Per-feature dicts from evaluate_features(...)['evaluations'] -- feature/iv/ks/auc/iv_band."},
            "vetoed": {"type": "array", "items": {"type": "string"}, "description": "Feature names the leakage check vetoed -- these are hard-dropped."},
            "redundant": {"type": "array", "items": {"type": "string"}, "description": "Feature names pruned as correlated duplicates of a stronger feature."},
            "iv_core": {"type": "number", "description": "IV at/above which a feature is 'core' (default 0.10)."},
            "iv_supporting": {"type": "number", "description": "IV at/above which a feature is KEPT as 'supporting' (default 0.02 -- deliberately permissive; lower it to keep even more)."},
            "title": {"type": "string", "description": "Catalog title."},
            "percentiles": {"type": "object", "description": "Optional {feature_name: percentile_table(fraud, nonfraud) result} map -- adds a percentile check to each feature's entry showing whether the fraud/non-fraud tails cross or where they diverge, which IV/KS/AUC alone cannot show. Build it by calling percentile_table(...) per feature before exporting."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "recall_history",
        "description": "Search the VERBATIM archive of earlier messages that were compacted out of the context window. Compaction summarises, it never deletes -- so if a number, code cell, error or decision is missing from the summary, look it up here instead of recomputing it. Use it whenever you are about to redo work that you suspect was already done.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words to search for (e.g. a variable name, a column, 'fraud rate', an error type)."},
            "max_hits": {"type": "integer", "description": "Max matching messages to return (default 5)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "set_goal",
        "description": "Declare (or refine) what this session is FOR: the objective, the unit of analysis the DELIVERABLE must be at (account or transaction), the entity key, and the label column. Call this early -- before building or exporting features. The goal is pinned into every turn and is ENFORCED: with unit_of_analysis='account', exporting a frame with duplicate entity keys is refused, not warned about. Exploration at any grain is always allowed; this constrains what you SHIP.",
        "parameters": {"type": "object", "properties": {
            "objective": {"type": "string", "description": "What the user actually wants out of this session, in one sentence."},
            "unit_of_analysis": {"type": "string", "description": "'account' (one row per entity in the deliverable) or 'transaction'."},
            "entity_key": {"type": "string", "description": "Column that identifies the entity, e.g. payer_account_number. Required when unit_of_analysis='account'."},
            "entity_key_secondary": {"type": "string", "description": "The OTHER side of a payment, e.g. payee_account_number, if this dataset has one. Aggregation defaults to entity_key alone (typically the payer/remitter side) -- set this so feature generation can ALSO be run against the beneficiary side, which is often where the account-level verdict actually lives."},
            "label_column": {"type": "string", "description": "The target/label column, if known."},
            "notes": {"type": "string", "description": "Any constraint worth pinning (e.g. 'features only, no modelling yet')."}},
            "required": ["objective"]}}},
    {"type": "function", "function": {
        "name": "request_modeling_authorisation",
        "description": "Ask the human to authorise moving from analysis/feature work into model training. Model training is REFUSED until this is granted. Call it only when you believe the feature set is genuinely ready, and summarise what you built so the human can judge.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "What you have built/found so far, and why you think modelling is the right next step."}},
            "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "save_understanding",
        "description": "Archive this session's understanding as a durable 'book' (summary + goal + data + findings) so it is recalled in future sessions on this data. Offer this when the analysis reaches a useful conclusion.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string", "description": "One-paragraph index summary."},
            "goal": {"type": "string"}},
            "required": ["title", "summary"]}}},
    {"type": "function", "function": {
        "name": "record_decision",
        "description": "Log a data-mutating or business decision (e.g. imputation strategy, threshold chosen, rows dropped) to the dataset's durable card, so future sessions on this data know it was already decided and why.",
        "parameters": {"type": "object", "properties": {
            "decision": {"type": "string", "description": "What was decided."},
            "rationale": {"type": "string", "description": "Why, in one line."}},
            "required": ["decision"]}}},
    {"type": "function", "function": {
        "name": "record_data_answer",
        "description": "Save the USER's answer to a question about the data (the [key] questions listed after profile_data: what a label's classes mean, which entity/time column, units, what codes mean). Stored on the dataset card and shared with the team, so nobody is asked again on this dataset. Only record what the user actually said -- never your own guess.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "The question key in brackets, e.g. 'label_classes:fraud_flag'."},
            "answer": {"type": "string", "description": "The user's answer, in their words."},
            "question": {"type": "string", "description": "The question as asked (optional)."}},
            "required": ["key", "answer"]}}},
    {"type": "function", "function": {
        "name": "open_chapter",
        "description": "Declare the APPROACH for the next stretch of analysis on this dataset (a chapter, from a fixed taxonomy -- e.g. 'percentile-profile', 'interaction-sweep'). Deduped against this exact data version: the same chapter with pages already recorded on UNCHANGED data is refused, showing what was already found -- this is what stops a second session on the same data from redoing the first session's work. A new content hash (new month, re-extracted file) auto-allows. Call this before starting a genuinely new line of analysis; add_chapter_page records progress under it.",
        "parameters": {"type": "object", "properties": {
            "chapter": {"type": "string", "description": "One name from the chapter taxonomy (ask if unsure -- an unknown name returns the list)."}},
            "required": ["chapter"]}}},
    {"type": "function", "function": {
        "name": "redo_chapter",
        "description": "Override a chapter refusal when the data has NOT changed but you have a real reason to redo it (the first pass was buggy, a wrong assumption, etc.). Always recorded, never silent.",
        "parameters": {"type": "object", "properties": {
            "chapter": {"type": "string"},
            "reason": {"type": "string", "description": "Why this needs to be redone despite unchanged data."}},
            "required": ["chapter", "reason"]}}},
    {"type": "function", "function": {
        "name": "add_chapter_page",
        "description": "Record one page of progress/findings under the currently open chapter. Append-only -- call this as you go, not just at the end.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "What this page covers -- the specific thing done or found."}},
            "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "record_percentile_snapshot",
        "description": "Record a feature's percentile table (from percentile_table(a, b) -- pass either side's flat {label: value} dict) so a LATER session on different data can check whether the pattern held. This is the numeric record cross-period confirmation compares against; add_chapter_page's prose notes are not enough for that.",
        "parameters": {"type": "object", "properties": {
            "feature": {"type": "string", "description": "The feature this snapshot is for."},
            "percentiles": {"type": "object", "description": "Flat {label: value} dict, e.g. percentile_table(fraud, nonfraud)['a']."},
            "chapter": {"type": "string", "description": "Defaults to the currently open chapter if omitted."}},
            "required": ["feature", "percentiles"]}}},
    {"type": "function", "function": {
        "name": "check_cross_period_stability",
        "description": "Compare a recorded percentile snapshot for `feature` between the CURRENT data version and an earlier story on this same dataset -- classifies STABLE or SHIFTED at each quantile. Requires record_percentile_snapshot to have been called in both stories first. Use this instead of eyeballing two percentile tables when confirming a finding holds across a later period.",
        "parameters": {"type": "object", "properties": {
            "feature": {"type": "string"},
            "chapter": {"type": "string", "description": "Defaults to the currently open chapter if omitted."},
            "against_story": {"type": "string", "description": "A story_id (see /story or /library) to compare against. Defaults to the story immediately before the current one."}},
            "required": ["feature"]}}},
    {"type": "function", "function": {
        "name": "confirm_roles",
        "description": "Confirm what each column IS (payer, payee, event_time, amount, label, label_time, rule_output, ...): guardrails key off these roles, not column names. Ledgerhound proposes roles on load; this asks the user to confirm them in one question. Correct a role with roles={column: role}; declare that the data has no label, label_time or label_source column with absent=[...]. label, label_time and label_source are always the user's call.",
        "parameters": {"type": "object", "properties": {
            "roles": {"type": "object", "description": "Corrections, {column: role}."},
            "absent": {"type": "array", "items": {"type": "string"},
                       "description": "Label roles the data does not have: label, label_time, label_source."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "kb_add",
        "description": "Add to the Knowledge Pack: one structured entry (type typology, rule, dictionary, known_finding or label_policy) or one document (.md, .txt, .pdf path). Entries are validated strictly; a rule's output_column is then blocked as a feature, a label_policy sets label maturity. scope=team is shared with everyone and asks the user first.",
        "parameters": {"type": "object", "properties": {
            "entry": {"type": "object", "description": "A structured entry with a type field."},
            "document": {"type": "string", "description": "Path to a .md/.txt/.pdf file."},
            "scope": {"type": "string", "enum": ["project", "team"]},
            "title": {"type": "string"}, "source": {"type": "string"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "undo_last_transform",
        "description": "Restore the dataframe to its state before the last run_python call that changed it -- use this when a mutation (a filter, drop, etc.) turns out to be wrong. Only available for small/medium in-session (exec-mode) data; kernel-mode or very large data has no automatic undo copy -- use load_data to start over instead.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]
