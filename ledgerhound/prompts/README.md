# `ledgerhound/prompts/` — every prompt, one tree

All text the model is *told* lives here. Nothing in this folder is code;
editing a file here changes behaviour without a Python change. Each file
is loaded at runtime with a code-embedded fallback, so a missing or
mangled file degrades to today's behaviour rather than to an empty prompt.

```
ledgerhound/prompts/
├── shell/          the conversational analyst  (ledgerhound shell / webui)
│   ├── system.md         behaviour, goal grounding, depth, leakage rules
│   └── tool_protocol.md  text tool-calling protocol (non-function-calling gateways)
├── agentic/        agent specs for the EDAAgent engine (YAML: role prompt,
│   │               allowed tools, required_tools, stop conditions)
│   ├── eda.yaml            feature_engineering.yaml   profiler.yaml
│   ├── comparison.yaml     typology_comparison.yaml   sampling.yaml
│   └── orchestrator.yaml
└── skills/         proven team procedures the analyst MAY consult
    ├── fraud_vs_nonfraud.md   mule_graph.md   signal_decay.md
```

## Who reads what

| File | Loaded by | Fallback |
|---|---|---|
| `shell/*.md` | `ledgerhound/shell/prompts/loader.py` → `load_prompt()` | `_SYSTEM_PROMPT_FALLBACK` in `conversation.py` |
| `agentic/*.yaml` | `ledgerhound/agents/registry.py` (`_BUILTIN_DIR`) | none — spec must parse |
| `skills/*.md` | `ledgerhound/skills/library.py` (`_BUILTIN_DIR`) | none — skill is optional |

**Two engines, deliberately separate.** `shell/` drives
`ConversationalAnalyst` (free-form, `run_python`) — this is what
`ledgerhound shell` runs and what produced the pilot reports. `agentic/`
drives `EDAAgent` (structured, fixed tool catalogue). A change to one does
NOT affect the other; check which engine you are changing before editing.

## Editing rules

1. `shell/system.md` has a byte-identical twin: `_SYSTEM_PROMPT_FALLBACK`
   in `ledgerhound/shell/conversation.py`. **Edit both.**
   `tests/test_prompt_loader.py` fails if they diverge.
2. `agentic/*.yaml` tool names must exist in
   `ledgerhound/agents/tool_registry.py`'s `TOOL_CATALOG`, or the spec is
   rejected at load and the agent silently disappears.
3. User overrides shadow built-ins: `~/.ledgerhound/skills/*.md` beats
   `skills/`, and project agent specs beat `agentic/`.

## Where CONTEXT lives (not here)

Prompts are static instructions. Context is what the session accumulates:

| Layer | Location | Lifetime |
|---|---|---|
| Live message list + pinned goal/schema block | in memory, `ConversationalAnalyst.messages` | one session |
| Session artifacts | `run_outputs/<timestamp>/` — `events.jsonl` (hash-chained), `findings.jsonl`, `plan.json`, `session_goal.json`, `transcript.jsonl`, `digest.json`, `session_code.py` | durable, auditable |
| Dataset card | `.ledgerhound/` via `ledgerhound/knowledge/dataset_card.py` | across sessions, per file |
| Books / lessons / vectors | `ledgerhound/memory/` | across sessions |
| **Domain profiles** | `ledgerhound/context/domain_profiles/*.yaml` | static priors — recognise a table on sight |

`domain_profiles/` is the one that answers "why didn't it just *know* this
was an account-level mule problem?": it matches column names at
`load_data` time and pre-seeds the session goal, so the grain gate is
armed before the first cell runs. Adding a new table shape is a YAML file.
