# ADR 0001 — Kernel isolation: sandboxed subprocess, containers deferred

- **Status:** accepted, 2026-11-04
- **Context:** Phase 2 exposes code execution (`run_analysis`) over MCP, so the
  kernel that runs agent-written code needs a decided isolation model. Phase 1
  kept the exec-mode `ExecBlocklist` and marked it as no security boundary.

## The guarantee, stated narrowly

> **The analysis kernel cannot open network connections.**

That is the whole network claim. It is **not** "data stays home": the host
agent reads every tool result and can send what it reads anywhere it likes.
Isolation stops code *inside the kernel* (for example, code a poisoned
Knowledge Pack document or a crafted column value talked the agent into
writing) from shipping raw rows out directly. It does not, and cannot, contain
the agent itself, which already has a shell on the user's machine. Nobody
should describe Ledgerhound's isolation more broadly than this paragraph does.

## Decision

Run the kernel as a local subprocess (the existing Jupyter kernel in
`ledgerhound/auto/kernel.py`) wrapped in OS-level sandboxing, not in a
container. MCP mode has no `exec` fallback: code runs only in the kernel.

| Control | How | Where available |
|---|---|---|
| No network | `bwrap --unshare-net`; kernel talks to the server over Jupyter's `ipc` transport (Unix sockets on disk), since a new network namespace has its own loopback | Linux, WSL |
| Writes confined | `bwrap` mounts `/` read-only; read-write only for the run folder and a private temp folder | Linux, WSL |
| No host sockets | `bwrap` gives the kernel a private `/run`, `/tmp` and `/dev/shm`. Unix sockets ignore the network namespace, and the host's `/run` holds the user D-Bus (`systemd-run --user` would start a process outside the sandbox), the WSL interop socket (runs Windows `.exe` files with full network) and `docker.sock` | Linux, WSL |
| Clean environment | Kernel starts with an allow-listed environment: no API keys, cloud credentials or tokens | All platforms |
| Memory cap | cgroup `MemoryMax` via `systemd-run --user --scope` when a user systemd and the memory controller exist; otherwise the server polls the kernel's real resident memory and kills it past a threshold | cgroup: Linux/WSL with systemd; polling: everywhere |
| Runaway cells | Per-cell wall-clock timeout, high and configurable (`kernel.cell_timeout_seconds`, default 30 min) | All platforms |

Deliberately **not** used:

- **No virtual-memory limit (`RLIMIT_AS`).** It counts reserved address space,
  not memory used. cuDF/CUDA and Arrow reserve far more than they touch and
  fail at startup under it.
- **No CPU-time limit (`RLIMIT_CPU`).** Multi-hour runs on large data are
  expected; a CPU-time cap would kill legitimate work. The wall-clock timeout
  is the runaway net instead.

Today's kernel code only warns about memory; the kill-on-threshold poller is
new work in Phase 2.

## Honest levels

The level actually achieved is reported in the `load_dataset` result, written
to the hash-chained event log, and stated on the certificate. Each level names
only what is enforced:

| Level | Enforced | Typical platform |
|---|---|---|
| `sandboxed` | no network, writes confined, clean env, memory cap, cell timeout | Linux, WSL |
| `partial` | clean env, memory cap (polling), cell timeout. **Network and writes not restricted** | macOS, native Windows |

Native Windows runs at `partial` with a warning on the certificate rather than
refusing: bank laptops are often Windows and refusing would block the main
user. Teams that want a hard stop pass `--require-isolation`, which refuses to
start a kernel below `sandboxed`.

## Approvals

High-risk actions are approved through MCP elicitation when the client
supports it (a person answers). When it does not, the tool refuses and the
agent may retry with an explicit confirmation. That path means the agent
approved on the user's behalf, so it is logged as `approved_by: agent` and the
certificate flags those actions as **approved by the agent, not verified by a
person**.

## Alternatives considered

**Container (Docker/Podman, `--network none`).** Stronger and uniform across
platforms, but needs a container runtime that is often banned on bank
laptops, needs nvidia-container-toolkit for cuDF, adds bind-mount and startup
cost on large parquet, and puts the "install to first report in under 10
minutes" target out of reach. Deferred as a later opt-in level for servers.

**Keep exec mode with a blocklist.** Substring matching is trivially bypassed;
it was never a boundary. Removed from MCP mode.

## Consequences

- `jupyter_client` and `ipykernel` become dependencies of the `mcp` extra.
- `bwrap` is a soft dependency on Linux; without it the level drops to
  `partial` and says so.
- The isolation level becomes part of every certified report.
