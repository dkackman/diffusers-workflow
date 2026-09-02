# Desktop Installers — Design

Date: 2026-09-01

## Summary

Ship `diffusers-workflow` as native installers for Windows (x64),
Linux (x64) and macOS (Apple Silicon), each carrying a **Tauri 2**
desktop shell that supervises the existing `dw.serve` process and
displays the existing SPA.

The installer is thin (~60–100 MB). It carries a
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
interpreter and `uv`, and on first launch provisions a real virtual
environment by installing the published
`diffusers-workflow` wheel from PyPI. The multi-gigabyte dependency set
is fetched, not bundled.

Nothing is frozen. This is a deliberate constraint, not a shortcut —
see [Why not PyInstaller](#why-not-pyinstaller).

## Constraints that shaped the design

Three properties of the existing codebase rule out the obvious
approaches, and each one independently requires a real, writable
interpreter on the user's disk:

1. **Runtime type resolution.** `dw/arguments.py` and
   `dw/config_objects.py` resolve classes by name through `importlib`
   at workflow-load time — `"FluxPipeline"`, `"sdnq.SDNQConfig"`,
   `constant:` paths, community pipelines. A static analyzer cannot see
   any of it, so a frozen bundle would ship a hidden-imports list that
   must enumerate every diffusers pipeline, and would forfeit the
   documented property that new quantization backends work
   automatically via dynamic import.
2. **The spawn worker.** `dw/worker.py` uses `multiprocessing` with the
   `spawn` start method, which re-execs the interpreter. A real venv
   satisfies this natively.
3. **In-app diffusers upgrades.** `dw/server/updater.py` runs
   `sys.executable -m pip install --upgrade git+…/diffusers` against
   its own venv. That mechanism only works if the venv is genuine and
   writable.

### Why not PyInstaller

Freezing breaks all three. It was considered and rejected; a
provisioned venv preserves all three at the cost of a first-run
download, which users already accept because models are multi-gigabyte
downloads regardless.

## Architecture

### Topology

```
┌─ Tauri shell (Rust) ────────────────────────────────┐
│  provisioning UI  ·  process supervisor  ·  menus   │
│                                                      │
│   spawns ──► <data>/venv/bin/python -m dw.serve      │
│              --port 8765 --workflow-dir … etc        │
│                                                      │
│   webview ──► http://127.0.0.1:8765  (the SPA,       │
│               served by FastAPI, unmodified)         │
└──────────────────────────────────────────────────────┘
```

The shell owns provisioning, lifecycle, port allocation, crash
handling, logs and menus. It owns no application logic. Three consumers
— browser, desktop app, MCP client — stay on one server contract.

**Rejected alternative:** bundling `ui/dist` as Tauri's own frontend
and calling the API cross-origin. It buys native menus at the cost of
giving the SPA a second deployment target — relative vs. absolute API
base, CORS, CSP, SSE across origins — so every `ui/` change gains a
second way to break.

### Repository layout

A new top-level `desktop/`. **`ui/` is not modified.**

| Path | Responsibility |
| --- | --- |
| `desktop/src-tauri/src/paths.rs` | platform data dirs; first-run seeding |
| `desktop/src-tauri/src/gpu.rs` | `nvidia-smi` probe → torch index URL |
| `desktop/src-tauri/src/provision.rs` | venv creation, `uv` invocation, progress events |
| `desktop/src-tauri/src/server.rs` | spawn/health-poll/stop `dw.serve`; port selection; log capture |
| `desktop/src-tauri/src/menu.rs` | native menu; opens the auxiliary windows |
| `desktop/src/` | small Vite app: provisioning, Connect, Developer, Logs screens |
| `desktop/src-tauri/resources/` | python-build-standalone, `uv`, seed `workflows/` + `prompts/` |

### Window model

The main window has two navigation states: the shell page during
provisioning and startup, then `http://127.0.0.1:<port>` once healthy.

Because navigating to the server would orphan them, the **Connect**,
**Developer** and **Logs** panels are separate Tauri windows opened from
the native menu. This is what keeps `ui/` untouched instead of growing
desktop-only pages inside the SPA.

## Provisioning

### State machine

`detect → confirm accelerator → install → seed → ready`

Every state is **idempotent and resumable**. An interrupted install
leaves a venv the next launch detects as incomplete and repairs; a
"Repair installation" menu item forces the same path. Completion is
recorded by a marker file carrying the provisioned wheel version, not
by the mere existence of the venv directory.

### Install steps

```
uv venv <data>/venv --python <bundled interpreter>
uv pip install torch torchvision --index-url <chosen index>
uv pip install "diffusers-workflow[server]==<app version>"
```

Torch goes first so the resolver sees it satisfied, mirroring what
`install.sh` already does. Progress streams to the shell UI as Tauri
events.

`diffusers` is installed at its released version via the wheel's
`diffusers>=0.40.0` floor. It is **not** pinned to a git commit: the
in-app updater (`dw/server/updater.py`, the Models page) is the
supported route to git HEAD, and it continues to work unchanged.

### Accelerator selection

`gpu.rs` probes for an NVIDIA GPU and driver version and chooses the
matching pytorch.org index; the choice is shown in the UI with a
dropdown to override to CPU. macOS is unambiguous — the default PyPI
wheel, MPS. A machine with an NVIDIA card but no usable driver falls
back to CPU with an explanation rather than installing a CUDA build
that cannot run.

ROCm and Intel XPU are out of scope: they fall back to CPU, with the
manual upgrade documented.

### Seeding

The repo's `workflows/` and `prompts/` ship as Tauri bundle resources
and are copied into the user data root on first run. They are
deliberately absent from the wheel — which carries only the
`dw/workflows/` builtins — so the app bundle is the only place the
examples can come from. Seeding never overwrites an existing file.

## Paths and ports

| What | Location |
| --- | --- |
| venv, logs, provisioning marker | platform data dir (`~/Library/Application Support/diffusers-workflow`, `%LOCALAPPDATA%\diffusers-workflow`, `~/.local/share/diffusers-workflow`) |
| `workflows/`, `prompts/`, `outputs/` | `~/Documents/diffusers-workflow` |
| models | standard HF cache (unchanged) |
| settings, `jobs.sqlite` | `~/.diffusers_helper` (unchanged) |

The shell passes `--workflow-dir`, `--output-dir` and `--prompt-dir`
explicitly, so `dw/serve.py`'s cwd-relative defaults never apply and
**`dw/serve.py` requires no change**.

`server.rs` prefers port **8765** — binding and releasing it to test
availability — and falls back to an ephemeral port only if it is taken.
It then writes `~/.diffusers_helper/server.json`:

```json
{ "base_url": "http://127.0.0.1:8765", "port": 8765, "pid": 12345 }
```

## Changes to existing code

Everything above is additive. Three existing-code changes earn their
place:

### `dw_mcp/client.py`

`resolve_base_url()` gains a `server.json` lookup, ordered **after** an
explicit `--url` / `$DW_MCP_URL` and **before** the hardcoded
`http://127.0.0.1:8765` default. A stale file (no live server) falls
through to the default rather than failing.

This is what lets the generated MCP config carry no port at all and
still resolve when the app has fallen back to a different port.

### `dw/repl.py`

Probe `/api/health` at startup; when a server is already running, warn
that the REPL's own persistent worker will compete for the same VRAM.
A **warning, not a refusal** — pinning a step to CPU while the server
runs is a legitimate workflow.

### Docs

New `docs/DESKTOP.md`; updates to `README.md` and `docs/RELEASING.md`.

## MCP integration

`dw-mcp` stays **stdio-only**. No HTTP transport is added.

The Connect window renders a ready-made client config — the absolute
path to the venv's `dw-mcp`, with no port, per the resolution order
above — with a copy button and a "write into Claude Desktop's config"
action that **reads, merges and writes back** the existing
`mcpServers` map rather than replacing the file.

The same window carries the Developer section: the venv path and an
"Open terminal here" button that launches the platform terminal with
the venv activated. **The installer never modifies `PATH` or shell
configuration.**

## Versioning and updates

The Tauri app version tracks `pyproject.toml`, and the app pins
`diffusers-workflow==<its own version>`. One `v<semver>` tag produces
the wheel and all three installers, so `scripts/release.sh` keeps
driving releases unchanged.

Tauri's updater ships the shell. On the first launch after a shell
update, the app compares the installed wheel version against its own
and runs `uv pip install --upgrade` when they differ.

## CI and signing

A `desktop` job matrix, gated on tags like the existing `wheel` job:

| Runner | Artifact | Note |
| --- | --- | --- |
| `macos-14` | `.dmg` (arm64) | signed + notarized |
| `windows-latest` | `.msi` | unsigned initially |
| `ubuntu-22.04` | `.AppImage` | older glibc/webkit2gtk for the widest reach |

The `release` job grows to attach these alongside the wheel and sdist.

macOS signs and notarizes through Tauri's built-in support. A signing
identity already exists — this is exporting the certificate into
`APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_ID`,
`APPLE_PASSWORD` and `APPLE_TEAM_ID` secrets, not enrolling.

Windows ships unsigned with the SmartScreen click-through documented.
The workflow is written so that enabling Windows signing later means
adding secrets and a conditional step, not restructuring the job.

## Testing

**Rust unit tests** — path resolution per platform; the `nvidia-smi`
output parser against fixture strings including the no-driver and
no-GPU cases; port selection including the "8765 is taken" fallback;
the config-merge logic against an existing `mcpServers` map.

**Integration** — one CI test on the Linux runner that provisions
end-to-end against the CPU torch index and asserts the server answers
`/api/health`. Slow, so tags-only.

**Python** — ordinary pytest coverage for `resolve_base_url`'s
`server.json` precedence (including the stale-file fallthrough) and the
REPL's already-running warning.

## Non-goals

- No offline installer.
- No ROCm or Intel XPU detection — CPU fallback, documented manual upgrade.
- No Windows code signing initially.
- No `.deb` or `.rpm`.
- No MCP HTTP transport.
- No changes to `ui/`.
- No `PATH` or shell-configuration modification.
- No x86-64 macOS build.
