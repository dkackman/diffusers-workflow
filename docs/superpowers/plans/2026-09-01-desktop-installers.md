# Desktop Installers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `diffusers-workflow` as signed/native installers for Windows, Linux and macOS (Apple Silicon), each carrying a Tauri 2 shell that provisions a real Python venv and supervises `dw.serve`.

**Architecture:** A thin installer bundles a python-build-standalone interpreter and `uv` as Tauri resources. On first launch the Rust shell provisions `<data>/venv`, installs the published wheel from PyPI, spawns `python -m dw.serve` on port 8765, and points its webview at the FastAPI-served SPA. Nothing is frozen; `ui/` is not modified.

**Tech Stack:** Rust (Tauri 2), TypeScript + Vite (shell screens only), Python 3.10–3.14, `uv`, GitHub Actions.

**Spec:** [../specs/2026-09-01-desktop-installers-design.md](../specs/2026-09-01-desktop-installers-design.md)

## Global Constraints

- **`ui/` is never modified.** Desktop-only screens live in `desktop/src/`, in separate Tauri windows.
- **`dw_mcp/` must never import `dw`.** Importing any `dw.*` submodule runs `dw/__init__.py` and pulls in torch; `tests/test_mcp_server.py::TestStartupWeight` guards this. Settings-directory logic must be duplicated in `dw_mcp/`, not imported.
- **`dw/serve.py` requires no change.** The shell passes `--workflow-dir`, `--output-dir` and `--prompt-dir` explicitly.
- **Settings root** is `$DIFFUSERS_HELPER_ROOT`, else `~/.diffusers_helper/` (matches `dw/settings.py:66`).
- **Port preference** is `8765`, falling back to an ephemeral port only when taken.
- **Default base URL** is `http://127.0.0.1:8765` (`dw_mcp/client.py:9`).
- **Python style:** `black` formatted; run `black --check dw dw_mcp tests` before every Python commit.
- **Rust style:** `cargo fmt --check` and `cargo clippy -- -D warnings` before every Rust commit.
- **Supported Python floor** is 3.10; the bundled interpreter is **3.12** (the version CI tests on).
- **No `PATH` or shell-configuration modification, ever.**

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `dw_mcp/client.py` (modify) | add `server.json` lookup to `resolve_base_url()` |
| `dw/repl.py` (modify) | warn when a server already holds the GPU |
| `desktop/src-tauri/src/paths.rs` | platform data dirs, seeding |
| `desktop/src-tauri/src/gpu.rs` | `nvidia-smi` probe → torch index URL |
| `desktop/src-tauri/src/ports.rs` | port selection + `server.json` writing |
| `desktop/src-tauri/src/provision.rs` | venv creation via `uv`, progress events |
| `desktop/src-tauri/src/server.rs` | spawn / health-poll / stop `dw.serve` |
| `desktop/src-tauri/src/connect.rs` | MCP client-config generation and merge |
| `desktop/src-tauri/src/menu.rs` | native menu, auxiliary windows |
| `desktop/src/` | Vite shell app: provisioning, Connect, Developer, Logs |
| `.github/workflows/ci.yml` (modify) | `desktop` job matrix; release attaches installers |
| `docs/DESKTOP.md` | user-facing install/troubleshooting doc |

Tasks 1–2 are Python and independently shippable. Tasks 3–8 are pure-function Rust
with real unit tests. Tasks 9–11 are UI and process wiring. Tasks 12–13 are release
and docs.

---

### Task 1: MCP client resolves the live server port

**Files:**
- Modify: `dw_mcp/client.py:9-56`
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `resolve_base_url(explicit=None) -> str` (unchanged signature);
  new module-level `SERVER_FILE_NAME = "server.json"` and
  `settings_dir() -> pathlib.Path`.

Resolution order becomes: explicit → `$DW_MCP_URL` → `server.json` → default.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_client.py — append

import json


def write_server_file(tmp_path, payload):
    (tmp_path / "server.json").write_text(json.dumps(payload))


def test_resolve_base_url_reads_the_server_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    write_server_file(tmp_path, {"base_url": "http://127.0.0.1:9001", "port": 9001})
    assert resolve_base_url(None) == "http://127.0.0.1:9001"


def test_explicit_value_beats_the_server_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    write_server_file(tmp_path, {"base_url": "http://127.0.0.1:9001"})
    assert resolve_base_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234"


def test_environment_beats_the_server_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DW_MCP_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    write_server_file(tmp_path, {"base_url": "http://127.0.0.1:9001"})
    assert resolve_base_url(None) == "http://127.0.0.1:9999"


def test_missing_server_file_falls_through_to_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    assert resolve_base_url(None) == "http://127.0.0.1:8765"


def test_unreadable_server_file_falls_through_to_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    (tmp_path / "server.json").write_text("{not json")
    assert resolve_base_url(None) == "http://127.0.0.1:8765"


def test_server_file_without_a_base_url_falls_through(tmp_path, monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    write_server_file(tmp_path, {"port": 9001})
    assert resolve_base_url(None) == "http://127.0.0.1:8765"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_client.py -k server_file -v`
Expected: FAIL — `resolve_base_url` still ignores the file and returns the default.

- [ ] **Step 3: Implement**

```python
# dw_mcp/client.py — replace resolve_base_url and add the two helpers above it

import json
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
SERVER_FILE_NAME = "server.json"


def settings_dir():
    """The ~/.diffusers_helper root, resolved the way dw.settings resolves it.

    Deliberately duplicated rather than imported: importing any dw.* module
    runs dw/__init__.py and pulls in torch, which this pure HTTP client has
    no use for (test_mcp_server.py::TestStartupWeight guards the boundary).
    """
    return Path(os.environ.get("DIFFUSERS_HELPER_ROOT") or "~/.diffusers_helper/").expanduser()


def _base_url_from_server_file():
    """The base URL the desktop shell recorded for the server it started,
    or None. Any unreadable or malformed file is simply absent - a stale
    file must never be an error, only a miss."""
    try:
        with open(settings_dir() / SERVER_FILE_NAME, encoding="utf-8") as handle:
            url = json.load(handle).get("base_url")
    except (OSError, ValueError):
        return None
    return url if isinstance(url, str) and url else None


def resolve_base_url(explicit=None):
    """Where dw.serve is: the explicit value, else DW_MCP_URL, else the port
    the desktop shell recorded in server.json, else the default port."""
    url = (
        explicit
        or os.environ.get("DW_MCP_URL")
        or _base_url_from_server_file()
        or DEFAULT_BASE_URL
    )
    return url.rstrip("/")
```

- [ ] **Step 4: Run the full MCP suite**

Run: `pytest tests/test_mcp_client.py -q && pytest tests/test_mcp_server.py -k StartupWeight -q`
Expected: PASS, including the boundary test that forbids importing `dw`.

- [ ] **Step 5: Format and commit**

```bash
black dw_mcp tests
git add dw_mcp/client.py tests/test_mcp_client.py
git commit -m "feat(mcp): resolve the base URL from the desktop shell's server.json"
```

---

### Task 2: REPL warns when a server already holds the GPU

**Files:**
- Modify: `dw/repl.py:271-294`
- Test: `tests/test_repl_ergonomics.py`

**Interfaces:**
- Consumes: `dw_mcp`-independent stdlib only.
- Produces: `dw.repl.running_server_url() -> str | None` and
  `dw.repl.warn_if_server_running(probe=None) -> str | None`
  (returns the URL it warned about, or None).

Rationale: the REPL starts its own persistent worker (`dw/repl_worker.py`), so
running it beside the desktop app puts two processes on one GPU. This is a
**warning, not a refusal** — pinning a step to CPU while the server runs is
legitimate.

Uses `urllib.request` from the stdlib, not `httpx` (an `mcp` extra) and not
`requests` (a runtime dep, but heavier than a one-shot localhost probe needs).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repl_ergonomics.py — append

from dw.repl import warn_if_server_running


def test_warns_when_a_server_answers(capsys, monkeypatch):
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", "/nonexistent")
    url = warn_if_server_running(probe=lambda base: True)
    assert url == "http://127.0.0.1:8765"
    out = capsys.readouterr().out
    assert "already running" in out
    assert "GPU memory" in out


def test_silent_when_nothing_answers(capsys, monkeypatch):
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", "/nonexistent")
    assert warn_if_server_running(probe=lambda base: False) is None
    assert capsys.readouterr().out == ""


def test_probe_failure_is_not_fatal(capsys, monkeypatch):
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", "/nonexistent")

    def boom(base):
        raise OSError("network down")

    assert warn_if_server_running(probe=boom) is None
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_repl_ergonomics.py -k server_running -v`
Expected: FAIL — `ImportError: cannot import name 'warn_if_server_running'`.

- [ ] **Step 3: Implement**

```python
# dw/repl.py — add above main()

import json
import urllib.request
from pathlib import Path

HEALTH_TIMEOUT_SECONDS = 0.5


def running_server_url():
    """The base URL a desktop shell recorded in server.json, else the
    default serve port."""
    root = Path(
        os.environ.get("DIFFUSERS_HELPER_ROOT") or "~/.diffusers_helper/"
    ).expanduser()
    try:
        with open(root / "server.json", encoding="utf-8") as handle:
            url = json.load(handle).get("base_url")
        if isinstance(url, str) and url:
            return url.rstrip("/")
    except (OSError, ValueError):
        pass
    return "http://127.0.0.1:8765"


def _health_probe(base_url):
    with urllib.request.urlopen(
        f"{base_url}/api/health", timeout=HEALTH_TIMEOUT_SECONDS
    ) as response:
        return response.status == 200


def warn_if_server_running(probe=None):
    """Warn - but do not refuse - when a dw.serve is already up. The REPL
    starts its own persistent worker, so the two would compete for the same
    GPU memory. Returns the URL warned about, else None."""
    base_url = running_server_url()
    try:
        if not (probe or _health_probe)(base_url):
            return None
    except Exception:
        return None

    print(f"\nNote: a diffusers-workflow server is already running at {base_url}.")
    print("The REPL loads models into its own worker process, so both will")
    print("hold GPU memory at once. Quit the other one first unless you mean")
    print("to run them side by side (for example, pinning a step to CPU).\n")
    return base_url
```

Then call it from `main()`, immediately after `startup(args.log_level)`:

```python
    startup(args.log_level)
    warn_if_server_running()
```

- [ ] **Step 4: Run the REPL suite**

Run: `pytest tests/test_repl_ergonomics.py -q && pytest -q -k repl`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black dw tests
git add dw/repl.py tests/test_repl_ergonomics.py
git commit -m "feat(repl): warn when a server already holds the GPU"
```

---

### Task 3: Scaffold the Tauri project

**Files:**
- Create: `desktop/package.json`, `desktop/vite.config.ts`, `desktop/index.html`,
  `desktop/src/main.ts`, `desktop/src-tauri/Cargo.toml`,
  `desktop/src-tauri/tauri.conf.json`, `desktop/src-tauri/build.rs`,
  `desktop/src-tauri/src/main.rs`, `desktop/src-tauri/src/lib.rs`
- Modify: `.gitignore`

**Interfaces:**
- Produces: a crate named `dw_desktop` whose `lib.rs` declares
  `pub mod paths; pub mod gpu; pub mod ports; pub mod provision; pub mod server; pub mod connect;`
  so later tasks add modules without touching `main.rs`.

- [ ] **Step 1: Create the npm side**

```json
// desktop/package.json
{
  "name": "dw-desktop",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "tauri": "tauri"
  },
  "devDependencies": {
    "@tauri-apps/cli": "^2.9.0",
    "typescript": "~6.0.2",
    "vite": "^8.2.2"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.9.0"
  }
}
```

- [ ] **Step 2: Create the crate**

```toml
# desktop/src-tauri/Cargo.toml
[package]
name = "dw_desktop"
version = "0.0.0"
edition = "2021"
rust-version = "1.77"

[lib]
name = "dw_desktop"
path = "src/lib.rs"

[[bin]]
name = "dw-desktop"
path = "src/main.rs"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
dirs = "6"

[dev-dependencies]
tempfile = "3"
```

```rust
// desktop/src-tauri/src/lib.rs
//! The desktop shell's logic, split so each module is unit-testable
//! without a running Tauri app.

pub mod connect;
pub mod gpu;
pub mod paths;
pub mod ports;
pub mod provision;
pub mod server;
```

```rust
// desktop/src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    dw_desktop::run();
}
```

Add `pub fn run()` to `lib.rs` once Task 10 wires the menu; until then it may
be `pub fn run() { tauri::Builder::default().run(tauri::generate_context!()).expect("tauri"); }`.

- [ ] **Step 3: Add build artifacts to .gitignore**

```
desktop/node_modules/
desktop/dist/
desktop/src-tauri/target/
desktop/src-tauri/resources/python/
desktop/src-tauri/resources/uv*
```

- [ ] **Step 4: Verify the crate compiles**

Run: `cd desktop/src-tauri && cargo check`
Expected: compiles (empty modules are fine as long as each file exists).

- [ ] **Step 5: Commit**

```bash
git add desktop .gitignore
git commit -m "build(desktop): scaffold the Tauri shell crate"
```

---

### Task 4: Platform paths

**Files:**
- Create: `desktop/src-tauri/src/paths.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct Layout { pub data_dir: PathBuf, pub venv_dir: PathBuf,
                      pub documents_dir: PathBuf, pub settings_dir: PathBuf }
  pub fn layout() -> Layout;
  pub fn layout_from(data: PathBuf, documents: PathBuf, settings: PathBuf) -> Layout;
  pub fn venv_python(venv_dir: &Path) -> PathBuf;
  pub fn venv_bin(venv_dir: &Path, exe: &str) -> PathBuf;
  ```

`venv_python` is `bin/python` on unix and `Scripts\python.exe` on Windows —
the one place that difference is encoded.

- [ ] **Step 1: Write the failing tests**

```rust
// desktop/src-tauri/src/paths.rs — #[cfg(test)] mod tests
#[test]
fn venv_python_is_platform_specific() {
    let venv = Path::new("/tmp/venv");
    let python = venv_python(venv);
    if cfg!(windows) {
        assert!(python.ends_with("Scripts/python.exe") || python.ends_with("Scripts\\python.exe"));
    } else {
        assert!(python.ends_with("bin/python"));
    }
}

#[test]
fn layout_from_places_the_venv_under_the_data_dir() {
    let l = layout_from("/d".into(), "/doc".into(), "/s".into());
    assert_eq!(l.venv_dir, PathBuf::from("/d/venv"));
    assert_eq!(l.documents_dir, PathBuf::from("/doc"));
}

#[test]
fn venv_bin_appends_exe_on_windows_only() {
    let got = venv_bin(Path::new("/v"), "dw-mcp");
    if cfg!(windows) {
        assert!(got.to_string_lossy().ends_with("dw-mcp.exe"));
    } else {
        assert!(got.to_string_lossy().ends_with("dw-mcp"));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd desktop/src-tauri && cargo test paths`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement**

```rust
use std::path::{Path, PathBuf};

pub const APP_DIR_NAME: &str = "diffusers-workflow";

pub struct Layout {
    pub data_dir: PathBuf,
    pub venv_dir: PathBuf,
    pub documents_dir: PathBuf,
    pub settings_dir: PathBuf,
}

pub fn layout_from(data: PathBuf, documents: PathBuf, settings: PathBuf) -> Layout {
    Layout { venv_dir: data.join("venv"), data_dir: data, documents_dir: documents, settings_dir: settings }
}

pub fn layout() -> Layout {
    let data = dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join(APP_DIR_NAME);
    let documents = dirs::document_dir().unwrap_or_else(|| data.clone()).join(APP_DIR_NAME);
    // Matches dw/settings.py:66 - the env override included
    let settings = std::env::var_os("DIFFUSERS_HELPER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".diffusers_helper"));
    layout_from(data, documents, settings)
}

pub fn venv_python(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) { venv_dir.join("Scripts").join("python.exe") }
    else { venv_dir.join("bin").join("python") }
}

pub fn venv_bin(venv_dir: &Path, exe: &str) -> PathBuf {
    let dir = if cfg!(windows) { venv_dir.join("Scripts") } else { venv_dir.join("bin") };
    if cfg!(windows) { dir.join(format!("{exe}.exe")) } else { dir.join(exe) }
}
```

- [ ] **Step 4: Run tests**

Run: `cd desktop/src-tauri && cargo test paths`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy -- -D warnings
git add desktop/src-tauri/src/paths.rs
git commit -m "feat(desktop): platform data-directory layout"
```

---

### Task 5: NVIDIA detection and torch index selection

**Files:**
- Create: `desktop/src-tauri/src/gpu.rs`

**Interfaces:**
- Produces:
  ```rust
  pub enum Accelerator { Cuda { index_url: String, driver: String, name: String }, Cpu { reason: String } }
  pub fn parse_nvidia_smi(stdout: &str) -> Option<(String, String)>;  // (driver, name)
  pub fn index_for_driver(driver: &str) -> Option<&'static str>;
  pub fn detect() -> Accelerator;
  ```

**Driver → index table.** PyTorch CUDA wheels bundle the CUDA runtime, so only
the driver floor matters. CUDA 12.x needs driver ≥ 525 (Linux) / 528 (Windows)
via minor-version compatibility; CUDA 13.x needs ≥ 580 / 581.

| Driver major | Index |
| --- | --- |
| ≥ 580 (Linux) / ≥ 581 (Windows) | `https://download.pytorch.org/whl/cu130` |
| ≥ 525 (Linux) / ≥ 528 (Windows) | `https://download.pytorch.org/whl/cu128` |
| anything lower, or no GPU | CPU |

macOS always returns `Cpu { reason: "macOS uses the MPS backend from the default PyPI wheel" }`
— on Apple Silicon the plain wheel *is* the accelerated one, so the CPU variant
here means "no extra index", not "no acceleration".

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn parses_a_normal_nvidia_smi_line() {
    let out = "580.65.06, NVIDIA GeForce RTX 4090\n";
    assert_eq!(parse_nvidia_smi(out),
        Some(("580.65.06".into(), "NVIDIA GeForce RTX 4090".into())));
}

#[test]
fn parses_the_first_gpu_when_several_are_present() {
    let out = "550.54.14, NVIDIA A100\n550.54.14, NVIDIA A100\n";
    assert_eq!(parse_nvidia_smi(out).unwrap().1, "NVIDIA A100");
}

#[test]
fn rejects_the_no_driver_output() {
    assert_eq!(parse_nvidia_smi("[N/A], [N/A]\n"), None);
    assert_eq!(parse_nvidia_smi(""), None);
    assert_eq!(parse_nvidia_smi("Failed to initialize NVML\n"), None);
}

#[test]
fn maps_drivers_to_indexes() {
    assert_eq!(index_for_driver("580.65.06"), Some("https://download.pytorch.org/whl/cu130"));
    assert_eq!(index_for_driver("550.54.14"), Some("https://download.pytorch.org/whl/cu128"));
    assert_eq!(index_for_driver("470.82"), None);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd desktop/src-tauri && cargo test gpu`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**

`parse_nvidia_smi` takes the first non-empty line, splits on `,`, trims, and
returns `None` when either field is empty or `[N/A]` or the driver's leading
component does not parse as a number. `detect()` runs
`nvidia-smi --query-gpu=driver_version,name --format=csv,noheader` and maps the
result through `index_for_driver`, returning a `Cpu` with a human-readable
`reason` when anything is missing.

- [ ] **Step 4: Run tests**

Run: `cd desktop/src-tauri && cargo test gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy -- -D warnings
git add desktop/src-tauri/src/gpu.rs
git commit -m "feat(desktop): detect NVIDIA driver and pick a torch index"
```

---

### Task 6: Port selection and `server.json`

**Files:**
- Create: `desktop/src-tauri/src/ports.rs`

**Interfaces:**
- Produces:
  ```rust
  pub const PREFERRED_PORT: u16 = 8765;
  pub fn pick_port() -> std::io::Result<u16>;
  pub fn write_server_file(settings_dir: &Path, port: u16, pid: u32) -> std::io::Result<()>;
  pub fn remove_server_file(settings_dir: &Path);
  ```

`write_server_file` must emit exactly the keys Task 1 and Task 2 read:
`base_url`, `port`, `pid`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn prefers_8765_when_free() {
    // Bind nothing; the preferred port is almost always free in CI.
    assert!(pick_port().is_ok());
}

#[test]
fn falls_back_when_the_preferred_port_is_taken() {
    let hog = std::net::TcpListener::bind(("127.0.0.1", PREFERRED_PORT));
    if let Ok(hog) = hog {
        let got = pick_port().unwrap();
        assert_ne!(got, PREFERRED_PORT);
        drop(hog);
    }
}

#[test]
fn writes_the_keys_the_python_side_reads() {
    let dir = tempfile::tempdir().unwrap();
    write_server_file(dir.path(), 9001, 42).unwrap();
    let text = std::fs::read_to_string(dir.path().join("server.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["base_url"], "http://127.0.0.1:9001");
    assert_eq!(v["port"], 9001);
    assert_eq!(v["pid"], 42);
}

#[test]
fn removing_a_missing_file_is_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    remove_server_file(dir.path());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd desktop/src-tauri && cargo test ports`
Expected: FAIL.

- [ ] **Step 3: Implement**

`pick_port` tries `TcpListener::bind(("127.0.0.1", PREFERRED_PORT))`; on success
it drops the listener and returns the port, otherwise it binds port `0` and
returns whatever the OS assigned. `write_server_file` creates the directory,
serializes the three keys, and writes atomically (write to `server.json.tmp`,
then `rename`) so a reader never sees a half-written file.

- [ ] **Step 4: Run tests**

Run: `cd desktop/src-tauri && cargo test ports`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy -- -D warnings
git add desktop/src-tauri/src/ports.rs
git commit -m "feat(desktop): port selection and server.json handshake"
```

---

### Task 7: MCP client-config generation and merge

**Files:**
- Create: `desktop/src-tauri/src/connect.rs`

**Interfaces:**
- Produces:
  ```rust
  pub fn mcp_server_entry(venv_dir: &Path) -> serde_json::Value;
  pub fn merge_mcp_config(existing: &str, entry: serde_json::Value) -> Result<String, serde_json::Error>;
  ```

`mcp_server_entry` carries **no port** — Task 1's `server.json` resolution is
what makes that safe.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn entry_names_the_venv_binary_and_no_port() {
    let v = mcp_server_entry(Path::new("/v"));
    let cmd = v["command"].as_str().unwrap();
    assert!(cmd.contains("dw-mcp"));
    assert!(v.get("env").is_none() || v["env"].get("DW_MCP_URL").is_none());
}

#[test]
fn merge_preserves_other_servers() {
    let existing = r#"{"mcpServers":{"other":{"command":"x"}}}"#;
    let merged = merge_mcp_config(existing, mcp_server_entry(Path::new("/v"))).unwrap();
    let v: serde_json::Value = serde_json::from_str(&merged).unwrap();
    assert_eq!(v["mcpServers"]["other"]["command"], "x");
    assert!(v["mcpServers"]["diffusers-workflow"].is_object());
}

#[test]
fn merge_preserves_unrelated_top_level_keys() {
    let existing = r#"{"theme":"dark","mcpServers":{}}"#;
    let merged = merge_mcp_config(existing, mcp_server_entry(Path::new("/v"))).unwrap();
    let v: serde_json::Value = serde_json::from_str(&merged).unwrap();
    assert_eq!(v["theme"], "dark");
}

#[test]
fn merge_creates_the_map_when_absent() {
    let merged = merge_mcp_config("{}", mcp_server_entry(Path::new("/v"))).unwrap();
    let v: serde_json::Value = serde_json::from_str(&merged).unwrap();
    assert!(v["mcpServers"]["diffusers-workflow"].is_object());
}

#[test]
fn merge_replaces_our_own_stale_entry() {
    let existing = r#"{"mcpServers":{"diffusers-workflow":{"command":"/old/dw-mcp"}}}"#;
    let merged = merge_mcp_config(existing, mcp_server_entry(Path::new("/new"))).unwrap();
    let v: serde_json::Value = serde_json::from_str(&merged).unwrap();
    assert!(!v["mcpServers"]["diffusers-workflow"]["command"].as_str().unwrap().contains("/old/"));
}

#[test]
fn empty_input_is_treated_as_an_empty_object() {
    assert!(merge_mcp_config("", mcp_server_entry(Path::new("/v"))).is_ok());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd desktop/src-tauri && cargo test connect`
Expected: FAIL.

- [ ] **Step 3: Implement**

`mcp_server_entry` returns `{"command": venv_bin(venv, "dw-mcp"), "args": []}`.
`merge_mcp_config` parses the existing text (treating empty/whitespace as `{}`),
ensures `mcpServers` is an object, inserts under the key `diffusers-workflow`,
and re-serializes with `to_string_pretty`.

- [ ] **Step 4: Run tests**

Run: `cd desktop/src-tauri && cargo test connect`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy -- -D warnings
git add desktop/src-tauri/src/connect.rs
git commit -m "feat(desktop): generate and merge the MCP client config"
```

---

### Task 8: Provisioning

**Files:**
- Create: `desktop/src-tauri/src/provision.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct Marker { pub wheel_version: String }
  pub fn read_marker(data_dir: &Path) -> Option<Marker>;
  pub fn write_marker(data_dir: &Path, version: &str) -> std::io::Result<()>;
  pub fn needs_provisioning(data_dir: &Path, app_version: &str) -> bool;
  pub fn install_commands(venv: &Path, uv: &Path, python: &Path, index: Option<&str>, version: &str) -> Vec<Vec<String>>;
  ```

Completion is recorded by `provisioned.json` carrying the wheel version — **not**
by the venv directory existing, so an interrupted install is detected and
repaired. `needs_provisioning` is true when the marker is missing or its version
differs from the app's, which makes upgrade and downgrade the same code path.

`install_commands` is a pure function returning the argv lists, so the exact
command sequence is unit-testable without running anything.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_fresh_dir_needs_provisioning() {
    let d = tempfile::tempdir().unwrap();
    assert!(needs_provisioning(d.path(), "0.4.0a10"));
}

#[test]
fn a_matching_marker_does_not() {
    let d = tempfile::tempdir().unwrap();
    write_marker(d.path(), "0.4.0a10").unwrap();
    assert!(!needs_provisioning(d.path(), "0.4.0a10"));
}

#[test]
fn a_version_mismatch_needs_reprovisioning_in_both_directions() {
    let d = tempfile::tempdir().unwrap();
    write_marker(d.path(), "0.4.0a9").unwrap();
    assert!(needs_provisioning(d.path(), "0.4.0a10"));
    write_marker(d.path(), "0.5.0").unwrap();
    assert!(needs_provisioning(d.path(), "0.4.0a10"));
}

#[test]
fn install_commands_put_torch_before_the_wheel() {
    let cmds = install_commands(Path::new("/v"), Path::new("/uv"), Path::new("/py"),
                               Some("https://download.pytorch.org/whl/cu128"), "0.4.0a10");
    assert_eq!(cmds.len(), 3);
    assert!(cmds[0].join(" ").contains("venv"));
    assert!(cmds[1].join(" ").contains("torch"));
    assert!(cmds[1].join(" ").contains("cu128"));
    assert!(cmds[2].join(" ").contains("diffusers-workflow[server]==0.4.0a10"));
}

#[test]
fn no_index_url_is_passed_when_none_is_chosen() {
    let cmds = install_commands(Path::new("/v"), Path::new("/uv"), Path::new("/py"), None, "0.4.0a10");
    assert!(!cmds[1].join(" ").contains("--index-url"));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd desktop/src-tauri && cargo test provision`
Expected: FAIL.

- [ ] **Step 3: Implement**

Three commands, mirroring `install.sh`'s ordering so the resolver leaves torch
alone: `uv venv <venv> --python <bundled>`, then
`uv pip install --python <venv python> torch torchvision [--index-url <index>]`,
then `uv pip install --python <venv python> "diffusers-workflow[server]==<version>"`.

- [ ] **Step 4: Run tests**

Run: `cd desktop/src-tauri && cargo test provision`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy -- -D warnings
git add desktop/src-tauri/src/provision.rs
git commit -m "feat(desktop): provisioning marker and install command sequence"
```

---

### Task 9: Server supervision

**Files:**
- Create: `desktop/src-tauri/src/server.rs`

**Interfaces:**
- Produces:
  ```rust
  pub fn serve_args(layout: &Layout, port: u16) -> Vec<String>;
  pub struct Supervisor { /* holds Child, log buffer */ }
  impl Supervisor {
      pub fn spawn(layout: &Layout, port: u16) -> std::io::Result<Self>;
      pub fn wait_healthy(&self, base_url: &str, timeout: Duration) -> bool;
      pub fn stop(&mut self);
      pub fn log_tail(&self, lines: usize) -> String;
  }
  ```

`serve_args` is pure and therefore the testable part; it must pass
`--workflow-dir`, `--output-dir` and `--prompt-dir` explicitly so
`dw/serve.py`'s cwd-relative defaults never apply.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn serve_args_pin_every_directory_and_the_port() {
    let l = layout_from("/d".into(), "/doc".into(), "/s".into());
    let args = serve_args(&l, 8765);
    let joined = args.join(" ");
    assert!(joined.contains("-m dw.serve"));
    assert!(joined.contains("--port 8765"));
    assert!(joined.contains("--workflow-dir /doc/workflows"));
    assert!(joined.contains("--output-dir /doc/outputs"));
    assert!(joined.contains("--prompt-dir /doc/prompts"));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd desktop/src-tauri && cargo test server`
Expected: FAIL.

- [ ] **Step 3: Implement**

`spawn` runs `venv_python(&layout.venv_dir)` with `serve_args`, capturing stdout
and stderr into a bounded ring buffer (last 500 lines) on a reader thread.
`wait_healthy` polls `GET {base_url}/api/health` every 250 ms until the timeout.
`stop` terminates the child and calls `ports::remove_server_file`.

- [ ] **Step 4: Run tests**

Run: `cd desktop/src-tauri && cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy -- -D warnings
git add desktop/src-tauri/src/server.rs
git commit -m "feat(desktop): supervise the dw.serve child process"
```

---

### Task 10: Shell UI and window wiring

**Files:**
- Create: `desktop/src/main.ts`, `desktop/src/provision.ts`, `desktop/src/connect.ts`, `desktop/src/styles.css`
- Modify: `desktop/src-tauri/src/lib.rs` (the `run()` function), `desktop/src-tauri/src/menu.rs`

The main window shows the provisioning screen, subscribes to
`provision://progress` events, and on `ready` navigates to
`http://127.0.0.1:<port>`. Connect, Developer and Logs are separate
`WebviewWindow`s opened from the native menu, each loading a distinct route of
the same Vite bundle — this is what keeps `ui/` untouched.

Tauri commands exposed: `detect_accelerator`, `start_provisioning`,
`server_status`, `mcp_config_json`, `write_mcp_config`, `open_terminal`,
`log_tail`, `repair_installation`.

- [ ] **Step 1: Build the frontend**

Run: `cd desktop && npm install && npm run build`
Expected: `desktop/dist/` produced.

- [ ] **Step 2: Run the app in dev**

Run: `cd desktop && npm run tauri dev`
Expected: window opens on the provisioning screen and detection completes.

- [ ] **Step 3: Commit**

```bash
git add desktop/src desktop/src-tauri/src/lib.rs desktop/src-tauri/src/menu.rs
git commit -m "feat(desktop): provisioning UI and auxiliary windows"
```

---

### Task 11: Bundle resources and seeding

**Files:**
- Modify: `desktop/src-tauri/tauri.conf.json`, `desktop/src-tauri/src/paths.rs`
- Create: `desktop/scripts/fetch-runtime.sh`, `desktop/scripts/fetch-runtime.ps1`

`fetch-runtime` downloads the python-build-standalone 3.12 archive and the `uv`
binary for the target triple into `desktop/src-tauri/resources/` (gitignored),
verifying a pinned SHA256 for each. CI runs it before `tauri build`.

Seeding copies bundled `workflows/` and `prompts/` into
`documents_dir`, **never overwriting an existing file**.

- [ ] **Step 1: Write the failing seeding test**

```rust
#[test]
fn seeding_never_overwrites_an_existing_file() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    std::fs::write(src.path().join("a.json"), "new").unwrap();
    std::fs::write(dst.path().join("a.json"), "mine").unwrap();
    seed_dir(src.path(), dst.path()).unwrap();
    assert_eq!(std::fs::read_to_string(dst.path().join("a.json")).unwrap(), "mine");
}

#[test]
fn seeding_copies_missing_files() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    std::fs::write(src.path().join("b.json"), "new").unwrap();
    seed_dir(src.path(), dst.path()).unwrap();
    assert_eq!(std::fs::read_to_string(dst.path().join("b.json")).unwrap(), "new");
}
```

- [ ] **Step 2: Run to verify failure, then implement `seed_dir`, then re-run**

Run: `cd desktop/src-tauri && cargo test seed`

- [ ] **Step 3: Commit**

```bash
git add desktop
git commit -m "feat(desktop): bundle the runtime and seed example workflows"
```

---

### Task 12: CI desktop job and release attachment

**Files:**
- Modify: `.github/workflows/ci.yml`

Add a `desktop` job mirroring the existing `wheel` job's tag gate
(`if: startsWith(github.ref, 'refs/tags/v') || github.event_name == 'workflow_dispatch'`),
`needs: [backend, ui]`, with matrix `macos-14` / `windows-latest` / `ubuntu-22.04`.
Ubuntu installs `libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
patchelf`. macOS sets `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`,
`APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`.

`release` gains `needs: [wheel, desktop]` and downloads both artifact sets.

The Tauri updater also needs configuring here, which no earlier task covers:
generate a minisign keypair with `npm run tauri signer generate`, store the
private key as `TAURI_SIGNING_PRIVATE_KEY` (and its password), put the public
key in `tauri.conf.json` under `plugins.updater.pubkey`, and point
`plugins.updater.endpoints` at the GitHub release feed
`https://github.com/dkackman/diffusers-workflow/releases/latest/download/latest.json`.
This key is independent of Apple code signing, so the unsigned Windows build
still auto-updates. The `desktop` job must upload `latest.json` alongside the
installers.

Also add a tags-only `desktop-provision-test` step on the Linux runner that runs
the provisioning command sequence against the CPU torch index and asserts
`/api/health` answers.

- [ ] **Step 1: Validate the workflow parses**

Run: `gh workflow view ci.yml` or `act -n` if available; at minimum
`python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"`.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build and release desktop installers"
```

---

### Task 13: Documentation

**Files:**
- Create: `docs/DESKTOP.md`
- Modify: `README.md`, `docs/RELEASING.md`, `CLAUDE.md`

`docs/DESKTOP.md` covers: what the installer does and does not download, the
first-run provisioning flow and how long it takes, where files land per
platform, the SmartScreen click-through on Windows, using the CLI/REPL from the
app's venv, connecting an MCP client, and repairing a failed install.

`CLAUDE.md` gains a short "Desktop Shell" section under Architecture, matching
the existing "Server & Web UI" and "MCP Server" sections in tone and depth.

Note: `tests/test_docs_links.py` exists — run it after editing docs.

- [ ] **Step 1: Write the docs, then verify links**

Run: `pytest tests/test_docs_links.py -q`
Expected: PASS.

- [ ] **Step 2: Commit**

```bash
git add docs/DESKTOP.md README.md docs/RELEASING.md CLAUDE.md
git commit -m "docs: desktop installer guide"
```
