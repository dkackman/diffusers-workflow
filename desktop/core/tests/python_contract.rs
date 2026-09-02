//! The one test that proves the handshake actually works.
//!
//! `ports::write_server_file` and the two Python readers
//! (`dw_mcp.client.resolve_base_url`, `dw.repl.running_server_url`) are the
//! two halves of a file format, and each side's unit tests only prove its
//! own half. This runs the real Python against a file the real Rust wrote,
//! so a rename of any key fails here instead of in a user's install.
//!
//! Each half skips when its Python module cannot be imported, so this is
//! never the reason a checkout fails to build. The two halves have very
//! different costs: `dw_mcp` deliberately imports nothing from `dw`, so it
//! needs only httpx and the repo on `PYTHONPATH` - which is why CI can run
//! that half for real in a job with no torch in it. `dw.repl` pulls in the
//! whole engine, so its half runs where a full install already exists.

use std::path::PathBuf;
use std::process::Command;

/// The repo's development venv, if this checkout has one.
fn dev_python() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("DW_CONTRACT_PYTHON") {
        return Some(PathBuf::from(explicit));
    }
    // core/ -> desktop/ -> repo root
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let candidate = if cfg!(windows) {
        root.join("venv").join("Scripts").join("python.exe")
    } else {
        root.join("venv").join("bin").join("python")
    };
    candidate.exists().then_some(candidate)
}

/// Whether this Python can import the module a half needs.
///
/// A skipped test still reports `ok`, so an environment that quietly stops
/// being able to import anything would leave this file green while proving
/// nothing. `DW_CONTRACT_REQUIRE` (a comma-separated list of modules) turns
/// a skip into a failure, and CI sets it for the half it intends to run.
fn can_import(python: &PathBuf, module: &str) -> bool {
    let importable = Command::new(python)
        .args(["-c", &format!("import {module}")])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !importable && required(module) {
        panic!(
            "DW_CONTRACT_REQUIRE names {module}, but {} cannot import it - \
             this half of the contract would have silently skipped",
            python.display()
        );
    }
    importable
}

fn required(module: &str) -> bool {
    std::env::var("DW_CONTRACT_REQUIRE")
        .map(|v| v.split(',').any(|m| m.trim() == module))
        .unwrap_or(false)
}

fn read_back(python: &PathBuf, settings_dir: &std::path::Path, code: &str) -> String {
    let output = Command::new(python)
        .args(["-c", code])
        .env("DIFFUSERS_HELPER_ROOT", settings_dir)
        .env_remove("DW_MCP_URL")
        .output()
        .expect("python should run");
    assert!(
        output.status.success(),
        "python failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

#[test]
fn the_mcp_client_reads_the_port_the_shell_wrote() {
    let Some(python) = dev_python() else {
        eprintln!("skipping: no Python available");
        return;
    };
    if !can_import(&python, "dw_mcp.client") {
        eprintln!("skipping: dw_mcp is not importable by {}", python.display());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    dw_desktop_core::ports::write_server_file(dir.path(), 9123, 4242).unwrap();

    let got = read_back(
        &python,
        dir.path(),
        "from dw_mcp.client import resolve_base_url; print(resolve_base_url(None))",
    );
    assert_eq!(got, "http://127.0.0.1:9123");
}

#[test]
fn the_repl_reads_the_port_the_shell_wrote() {
    let Some(python) = dev_python() else {
        eprintln!("skipping: no Python available");
        return;
    };
    // dw.repl imports the engine, and therefore torch - only a full install
    if !can_import(&python, "dw.repl") {
        eprintln!("skipping: dw is not importable by {}", python.display());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    dw_desktop_core::ports::write_server_file(dir.path(), 9124, 4243).unwrap();

    let got = read_back(
        &python,
        dir.path(),
        "from dw.repl import running_server_url; print(running_server_url())",
    );
    assert_eq!(got, "http://127.0.0.1:9124");
}
