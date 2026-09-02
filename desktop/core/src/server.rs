//! Starting and watching `python -m dw.serve`.
//!
//! The shell never reimplements anything the server does. It starts the
//! same process a terminal would, waits for the same health endpoint, and
//! points a webview at the same SPA. Three consumers - the browser, this
//! app, and an MCP client - stay on one server contract.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::paths::{venv_python, Layout};

/// How many lines of server output the Logs window can show.
pub const LOG_LINES: usize = 500;

/// The command line for the server.
///
/// Every directory is passed explicitly. `dw/serve.py` defaults
/// `--workflow-dir` and `--output-dir` to paths relative to the working
/// directory, which is meaningless for a launched app, so pinning them here
/// is what lets that file stay unchanged.
pub fn serve_args(layout: &Layout, port: u16) -> Vec<String> {
    vec![
        "-m".into(),
        "dw.serve".into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--workflow-dir".into(),
        layout.workflows_dir().to_string_lossy().into_owned(),
        "--output-dir".into(),
        layout.outputs_dir().to_string_lossy().into_owned(),
        "--prompt-dir".into(),
        layout.prompts_dir().to_string_lossy().into_owned(),
    ]
}

/// A bounded ring of the most recent output lines.
#[derive(Default, Clone)]
pub struct LogBuffer(Arc<Mutex<Vec<String>>>);

impl LogBuffer {
    pub fn push(&self, line: String) {
        let mut lines = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if lines.len() >= LOG_LINES {
            lines.remove(0);
        }
        lines.push(line);
    }

    pub fn tail(&self, count: usize) -> String {
        let lines = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let start = lines.len().saturating_sub(count);
        lines[start..].join("\n")
    }
}

pub struct Supervisor {
    child: Child,
    pub port: u16,
    pub logs: LogBuffer,
    settings_dir: std::path::PathBuf,
}

impl Supervisor {
    pub fn spawn(layout: &Layout, port: u16) -> std::io::Result<Self> {
        let mut child = Command::new(venv_python(&layout.venv_dir))
            .args(serve_args(layout, port))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let logs = LogBuffer::default();
        for stream in [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let logs = logs.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    logs.push(line);
                }
            });
        }

        crate::ports::write_server_file(&layout.settings_dir, port, child.id())?;

        Ok(Self {
            child,
            port,
            logs,
            settings_dir: layout.settings_dir.clone(),
        })
    }

    /// Poll `/api/health` until it answers or the deadline passes.
    ///
    /// Startup is not instant - the engine imports torch before the first
    /// request is served - so this is what the provisioning screen waits on
    /// rather than assuming a spawned process is a ready one.
    pub fn wait_healthy(&mut self, probe: &dyn Fn(&str) -> bool, timeout: Duration) -> bool {
        let url = crate::ports::base_url(self.port);
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return false; // it exited; no amount of waiting will help
            }
            if probe(&url) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        false
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        crate::ports::remove_server_file(&self.settings_dir);
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Whether a venv looks provisioned enough to start.
pub fn venv_is_usable(venv_dir: &Path) -> bool {
    venv_python(venv_dir).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::layout_from;

    fn layout() -> Layout {
        layout_from("/d".into(), "/doc".into(), "/s".into())
    }

    #[test]
    fn serve_args_pin_every_directory_and_the_port() {
        let joined = serve_args(&layout(), 8765).join(" ");
        assert!(joined.contains("-m dw.serve"), "{joined}");
        assert!(joined.contains("--port 8765"), "{joined}");
        assert!(joined.contains("--workflow-dir /doc/workflows"), "{joined}");
        assert!(joined.contains("--output-dir /doc/outputs"), "{joined}");
        assert!(joined.contains("--prompt-dir /doc/prompts"), "{joined}");
    }

    #[test]
    fn serve_binds_localhost_only() {
        // This app serves the user's GPU to the user, not to their network
        let joined = serve_args(&layout(), 8765).join(" ");
        assert!(joined.contains("--host 127.0.0.1"), "{joined}");
    }

    #[test]
    fn no_directory_is_left_to_the_working_directory() {
        let args = serve_args(&layout(), 8765);
        for flag in ["--workflow-dir", "--output-dir", "--prompt-dir"] {
            let i = args.iter().position(|a| a == flag).expect(flag);
            assert!(args[i + 1].starts_with('/'), "{flag} must be absolute");
        }
    }

    #[test]
    fn the_log_buffer_keeps_only_the_most_recent_lines() {
        let logs = LogBuffer::default();
        for i in 0..(LOG_LINES + 50) {
            logs.push(format!("line {i}"));
        }
        let tail = logs.tail(LOG_LINES + 50);
        assert_eq!(tail.lines().count(), LOG_LINES);
        assert!(
            tail.starts_with("line 50"),
            "oldest lines should be dropped"
        );
    }

    #[test]
    fn the_log_tail_can_ask_for_fewer_lines_than_it_holds() {
        let logs = LogBuffer::default();
        for i in 0..10 {
            logs.push(format!("line {i}"));
        }
        assert_eq!(logs.tail(3).lines().count(), 3);
        assert!(logs.tail(3).starts_with("line 7"));
    }

    #[test]
    fn an_unprovisioned_venv_is_not_usable() {
        let d = tempfile::tempdir().unwrap();
        assert!(!venv_is_usable(d.path()));
    }
}
