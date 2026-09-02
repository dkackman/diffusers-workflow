//! The command surface the shell's own screens call.
//!
//! Each one is a thin adapter over `dw_desktop_core`, where the decisions
//! and their tests live. Anything with real logic in it belongs there
//! instead.

use std::time::Duration;

use dw_desktop_core::{connect, gpu, paths, ports, provision, server};
use tauri::{Emitter, Manager, State};

use crate::state::Shell;

const HEALTH_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(serde::Serialize)]
pub struct Status {
    pub needs_provisioning: bool,
    pub running: bool,
    pub base_url: Option<String>,
    pub version: String,
}

#[tauri::command]
pub fn shell_status(shell: State<Shell>) -> Status {
    let server = shell.server.lock().expect("server lock");
    Status {
        needs_provisioning: provision::needs_provisioning(
            &shell.layout.data_dir,
            Shell::wheel_version(),
        ),
        running: server.is_some(),
        base_url: server.as_ref().map(|s| ports::base_url(s.port)),
        version: Shell::wheel_version().to_string(),
    }
}

#[tauri::command]
pub fn detect_accelerator() -> gpu::Accelerator {
    gpu::detect()
}

/// Build the environment, then start the server, reporting each phase.
#[tauri::command]
pub async fn start_provisioning(
    app: tauri::AppHandle,
    index_url: Option<String>,
) -> Result<String, String> {
    let shell = app.state::<Shell>();
    let resources = app
        .path()
        .resource_dir()
        .map_err(|e| format!("no resource directory: {e}"))?;

    let uv = provision::bundled_uv(&resources);
    let python = resources.join("python").join(if cfg!(windows) {
        "python.exe"
    } else {
        "bin/python3"
    });

    let commands = provision::install_commands(
        &shell.layout.venv_dir,
        &uv,
        &python,
        index_url.as_deref(),
        Shell::wheel_version(),
    );

    for (step, argv) in commands.iter().enumerate() {
        let _ = app.emit(
            "provision://progress",
            serde_json::json!({ "step": step + 1, "of": commands.len(), "command": argv }),
        );
        let output = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|e| format!("could not run {}: {e}", argv[0]))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
    }

    // Seed only after a successful install, so a failed run leaves nothing
    for (name, destination) in [
        ("workflows", shell.layout.workflows_dir()),
        ("prompts", shell.layout.prompts_dir()),
    ] {
        let source = resources.join("seed").join(name);
        if source.exists() {
            provision::seed_dir(&source, &destination).map_err(|e| e.to_string())?;
        }
    }
    std::fs::create_dir_all(shell.layout.outputs_dir()).map_err(|e| e.to_string())?;

    provision::write_marker(&shell.layout.data_dir, Shell::wheel_version())
        .map_err(|e| e.to_string())?;

    start_server(app.clone()).await
}

/// Start the server and return the URL once it answers.
#[tauri::command]
pub async fn start_server(app: tauri::AppHandle) -> Result<String, String> {
    let shell = app.state::<Shell>();
    if !server::venv_is_usable(&shell.layout.venv_dir) {
        return Err("The Python environment is not installed yet.".into());
    }

    let port = ports::pick_port().map_err(|e| e.to_string())?;
    let mut supervisor =
        server::Supervisor::spawn(&shell.layout, port).map_err(|e| e.to_string())?;

    let healthy = supervisor.wait_healthy(&health_probe, HEALTH_TIMEOUT);
    if !healthy {
        let logs = supervisor.logs.tail(30);
        supervisor.stop();
        return Err(format!("The server did not start.\n\n{logs}"));
    }

    let url = ports::base_url(port);
    *shell.server.lock().expect("server lock") = Some(supervisor);
    Ok(url)
}

/// Liveness by an actual `GET /api/health`.
///
/// A raw request over `std::net` rather than an HTTP client crate: one
/// localhost liveness check does not earn a dependency. Accepting the TCP
/// connection is not sufficient evidence - the port can be open while the
/// engine is still importing torch - so this insists on a 200.
fn health_probe(base_url: &str) -> bool {
    use std::io::{Read, Write};

    let Some(port) = base_url
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
    else {
        return false;
    };
    let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&address, Duration::from_millis(500))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    if stream
        .write_all(b"GET /api/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = String::new();
    let _ = stream.take(256).read_to_string(&mut response);
    response.starts_with("HTTP/1.") && response.contains(" 200 ")
}

#[tauri::command]
pub fn stop_server(shell: State<Shell>) {
    if let Some(mut supervisor) = shell.server.lock().expect("server lock").take() {
        supervisor.stop();
    }
}

#[tauri::command]
pub fn mcp_config_json(shell: State<Shell>) -> String {
    connect::standalone_config(&shell.layout.venv_dir)
}

/// Merge our entry into a client's config file, preserving everything else.
#[tauri::command]
pub fn write_mcp_config(shell: State<Shell>, path: String) -> Result<(), String> {
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let merged =
        connect::merge_mcp_config(&existing, connect::mcp_server_entry(&shell.layout.venv_dir))
            .map_err(|e| format!("{path} is not valid JSON, so it was left alone: {e}"))?;
    std::fs::write(&path, merged).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn venv_path(shell: State<Shell>) -> String {
    shell.layout.venv_dir.to_string_lossy().into_owned()
}

#[tauri::command]
pub fn documents_path(shell: State<Shell>) -> String {
    shell.layout.documents_dir.to_string_lossy().into_owned()
}

#[tauri::command]
pub fn log_tail(shell: State<Shell>, lines: usize) -> String {
    shell
        .server
        .lock()
        .expect("server lock")
        .as_ref()
        .map(|s| s.logs.tail(lines))
        .unwrap_or_else(|| "The server is not running.".into())
}

/// Forget the provisioning marker so the next launch rebuilds the venv.
#[tauri::command]
pub fn repair_installation(shell: State<Shell>) -> Result<(), String> {
    std::fs::remove_file(shell.layout.data_dir.join(provision::MARKER_FILE_NAME))
        .or_else(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(e)
            }
        })
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn open_terminal(shell: State<Shell>) -> Result<(), String> {
    let activate_dir = paths::venv_python(&shell.layout.venv_dir);
    let dir = activate_dir.parent().unwrap_or(&shell.layout.venv_dir);
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
            .args(["-a", "Terminal"])
            .arg(dir)
            .spawn()
    } else if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", "start", "cmd", "/K"])
            .arg(dir)
            .spawn()
    } else {
        std::process::Command::new("x-terminal-emulator")
            .current_dir(dir)
            .spawn()
    };
    result.map(|_| ()).map_err(|e| e.to_string())
}

pub fn handlers() -> impl Fn(tauri::ipc::Invoke) -> bool + Send + Sync + 'static {
    tauri::generate_handler![
        shell_status,
        detect_accelerator,
        start_provisioning,
        start_server,
        stop_server,
        mcp_config_json,
        write_mcp_config,
        venv_path,
        documents_path,
        log_tail,
        repair_installation,
        open_terminal
    ]
}
