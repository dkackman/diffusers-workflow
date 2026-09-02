//! Choosing a port and publishing it.
//!
//! 8765 is not just a default, it is a contract: `dw_mcp/client.py` falls
//! back to it, docs quote it, and a person who bookmarked the UI expects
//! it. So the shell asks for 8765 first and only accepts an ephemeral port
//! when something else already holds it.
//!
//! Whichever port it lands on is published to `server.json` in the settings
//! directory, which is what lets a generated MCP config carry no port at
//! all. The file is written atomically - a reader that catches it
//! mid-write would see truncated JSON and silently fall back to 8765,
//! which is exactly the wrong answer.

use std::io;
use std::net::TcpListener;
use std::path::Path;

pub const PREFERRED_PORT: u16 = 8765;
pub const SERVER_FILE_NAME: &str = "server.json";

/// Bind-and-release to find a usable port, preferring [`PREFERRED_PORT`].
///
/// There is an unavoidable race between releasing the port here and the
/// server binding it; it is narrow, and losing it surfaces as a normal
/// startup failure the shell already reports.
pub fn pick_port() -> io::Result<u16> {
    if let Ok(listener) = TcpListener::bind(("127.0.0.1", PREFERRED_PORT)) {
        drop(listener);
        return Ok(PREFERRED_PORT);
    }
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

pub fn base_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// Publish the live port. The three keys here are read by
/// `dw_mcp/client.py` and `dw/repl.py` - changing them breaks both.
pub fn write_server_file(settings_dir: &Path, port: u16, pid: u32) -> io::Result<()> {
    std::fs::create_dir_all(settings_dir)?;
    let body = serde_json::json!({
        "base_url": base_url(port),
        "port": port,
        "pid": pid,
    });
    let final_path = settings_dir.join(SERVER_FILE_NAME);
    let temp_path = settings_dir.join(format!("{SERVER_FILE_NAME}.tmp"));
    std::fs::write(&temp_path, serde_json::to_vec_pretty(&body)?)?;
    std::fs::rename(&temp_path, &final_path)
}

/// Withdraw the published port. A file that is already gone is success -
/// this runs on shutdown paths that must not fail.
pub fn remove_server_file(settings_dir: &Path) {
    let _ = std::fs::remove_file(settings_dir.join(SERVER_FILE_NAME));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_a_usable_port() {
        let port = pick_port().expect("a port should be available");
        assert!(port > 0);
        // Whatever it returned must actually be bindable now
        TcpListener::bind(("127.0.0.1", port)).expect("returned port should be free");
    }

    #[test]
    fn falls_back_when_the_preferred_port_is_taken() {
        // If 8765 is already held by something else on this machine the
        // bind fails and the test still exercises the fallback path.
        let hog = TcpListener::bind(("127.0.0.1", PREFERRED_PORT));
        let got = pick_port().expect("a port should be available");
        if hog.is_ok() {
            assert_ne!(got, PREFERRED_PORT, "should not hand out a held port");
        }
        drop(hog);
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
    fn creates_the_settings_directory_if_absent() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("does").join("not").join("exist");
        write_server_file(&nested, 8765, 1).unwrap();
        assert!(nested.join("server.json").exists());
    }

    #[test]
    fn leaves_no_temporary_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        write_server_file(dir.path(), 8765, 1).unwrap();
        assert!(!dir.path().join("server.json.tmp").exists());
    }

    #[test]
    fn removing_a_missing_file_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        remove_server_file(dir.path());
    }

    #[test]
    fn removing_deletes_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        write_server_file(dir.path(), 8765, 1).unwrap();
        remove_server_file(dir.path());
        assert!(!dir.path().join("server.json").exists());
    }
}
