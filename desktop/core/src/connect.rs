//! The MCP client configuration the Connect window hands out.
//!
//! `dw-mcp` stays a stdio server, so the client - Claude Code, Claude
//! Desktop - spawns it. All it needs is the absolute path to the console
//! script in the provisioned venv.
//!
//! Deliberately no port and no `DW_MCP_URL`: `dw_mcp/client.py` reads the
//! live port out of `server.json`, so a config written today keeps working
//! after the shell has had to fall back to a different port. Baking a port
//! in here would be the one thing that makes this config go stale.

use serde_json::{Map, Value};
use std::path::Path;

/// The key this shell owns inside `mcpServers`.
pub const SERVER_KEY: &str = "diffusers-workflow";

pub fn mcp_server_entry(venv_dir: &Path) -> Value {
    serde_json::json!({
        "command": crate::paths::venv_bin(venv_dir, "dw-mcp").to_string_lossy(),
        "args": [],
    })
}

/// A whole config file, ready to show in the Connect window.
pub fn standalone_config(venv_dir: &Path) -> String {
    let config = serde_json::json!({
        "mcpServers": { SERVER_KEY: mcp_server_entry(venv_dir) }
    });
    serde_json::to_string_pretty(&config).expect("a literal object always serializes")
}

/// Insert our entry into a client's existing config, preserving everything
/// else in the file.
///
/// This runs against a file the user owns and other tools also write to, so
/// the rule is: touch exactly one key. Unrelated top-level settings and
/// every other MCP server survive untouched; only a previous entry of ours
/// is replaced, since a stale path to an old venv is worse than no entry.
pub fn merge_mcp_config(existing: &str, entry: Value) -> Result<String, serde_json::Error> {
    let mut root: Value = if existing.trim().is_empty() {
        Value::Object(Map::new())
    } else {
        serde_json::from_str(existing)?
    };

    if !root.is_object() {
        root = Value::Object(Map::new());
    }
    let object = root.as_object_mut().expect("just ensured an object");

    // A non-object mcpServers is a broken file, not a merge conflict -
    // replacing it is the only way to produce a usable config
    if !object
        .get("mcpServers")
        .map(Value::is_object)
        .unwrap_or(false)
    {
        object.insert("mcpServers".into(), Value::Object(Map::new()));
    }
    object
        .get_mut("mcpServers")
        .and_then(Value::as_object_mut)
        .expect("just ensured an object")
        .insert(SERVER_KEY.into(), entry);

    serde_json::to_string_pretty(&root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> Value {
        mcp_server_entry(Path::new("/v"))
    }

    fn merged(existing: &str) -> Value {
        serde_json::from_str(&merge_mcp_config(existing, entry()).unwrap()).unwrap()
    }

    #[test]
    fn entry_names_the_venv_binary_and_carries_no_port() {
        let v = entry();
        assert!(v["command"].as_str().unwrap().contains("dw-mcp"));
        assert!(v.get("env").is_none());
        assert!(!serde_json::to_string(&v).unwrap().contains("8765"));
    }

    #[test]
    fn merge_preserves_other_servers() {
        let v = merged(r#"{"mcpServers":{"other":{"command":"x"}}}"#);
        assert_eq!(v["mcpServers"]["other"]["command"], "x");
        assert!(v["mcpServers"][SERVER_KEY].is_object());
    }

    #[test]
    fn merge_preserves_unrelated_top_level_keys() {
        let v = merged(r#"{"theme":"dark","mcpServers":{}}"#);
        assert_eq!(v["theme"], "dark");
    }

    #[test]
    fn merge_creates_the_map_when_absent() {
        assert!(merged("{}")["mcpServers"][SERVER_KEY].is_object());
    }

    #[test]
    fn merge_replaces_our_own_stale_entry() {
        let v = merged(r#"{"mcpServers":{"diffusers-workflow":{"command":"/old/dw-mcp"}}}"#);
        let command = v["mcpServers"][SERVER_KEY]["command"].as_str().unwrap();
        assert!(!command.contains("/old/"), "{command}");
    }

    #[test]
    fn empty_input_is_treated_as_an_empty_object() {
        assert!(merged("")["mcpServers"][SERVER_KEY].is_object());
        assert!(merged("   \n")["mcpServers"][SERVER_KEY].is_object());
    }

    #[test]
    fn malformed_json_is_an_error_rather_than_a_silent_overwrite() {
        // The user's file is not ours to discard - the caller must be able
        // to tell them it could not be parsed
        assert!(merge_mcp_config("{not json", entry()).is_err());
    }

    #[test]
    fn a_non_object_mcp_servers_is_replaced() {
        assert!(merged(r#"{"mcpServers":[]}"#)["mcpServers"][SERVER_KEY].is_object());
    }

    #[test]
    fn standalone_config_is_valid_json_with_our_key() {
        let v: Value = serde_json::from_str(&standalone_config(Path::new("/v"))).unwrap();
        assert!(v["mcpServers"][SERVER_KEY]["command"]
            .as_str()
            .unwrap()
            .contains("dw-mcp"));
    }
}
