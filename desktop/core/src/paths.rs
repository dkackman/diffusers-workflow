//! Where everything lives on each platform.
//!
//! Two roots, deliberately separate: machine-managed state (the virtual
//! environment, logs, the provisioning marker) goes in the platform data
//! directory, while the things a person opens in Finder or Explorer -
//! workflows, prompts, outputs - go somewhere visible under Documents.
//! Settings and the job database stay in `~/.diffusers_helper`, where the
//! CLI and REPL already keep them.

use std::path::{Path, PathBuf};

pub const APP_DIR_NAME: &str = "diffusers-workflow";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub data_dir: PathBuf,
    pub venv_dir: PathBuf,
    pub documents_dir: PathBuf,
    pub settings_dir: PathBuf,
}

impl Layout {
    pub fn workflows_dir(&self) -> PathBuf {
        self.documents_dir.join("workflows")
    }

    pub fn prompts_dir(&self) -> PathBuf {
        self.documents_dir.join("prompts")
    }

    pub fn outputs_dir(&self) -> PathBuf {
        self.documents_dir.join("outputs")
    }
}

/// Build a layout from explicit roots. Everything testable goes through
/// here; `layout()` only decides what the roots are.
pub fn layout_from(data: PathBuf, documents: PathBuf, settings: PathBuf) -> Layout {
    Layout {
        venv_dir: data.join("venv"),
        data_dir: data,
        documents_dir: documents,
        settings_dir: settings,
    }
}

/// The real layout for this machine.
pub fn layout() -> Layout {
    let data = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_DIR_NAME);
    let documents = dirs::document_dir()
        .unwrap_or_else(|| data.clone())
        .join(APP_DIR_NAME);
    // Mirrors dw/settings.py: $DIFFUSERS_HELPER_ROOT, else ~/.diffusers_helper
    let settings = std::env::var_os("DIFFUSERS_HELPER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".diffusers_helper")
        });
    layout_from(data, documents, settings)
}

/// The directory a venv puts its executables in. The only place this
/// platform difference is spelled out.
fn scripts_dir(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts")
    } else {
        venv_dir.join("bin")
    }
}

pub fn venv_python(venv_dir: &Path) -> PathBuf {
    venv_bin(venv_dir, "python")
}

/// The path to one of the venv's console scripts - `dw-mcp`, `dw-repl`.
pub fn venv_bin(venv_dir: &Path, exe: &str) -> PathBuf {
    let dir = scripts_dir(venv_dir);
    if cfg!(windows) {
        dir.join(format!("{exe}.exe"))
    } else {
        dir.join(exe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venv_python_is_platform_specific() {
        let python = venv_python(Path::new("/tmp/venv"));
        let shown = python.to_string_lossy().replace('\\', "/");
        if cfg!(windows) {
            assert!(shown.ends_with("Scripts/python.exe"), "{shown}");
        } else {
            assert!(shown.ends_with("bin/python"), "{shown}");
        }
    }

    #[test]
    fn layout_from_places_the_venv_under_the_data_dir() {
        let l = layout_from("/d".into(), "/doc".into(), "/s".into());
        assert_eq!(l.venv_dir, PathBuf::from("/d/venv"));
        assert_eq!(l.documents_dir, PathBuf::from("/doc"));
        assert_eq!(l.settings_dir, PathBuf::from("/s"));
    }

    #[test]
    fn user_facing_directories_hang_off_documents() {
        let l = layout_from("/d".into(), "/doc".into(), "/s".into());
        assert_eq!(l.workflows_dir(), PathBuf::from("/doc/workflows"));
        assert_eq!(l.prompts_dir(), PathBuf::from("/doc/prompts"));
        assert_eq!(l.outputs_dir(), PathBuf::from("/doc/outputs"));
    }

    #[test]
    fn venv_bin_appends_exe_on_windows_only() {
        let shown = venv_bin(Path::new("/v"), "dw-mcp")
            .to_string_lossy()
            .to_string();
        if cfg!(windows) {
            assert!(shown.ends_with("dw-mcp.exe"), "{shown}");
        } else {
            assert!(shown.ends_with("dw-mcp"), "{shown}");
        }
    }
}
