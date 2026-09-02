//! Building the Python environment the shell then supervises.
//!
//! Nothing here is frozen or vendored: the installer carries an
//! interpreter and `uv`, and this module drives them to build a real,
//! writable virtual environment. That is a requirement, not a convenience -
//! the engine resolves pipeline and quantization classes by name through
//! `importlib` at runtime, `dw/worker.py` re-execs the interpreter under
//! the `spawn` start method, and the Models page upgrades diffusers by
//! running pip against its own venv. A frozen bundle breaks all three.

use std::io;
use std::path::{Path, PathBuf};

pub const MARKER_FILE_NAME: &str = "provisioned.json";
pub const PACKAGE_NAME: &str = "diffusers-workflow";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Marker {
    pub wheel_version: String,
}

pub fn read_marker(data_dir: &Path) -> Option<Marker> {
    let text = std::fs::read_to_string(data_dir.join(MARKER_FILE_NAME)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Record a finished install. Written last, so its presence means the whole
/// sequence succeeded.
pub fn write_marker(data_dir: &Path, version: &str) -> io::Result<()> {
    std::fs::create_dir_all(data_dir)?;
    let marker = Marker {
        wheel_version: version.to_string(),
    };
    std::fs::write(
        data_dir.join(MARKER_FILE_NAME),
        serde_json::to_vec_pretty(&marker)?,
    )
}

/// Whether the environment must be built before the server can start.
///
/// Keyed on the marker rather than on the venv directory existing, so an
/// install killed halfway leaves no marker and is repaired on next launch.
/// Any version difference qualifies, which makes a downgrade take exactly
/// the same path as an upgrade.
pub fn needs_provisioning(data_dir: &Path, app_version: &str) -> bool {
    match read_marker(data_dir) {
        Some(marker) => marker.wheel_version != app_version,
        None => true,
    }
}

/// The exact commands a provisioning run executes, in order.
///
/// Pure, so the sequence is testable without running anything. torch goes
/// in before the wheel for the same reason `install.sh` does it: the
/// resolver then sees it satisfied and leaves the accelerator-specific
/// build alone instead of replacing it with the default PyPI one.
pub fn install_commands(
    venv: &Path,
    uv: &Path,
    python: &Path,
    index_url: Option<&str>,
    version: &str,
) -> Vec<Vec<String>> {
    let uv = uv.to_string_lossy().to_string();
    let venv_s = venv.to_string_lossy().to_string();
    let venv_python = crate::paths::venv_python(venv)
        .to_string_lossy()
        .to_string();

    let create = vec![
        uv.clone(),
        "venv".into(),
        venv_s,
        "--python".into(),
        python.to_string_lossy().to_string(),
    ];

    let mut torch = vec![
        uv.clone(),
        "pip".into(),
        "install".into(),
        "--python".into(),
        venv_python.clone(),
        "torch".into(),
        "torchvision".into(),
    ];
    if let Some(index_url) = index_url {
        torch.push("--index-url".into());
        torch.push(index_url.into());
    }

    // An exact pin, not >=: the shell and the engine are released together,
    // so this is also what makes a downgrade work like an upgrade. pip
    // normalizes the project's semver pre-releases (0.4.0-alpha.10 becomes
    // 0.4.0a10) and an exact pin installs one without needing --pre.
    let wheel = vec![
        uv,
        "pip".into(),
        "install".into(),
        "--python".into(),
        venv_python,
        format!("{PACKAGE_NAME}[server]=={version}"),
    ];

    vec![create, torch, wheel]
}

/// Copy seed content into a user directory without ever overwriting.
///
/// The examples are a starting point, not managed content: once a file is
/// in the user's Documents folder it is theirs, and a later app version
/// must not quietly revert their edits.
pub fn seed_dir(source: &Path, destination: &Path) -> io::Result<usize> {
    std::fs::create_dir_all(destination)?;
    let mut copied = 0;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copied += seed_dir(&entry.path(), &target)?;
        } else if !target.exists() {
            std::fs::copy(entry.path(), &target)?;
            copied += 1;
        }
    }
    Ok(copied)
}

/// Where the bundled interpreter and `uv` live inside the app bundle.
pub fn bundled_uv(resource_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        resource_dir.join("uv.exe")
    } else {
        resource_dir.join("uv")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CU128: &str = "https://download.pytorch.org/whl/cu128";

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
    fn a_version_mismatch_reprovisions_in_both_directions() {
        let d = tempfile::tempdir().unwrap();
        write_marker(d.path(), "0.4.0a9").unwrap();
        assert!(needs_provisioning(d.path(), "0.4.0a10"), "upgrade");
        write_marker(d.path(), "0.5.0").unwrap();
        assert!(needs_provisioning(d.path(), "0.4.0a10"), "downgrade");
    }

    #[test]
    fn a_corrupt_marker_is_treated_as_no_marker() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(MARKER_FILE_NAME), "{not json").unwrap();
        assert!(needs_provisioning(d.path(), "0.4.0a10"));
    }

    #[test]
    fn install_commands_put_torch_before_the_wheel() {
        let cmds = install_commands(
            Path::new("/v"),
            Path::new("/uv"),
            Path::new("/py"),
            Some(CU128),
            "0.4.0a10",
        );
        assert_eq!(cmds.len(), 3);
        assert!(cmds[0].join(" ").contains("venv"));
        assert!(cmds[1].join(" ").contains("torch"));
        assert!(cmds[1].join(" ").contains("cu128"));
        assert!(cmds[2]
            .join(" ")
            .contains("diffusers-workflow[server]==0.4.0a10"));
    }

    #[test]
    fn no_index_url_is_passed_when_none_is_chosen() {
        let cmds = install_commands(
            Path::new("/v"),
            Path::new("/uv"),
            Path::new("/py"),
            None,
            "0.4.0a10",
        );
        assert!(!cmds[1].join(" ").contains("--index-url"));
        assert!(cmds[1].join(" ").contains("torch"));
    }

    #[test]
    fn every_install_targets_the_venv_not_the_bundled_interpreter() {
        let cmds = install_commands(
            Path::new("/v"),
            Path::new("/uv"),
            Path::new("/py"),
            None,
            "0.4.0a10",
        );
        for cmd in &cmds[1..] {
            assert!(cmd.contains(&"--python".to_string()));
            assert!(cmd.join(" ").contains("/v/"), "{cmd:?}");
        }
    }

    #[test]
    fn seeding_never_overwrites_an_existing_file() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("a.json"), "new").unwrap();
        std::fs::write(dst.path().join("a.json"), "mine").unwrap();
        assert_eq!(seed_dir(src.path(), dst.path()).unwrap(), 0);
        assert_eq!(
            std::fs::read_to_string(dst.path().join("a.json")).unwrap(),
            "mine"
        );
    }

    #[test]
    fn seeding_copies_missing_files() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("b.json"), "new").unwrap();
        assert_eq!(seed_dir(src.path(), dst.path()).unwrap(), 1);
        assert_eq!(
            std::fs::read_to_string(dst.path().join("b.json")).unwrap(),
            "new"
        );
    }

    #[test]
    fn seeding_recurses_into_subdirectories() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        std::fs::create_dir(src.path().join("sitcom")).unwrap();
        std::fs::write(src.path().join("sitcom").join("c.json"), "x").unwrap();
        assert_eq!(seed_dir(src.path(), dst.path()).unwrap(), 1);
        assert!(dst.path().join("sitcom").join("c.json").exists());
    }
}
