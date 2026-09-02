//! Which torch build to install.
//!
//! PyTorch's CUDA wheels bundle the CUDA runtime, so the only thing that
//! matters on the user's machine is the *driver* version - there is no need
//! for a CUDA toolkit install. CUDA 12.x runs on driver >= 525 (Linux) /
//! 528 (Windows) through minor-version compatibility; CUDA 13.x needs
//! >= 580 / 581.
//!
//! macOS never reaches the table: on Apple Silicon the plain PyPI wheel is
//! already the accelerated one (MPS), so "no extra index" is the correct
//! answer there and does not mean "no acceleration".

use std::process::Command;

pub const CU130: &str = "https://download.pytorch.org/whl/cu130";
pub const CU128: &str = "https://download.pytorch.org/whl/cu128";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Accelerator {
    Cuda {
        index_url: String,
        driver: String,
        name: String,
    },
    /// No extra index. `reason` is shown to the user, so it must explain
    /// itself without jargon.
    Cpu { reason: String },
}

/// Pull `(driver_version, gpu_name)` out of
/// `nvidia-smi --query-gpu=driver_version,name --format=csv,noheader`.
///
/// Returns None for every "there is no usable GPU here" shape: no output,
/// an NVML error message, or the `[N/A]` placeholders nvidia-smi prints
/// when the driver is present but not functioning.
pub fn parse_nvidia_smi(stdout: &str) -> Option<(String, String)> {
    let line = stdout.lines().find(|l| !l.trim().is_empty())?;
    let (driver, name) = line.split_once(',')?;
    let (driver, name) = (driver.trim(), name.trim());

    if driver.is_empty() || name.is_empty() || driver == "[N/A]" || name == "[N/A]" {
        return None;
    }
    // "580.65.06" -> 580. Anything that does not start with a number is an
    // error message, not a version.
    driver.split('.').next()?.parse::<u32>().ok()?;
    Some((driver.to_string(), name.to_string()))
}

/// The torch index for a driver version, or None when the driver is too old
/// for any CUDA build we ship.
pub fn index_for_driver(driver: &str) -> Option<&'static str> {
    let major: u32 = driver.split('.').next()?.parse().ok()?;
    // Windows driver numbering runs a few points ahead of Linux for the
    // same CUDA floor, so the higher of the two pairs is used on both -
    // erring toward the older, wider-compatibility wheel is harmless,
    // while erring the other way installs a wheel that cannot run.
    let (cuda13_floor, cuda12_floor) = if cfg!(windows) {
        (581, 528)
    } else {
        (580, 525)
    };
    if major >= cuda13_floor {
        Some(CU130)
    } else if major >= cuda12_floor {
        Some(CU128)
    } else {
        None
    }
}

/// Probe this machine. Never fails: everything unknown becomes CPU with an
/// explanation.
pub fn detect() -> Accelerator {
    if cfg!(target_os = "macos") {
        return Accelerator::Cpu {
            reason: "macOS uses the MPS backend, which ships in the standard PyTorch wheel".into(),
        };
    }
    detect_from(run_nvidia_smi().as_deref())
}

/// The decision, separated from the process call so it can be tested.
pub fn detect_from(stdout: Option<&str>) -> Accelerator {
    let Some(stdout) = stdout else {
        return Accelerator::Cpu {
            reason: "No NVIDIA GPU detected (nvidia-smi is not available)".into(),
        };
    };
    let Some((driver, name)) = parse_nvidia_smi(stdout) else {
        return Accelerator::Cpu {
            reason: "An NVIDIA driver was found but reported no usable GPU".into(),
        };
    };
    match index_for_driver(&driver) {
        Some(index_url) => Accelerator::Cuda {
            index_url: index_url.to_string(),
            driver,
            name,
        },
        None => Accelerator::Cpu {
            reason: format!(
                "{name} has driver {driver}, which is older than the 525 minimum \
                 for the CUDA builds of PyTorch. Update the NVIDIA driver and \
                 use Repair installation to switch to a GPU build."
            ),
        },
    }
}

fn run_nvidia_smi() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version,name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_normal_nvidia_smi_line() {
        let out = "580.65.06, NVIDIA GeForce RTX 4090\n";
        assert_eq!(
            parse_nvidia_smi(out),
            Some(("580.65.06".into(), "NVIDIA GeForce RTX 4090".into()))
        );
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
        assert_eq!(parse_nvidia_smi("   \n"), None);
    }

    #[test]
    fn maps_drivers_to_indexes() {
        // Floors differ by platform; assert against this platform's pair.
        let (cuda13, cuda12) = if cfg!(windows) {
            (581, 528)
        } else {
            (580, 525)
        };
        assert_eq!(index_for_driver(&format!("{cuda13}.65.06")), Some(CU130));
        assert_eq!(index_for_driver(&format!("{cuda12}.10")), Some(CU128));
        // One below the CUDA 12 floor is not merely a lower index - it is no index
        assert_eq!(index_for_driver(&format!("{}.82", cuda12 - 1)), None);
        assert_eq!(index_for_driver("470.82"), None);
        assert_eq!(index_for_driver("not-a-version"), None);
    }

    #[test]
    fn a_modern_driver_selects_a_cuda_build() {
        let acc = detect_from(Some("580.65.06, NVIDIA GeForce RTX 4090\n"));
        match acc {
            Accelerator::Cuda {
                index_url, name, ..
            } => {
                assert_eq!(index_url, CU130);
                assert_eq!(name, "NVIDIA GeForce RTX 4090");
            }
            other => panic!("expected CUDA, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_nvidia_smi_falls_back_to_cpu() {
        match detect_from(None) {
            Accelerator::Cpu { reason } => assert!(reason.contains("nvidia-smi")),
            other => panic!("expected CPU, got {other:?}"),
        }
    }

    #[test]
    fn an_old_driver_explains_itself_and_falls_back_to_cpu() {
        match detect_from(Some("470.82, NVIDIA GeForce GTX 1080\n")) {
            Accelerator::Cpu { reason } => {
                assert!(reason.contains("470"), "{reason}");
                assert!(reason.contains("Repair installation"), "{reason}");
            }
            other => panic!("expected CPU, got {other:?}"),
        }
    }
}
