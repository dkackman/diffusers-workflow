//! Every decision the desktop shell makes, with no dependency on Tauri.
//!
//! The shell is a supervisor, not an application: it provisions a Python
//! virtual environment, starts `python -m dw.serve` in it, and points a
//! webview at the SPA that server already serves. The logic behind each of
//! those steps lives here so it can be tested without building a GUI.

pub mod connect;
pub mod gpu;
pub mod paths;
pub mod ports;
pub mod provision;
pub mod server;
