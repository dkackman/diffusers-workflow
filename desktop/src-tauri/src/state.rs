//! What the shell knows between commands.

use std::sync::Mutex;

use dw_desktop_core::paths::{layout, Layout};
use dw_desktop_core::server::Supervisor;

pub struct Shell {
    pub layout: Layout,
    pub server: Mutex<Option<Supervisor>>,
}

impl Default for Shell {
    fn default() -> Self {
        Self {
            layout: layout(),
            server: Mutex::new(None),
        }
    }
}

impl Shell {
    /// The version of the wheel this shell expects, which is its own.
    pub fn wheel_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}
