//! The native menu, and the windows it opens.
//!
//! Once the main window navigates to the server, the shell's own screens
//! would be unreachable inside it - so Connect, Developer and Logs are
//! separate windows on distinct routes of the same bundle. That is what
//! lets the SPA in `ui/` stay entirely unaware that a desktop app exists.

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{Manager, Runtime, WebviewUrl, WebviewWindowBuilder};

pub fn build<R: Runtime>(app: &tauri::AppHandle<R>) -> tauri::Result<Menu<R>> {
    let connect = MenuItem::with_id(app, "connect", "Connect to Claude…", true, None::<&str>)?;
    let developer = MenuItem::with_id(app, "developer", "Developer…", true, None::<&str>)?;
    let logs = MenuItem::with_id(app, "logs", "Server Logs…", true, None::<&str>)?;
    let repair = MenuItem::with_id(app, "repair", "Repair Installation…", true, None::<&str>)?;

    let app_menu = Submenu::with_items(
        app,
        "diffusers-workflow",
        true,
        &[
            &connect,
            &developer,
            &logs,
            &PredefinedMenuItem::separator(app)?,
            &repair,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;
    Menu::with_items(app, &[&app_menu])
}

/// Open (or focus) one of the shell's auxiliary windows.
pub fn open_panel<R: Runtime>(app: &tauri::AppHandle<R>, id: &str) -> tauri::Result<()> {
    if let Some(window) = app.get_webview_window(id) {
        return window.set_focus();
    }
    let title = match id {
        "connect" => "Connect to Claude",
        "developer" => "Developer",
        _ => "Server Logs",
    };
    WebviewWindowBuilder::new(app, id, WebviewUrl::App(format!("{id}.html").into()))
        .title(title)
        .inner_size(720.0, 560.0)
        .build()?;
    Ok(())
}
