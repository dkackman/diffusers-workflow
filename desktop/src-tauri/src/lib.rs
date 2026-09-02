//! The Tauri shell. Every decision lives in `dw_desktop_core`; this crate
//! only wires those decisions to windows, menus and child processes.

use tauri::Manager;

pub mod commands;
pub mod menu;
pub mod state;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(state::Shell::default())
        .menu(menu::build)
        .on_menu_event(|app, event| {
            let id = event.id().0.as_str();
            if id == "repair" {
                // Clearing the marker is all it takes; the next launch
                // rebuilds. Surfacing the panel keeps the user informed.
                let _ = menu::open_panel(app, "developer");
            } else {
                let _ = menu::open_panel(app, id);
            }
        })
        .on_window_event(|window, event| {
            // The server is a child process, not a daemon: closing the app
            // must not leave a GPU worker holding models.
            if let tauri::WindowEvent::Destroyed = event {
                if window.label() == "main" {
                    if let Some(shell) = window.app_handle().try_state::<state::Shell>() {
                        if let Some(mut supervisor) =
                            shell.server.lock().expect("server lock").take()
                        {
                            supervisor.stop();
                        }
                    }
                }
            }
        })
        .invoke_handler(commands::handlers())
        .run(tauri::generate_context!())
        .expect("error while running the diffusers-workflow shell");
}
