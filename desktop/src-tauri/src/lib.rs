//! The Tauri shell. Every decision lives in `dw_desktop_core`; this crate
//! only wires those decisions to windows, menus and child processes.

pub mod commands;
pub mod menu;
pub mod state;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(state::Shell::default())
        .menu(menu::build)
        .invoke_handler(commands::handlers())
        .run(tauri::generate_context!())
        .expect("error while running the diffusers-workflow shell");
}
