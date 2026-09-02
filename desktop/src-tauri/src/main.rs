// Hide the console window on a release build for Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    dw_desktop::run();
}
