import json
import os
from pathlib import Path


class Settings:
    log_level: str = "WARNING"
    log_filename: str = "log/dw.log"
    log_to_console: bool = False


def load_settings():
    settings = Settings()
    try:
        with open(get_settings_full_path(), "r") as file:
            settings_dict = json.load(file)
    except FileNotFoundError:
        settings_dict = {}
    except json.JSONDecodeError:
        print("invalid settings file")
        settings_dict = {}

    settings.log_level = settings_dict.get("log_level", "WARNING")
    settings.log_filename = settings_dict.get("log_filename", "log/dw.log")
    settings.log_to_console = settings_dict.get("log_to_console", False)
    return settings


def save_settings(settings):
    settings_dict = settings.__dict__
    with open(get_settings_full_path(), "w") as file:
        json.dump(settings_dict, file, indent=2)


def settings_exist():
    return get_settings_full_path().is_file()


def resolve_path(path):
    full_path = get_settings_dir().joinpath(path)
    # make the directory if it doesn't exist
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


def get_settings_dir():
    dir_path = os.environ.get("DIFFUSERS_HELPER_ROOT") or "~/.diffusers_helper/"

    return Path(dir_path).expanduser()


def save_file(data, filename):
    with open(resolve_path(filename), "w") as file:
        json.dump(data, file, indent=2)


def get_settings_full_path():
    return resolve_path("settings.json")
