"""docker/entrypoint.py - the diffusers overlay's lifecycle, the API token and
dw.serve's command line - and the pins between docker/ and the rest of the
repo. Subprocesses are faked: nothing here runs pip, a venv or docker."""

import importlib.util
import json
import re
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCKER = REPO / "docker"


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location(
        "dw_docker_entrypoint", DOCKER / "entrypoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRun:
    """Stands in for subprocess.run: records each command, creates the venv's
    interpreter when `-m venv` runs, and answers `probe` with the diffusers
    the overlay imports - `before` until pip has installed, `installed` after -
    or fails it when that is None."""

    def __init__(self, installed=None, fail_pip=False, before=None):
        self.installed = installed
        self.before = installed if before is None else before
        self.fail_pip = fail_pip
        self.pip_ran = False
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[-1] == "probe":
            installed = self.installed if self.pip_ran else self.before
            if installed is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="ImportError")
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(installed) + "\n", stderr=""
            )
        if "pip" in command:
            if self.fail_pip:
                return SimpleNamespace(returncode=1, stdout="", stderr="build failed")
            self.pip_ran = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def pip_targets(self):
        return [c[-1] for c in self.commands if c[1:4] == ["-m", "pip", "install"]]


IMAGE = {
    "build_id": "b1",
    "python": "3.13.9",
    "diffusers": {"version": "0.43.0.dev0", "commit": "c" * 40},
}
HEAD = {"version": "0.43.0.dev0", "commit": "a" * 40}


def environ_for(tmp_path, **extra):
    return {"DW_ENV_DIR": str(tmp_path / "env"), **extra}


def marker(entry, environ):
    return json.loads(entry.marker_path(environ).read_text())


class TestParseRef:
    def test_keywords(self, entry):
        assert entry.parse_ref(None) == ("head", None)
        assert entry.parse_ref(" HEAD ") == ("head", None)
        assert entry.parse_ref("image") == ("image", None)

    def test_commit(self, entry):
        assert entry.parse_ref("abc1234") == ("commit", "abc1234")

    def test_anything_else_refuses_to_start(self, entry):
        with pytest.raises(SystemExit):
            entry.parse_ref("main; rm -rf /")


class TestInstallTarget:
    def test_pinned_ref_wins(self, entry):
        assert entry.install_target(
            "commit", "abc1234", {"commit": "f" * 40}, None
        ) == {"commit": "abc1234"}

    def test_old_overlay_beats_marker(self, entry):
        """The Models page may have moved diffusers since the marker was written."""
        assert entry.install_target(
            "head", None, {"commit": "f" * 40}, {"commit": "e" * 40}
        ) == {"commit": "e" * 40}

    def test_marker_when_old_overlay_will_not_run(self, entry):
        assert entry.install_target("head", None, {"commit": "f" * 40}, None) == {
            "commit": "f" * 40
        }

    def test_held_release_reverts(self, entry):
        assert entry.install_target(
            "head", None, {"version": "0.42.0", "commit": None}, None
        ) == {"revert": True}

    def test_nothing_held_is_head(self, entry):
        assert entry.install_target("head", None, None, None) == {}


class TestPrepareOverlay:
    def test_image_ref_skips_the_overlay(self, entry, tmp_path):
        run = FakeRun(installed=HEAD)
        environ = environ_for(tmp_path, DW_DIFFUSERS_REF="image")
        assert entry.prepare_overlay(environ, IMAGE, run) == entry.BASE_PYTHON
        assert run.commands == []

    def test_first_start_installs_head(self, entry, tmp_path):
        run = FakeRun(installed=HEAD)
        environ = environ_for(tmp_path)
        python = entry.prepare_overlay(environ, IMAGE, run)
        assert python == str(entry.overlay_python(environ))
        assert "--system-site-packages" in run.commands[0]
        targets = run.pip_targets()
        assert targets[0] == "git+https://github.com/huggingface/diffusers"
        assert all(c[0] == python for c in run.commands if "pip" in c)
        assert marker(entry, environ) == {
            "build_id": "b1",
            "python": "3.13.9",
            "diffusers": HEAD,
        }

    def test_restart_installs_nothing(self, entry, tmp_path):
        environ = environ_for(tmp_path)
        entry.prepare_overlay(environ, IMAGE, FakeRun(installed=HEAD))
        run = FakeRun(installed=HEAD)
        assert entry.prepare_overlay(environ, IMAGE, run) == str(
            entry.overlay_python(environ)
        )
        assert run.pip_targets() == []

    def test_restart_records_a_models_page_update(self, entry, tmp_path):
        environ = environ_for(tmp_path)
        entry.prepare_overlay(environ, IMAGE, FakeRun(installed=HEAD))
        moved = {"version": "0.43.0.dev0", "commit": "e" * 40}
        entry.prepare_overlay(environ, IMAGE, FakeRun(installed=moved))
        assert marker(entry, environ)["diffusers"] == moved

    def test_new_image_reinstalls_the_held_commit_not_head(self, entry, tmp_path):
        environ = environ_for(tmp_path)
        entry.prepare_overlay(environ, IMAGE, FakeRun(installed=HEAD))
        run = FakeRun(installed=HEAD)
        entry.prepare_overlay(environ, {**IMAGE, "build_id": "b2"}, run)
        assert (
            run.pip_targets()[0]
            == "git+https://github.com/huggingface/diffusers@" + "a" * 40
        )
        assert marker(entry, environ)["build_id"] == "b2"

    def test_pinned_ref_moves_an_overlay_that_differs(self, entry, tmp_path):
        environ = environ_for(tmp_path)
        entry.prepare_overlay(environ, IMAGE, FakeRun(installed=HEAD))
        environ["DW_DIFFUSERS_REF"] = "b" * 40
        run = FakeRun(
            installed={"version": "0.43.0.dev0", "commit": "b" * 40}, before=HEAD
        )
        entry.prepare_overlay(environ, IMAGE, run)
        assert run.pip_targets()[0].endswith("@" + "b" * 40)

    def test_failed_install_serves_the_baked_diffusers(self, entry, tmp_path):
        environ = environ_for(tmp_path)
        run = FakeRun(installed=None, fail_pip=True)
        assert entry.prepare_overlay(environ, IMAGE, run) == entry.BASE_PYTHON
        # build_id None: the next start retries
        assert marker(entry, environ)["build_id"] is None

    def test_overlay_that_will_not_import_serves_the_baked_diffusers(
        self, entry, tmp_path
    ):
        environ = environ_for(tmp_path)
        assert (
            entry.prepare_overlay(environ, IMAGE, FakeRun(installed=None))
            == entry.BASE_PYTHON
        )


class TestToken:
    def test_environment_wins(self, entry, tmp_path):
        environ = {"DW_API_TOKEN": " xyz ", "DIFFUSERS_HELPER_ROOT": str(tmp_path)}
        assert entry.resolve_token(environ) == "xyz"
        assert not (tmp_path / entry.TOKEN_FILE_NAME).exists()

    def test_insecure_opt_out(self, entry, tmp_path):
        environ = {"DW_INSECURE_NO_TOKEN": "1", "DIFFUSERS_HELPER_ROOT": str(tmp_path)}
        assert entry.resolve_token(environ) is None

    def test_generated_once_private_and_kept(self, entry, tmp_path):
        environ = {"DIFFUSERS_HELPER_ROOT": str(tmp_path / "settings")}
        token = entry.resolve_token(environ)
        path = tmp_path / "settings" / entry.TOKEN_FILE_NAME
        assert re.fullmatch(r"[0-9a-f]{64}", token)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert entry.resolve_token(environ) == token

    def test_login_url_prints_the_saved_token_and_never_mints_one(
        self, entry, tmp_path, capsys
    ):
        environ = {"DIFFUSERS_HELPER_ROOT": str(tmp_path), "DW_HOST_PORT": "9000"}
        assert entry.login_url(environ) == 0
        assert capsys.readouterr().out == "http://localhost:9000/\n"
        assert not (tmp_path / entry.TOKEN_FILE_NAME).exists()
        token = entry.resolve_token(environ)
        entry.login_url(environ)
        assert capsys.readouterr().out == f"http://localhost:9000/?token={token}\n"

    def test_healthcheck_never_mints_one(self, entry, tmp_path, monkeypatch):
        environ = {"DIFFUSERS_HELPER_ROOT": str(tmp_path)}
        monkeypatch.setattr(
            entry.urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(OSError()),
        )
        assert entry.healthcheck(environ) == 1
        assert not (tmp_path / entry.TOKEN_FILE_NAME).exists()


class TestServe:
    def test_defaults_unless_named(self, entry):
        command = entry.serve_command("/py", ["--mcp"])
        assert command[:3] == ["/py", "-m", "dw.serve"]
        assert command[command.index("--host") + 1] == "0.0.0.0"
        assert command[-1] == "--mcp"
        command = entry.serve_command("/py", ["--host", "127.0.0.1", "--port", "9000"])
        assert command.count("--host") == 1 and command.count("--port") == 1

    def test_token_goes_in_the_environment_not_argv(self, entry, tmp_path):
        calls = []
        environ = environ_for(
            tmp_path,
            DW_DIFFUSERS_REF="image",
            DW_API_TOKEN="xyz",
            HOME=str(tmp_path / "home"),
        )
        entry.serve(
            ["--mcp"],
            environ,
            FakeRun(),
            lambda path, argv, env: calls.append((path, argv, env)),
        )
        path, argv, env = calls[0]
        assert path == entry.BASE_PYTHON
        assert env["DW_API_TOKEN"] == "xyz"
        assert "xyz" not in argv and "--token" not in argv

    def test_token_never_reaches_the_log(self, entry, tmp_path, capsys):
        """docker logs output gets pasted into bug reports, and the token
        outlives the container; login-url is how a person gets it."""
        environ = environ_for(
            tmp_path,
            DW_DIFFUSERS_REF="image",
            HOME=str(tmp_path / "home"),
            DIFFUSERS_HELPER_ROOT=str(tmp_path / "settings"),
        )
        entry.serve([], environ, FakeRun(), lambda *a: None)
        token = entry.saved_token(environ)
        captured = capsys.readouterr()
        assert token and token not in captured.err + captured.out
        assert "login-url" in captured.err


class TestPins:
    def test_node_follows_ui_nvmrc(self):
        nvmrc = (REPO / "ui" / ".nvmrc").read_text().strip().lstrip("v")
        assert f"FROM node:{nvmrc}-" in (DOCKER / "Dockerfile").read_text()

    def test_shared_volumes_mount_at_their_host_path(self):
        """Manifests and cache entries record absolute paths; a container and a
        native server share them only when the paths agree."""
        compose = (DOCKER / "compose.yaml").read_text()
        mounts = re.findall(r"^\s+- \$\{(\w+)[^}]*\}:(\S+)$", compose, re.MULTILINE)
        assert mounts
        for variable, target in mounts:
            if variable == "DW_ENV_DIR":
                # docker/.env reaches the container too: the host path must not
                assert target == "/data/env"
                assert re.search(r"^\s+DW_ENV_DIR: /data/env$", compose, re.MULTILINE)
            else:
                assert target == "${%s}" % variable
                assert re.search(
                    rf"^\s+{variable}: \${{{variable}}}$", compose, re.MULTILINE
                )

    def test_init_env_writes_every_variable_compose_requires(self):
        compose = (DOCKER / "compose.yaml").read_text()
        required = set(re.findall(r"\$\{(\w+):\?", compose))
        init_env = (DOCKER / "init-env.sh").read_text()
        assert required
        for variable in required:
            assert re.search(rf"^{variable}=", init_env, re.MULTILINE), variable
