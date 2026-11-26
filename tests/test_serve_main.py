"""dw.serve's argument handling, without starting uvicorn."""

import os

import pytest


@pytest.fixture
def serve(monkeypatch, tmp_path):
    """dw.serve.main with uvicorn and the app factory replaced, so a call
    returns what it would have served instead of serving it."""
    import uvicorn

    import dw.serve as serve_module
    from dw.server import app as app_module

    calls = {}

    def fake_create_app(**kwargs):
        calls["create_app"] = kwargs
        return object()

    def fake_run(app, **kwargs):
        calls["uvicorn"] = kwargs

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("DW_API_TOKEN", raising=False)
    # main() pins DW_PROMPT_DIR in os.environ; monkeypatch restores it
    monkeypatch.setenv("DW_PROMPT_DIR", str(tmp_path / "prompts"))
    # And a workspace of its own: without one main() resolves the working
    # directory, which for the test suite is the checkout - and then creates
    # the asset library inside it
    monkeypatch.setenv("DW_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("DW_WORKSPACE_SOURCE", "flag")
    (tmp_path / "workflows").mkdir()

    def run(*argv):
        monkeypatch.setattr(
            "sys.argv",
            ["dw-serve", "--workflow-dir", str(tmp_path / "workflows"), *argv],
        )
        serve_module.main()
        return calls

    return run


def test_mcp_is_off_by_default(serve):
    calls = serve()
    assert calls["create_app"]["mcp"] is False
    assert calls["create_app"]["port"] == 8765


def test_mcp_flag_is_passed_through(serve, capsys):
    calls = serve("--mcp", "--port", "9000", "--token", "t")
    assert calls["create_app"]["mcp"] is True
    assert calls["create_app"]["port"] == 9000
    # the banner tells the operator where the MCP endpoint is
    assert "/mcp" in capsys.readouterr().out


def test_mcp_on_a_non_loopback_bind_requires_a_token(serve, capsys):
    with pytest.raises(SystemExit) as exit_info:
        serve("--mcp", "--host", "0.0.0.0")
    assert exit_info.value.code == 2
    assert "--token" in capsys.readouterr().err


def test_the_refusal_comes_before_the_worker_is_started(serve, monkeypatch):
    """The point of a hard error is that nothing has happened yet - no
    startup(), no spawned worker to leave behind."""
    import dw

    def fail(*args, **kwargs):
        raise AssertionError("startup() ran before the --mcp check")

    monkeypatch.setattr(dw, "startup", fail)
    with pytest.raises(SystemExit):
        serve("--mcp", "--host", "0.0.0.0")


def test_mcp_on_loopback_needs_no_token(serve):
    calls = serve("--mcp")
    assert calls["create_app"]["mcp"] is True


def test_a_non_loopback_bind_without_mcp_is_only_warned_about(serve):
    calls = serve("--host", "0.0.0.0")
    assert calls["create_app"]["mcp"] is False
    assert calls["uvicorn"]["host"] == "0.0.0.0"


def test_shutdown_has_a_grace_period_so_an_open_mcp_connection_cannot_block_it(serve):
    """#477: uvicorn otherwise waits forever on SIGTERM for a connected
    streamable-HTTP MCP client to disconnect."""
    calls = serve()
    assert calls["uvicorn"]["timeout_graceful_shutdown"] == 5


@pytest.fixture
def two_cards(monkeypatch):
    """A box with two CUDA cards, whatever the machine running the test has."""
    import torch

    names = {0: "NVIDIA GeForce RTX 4090", 1: "NVIDIA GeForce RTX 3090"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: names[index])


@pytest.fixture
def one_card(monkeypatch):
    """A box that had two cards and now has one - lem with a card pulled."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index=0: "NVIDIA GeForce RTX 3090"
    )


@pytest.fixture
def settings_devices(monkeypatch):
    """Sets what the `devices` setting holds."""
    import importlib
    import types

    # `dw.settings` the attribute is a Settings instance; the module is
    # what configure_devices imports load_settings from
    settings_module = importlib.import_module("dw.settings")

    def set_value(value):
        monkeypatch.setattr(
            settings_module,
            "load_settings",
            lambda: types.SimpleNamespace(devices=value),
        )

    set_value(None)
    return set_value


class TestResolveServeDevices:
    def test_two_entries_are_both_kept(self, two_cards):
        from dw.devices import resolve_serve_devices

        assert resolve_serve_devices("cuda:0,cuda:1", None) == ["cuda:0", "cuda:1"]

    def test_a_card_named_twice_is_refused(self, two_cards):
        from dw.devices import DeviceConfigError, resolve_serve_devices

        with pytest.raises(DeviceConfigError, match="more than once"):
            resolve_serve_devices("cuda:0,cuda:0", None)

    def test_a_bare_cuda_beside_another_card_is_refused(self, two_cards):
        from dw.devices import DeviceConfigError, resolve_serve_devices

        with pytest.raises(DeviceConfigError, match="not a bare 'cuda'"):
            resolve_serve_devices("cuda,cuda:1", None)

    def test_a_card_the_machine_lacks_is_refused_naming_the_cards_present(
        self, two_cards
    ):
        from dw.devices import DeviceConfigError, resolve_serve_devices

        with pytest.raises(DeviceConfigError) as refusal:
            resolve_serve_devices("cuda:7", None)
        message = str(refusal.value)
        assert "cuda:0 (NVIDIA GeForce RTX 4090)" in message
        assert "cuda:1 (NVIDIA GeForce RTX 3090)" in message

    def test_a_card_gone_from_a_list_is_dropped_and_reported(self, one_card):
        from dw.devices import resolve_serve_devices

        absent = []
        devices = resolve_serve_devices("cuda:0,cuda:1", None, on_absent=absent.append)
        assert devices == ["cuda:0"]
        assert len(absent) == 1
        assert "There is no cuda:1" in absent[0]
        assert "cuda:0 (NVIDIA GeForce RTX 3090)" in absent[0]

    def test_a_list_with_no_card_present_is_refused(self, one_card):
        from dw.devices import DeviceConfigError, resolve_serve_devices

        with pytest.raises(DeviceConfigError, match="There is no cuda:1"):
            resolve_serve_devices("cuda:1,cuda:2", None, on_absent=lambda _: None)

    @pytest.mark.parametrize("value", [None, ""])
    def test_naming_nothing_is_none(self, value):
        from dw.devices import resolve_serve_devices

        assert resolve_serve_devices(value, None) is None
        assert resolve_serve_devices(value, "") is None

    def test_a_present_card_is_returned(self, two_cards):
        from dw.devices import resolve_serve_devices

        assert resolve_serve_devices("cuda:1", None) == ["cuda:1"]

    def test_the_cli_value_wins_over_the_setting(self, two_cards):
        from dw.devices import resolve_serve_devices

        assert resolve_serve_devices("cuda:1", "cuda:0") == ["cuda:1"]

    def test_the_setting_is_used_when_the_cli_names_nothing(self, two_cards):
        from dw.devices import resolve_serve_devices

        assert resolve_serve_devices(None, "cuda:0") == ["cuda:0"]


class TestConfigureDevices:
    def test_a_refusal_exits_with_code_2(
        self, two_cards, settings_devices, monkeypatch, capsys
    ):
        import types

        from dw.serve import configure_devices

        # setenv first so monkeypatch records the original state and undoes
        # whatever configure_devices assigns
        monkeypatch.setenv("DW_DEVICE", "unset")
        monkeypatch.delenv("DW_DEVICE")
        with pytest.raises(SystemExit) as exit_info:
            configure_devices(types.SimpleNamespace(devices="cuda:7"))
        assert exit_info.value.code == 2
        assert "cuda:1 (NVIDIA GeForce RTX 3090)" in capsys.readouterr().err
        assert "DW_DEVICE" not in os.environ

    def test_a_pulled_card_in_the_setting_starts_on_the_rest_and_warns(
        self, one_card, settings_devices, monkeypatch, capsys
    ):
        import types

        from dw.serve import configure_devices

        # setenv first so monkeypatch records the original state and undoes
        # whatever configure_devices assigns
        monkeypatch.setenv("DW_DEVICE", "unset")
        monkeypatch.delenv("DW_DEVICE")
        settings_devices("cuda:0,cuda:1")
        devices = configure_devices(types.SimpleNamespace(devices=None))
        assert devices == ["cuda:0"]
        assert os.environ["DW_DEVICE"] == "cuda:0"
        err = capsys.readouterr().err
        assert "There is no cuda:1" in err
        assert "cuda:0" in err

    def test_success_sets_dw_device(self, two_cards, settings_devices, monkeypatch):
        import types

        from dw.serve import configure_devices

        # setenv first so monkeypatch records the original state and undoes
        # whatever configure_devices assigns
        monkeypatch.setenv("DW_DEVICE", "unset")
        monkeypatch.delenv("DW_DEVICE")
        configure_devices(types.SimpleNamespace(devices="cuda:1"))
        assert os.environ["DW_DEVICE"] == "cuda:1"

    def test_the_setting_is_used_when_the_flag_is_absent(
        self, two_cards, settings_devices, monkeypatch
    ):
        import types

        from dw.serve import configure_devices

        # setenv first so monkeypatch records the original state and undoes
        # whatever configure_devices assigns
        monkeypatch.setenv("DW_DEVICE", "unset")
        monkeypatch.delenv("DW_DEVICE")
        settings_devices("cuda:0")
        configure_devices(types.SimpleNamespace(devices=None))
        assert os.environ["DW_DEVICE"] == "cuda:0"

    def test_naming_nothing_leaves_dw_device_alone(self, settings_devices, monkeypatch):
        import types

        from dw.serve import configure_devices

        # setenv first so monkeypatch records the original state and undoes
        # whatever configure_devices assigns
        monkeypatch.setenv("DW_DEVICE", "unset")
        monkeypatch.delenv("DW_DEVICE")
        configure_devices(types.SimpleNamespace(devices=None))
        assert "DW_DEVICE" not in os.environ


def test_managed_is_passed_to_the_app_and_off_by_default(serve):
    assert serve()["create_app"]["managed"] is False
    assert serve("--managed")["create_app"]["managed"] is True


@pytest.mark.parametrize("flag", [True, False])
def test_the_app_records_managed_on_its_state(tmp_path, flag):
    from dw.server.app import create_app
    from dw.server.jobs import JobManager

    from .test_server import ScriptedWorkerManager, success_script

    (tmp_path / "workflows").mkdir()
    manager = JobManager(
        str(tmp_path / "out"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(tmp_path / "out"),
        job_manager=manager,
        prompt_dir=str(tmp_path / "prompts"),
        managed=flag,
    )
    assert app.state.managed is flag


def test_install_root_flag_sets_the_environment(serve, monkeypatch, tmp_path):
    # setenv first so monkeypatch restores whatever main() assigns
    monkeypatch.setenv("DW_INSTALL_ROOT", "unset")
    serve("--install-root", str(tmp_path / "root"))
    assert os.environ["DW_INSTALL_ROOT"] == str(tmp_path / "root")


def test_the_packaged_catalog_is_the_default_examples_dir(serve, monkeypatch, tmp_path):
    import dw.serve as serve_module

    packaged = tmp_path / "pkg" / "examples" / "workflows"
    packaged.mkdir(parents=True)
    monkeypatch.setattr(serve_module, "packaged_examples_dir", lambda: str(packaged))
    assert serve()["create_app"]["examples_dirs"] == [str(packaged)]
    given = serve("--examples-dir", str(tmp_path / "mine"))["create_app"]
    assert given["examples_dirs"] == [str(tmp_path / "mine")]


def test_packaged_examples_dir_is_an_install_only_default(tmp_path):
    from dw.serve import packaged_examples_dir

    installed = tmp_path / "site-packages" / "dw"
    assert packaged_examples_dir(installed) is None  # no catalog shipped
    (installed / "examples" / "workflows").mkdir(parents=True)
    assert packaged_examples_dir(installed) == str(installed / "examples" / "workflows")
    checkout = tmp_path / "checkout"
    (checkout / "dw" / "examples" / "workflows").mkdir(parents=True)
    (checkout / "workflows").mkdir()
    (checkout / "pyproject.toml").write_text("")
    assert packaged_examples_dir(checkout / "dw") is None


def test_this_checkout_never_serves_a_packaged_copy():
    from dw.serve import packaged_examples_dir

    assert packaged_examples_dir() is None


def test_build_dist_copies_the_tracked_catalog_into_the_package(tmp_path):
    """Runs build_dist.sh's catalog step in a scratch git repo: the three
    trees land in dw/examples/, generated media (untracked) does not."""
    import re
    import subprocess
    import tomllib
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts" / "build_dist.sh").read_text()
    step = re.search(r"^rm -rf dw/examples\n.*?^done\n", script, re.S | re.M)
    assert step, "build_dist.sh lost its dw/examples step"
    for tree in ("workflows/t/a.json", "prompts/p.json", "loras/l.json"):
        (tmp_path / tree).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / tree).write_text("{}")
    (tmp_path / "workflows/t/assets").mkdir()
    (tmp_path / "workflows/t/assets/big.mp4").write_text("media")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [*git, "add", "workflows/t/a.json", "prompts", "loras"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["bash", "-c", step.group(0)], cwd=tmp_path, check=True)
    examples = tmp_path / "dw" / "examples"
    assert sorted(
        str(p.relative_to(examples)) for p in examples.rglob("*") if p.is_file()
    ) == ["loras/l.json", "prompts/p.json", "workflows/t/a.json"]
    # and the wheel and sdist carry what it copied
    package_data = tomllib.loads((repo / "pyproject.toml").read_text())["tool"][
        "setuptools"
    ]["package-data"]
    assert "examples/**" in package_data["dw"]
    assert "recursive-include dw/examples *" in (repo / "MANIFEST.in").read_text()
    assert "dw/examples/" in (repo / ".gitignore").read_text().splitlines()


def _run_with_uvicorn_interrupted(serve, monkeypatch, restart_code):
    """dw.serve.main when uvicorn.run ends in the KeyboardInterrupt that a
    SIGINT leaves behind, with a restart asked for (or not)."""
    import uvicorn

    import dw.serve as serve_module
    from dw.server import self_update

    serve()  # the fixture's stubs and environment are in place

    def interrupted(app, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn, "run", interrupted)
    monkeypatch.setitem(self_update._restart, "code", restart_code)
    return serve_module.main()


def test_an_update_restart_exits_with_75(serve, monkeypatch):
    from dw.server.self_update import RESTART_EXIT_CODE

    code = _run_with_uvicorn_interrupted(serve, monkeypatch, RESTART_EXIT_CODE)
    assert code == 75


def test_ctrl_c_without_a_restart_request_still_interrupts(serve, monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        _run_with_uvicorn_interrupted(serve, monkeypatch, None)


def test_an_install_root_flag_with_traversal_exits_2(serve, monkeypatch, capsys):
    monkeypatch.setenv("DW_INSTALL_ROOT", "unset")
    with pytest.raises(SystemExit) as exit_info:
        serve("--install-root", "/x/../y")
    assert exit_info.value.code == 2
    assert "--install-root refused" in capsys.readouterr().err
    assert os.environ["DW_INSTALL_ROOT"] == "unset"
