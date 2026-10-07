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
    def test_two_entries_are_refused_with_a_message_to_name_one_card(self, two_cards):
        from dw.devices import DeviceConfigError, resolve_serve_devices

        with pytest.raises(DeviceConfigError, match="name one card"):
            resolve_serve_devices("cuda:0,cuda:1", None)

    def test_a_card_the_machine_lacks_is_refused_naming_the_cards_present(
        self, two_cards
    ):
        from dw.devices import DeviceConfigError, resolve_serve_devices

        with pytest.raises(DeviceConfigError) as refusal:
            resolve_serve_devices("cuda:7", None)
        message = str(refusal.value)
        assert "cuda:0 (NVIDIA GeForce RTX 4090)" in message
        assert "cuda:1 (NVIDIA GeForce RTX 3090)" in message

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
