"""The REPL's command groups: help routing, and what each group answers
before any workflow is loaded or any worker has started."""

import pytest

from dw.repl import DiffusersWorkflowREPL


def output_of(repl, capsys, line):
    capsys.readouterr()
    repl.onecmd(line)
    return capsys.readouterr().out


def test_help_lists_every_command_group(capsys):
    out = output_of(DiffusersWorkflowREPL(), capsys, "help")
    for group in DiffusersWorkflowREPL.COMMAND_GROUPS:
        assert f"  {group}" in out


@pytest.mark.parametrize("group", DiffusersWorkflowREPL.COMMAND_GROUPS)
def test_group_help_and_help_group_tell_the_same_story(capsys, group):
    repl = DiffusersWorkflowREPL()
    via_group = output_of(repl, capsys, f"{group} ?")
    via_help = output_of(repl, capsys, f"help {group}")
    assert f"{group.capitalize()} commands:" in via_group
    assert via_help == via_group


@pytest.mark.parametrize(
    "line, expected",
    [
        ("workflow status", "No workflow currently loaded"),
        ("arg show", "No workflow loaded"),
        ("memory show", "No worker process running"),
        ("memory clear", "No worker process running"),
        ("workflow bogus", "Unknown workflow subcommand: bogus"),
        ("memory bogus", "Unknown memory subcommand: bogus"),
        ("config bogus", "Unknown config subcommand: bogus"),
    ],
)
def test_commands_answer_before_a_workflow_or_worker_exists(capsys, line, expected):
    assert expected in output_of(DiffusersWorkflowREPL(), capsys, line)


def test_config_show_lists_the_session_settings(capsys):
    repl = DiffusersWorkflowREPL()
    out = output_of(repl, capsys, "config show")
    for name, value in repl.globals.items():
        assert f"  {name}={value}" in out


def test_unknown_command_suggests_the_close_match(capsys):
    out = output_of(DiffusersWorkflowREPL(), capsys, "worklfow status")
    assert "Unknown command: worklfow status" in out
    assert "Did you mean: workflow?" in out
