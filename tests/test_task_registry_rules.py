"""A command's validate-time rules are declared on its registration (#692).

`TASK_ARGUMENT_DOMAINS`, `TASK_ARGUMENT_CHOICES`, `TASK_MEDIA_ARGUMENTS`,
`TASK_WHOLE_NUMBER_ARGUMENTS` and `TASK_STATIC_CHECKS` used to be hand-kept dicts beside the registry, and a
command left out of one lost that check with no error - for a media argument,
its path confinement. They are now views of `register_command`'s `domains`,
`choices`, `media_arguments`, `whole_numbers` and `static_check`, pinned here to the registry
and to the signatures the dispatch forwards into.
"""

import collections
import inspect
import subprocess
import sys

import pytest

import dw.task_domains as task_domains
import dw.task_problems as task_problems
import dw.tasks.task  # noqa: F401 - registers every command, as validation does
from dw.introspection import describe_task
from dw.locations import TASK_MEDIA_ARGUMENTS
from dw.task_domains import (
    TASK_ARGUMENT_CHOICES,
    TASK_ARGUMENT_DOMAINS,
    TASK_STATIC_CHECKS,
    TASK_WHOLE_NUMBER_ARGUMENTS,
)
from dw.tasks.registry import _COMMAND_RULES

TABLES = [
    (TASK_ARGUMENT_DOMAINS, "domains"),
    (TASK_ARGUMENT_CHOICES, "choices"),
    (TASK_MEDIA_ARGUMENTS, "media_arguments"),
    (TASK_WHOLE_NUMBER_ARGUMENTS, "whole_numbers"),
    (TASK_STATIC_CHECKS, "static_check"),
]


def _parameters(command):
    return {p["name"] for p in describe_task(command)["parameters"]}


@pytest.mark.parametrize("table, rule", TABLES, ids=[rule for _, rule in TABLES])
def test_every_derived_table_is_the_registrys_view(table, rule):
    expected = {
        command: rules[rule]
        for command, rules in _COMMAND_RULES.items()
        if rule in rules
    }
    assert expected, rule
    assert {command: table[command] for command in table} == expected


@pytest.mark.parametrize("table, rule", TABLES, ids=[rule for _, rule in TABLES])
def test_a_derived_table_is_read_only(table, rule):
    command = next(iter(table))
    with pytest.raises(TypeError):
        table[command] = table[command]
    if isinstance(table[command], collections.abc.Mapping):
        name = next(iter(table[command]))
        with pytest.raises(TypeError):
            table[command][name] = table[command][name]


def test_a_table_read_first_holds_the_self_registering_commands():
    """The view is built on read, after importing dw.tasks.task - so a
    process that reads it before anything imported a self-registering task
    module (beats, cuts, trim) still sees their commands."""
    script = (
        "from dw.task_domains import TASK_ARGUMENT_DOMAINS, TASK_STATIC_CHECKS\n"
        "assert 'analyze_beats' in TASK_STATIC_CHECKS\n"
        "assert 'plan_cuts' in TASK_STATIC_CHECKS\n"
        "assert 'trim_video' in TASK_ARGUMENT_DOMAINS\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def _static_check_functions():
    """Every rule function in task_domains and task_problems shaped as a
    static check: named `*_errors`, taking a step's `arguments` first."""
    found = []
    for module in (task_domains, task_problems):
        for name, value in vars(module).items():
            if not (inspect.isfunction(value) and name.endswith("_errors")):
                continue
            if value.__module__ != module.__name__:
                continue
            parameters = list(inspect.signature(value).parameters)
            if parameters and parameters[0] == "arguments":
                found.append(value)
    return found


def test_every_static_check_is_registered_to_exactly_one_command():
    registered = collections.Counter(TASK_STATIC_CHECKS.values())
    for check in _static_check_functions():
        assert registered[check] == 1, check.__name__
    # and one kept in its task's own module (attribute_voices') is not shared
    assert set(registered.values()) == {1}


def test_every_media_argument_is_a_parameter_of_its_command():
    for command, names in TASK_MEDIA_ARGUMENTS.items():
        assert set(names) <= _parameters(command), command


def test_every_choice_is_a_parameter_of_its_command():
    for command, choices in TASK_ARGUMENT_CHOICES.items():
        assert set(choices) <= _parameters(command), command


def test_every_whole_number_is_a_parameter_with_a_declared_domain():
    assert TASK_WHOLE_NUMBER_ARGUMENTS
    for command, names in TASK_WHOLE_NUMBER_ARGUMENTS.items():
        assert set(names) <= _parameters(command), command
        assert set(names) <= set(TASK_ARGUMENT_DOMAINS.get(command, {})), command


def test_a_registered_static_check_reaches_validate():
    """attribute_voices' check used to be its own validation pass outside the
    registry; it is now one more registered check task_argument_errors runs."""
    definition = {
        "steps": [
            {
                "name": "who",
                "task": {
                    "command": "attribute_voices",
                    "arguments": {
                        "audio": "asset:song.wav",
                        "voices": {"a": {"start_seconds": 0, "duration_seconds": 4}},
                    },
                },
            }
        ]
    }
    errors = task_domains.task_argument_errors(definition)
    assert [e["path"] for e in errors] == ["steps[0].task.arguments.voices"]
    assert "at least 2 voices" in errors[0]["message"]
