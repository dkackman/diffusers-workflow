"""Task argument discovery: the registered implementation's signature is the
command's argument schema, so the editor's forms and the pre-run warnings
read the same truth the dispatch calls."""

import importlib
import inspect

import pytest

from dw.introspection import (
    describe_task,
    unknown_task_arguments,
    workflow_argument_warnings,
)
from dw.tasks.task import _COMMAND_INFO, _VIDEO_PROCESSOR_INFO


def resolve(path):
    module_name, _, function_name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), function_name)


class TestRegistryIntegrity:
    """The drift-killers: a registration that rots fails here, not in a form."""

    def all_info(self):
        entries = dict(_COMMAND_INFO)
        entries.update(_VIDEO_PROCESSOR_INFO)
        return entries

    def test_every_implementation_path_resolves_to_a_callable(self):
        for command, info in self.all_info().items():
            if info["implementation"] is None:
                continue
            implementation = resolve(info["implementation"])
            assert callable(implementation), command

    def test_every_provided_name_is_a_real_parameter(self):
        for command, info in self.all_info().items():
            if info["implementation"] is None or not info["provided"]:
                continue
            parameters = inspect.signature(resolve(info["implementation"])).parameters
            for name in info["provided"]:
                assert name in parameters, f"{command}: '{name}' not in signature"


class TestDescribeTask:
    def names(self, description):
        return [p["name"] for p in description["parameters"]]

    def test_signature_becomes_the_schema(self):
        description = describe_task("qr_code")
        by_name = {p["name"]: p for p in description["parameters"]}
        assert by_name["qr_code_contents"]["required"]
        assert by_name["height"]["default"] == 768
        assert not description["accepts_kwargs"]

    def test_provided_parameters_are_hidden(self):
        # get_last_frame pins frame_index itself; batch_decode gets its
        # processor from a pipeline reference, not from arguments
        assert "frame_index" not in self.names(describe_task("get_last_frame"))
        assert "frame_index" in self.names(describe_task("get_frame"))
        assert "processor" not in self.names(describe_task("batch_decode_post_process"))

    def test_device_is_always_offered(self):
        # In the signature where the implementation takes it, appended where
        # the dispatch alone consumes it
        assert "device" in self.names(describe_task("upscale"))
        assert "device" in self.names(describe_task("concat_videos"))

    def test_image_processors_and_free_form_commands_are_open_ended(self):
        canny = describe_task("canny")
        assert canny["accepts_kwargs"]
        assert "image" in self.names(canny)
        gather = describe_task("gather_inputs")
        assert gather["accepts_kwargs"]
        assert gather["parameters"] == []

    def test_unknown_command_raises(self):
        with pytest.raises(ValueError, match="Unknown task command"):
            describe_task("not_a_task")


class TestTaskArgumentWarnings:
    def test_a_typo_is_flagged(self):
        assert unknown_task_arguments("concat_videos", ["videos", "trim_framse"]) == [
            "trim_framse"
        ]

    def test_device_is_never_flagged(self):
        assert unknown_task_arguments("concat_videos", ["videos", "device"]) == []

    def test_kwargs_and_unknown_commands_stay_silent(self):
        # A warning must never be wrong
        assert unknown_task_arguments("upscale", ["anything_at_all"]) == []
        assert unknown_task_arguments("not_a_task", ["x"]) == []

    def test_workflow_warnings_cover_task_steps(self):
        definition = {
            "steps": [
                {
                    "name": "join",
                    "task": {
                        "command": "concat_videos",
                        "arguments": {"videos": [], "trim_framse": 1},
                    },
                }
            ]
        }
        warnings = workflow_argument_warnings(definition)
        assert len(warnings) == 1
        assert "trim_framse" in warnings[0]
        assert "concat_videos" in warnings[0]

    def test_inputs_style_task_steps_are_left_alone(self):
        definition = {
            "steps": [
                {
                    "name": "fan",
                    "task": {"command": "concat_videos", "inputs": [{"videos": []}]},
                }
            ]
        }
        assert workflow_argument_warnings(definition) == []


class TestDocumentedKwargs:
    """A task whose real arguments live behind **kwargs still has to offer
    them: the docstring's nested block is the only place they are declared,
    so discovery reads it."""

    def names(self, description):
        return [p["name"] for p in description["parameters"]]

    def test_nested_kwargs_become_parameters(self):
        description = describe_task("text_generation")
        names = self.names(description)
        assert description["accepts_kwargs"]
        for name in (
            "prompt",
            "device",
            "model_name",
            "system_prompt",
            "max_new_tokens",
            "image",
            "repetition_penalty",
            "generate_kwargs",
        ):
            assert name in names, name

    def test_documented_kwargs_are_optional_and_described(self):
        by_name = {p["name"]: p for p in describe_task("text_generation")["parameters"]}
        assert not by_name["system_prompt"]["required"]
        assert by_name["system_prompt"]["default"] is None
        assert "system instruction" in by_name["system_prompt"]["description"].lower()

    def test_a_typeless_docstring_header_still_describes(self):
        # Google-style 'name: description' has no parenthesized type, and the
        # named parameters of these tasks are written that way
        by_name = {p["name"]: p for p in describe_task("text_generation")["parameters"]}
        assert "prompt" in by_name["prompt"]["description"].lower()

    def test_named_parameters_win_over_the_nested_block(self):
        # device is a real parameter; it must not be duplicated by anything
        # the kwargs block says
        names = self.names(describe_task("text_generation"))
        assert names.count("device") == 1
