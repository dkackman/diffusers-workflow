"""Realization: a copy of a workflow with every mutable input pinned, so the
file beside a run's manifest reproduces that run whatever changes later."""

import copy
import hashlib
import json
import os

import pytest

from dw.realize import realize_workflow, strings_with_prefix
from dw.runs import new_run_id
from dw.schema import load_schema, validate_data


def definition():
    return {
        "id": "realize_test",
        "variables": {"prompt": "a default", "steps": 25},
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {
                        "prompt": "variable:prompt",
                        "num_inference_steps": "variable:steps",
                    },
                },
            }
        ],
    }


@pytest.fixture
def prompt_library(tmp_path):
    library = tmp_path / "prompts"
    (library / "scenic").mkdir(parents=True)
    (library / "scenic" / "dusk.json").write_text(
        json.dumps({"text": "a harbour at dusk"})
    )
    return str(library)


@pytest.fixture
def output_root(tmp_path):
    """An output root holding one finished run of 'ltx2/Gyre'."""
    root = tmp_path / "outputs"
    run_id = new_run_id({"a": 1})
    run = root / "ltx2" / "Gyre" / run_id
    run.mkdir(parents=True)
    (run / "still.png").write_bytes(b"not really a png")
    return str(root), run_id


class TestVariablesAndSeed:
    def test_arguments_become_the_variable_defaults(self):
        realized, _ = realize_workflow(definition(), {"prompt": "a cat", "steps": 4}, 7)
        assert realized["variables"] == {"prompt": "a cat", "steps": 4}

    def test_variable_references_are_left_alone(self):
        realized, _ = realize_workflow(definition(), {"prompt": "a cat"}, 7)
        arguments = realized["steps"][0]["pipeline"]["arguments"]
        assert arguments["prompt"] == "variable:prompt"

    def test_the_seed_is_written_even_when_the_definition_had_none(self):
        realized, _ = realize_workflow(definition(), {}, 991)
        assert realized["seed"] == 991

    def test_the_input_definition_is_not_mutated(self):
        original = definition()
        before = copy.deepcopy(original)
        realize_workflow(original, {"prompt": "a cat"}, 7)
        assert original == before


class TestPrompts:
    def test_a_stored_prompt_is_inlined_and_annotated(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"

        realized, annotations = realize_workflow(
            source, {}, 7, prompt_dir=prompt_library
        )

        arguments = realized["steps"][0]["pipeline"]["arguments"]
        assert arguments["prompt"] == "a harbour at dusk"
        assert annotations["prompts"] == ["scenic/dusk"]

    def test_a_name_is_annotated_once_in_first_seen_order(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"
        source["steps"][0]["pipeline"]["arguments"][
            "negative_prompt"
        ] = "prompt:scenic/dusk"

        _, annotations = realize_workflow(source, {}, 7, prompt_dir=prompt_library)

        assert annotations["prompts"] == ["scenic/dusk"]

    def test_an_unresolvable_prompt_is_left_as_written(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:missing"

        realized, annotations = realize_workflow(
            source, {}, 7, prompt_dir=prompt_library
        )

        assert realized["steps"][0]["pipeline"]["arguments"]["prompt"] == (
            "prompt:missing"
        )
        assert annotations["prompts"] == []


class TestOutputReferences:
    def test_latest_is_pinned_to_the_run_it_resolved_to(self, output_root):
        root, run_id = output_root
        source = definition()
        source["steps"][0]["pipeline"]["arguments"][
            "image"
        ] = "output:ltx2/Gyre/latest/still.png"

        realized, _ = realize_workflow(source, {}, 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == (
            f"output:ltx2/Gyre/{run_id}/still.png"
        )

    def test_an_explicit_run_id_is_kept_as_written(self, output_root):
        root, run_id = output_root
        written = f"output:ltx2/Gyre/{run_id}/still.png"
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = written

        realized, _ = realize_workflow(source, {}, 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == written

    def test_an_unresolvable_output_is_left_as_written(self, output_root):
        root, _ = output_root
        written = "output:ltx2/Nothing/latest/still.png"
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = written

        realized, _ = realize_workflow(source, {}, 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == written


class TestReferencesThatAreKept:
    @pytest.mark.parametrize(
        "value",
        [
            "asset:iris.png",
            "constant:diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES",
            "previous_result:gen",
        ],
    )
    def test_kept_verbatim(self, value):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["thing"] = value

        realized, _ = realize_workflow(source, {}, 7)

        assert realized["steps"][0]["pipeline"]["arguments"]["thing"] == value


class TestSubWorkflows:
    def test_a_local_path_is_kept_and_digested(self, tmp_path):
        tree = tmp_path / "workflows"
        (tree / "steps").mkdir(parents=True)
        child = tree / "steps" / "upscale.json"
        child.write_text(json.dumps({"id": "child", "steps": []}))

        source = definition()
        source["steps"].append(
            {"name": "up", "workflow": {"path": "steps/upscale.json"}}
        )

        realized, annotations = realize_workflow(
            source, {}, 7, base_dir=str(tree), workflow_dir=str(tree)
        )

        assert realized["steps"][1]["workflow"]["path"] == "steps/upscale.json"
        digest = annotations["sub_workflows"]["steps/upscale.json"]
        assert digest == hashlib.sha256(child.read_bytes()).hexdigest()

    def test_an_unreadable_sub_workflow_digests_to_null(self, tmp_path):
        tree = tmp_path / "workflows"
        tree.mkdir()
        source = definition()
        source["steps"].append({"name": "up", "workflow": {"path": "gone.json"}})

        _, annotations = realize_workflow(
            source, {}, 7, base_dir=str(tree), workflow_dir=str(tree)
        )

        assert annotations["sub_workflows"] == {"gone.json": None}

    def test_a_builtin_is_not_digested(self):
        source = definition()
        source["steps"].append(
            {"name": "up", "workflow": {"path": "builtin:upscale.json"}}
        )

        realized, annotations = realize_workflow(source, {}, 7)

        assert realized["steps"][1]["workflow"]["path"] == "builtin:upscale.json"
        assert annotations["sub_workflows"] == {}


def test_the_realized_file_validates_against_the_schema(prompt_library):
    source = definition()
    source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"

    realized, _ = realize_workflow(source, {"steps": 4}, 991, prompt_dir=prompt_library)

    ok, message = validate_data(realized, load_schema("workflow"))
    assert ok, message


class TestStringsWithPrefix:
    def test_finds_every_matching_string_deduplicated_in_first_seen_order(self):
        tree = {
            "a": "asset:iris.png",
            "b": ["asset:iris.png", "output:x/y/still.png"],
            "c": {"d": "asset:mask.png", "e": 5, "f": None},
        }

        assert strings_with_prefix(tree, "asset:") == [
            "asset:iris.png",
            "asset:mask.png",
        ]
