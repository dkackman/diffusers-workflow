"""Realization: a copy of a workflow with every mutable input pinned, so the
file beside a run's manifest reproduces that run whatever changes later."""

import copy
import hashlib
import json

import pytest

from dw.realize import realize_workflow, strings_with_prefix
from dw.runs import new_run_id
from dw.schema import load_schema, validate_data
from dw.workflow import Workflow
from dw.workflow_run import prepare_definition


def folded(source, arguments=None):
    """The variables a run of `source` with `arguments` folds - what the
    run hands realize_workflow to record."""
    return Workflow(copy.deepcopy(source), "outputs", None).folded_variables(arguments)


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
        realized, _ = realize_workflow(
            definition(), folded(definition(), {"prompt": "a cat", "steps": 4}), 7
        )
        assert realized["variables"] == {"prompt": "a cat", "steps": 4}

    def test_variable_references_are_left_alone(self):
        realized, _ = realize_workflow(
            definition(), folded(definition(), {"prompt": "a cat"}), 7
        )
        arguments = realized["steps"][0]["pipeline"]["arguments"]
        assert arguments["prompt"] == "variable:prompt"

    def test_the_seed_is_written_even_when_the_definition_had_none(self):
        realized, _ = realize_workflow(definition(), folded(definition(), {}), 991)
        assert realized["seed"] == 991

    def test_a_seed_pinned_through_a_variable_reference_updates_that_variable(self):
        source = definition()
        source["seed"] = "variable:seed_arg"
        source["variables"]["seed_arg"] = None

        realized, _ = realize_workflow(source, folded(source, {}), 991)

        assert realized["seed"] == 991
        assert realized["variables"]["seed_arg"] == 991

    def test_the_input_definition_is_not_mutated(self):
        original = definition()
        before = copy.deepcopy(original)
        realize_workflow(original, folded(original, {"prompt": "a cat"}), 7)
        assert original == before


class TestPrompts:
    def test_a_stored_prompt_is_inlined_and_annotated(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"

        realized, annotations = realize_workflow(
            source, folded(source, {}), 7, prompt_dir=prompt_library
        )

        arguments = realized["steps"][0]["pipeline"]["arguments"]
        assert arguments["prompt"] == "a harbour at dusk"
        assert annotations["prompts"] == ["scenic/dusk"]

    def test_a_name_is_annotated_once_in_first_seen_order(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"
        source["steps"][0]["pipeline"]["arguments"]["negative_prompt"] = (
            "prompt:scenic/dusk"
        )

        _, annotations = realize_workflow(
            source, folded(source, {}), 7, prompt_dir=prompt_library
        )

        assert annotations["prompts"] == ["scenic/dusk"]

    def test_an_unresolvable_prompt_is_left_as_written(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:missing"

        realized, annotations = realize_workflow(
            source, folded(source, {}), 7, prompt_dir=prompt_library
        )

        assert realized["steps"][0]["pipeline"]["arguments"]["prompt"] == (
            "prompt:missing"
        )
        assert annotations["prompts"] == []


class TestOutputReferences:
    def test_latest_is_pinned_to_the_run_it_resolved_to(self, output_root):
        root, run_id = output_root
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = (
            "output:ltx2/Gyre/latest/still.png"
        )

        realized, _ = realize_workflow(source, folded(source, {}), 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == (
            f"output:ltx2/Gyre/{run_id}/still.png"
        )

    def test_a_version_is_pinned_to_the_run_it_named(self, output_root):
        # A version is stable, but deleting the newest run frees its number,
        # so the realized copy names the run id as it does for 'latest'
        root, run_id = output_root
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = (
            "output:ltx2/Gyre/v1/still.png"
        )

        realized, _ = realize_workflow(source, folded(source, {}), 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == (
            f"output:ltx2/Gyre/{run_id}/still.png"
        )

    def test_an_explicit_run_id_is_kept_as_written(self, output_root):
        root, run_id = output_root
        written = f"output:ltx2/Gyre/{run_id}/still.png"
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = written

        realized, _ = realize_workflow(source, folded(source, {}), 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == written

    def test_an_unresolvable_output_is_left_as_written(self, output_root):
        root, _ = output_root
        written = "output:ltx2/Nothing/latest/still.png"
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = written

        realized, _ = realize_workflow(source, folded(source, {}), 7, output_root=root)

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

        realized, _ = realize_workflow(source, folded(source, {}), 7)

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
            source, folded(source, {}), 7, base_dir=str(tree), workflow_dir=str(tree)
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
            source, folded(source, {}), 7, base_dir=str(tree), workflow_dir=str(tree)
        )

        assert annotations["sub_workflows"] == {"gone.json": None}

    def test_a_builtin_is_not_digested(self):
        source = definition()
        source["steps"].append(
            {"name": "up", "workflow": {"path": "builtin:upscale.json"}}
        )

        realized, annotations = realize_workflow(source, folded(source, {}), 7)

        assert realized["steps"][1]["workflow"]["path"] == "builtin:upscale.json"
        assert annotations["sub_workflows"] == {}


def test_the_realized_file_validates_against_the_schema(prompt_library):
    source = definition()
    source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"

    realized, _ = realize_workflow(
        source, folded(source, {"steps": 4}), 991, prompt_dir=prompt_library
    )

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


class TestUnpinnedOutputs:
    def test_pin_outputs_false_leaves_latest_as_written(self, output_root):
        root, _ = output_root
        spec = definition()
        spec["steps"][0]["pipeline"]["arguments"]["image"] = (
            "output:ltx2/Gyre/latest/still.png"
        )
        realized, _ = realize_workflow(
            spec, folded(spec, {}), 7, output_root=root, pin_outputs=False
        )
        assert (
            realized["steps"][0]["pipeline"]["arguments"]["image"]
            == "output:ltx2/Gyre/latest/still.png"
        )

    def test_pin_outputs_false_still_inlines_prompts(self, prompt_library):
        spec = definition()
        spec["variables"]["prompt"] = "prompt:scenic/dusk"
        realized, annotations = realize_workflow(
            spec, folded(spec, {}), 7, prompt_dir=prompt_library, pin_outputs=False
        )
        assert realized["variables"]["prompt"] == "a harbour at dusk"
        assert annotations["prompts"] == ["scenic/dusk"]


class TestReadSubWorkflow:
    def test_reads_a_child_beside_the_parent(self, tmp_path):
        from dw.realize import read_sub_workflow

        child = {"id": "child", "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        raw = read_sub_workflow("child.json", str(tmp_path), str(tmp_path))
        assert json.loads(raw) == child

    def test_a_missing_child_reads_as_none(self, tmp_path):
        from dw.realize import read_sub_workflow

        assert read_sub_workflow("nope.json", str(tmp_path), str(tmp_path)) is None

    def test_a_child_outside_the_confinement_reads_as_none(self, tmp_path):
        from dw.realize import read_sub_workflow

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "child.json").write_text(json.dumps({"id": "c", "steps": []}))
        confined = tmp_path / "confined"
        confined.mkdir()
        assert (
            read_sub_workflow("../outside/child.json", str(confined), str(confined))
            is None
        )

    def test_a_child_climbing_out_of_the_catalog_reads_as_none_unconfined(
        self, tmp_path
    ):
        """An unconfined run (no workflow_dir - a bare CLI run) still confines
        a relative reference to the catalog root, so the read refuses the same
        climb the run does rather than reaching outside it - the read the
        unguarded stat used to allow."""
        from dw.realize import read_sub_workflow

        catalog = tmp_path / "workflows"
        (catalog / "templates").mkdir(parents=True)
        (tmp_path / "Outside.json").write_text(json.dumps({"id": "c", "steps": []}))

        assert (
            read_sub_workflow("../../Outside.json", str(catalog / "templates"), None)
            is None
        )

    def test_a_child_climbing_to_a_sibling_catalog_folder_still_reads(self, tmp_path):
        """The confinement is the catalog root, not the referencing file's own
        directory - '../models/x.json' from templates/ is the form every
        template uses, and stays readable."""
        from dw.realize import read_sub_workflow

        catalog = tmp_path / "workflows"
        (catalog / "templates").mkdir(parents=True)
        (catalog / "models").mkdir()
        (catalog / "models" / "Child.json").write_text(
            json.dumps({"id": "c", "steps": []})
        )

        raw = read_sub_workflow(
            "../models/Child.json", str(catalog / "templates"), None
        )

        assert json.loads(raw) == {"id": "c", "steps": []}


def test_the_realized_record_carries_the_snapped_value():
    from dw.realize import realize_workflow
    from tests.test_variable_constraints import H3, workflow_with

    source = workflow_with({"num_frames": H3}, {"num_frames": 124})
    realized, _ = realize_workflow(source, folded(source, {"num_frames": 130}), 1)
    assert realized["variables"]["num_frames"] == 141


class Undeepcopyable:
    """A stand-in for realized media - an image, a frame list's tensor - that
    a record must share rather than duplicate: copying it deeply raises."""

    def __deepcopy__(self, memo):
        raise AssertionError("a variable's leaf was deep-copied")


def media_definition():
    """A child-shaped workflow: a variable that a composing parent fills with
    media it has already realized, and a step that reads it."""
    source = definition()
    source["variables"]["image"] = None
    source["variables"]["frames"] = []
    source["steps"][0]["pipeline"]["arguments"]["reference"] = "variable:image"
    return source


class TestRecordedVariablesShareTheirLeaves:
    def test_realize_workflow_records_the_object_itself(self):
        media = Undeepcopyable()
        variables = {"prompt": "a cat", "steps": 4, "image": media, "frames": [media]}

        realized, _ = realize_workflow(media_definition(), variables, 7)

        assert realized["variables"]["image"] is media
        assert realized["variables"]["frames"][0] is media
        # The containers are the record's own
        assert realized["variables"] is not variables
        assert realized["variables"]["frames"] is not variables["frames"]

    def test_prepare_definition_records_the_object_itself(self, tmp_path):
        media = Undeepcopyable()
        frames = [media, (media, media)]
        wf = Workflow(media_definition(), str(tmp_path), None)

        prepared, _, recorded = prepare_definition(
            wf,
            copy.deepcopy(wf.workflow_definition),
            {"image": media, "frames": frames},
            str(tmp_path),
        )

        assert recorded["image"] is media
        assert recorded["frames"][0] is media
        assert recorded["frames"][1][0] is media
        assert isinstance(recorded["frames"][1], tuple)
        assert recorded["frames"] is not frames
        assert prepared["steps"][0]["pipeline"]["arguments"]["reference"] is media

    def test_a_composed_child_s_record_realizes_without_copying_media(self, tmp_path):
        """What a composed child prepares from: arguments carrying media -
        already the child's own copy, taken once where the parent hands them
        over (TestAComposedChildOwnsItsMedia) - folded and then recorded
        with no further copy, so the record holds that same object."""
        media = Undeepcopyable()
        wf = Workflow(media_definition(), str(tmp_path), None)

        _, _, recorded = prepare_definition(
            wf,
            copy.deepcopy(wf.workflow_definition),
            {"image": media, "frames": [media]},
            str(tmp_path),
        )
        realized, _ = realize_workflow(wf.workflow_definition, recorded, 7)

        assert realized["variables"]["image"] is media
        assert realized["variables"]["frames"][0] is media

    def test_cache_hits_prepares_without_copying_media(self, tmp_path):
        media = Undeepcopyable()
        source = media_definition()
        source["seed"] = 3
        wf = Workflow(source, str(tmp_path), None)

        assert wf.cache_hits({"image": media, "frames": [media]}) == []


class CountingMedia:
    """A realized video handed down to a composed child: an AudioVideo that
    counts how often it is deep-copied."""

    copies = 0

    def __new__(cls):
        from PIL import Image

        from dw.media_types import AudioVideo

        class Counted(AudioVideo):
            def __deepcopy__(self, memo):
                CountingMedia.copies += 1
                return Counted(
                    [frame.copy() for frame in self.frames],
                    self.audio,
                    self.sample_rate,
                    fps=self.fps,
                )

        frames = [Image.new("RGB", (16, 16), (i * 40, 0, 0)) for i in range(4)]
        return Counted(frames, None, None, fps=24)


class TestAComposedChildOwnsItsMedia:
    """A parent's realized media passed into a child through the workflow
    step's arguments is the child's own copy: a child step that writes onto
    its artifact in place - `select` hands its candidate back by identity, and
    a declared `result.fps` is stamped onto that artifact (conform_artifact)
    - restamps the child's copy, never the parent's object."""

    def run_parent(self, tmp_path, clip):
        child = {
            "id": "child",
            "variables": {"clips": []},
            "steps": [
                {
                    "name": "pick",
                    "task": {
                        "command": "select",
                        "arguments": {
                            "candidates": "variable:clips",
                            "scores": [1],
                            "rule": "index",
                            "index": 0,
                        },
                    },
                    "result": {"content_type": "video/mp4", "fps": 12},
                }
            ],
        }
        (tmp_path / "child.json").write_text(json.dumps(child))
        parent = {
            "id": "parent",
            "variables": {"clip": None},
            "steps": [
                # The parent's own step result, as a generating step would
                # leave it: the clip itself, by identity
                {
                    "name": "shot",
                    "task": {
                        "command": "select",
                        "arguments": {
                            "candidates": ["variable:clip"],
                            "scores": [1],
                            "rule": "index",
                            "index": 0,
                        },
                    },
                },
                {
                    "name": "compose",
                    "workflow": {
                        "path": "child.json",
                        "arguments": {"clips": ["previous_result:shot"]},
                    },
                    "result": {"content_type": "video/mp4"},
                },
            ],
        }
        parent_file = tmp_path / "parent.json"
        parent_file.write_text(json.dumps(parent))
        from dw.workflow import workflow_from_file

        outputs = tmp_path / "outputs"
        outputs.mkdir()
        wf = workflow_from_file(str(parent_file), str(outputs), str(tmp_path))
        wf.run({"clip": clip})

    def test_the_parent_s_object_is_not_restamped(self, tmp_path):
        clip = CountingMedia()

        self.run_parent(tmp_path, clip)

        assert clip.fps == 24

    def test_the_child_copies_what_it_is_handed_once(self, tmp_path):
        clip = CountingMedia()
        CountingMedia.copies = 0

        self.run_parent(tmp_path, clip)

        assert CountingMedia.copies == 1


class Bomb:
    """An argument value deepcopy refuses."""

    def __deepcopy__(self, memo):
        raise TypeError("cannot deepcopy a Bomb")


def composed_child(variables):
    definition = {
        "id": "child",
        "steps": [
            {
                "name": "pick",
                "task": {
                    "command": "select",
                    "arguments": {
                        "candidates": ["x"],
                        "scores": [1],
                        "rule": "index",
                        "index": 0,
                    },
                },
                "result": {"content_type": "text/plain"},
            }
        ],
    }
    if variables is not None:
        definition["variables"] = variables
    return definition


class TestAComposedChildCopiesWithinItsCleanupAndOnlyWhatItDeclares:
    def run_child(self, tmp_path, variables, arguments, context=None):
        wf = Workflow(composed_child(variables), str(tmp_path), None)
        wf._composed = True
        return wf.run(arguments, context=context)

    def test_a_failed_copy_unwinds_the_run_context(self, tmp_path):
        from dw.events import RunContext

        ctx = RunContext()

        with pytest.raises(TypeError, match="Bomb"):
            self.run_child(tmp_path, {"clips": []}, {"clips": Bomb()}, ctx)

        assert ctx._run_depth == 0
        assert ctx._watchdog_thread is None

    def test_an_undeclared_argument_is_not_copied_and_is_refused_by_name(
        self, tmp_path
    ):
        with pytest.raises(ValueError, match="Unknown variable 'extra'"):
            self.run_child(tmp_path, {"clips": []}, {"extra": Bomb()})

    def test_a_child_declaring_no_variables_copies_nothing(self, tmp_path):
        # Its arguments are ignored, so a value deepcopy refuses is no failure
        self.run_child(tmp_path, None, {"clips": Bomb()})
