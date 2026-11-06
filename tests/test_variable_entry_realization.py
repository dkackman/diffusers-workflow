"""A variable list's entries reach the steps pass as written (#1199).

The variables pass realizes with key conventions off at every depth
(dw/arguments.py `realize_args`): a field of a variable's entry is not the
argument it fills, so its media, its type and the object it describes are
loaded and built by the steps pass, under the argument key the member names.
These run a definition the way a run does - prepare_definition, then the
steps pass over the prepared steps - so what is pinned is what a member
receives, not what one helper returns.
"""

import copy
from unittest.mock import patch

from PIL import Image

from dw.step_value_checks import null_media_errors
from dw.workflow import Workflow
from dw.workflow_run import prepare_definition
from dw.arguments import realize_args


class Reference:
    """Stands in for an object built from a file (MiniMaxH3ImageReference)."""

    def __init__(self, location):
        self.location = location

    @classmethod
    def from_file(cls, media):
        return cls(media)


REFERENCE = f"{__name__}.Reference"


def task_step(name, arguments, for_each="variable:entries"):
    step = {"name": name, "task": {"command": "noop", "arguments": arguments}}
    if for_each is not None:
        step["for_each"] = for_each
    return step


def run_steps(definition, tmp_path, arguments=None):
    """The steps a run executes for `definition`: prepared, then realized
    by the steps pass, as Workflow.run does them."""
    wf = Workflow(copy.deepcopy(definition), str(tmp_path / "out"), None)
    prepared, _, _ = prepare_definition(
        wf, copy.deepcopy(wf.workflow_definition), arguments or {}, str(tmp_path)
    )
    steps = prepared["steps"]
    realize_args(steps, str(tmp_path))
    return {step["name"]: step for step in steps}


def write_image(tmp_path, name, colour="red"):
    path = tmp_path / name
    Image.new("RGB", (8, 8), colour).save(path)
    return str(path)


def keep(definition):
    """A last step that reads every member, so elision keeps them all."""
    definition["steps"].append(
        task_step("sink", {"values": "gather:use"}, for_each=None)
    )
    return definition


class TestItemFieldsLoadUnderTheMemberKey:
    def test_item_image_loads_in_the_steps_pass(self, tmp_path):
        image = write_image(tmp_path, "a.png")
        steps = run_steps(
            keep(
                {
                    "id": "t",
                    "variables": {"entries": [{"name": "a", "image": image}]},
                    "steps": [task_step("use", {"image": "item:image"})],
                }
            ),
            tmp_path,
        )

        loaded = steps["use@a"]["task"]["arguments"]["image"]
        assert isinstance(loaded, Image.Image)

    def test_item_video_reaches_the_video_loader_as_the_file(self, tmp_path):
        # The #1199 drill: the entry's 'video' must not be decoded by the
        # variables pass - the member's 'video' argument loads the file
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"placeholder")
        with patch(
            "dw.arguments.fetch_video_with_context",
            side_effect=lambda v, base_dir, k, with_audio=False: ("loaded", v),
        ) as fetch:
            steps = run_steps(
                keep(
                    {
                        "id": "t",
                        "variables": {"entries": [{"name": "a", "video": str(video)}]},
                        "steps": [task_step("use", {"video": "item:video"})],
                    }
                ),
                tmp_path,
            )

        assert steps["use@a"]["task"]["arguments"]["video"] == ("loaded", str(video))
        # Once, by the steps pass - never by the variables pass as well
        assert fetch.call_count == 1

    def test_an_entry_field_under_a_non_media_key_stays_a_path(self, tmp_path):
        # A field named 'image' read into an argument that is not media-named
        # is the string it was written as; the steps pass does not guess
        image = write_image(tmp_path, "a.png")
        steps = run_steps(
            keep(
                {
                    "id": "t",
                    "variables": {"entries": [{"name": "a", "image": image}]},
                    "steps": [task_step("use", {"source": "item:image"})],
                }
            ),
            tmp_path,
        )

        assert steps["use@a"]["task"]["arguments"]["source"] == image


class TestObjectsNestedInAVariableList:
    def definition(self, references):
        return keep(
            {
                "id": "t",
                "variables": {
                    "entries": [{"name": "a", "references": references}],
                },
                "steps": [task_step("use", {"references": "item:references"})],
            }
        )

    def test_a_from_file_object_is_built_by_the_steps_pass(self, tmp_path):
        image = write_image(tmp_path, "a.png")
        steps = run_steps(
            self.definition([{"reference_type": REFERENCE, "from_file": image}]),
            tmp_path,
        )

        built = steps["use@a"]["task"]["arguments"]["references"]
        assert len(built) == 1
        assert isinstance(built[0], Reference)

    def test_a_null_from_file_entry_leaves_the_member_s_list(self, tmp_path):
        image = write_image(tmp_path, "a.png")
        steps = run_steps(
            self.definition(
                [
                    {"reference_type": REFERENCE, "from_file": image},
                    {"reference_type": REFERENCE, "from_file": None},
                ]
            ),
            tmp_path,
        )

        built = steps["use@a"]["task"]["arguments"]["references"]
        assert [type(r) for r in built] == [Reference]


class TestForEachOverNullMediaEntries:
    """A for_each over a list of object descriptions, one of them null.

    The variables pass used to build each entry and drop the null one, so
    the run had one member fewer than validation expanded (validation never
    realizes). Now the run expands what validation does: every entry is a
    member, and the null one is left out wherever the member puts it."""

    def definition(self, arguments):
        image = "variable:image"
        return keep(
            {
                "id": "t",
                "variables": {
                    "image": None,
                    "entries": [
                        # Keyed by index: an object entry's other fields
                        # go onto the object it builds
                        {"reference_type": REFERENCE, "from_file": image},
                        {"reference_type": REFERENCE, "from_file": None},
                    ],
                },
                "steps": [task_step("use", arguments)],
            }
        )

    def test_every_entry_is_a_member_as_in_validation(self, tmp_path):
        definition = self.definition({"references": ["item:"]})
        image = write_image(tmp_path, "a.png")
        steps = run_steps(definition, tmp_path, {"image": image})

        expanded = Workflow(copy.deepcopy(definition), str(tmp_path), None)
        validated = expanded.expanded_definition({"image": image})["steps"]
        assert [s["name"] for s in validated if "@" in s["name"]] == [
            "use@0",
            "use@1",
        ]
        assert "use@0" in steps and "use@1" in steps
        # In a list the null one is left out; the other is built
        assert steps["use@1"]["task"]["arguments"]["references"] == []
        (built,) = steps["use@0"]["task"]["arguments"]["references"]
        assert isinstance(built, Reference)

    def test_a_null_entry_on_its_own_is_refused_by_validation(self, tmp_path):
        # Under a bare key there is nothing to leave it out of: validation
        # refuses the member before anything loads (#478)
        definition = self.definition({"reference": "item:"})
        member = definition["steps"][0]
        member["pipeline"] = {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {"model_name": "m"},
            "arguments": member.pop("task")["arguments"],
        }
        wf = Workflow(copy.deepcopy(definition), str(tmp_path), None)
        errors = null_media_errors(wf.expanded_definition())

        # Both members: 'image' is declared null, and 'b' wrote null
        assert len(errors) == 2
        assert all("names an object to build" in e["message"] for e in errors)


class TestGatherOverMediaEntries:
    def test_gather_lists_every_member_and_each_loaded_its_own(self, tmp_path):
        first = write_image(tmp_path, "a.png", "red")
        second = write_image(tmp_path, "b.png", "blue")
        steps = run_steps(
            keep(
                {
                    "id": "t",
                    "variables": {
                        "entries": [
                            {"name": "a", "image": first},
                            {"name": "b", "image": second},
                        ]
                    },
                    "steps": [task_step("use", {"image": "item:image"})],
                }
            ),
            tmp_path,
        )

        assert steps["sink"]["task"]["arguments"]["values"] == [
            "previous_result:use@a",
            "previous_result:use@b",
        ]
        for name, colour in (("use@a", (255, 0, 0)), ("use@b", (0, 0, 255))):
            loaded = steps[name]["task"]["arguments"]["image"]
            assert isinstance(loaded, Image.Image)
            assert loaded.convert("RGB").getpixel((0, 0)) == colour
