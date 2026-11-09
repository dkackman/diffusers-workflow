"""repair-segment, the template (#1169): its variables, tiers and
constraints, its steps' wiring, and the cut feeding the render and the
splice through a workflow's references. The task itself is
tests/test_chain_repair.py, whose fixtures these share."""

import pytest

from dw.tasks.chain_repair import chain_repair_cut, repair_geometry
from dw.task_domains import frames_to_samples

from .test_chain_repair import CHAIN, FPS, write_run


@pytest.fixture
def run(tmp_path):
    return write_run(str(tmp_path), [40, 51, 51, 51])


TEMPLATES = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "workflows"
    / "templates"
    / "minimax"
)
TEMPLATE = TEMPLATES / "repair-segment.json"


def template_errors(arguments=None):
    import json

    from dw.workflow import Workflow

    definition = json.loads(TEMPLATE.read_text())
    workflow = Workflow(definition, str(TEMPLATE.parent), str(TEMPLATE))
    return workflow.validation_errors(arguments)


def chain_blocks(definition):
    if isinstance(definition, dict):
        if isinstance(definition.get("chain"), dict):
            yield definition["chain"]
        for value in definition.values():
            yield from chain_blocks(value)
    elif isinstance(definition, list):
        for value in definition:
            yield from chain_blocks(value)


class TestTemplate:
    def test_validates_with_its_defaults(self):
        assert template_errors() == []

    def test_validates_with_sample_arguments(self):
        assert (
            template_errors(
                {
                    "segment": 3,
                    "num_frames": 192,
                    "tail_frames": 39,
                    "prompt": "A glass figure turns towards the light.",
                    "seed": 7,
                }
            )
            == []
        )

    def test_segment_1_is_refused_before_the_run(self):
        errors = template_errors({"segment": 1})
        assert any("segment" in str(error) for error in errors), errors

    @pytest.mark.parametrize(
        "arguments", [{"head_frames": 39}, {"tail_frames": 56}, {"tail_frames": 5}]
    )
    def test_holds_past_the_vram_fold_are_refused(self, arguments):
        errors = template_errors(arguments)
        assert any(next(iter(arguments)) in str(error) for error in errors), errors

    def test_the_vram_fold_is_an_upper_bound_on_every_allowed_render(self):
        import json

        definition = json.loads(TEMPLATE.read_text())
        rate = definition["vram_estimate"]["bytes_per_voxel"]
        constraints = definition["variable_constraints"]
        head = constraints["head_frames"]
        tail = constraints["tail_frames"]
        assert head["min_frames"] == head["max_frames"] == 22
        assert (tail["min_frames"], tail["max_frames"]) == (22, 39)
        frames = constraints["num_frames"]
        for num_frames in range(frames["min_frames"], frames["max_frames"] + 1, 17):
            # The t2va rate on the render and on both guides' frames
            assert rate * num_frames >= 28.71 * (num_frames + 22 + 39), num_frames
        assert "exact" not in definition["vram_estimate"]["reason"]

    def test_the_segment_bound_says_it_is_no_frame_count(self):
        import json

        constraints = json.loads(TEMPLATE.read_text())["variable_constraints"]
        assert "not a frame count" in constraints["segment"]["reason"]

    def test_the_steps_cut_render_and_splice_the_take(self):
        import json

        definition = json.loads(TEMPLATE.read_text())
        steps = {s["name"]: s for s in definition["steps"]}
        assert list(steps) == ["cut", "repair", "splice"]
        assert steps["cut"]["task"]["command"] == "chain_repair_cut"
        assert steps["cut"]["task"]["arguments"]["chain_take"] == (
            "variable:chain_take"
        )
        arguments = steps["repair"]["pipeline"]["arguments"]
        # The cut lays the guides out: one for the last segment, and picture
        # only under a held narration
        assert arguments["guides"] == "previous_result:cut.guides"
        assert arguments["hold_audio"] == "previous_result:cut.held_audio"
        splice = steps["splice"]["task"]
        assert splice["command"] == "chain_repair_splice"
        assert splice["arguments"]["chain_take"] == "previous_result:cut.chain_take"
        assert splice["arguments"]["video"] == "previous_result:repair"
        # No made-up output: path stands in for the caller's take
        assert definition["variables"]["chain_take"] is None
        assert definition["variables"]["chain_output"] is None

    def test_the_last_segment_is_allowed_before_the_run(self):
        import json

        constraint = json.loads(TEMPLATE.read_text())["variable_constraints"]["segment"]
        assert "max_frames" not in constraint
        assert template_errors({"segment": 5, "num_frames": 124}) == []

    def test_the_default_tier_is_the_chained_segments_standard(self):
        import json

        definition = json.loads(TEMPLATE.read_text())
        variables = definition["variables"]
        tiers = definition["presets"]["tier"]
        assert variables["tier"] == "standard"
        chained = json.loads((TEMPLATES / "chained-segments.json").read_text())
        for name, bundle in chained["presets"]["tier"].items():
            assert {k: tiers[name][k] for k in bundle} == bundle, name
        # Today's literals are the standard bundle's, so a plain call is
        # unchanged
        for key, value in tiers["standard"].items():
            assert variables[key] == value, key

    @pytest.mark.parametrize("tier", ["standard", "best"])
    def test_a_narrated_tier_is_the_narrated_takes(self, tier):
        import json

        definition = json.loads(TEMPLATE.read_text())
        narrated = json.loads((TEMPLATES / "narrated-take.json").read_text())
        bundle = definition["presets"]["tier"][f"narrated_{tier}"]
        for key, value in narrated["presets"]["tier"][tier].items():
            assert bundle[key] == value, key
        for key in ("lora_model_name", "video_shift", "audio_shift"):
            assert bundle[key] == narrated["variables"][key], key

    def test_a_narrated_standard_interior_repair_validates(self):
        # 22 + 170 + 39 = 231, onto the grid and the 3-frame audio grid: 243
        assert template_errors({"tier": "narrated_standard", "num_frames": 243}) == []

    @pytest.mark.parametrize("segment, num_frames", [(3, 243), (5, 192)])
    def test_a_narrated_best_repair_projects_past_a_24_gb_card(
        self, segment, num_frames
    ):
        # The guide fold is an upper bound, not a measurement: a 1344x768
        # repair of a narrated take's segment projects past the 3090 row
        # until one is measured (the description says so)
        errors = template_errors(
            {"tier": "narrated_best", "segment": segment, "num_frames": num_frames}
        )
        assert [e["path"] for e in errors] == ["arguments"]
        assert "RTX 3090" in errors[0]["message"]

    def test_the_description_says_how_to_repair_a_narrated_take(self):
        import json

        description = json.loads(TEMPLATE.read_text())["description"]
        for phrase in ("narrated_best", "held", "last segment"):
            assert phrase in description, phrase

    def test_the_default_render_fits_the_default_chain(self):
        import json

        variables = json.loads(TEMPLATE.read_text())["variables"]
        geometry = repair_geometry(
            CHAIN,
            variables["segment"],
            variables["num_frames"],
            variables["head_frames"],
            variables["tail_frames"],
        )
        assert geometry["minimum_num_frames"] == variables["num_frames"]
        assert variables["tail_frames"] > variables["head_frames"]
        assert variables["seed"] is None

    @pytest.mark.parametrize(
        "name",
        [
            "chained-segments.json",
            "chain-matched-to-audio.json",
            "chain-matched-and-aligned.json",
            "chain-video-continuity.json",
        ],
    )
    def test_the_h3_chains_can_keep_their_segments(self, name):
        import json

        definition = json.loads((TEMPLATES / name).read_text())
        assert definition["variables"]["keep_segments"] is False
        (chain,) = list(chain_blocks(definition))
        assert chain["keep_segments"] == "variable:keep_segments"
        # The guide chain spills every take (Don, 2026-11-08): a chain that
        # dies leaves its finished segments for resume_from
        expected = (
            (True,)
            if name == "chained-segments.json"
            else (
                "variable:keep_segments",
                True,
            )
        )
        assert chain["save_segments"] in expected
        assert "repair-segment" in definition["description"]


@pytest.mark.parametrize("name", ["save_segments", "keep_segments"])
@pytest.mark.parametrize(
    "value, valid",
    [(True, True), ("variable:keep_segments", True), ("false", False)],
)
def test_the_segment_flags_take_a_boolean_or_a_variable(name, value, valid):
    # A literal "false" would read as bool("false") == True
    import json

    import jsonschema

    schema = json.loads(
        (TEMPLATES.parent.parent.parent / "dw" / "workflow_schema.json").read_text()
    )

    def find(node):
        if isinstance(node, dict):
            if name in node and isinstance(node[name], dict) and "type" in node[name]:
                return node[name]
            for child in node.values():
                if (found := find(child)) is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                if (found := find(child)) is not None:
                    return found
        return None

    entry = find(schema)
    errors = list(jsonschema.Draft7Validator(entry).iter_errors(value))
    assert (not errors) == valid, errors


def test_the_cut_feeds_the_splice_in_a_workflow(run, tmp_path, monkeypatch):
    """The template's task steps as a workflow, with a stand-in for the H3
    render: the record's fields reach the splice as one value each, and the
    saved splice keeps the take's length."""
    import av

    from dw.workflow import Workflow

    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    definition = {
        "id": "repair-wiring",
        "steps": [
            {
                "name": "cut",
                "task": {
                    "command": "chain_repair_cut",
                    "arguments": {
                        "chain_take": run["take"],
                        "segment": 2,
                        "num_frames": 107,
                        "tail_frames": 22,
                    },
                },
            },
            {
                # Stands in for the render: 107 frames of the take with audio
                "name": "repair",
                "task": {
                    "command": "trim_video",
                    "arguments": {
                        "video": run["take"],
                        "start_frame": 18,
                        "num_frames": 107,
                        "fps": 24,
                    },
                },
            },
            {
                "name": "splice",
                "task": {
                    "command": "chain_repair_splice",
                    "arguments": {
                        "chain_take": "previous_result:cut.chain_take",
                        "video": "previous_result:repair",
                        "segment": 2,
                        "start_frame": "previous_result:cut.start_frame",
                        "end_frame": "previous_result:cut.end_frame",
                        "keep_start": "previous_result:cut.keep_start",
                    },
                },
                "result": {"content_type": "video/mp4", "fps": 24},
            },
        ],
    }
    out = tmp_path / "out"
    Workflow(definition, str(out), "").run({})
    saved = [p for p in out.rglob("*.mp4")]
    assert len(saved) == 1, saved
    with av.open(str(saved[0])) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == 193


@pytest.mark.parametrize("match_audio", [False, True])
def test_the_cut_reaches_the_render_arguments(tmp_path, match_audio):
    """The template's render step's references, resolved against the cut's
    result: its guides list whole and its held track - None for a chain's
    own track, which the H3 step drops as no hold at all."""
    import json

    from dw.previous_results import get_iterations
    from dw.result import Result

    made = write_run(str(tmp_path), [40, 51, 51, 51], match_audio=match_audio)
    cut = chain_repair_cut(
        chain_take=made["take"], segment=4, num_frames=73, tail_frames=22
    )
    result = Result({})
    result.add_result(cut)
    steps = {s["name"]: s for s in json.loads(TEMPLATE.read_text())["steps"]}
    template = steps["repair"]["pipeline"]["arguments"]
    (arguments,) = get_iterations(
        {key: template[key] for key in ("guides", "hold_audio", "image")},
        {"cut": result},
    )
    assert arguments["guides"] == cut["guides"]
    assert len(arguments["guides"]) == 1
    assert arguments["image"] is cut["keyframe"]
    if match_audio:
        from dw.pipeline_processors.h3_hold import hold_audio_reference

        assert arguments["hold_audio"] is cut["held_audio"]
        assert arguments["guides"][0]["audio"] is False
        # What the H3 hold block is handed: the slice at its own rate
        held = hold_audio_reference(arguments["hold_audio"])
        assert held.sample_rate == 44100
        assert held.audio.shape[-1] == frames_to_samples(73, FPS, 44100)
    else:
        assert arguments["hold_audio"] is None
        assert arguments["guides"][0]["audio"] is True
