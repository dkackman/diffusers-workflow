"""`keystone_images`, the moving keystone (#1308 Amendment 1): one image per
segment, segment k's fl2va `last_image` (entry k - 1, 1-based k) in place of
segment 1's frame 0, under keystone 'segment_1' at keystone_slot 'last'.
Segment 1 takes entry 0 too; anchor_every spaces the later segments; entry k
stays segment k's on a resume; it resolves like the chain's prompts/lines; and
a held narration places it with guide audio forced off as before."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dw.events import RunContext, activate_context, deactivate_context
from dw.guides import anchor_errors
from dw.pipeline_processors import h3_rules
from dw.pipeline_processors.chain import ChainConfig, run_chain
from dw.pipeline_processors.chain_rules import (
    CHAIN_KEYSTONE_IMAGES,
    chain_keystone_images_problem,
)
from dw.previous_results import resolve_chain_lists
from dw.security import SecurityError
from tests import test_chain
from tests.test_chain import FakePipeline, solid_frame
from tests.test_chain_resume import (  # noqa: F401 - fixtures
    FIRST_RUN,
    SECOND_RUN,
    SegmentFailed,
    _factory,
    _first_file_reference,
    _pipeline,
    outputs,
)

GuideShape = test_chain.TestGuideContinuity
FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR
PATH = f"steps[0].pipeline.chain.{CHAIN_KEYSTONE_IMAGES}"


def keyframes(count):
    """`count` stills, each coloured by its entry: (200, entry, 9)."""
    return [solid_frame((200, entry, 9)) for entry in range(count)]


def entry_of(image):
    """Which `keyframes` entry an image is, or None for another image."""
    red, entry, blue = image.getpixel((0, 0))
    return entry if (red, blue) == (200, 9) else None


def chain(images, **overrides):
    return GuideShape().chain(
        keystone="segment_1",
        keystone_slot="last",
        **{CHAIN_KEYSTONE_IMAGES: images},
        **overrides,
    )


def arguments(**extra):
    return GuideShape().arguments(image=solid_frame((1, 2, 3)), **extra)


def render(images, **overrides):
    pipeline = FakePipeline(GuideShape().output)
    run_chain(pipeline, chain(images, **overrides), arguments())
    return pipeline.calls


def last_images(calls):
    return [
        entry_of(call["last_image"]) if "last_image" in call else None for call in calls
    ]


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def events():
    seen = []
    token = activate_context(RunContext(on_event=seen.append))
    yield seen
    deactivate_context(token)


def _workflow(block, workflow="fl2va", **step_arguments):
    return {
        "steps": [
            {
                "name": "take",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": workflow,
                    },
                    "chain": block,
                    "arguments": {"num_frames": 124} | step_arguments,
                },
            }
        ]
    }


def _messages(definition):
    return [e["message"] for e in anchor_errors(definition) if e["path"] == PATH]


# 1. The run


def test_segment_k_takes_entry_k_as_last_image():
    calls = render(keyframes(4), segments=4)
    assert len(calls) == 4
    assert last_images(calls) == [0, 1, 2, 3]
    for index, call in enumerate(calls[1:], 1):
        # the keyframe is still the carried guide's first frame, never an entry
        assert call["image"].getpixel((0, 0)) == ((index - 1) * 50, 102, 7)
    # segment 1 opens on the step's own keyframe
    assert calls[0]["image"].getpixel((0, 0)) == (1, 2, 3)


def test_a_longer_list_is_kept_and_its_tail_unused():
    assert last_images(render(keyframes(6), segments=4)) == [0, 1, 2, 3]


def test_anchor_every_skips_off_cadence_segments(flag_on):
    # The 0-based rule config.anchors(i) has: i % 2 == 0 for i >= 1, so
    # indices 2 and 4 (positions 3 and 5) take theirs; segment 1 always does
    calls = render(keyframes(5), segments=5, anchor_every=2)
    assert last_images(calls) == [0, None, 2, None, 4]


def test_the_job_log_names_each_segment_s_entry(events):
    render(keyframes(3))
    applied = [
        e["message"] for e in events if "keystone applied" in str(e.get("message"))
    ]
    assert len(applied) == 3
    assert "segment 1 ends on keystone_images[0] (its last_image)" in applied[0]
    assert "segment 3 ends on keystone_images[2] (its last_image)" in applied[2]


def test_resume_keeps_entry_k_for_segment_k(outputs, events):  # noqa: F811
    images = keyframes(4)
    block = {
        "segments": 4,
        "continuity": "guide",
        "guide_frames": 22,
        "carry_audio": False,
        "fps": 24,
        "save_segments": True,
        "keep_segments": True,
        "keystone": "segment_1",
        "keystone_slot": "last",
        CHAIN_KEYSTONE_IMAGES: images,
    }
    step = {
        "prompt": "test",
        "num_frames": 56,
        "image": solid_frame((1, 2, 3)).resize((16, 16)),
    }
    dead = _pipeline(outputs, FIRST_RUN, _factory(fail_at=2))
    with pytest.raises(SegmentFailed):
        run_chain(dead, block, step)
    assert last_images(dead.calls[:2]) == [0, 1]

    pipeline = _pipeline(outputs, SECOND_RUN, _factory())
    run_chain(pipeline, {**block, "resume_from": _first_file_reference()}, step)

    assert last_images(pipeline.calls) == [2, 3]


def test_a_held_narration_places_last_images_with_guide_audio_off():
    references = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.references"
    )
    from dw.media_types import AudioVideo
    from tests.test_chain_held_guide import (
        RATE,
        THREE_SEGMENTS,
        arguments as held_arguments,
        chain as held_chain,
        output,
        ramp,
    )

    held = references.MiniMaxH3AudioReference(
        audio=ramp(THREE_SEGMENTS), sample_rate=RATE
    )
    block = held_chain(
        keystone="segment_1",
        keystone_slot="last",
        carry_audio=True,
        **{CHAIN_KEYSTONE_IMAGES: keyframes(3)},
    )
    pipeline = FakePipeline(output)

    run_chain(pipeline, block, held_arguments(held) | {"image": solid_frame((1, 2, 3))})

    assert len(pipeline.calls) == 3
    assert last_images(pipeline.calls) == [0, 1, 2]
    for call in pipeline.calls:
        assert call["hold_audio"] is not None
    for call in pipeline.calls[1:]:
        (guide,) = call["guides"]
        assert guide.get("audio") is not True
        assert not isinstance(guide["video"], AudioVideo)


# 2. Resolution


class FakeResult:
    """A stored step result: `artifacts` whole, `properties` by name."""

    def __init__(self, artifacts=None, **properties):
        self.artifacts = artifacts or []
        self.properties = properties

    def get_artifacts(self):
        return list(self.artifacts)

    def get_artifact_properties(self, name):
        return list(self.properties[name])


def _action(images):
    return SimpleNamespace(
        name="take",
        pipeline_definition={"chain": {CHAIN_KEYSTONE_IMAGES: images}},
    )


def test_previous_result_list_resolves():
    images = keyframes(4)
    action = _action("previous_result:keys.images")
    resolve_chain_lists(action, {"keys": FakeResult(images=[images])})
    assert [entry_of(i) for i in action.chain_keystone_images] == [0, 1, 2, 3]


def test_a_for_each_step_hands_on_its_first_image_per_entry():
    # one result per entry, each an image list; the second image is dropped
    images = keyframes(4)
    per_entry = [[image, solid_frame((0, 0, 0))] for image in images]
    action = _action("previous_result:keys.images")
    resolve_chain_lists(action, {"keys": FakeResult(images=per_entry)})
    assert [entry_of(i) for i in action.chain_keystone_images] == [0, 1, 2, 3]
    # ...and the bare step name hands on its artifacts, one image each
    action = _action("previous_result:keys")
    resolve_chain_lists(action, {"keys": FakeResult(artifacts=images)})
    assert [entry_of(i) for i in action.chain_keystone_images] == [0, 1, 2, 3]


def test_reference_entries_resolve_one_image_each():
    images = keyframes(2)
    action = _action(["previous_result:a.images", images[1]])
    resolve_chain_lists(action, {"a": FakeResult(images=[[images[0]]])})
    assert [entry_of(i) for i in action.chain_keystone_images] == [0, 1]


def test_a_literal_list_is_left_to_the_run():
    action = _action(keyframes(2))
    resolve_chain_lists(action, {})
    assert not hasattr(action, "chain_keystone_images")


def test_the_resolved_list_reaches_the_run():
    pipeline = FakePipeline(GuideShape().output)
    pipeline.chain_keystone_images = keyframes(3)
    run_chain(pipeline, chain("previous_result:keys.images"), arguments())
    assert last_images(pipeline.calls) == [0, 1, 2]


def test_a_path_entry_loads_under_the_location_policy(tmp_path):
    keyframes(1)[0].save(tmp_path / "k0.png")
    config = ChainConfig(
        chain(["k0.png", solid_frame((200, 1, 9)), solid_frame((200, 2, 9))]),
        arguments(),
        base_dir=str(tmp_path),
    )
    assert [entry_of(i) for i in config.keystone_images] == [0, 1, 2]
    with pytest.raises(SecurityError):
        ChainConfig(chain(["../k0.png"] * 3), arguments(), base_dir=str(tmp_path))


# 3. The refusals, each naming the key


@pytest.mark.parametrize(
    "keys",
    [
        {"keystone": "none", "keystone_slot": None},
        {"keystone": "segment_1", "keystone_slot": "first"},
        {"keystone": "segment_1", "keystone_slot": None},
    ],
)
def test_the_key_needs_keystone_segment_1_at_the_last_slot(keys):
    block = GuideShape().chain(**keys, **{CHAIN_KEYSTONE_IMAGES: keyframes(3)})
    problem = chain_keystone_images_problem(block)
    assert CHAIN_KEYSTONE_IMAGES in problem
    assert "keystone 'segment_1'" in problem and "keystone_slot 'last'" in problem
    with pytest.raises(ValueError, match=CHAIN_KEYSTONE_IMAGES):
        ChainConfig(block, arguments())
    assert _messages(_workflow(block))


def test_a_short_list_is_refused_naming_both_counts():
    with pytest.raises(ValueError, match=r"holds 3 image\(s\) for 4 planned segment"):
        ChainConfig(chain(keyframes(3), segments=4), arguments())
    (message,) = _messages(_workflow(chain(["k.png"] * 3, segments=4)))
    assert "holds 3 image(s) for 4 planned segment(s)" in message


def test_a_probe_still_needs_the_full_plan_s_length():
    block = chain(
        keyframes(2),
        segments=4,
        stop_after=2,
        save_segments=True,
        keep_segments=True,
    )
    with pytest.raises(ValueError, match="2 image\\(s\\) for 4 planned"):
        ChainConfig(block, arguments())


def test_a_matched_plan_is_counted_at_the_run():
    from tests.test_chain import FakeAudioReference

    shape = GuideShape()
    block = chain(keyframes(1), match_audio=True)
    block.pop("segments")
    # 124 + 102 frames: two segments, which validate cannot count
    samples = round(226 / shape.FPS * shape.RATE)
    step = arguments(references=None) | {
        "hold_audio": FakeAudioReference(torch.zeros(2, samples), shape.RATE)
    }
    del step["references"]
    assert chain_keystone_images_problem(block, None, step) is None
    with pytest.raises(ValueError, match=r"holds 1 image\(s\) for 2 planned"):
        ChainConfig(block, step)


@pytest.mark.parametrize(
    "images, match",
    [
        (["take.mp4"], "entry 0 is not an image"),
        ([solid_frame((1, 1, 1)), 7], "entry 1 is not an image"),
        ([{"media_type": "video", "location": "a.mp4"}], "entry 0 is not an image"),
        (["prompt:walk"], "entry 0 is not an image"),
        ("k.png", "got the string"),
        ([], "list of images"),
        ({"location": "a.png"}, "list of images"),
    ],
)
def test_an_entry_that_is_not_an_image_is_refused(images, match):
    problem = chain_keystone_images_problem(chain(images))
    assert CHAIN_KEYSTONE_IMAGES in problem and match in problem
    with pytest.raises(ValueError, match=match):
        ChainConfig(chain(images), arguments())


def test_a_resolved_entry_that_is_not_an_image_is_refused_at_the_run():
    pipeline = FakePipeline(GuideShape().output)
    pipeline.chain_keystone_images = ["a caption, not a picture"]
    with pytest.raises(ValueError, match="entry 0 is not an image"):
        run_chain(pipeline, chain("previous_result:keys.images"), arguments())
    assert pipeline.calls == []


@pytest.mark.parametrize(
    "workflow, match", [("t2va", "'t2va' workflow"), ("ref2va", "'ref2va' workflow")]
)
def test_the_key_off_an_h3_fl2va_step_is_refused(workflow, match):
    (message,) = _messages(_workflow(chain(["k.png"] * 3), workflow))
    assert CHAIN_KEYSTONE_IMAGES in message and match in message


def test_the_key_off_h3_is_refused():
    definition = _workflow(chain(["k.png"] * 3))
    definition["steps"][0]["pipeline"]["configuration"]["component_type"] = (
        "LTX2Pipeline"
    )
    (message,) = _messages(definition)
    assert "MiniMax-H3" in message and CHAIN_KEYSTONE_IMAGES in message


def test_the_run_refuses_a_step_with_no_fl2va_keyframe():
    with pytest.raises(ValueError, match=f"{CHAIN_KEYSTONE_IMAGES}.*fl2va"):
        ChainConfig(chain(keyframes(3)), GuideShape().arguments())


def test_the_key_on_a_step_with_its_own_last_image_is_refused():
    own = {"last_image": solid_frame((4, 5, 6))}
    with pytest.raises(ValueError, match=f"{CHAIN_KEYSTONE_IMAGES}.*last_image"):
        ChainConfig(chain(keyframes(3)), arguments(**own))
    (message,) = _messages(_workflow(chain(["k.png"] * 3), last_image="still.png"))
    assert "last_image" in message


def test_absent_null_and_unresolved_are_off():
    assert chain_keystone_images_problem(GuideShape().chain()) is None
    assert chain_keystone_images_problem(chain(None)) is None
    assert chain_keystone_images_problem(chain("variable:keyframes")) is None
    # a whole-list reference is the run's to resolve and count
    assert chain_keystone_images_problem(chain("previous_result:keys.images")) is None
    assert _messages(_workflow(chain(["k.png"] * 3))) == []


def test_validation_refuses_a_path_outside_the_roots():
    errors = anchor_errors(_workflow(chain(["k.png"] * 3)))
    assert not [e for e in errors if CHAIN_KEYSTONE_IMAGES in e["path"]]
    from dw.locations import location_errors

    located = location_errors(_workflow(chain(["../k.png", "asset:k.png", "k.png"])))
    assert [e["path"] for e in located][:1] == [f"{PATH}[0]"]


# 4. The template


TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "workflows/templates/minimax/narrated-take.json"
)


def test_the_template_wires_keyframes_and_the_walk_preset():
    definition = json.loads(TEMPLATE.read_text())
    assert definition["variables"]["keyframes"] is None
    take = {s["name"]: s for s in definition["steps"]}["take"]
    assert take["pipeline"]["chain"][CHAIN_KEYSTONE_IMAGES] == "variable:keyframes"
    assert definition["presets"]["shot"]["walk_keyframed"] == {
        "continuity": "guide",
        "keystone": "segment_1",
        "keystone_slot": "last",
    }


def test_the_template_validates_keyframes_only_on_the_walk_preset(tmp_path):
    from dw.validation import workflow_errors
    from dw.workflow import workflow_from_definition
    from tests.test_narrated_take_template import ARGUMENTS

    def errors(extra):
        workflow = workflow_from_definition(
            json.loads(TEMPLATE.read_text()), str(tmp_path)
        )
        return workflow_errors(workflow, copy.deepcopy(ARGUMENTS) | extra)

    frames = [f"asset:keys/k{i}.png" for i in range(13)]
    assert errors({"shot": "walk_keyframed", "keyframes": frames}) == []
    refused = errors({"keyframes": frames})
    assert any(CHAIN_KEYSTONE_IMAGES in str(e.get("message")) for e in refused)
    # the dialogue default still validates with no keyframes
    assert errors({}) == []


def test_the_standing_preset_is_the_keyframe_free_keystone(tmp_path):
    """`standing`: guide continuity, keystone segment_1 at slot `last`, no
    keyframes - the narrated take held to about 85 s with no keyframe built
    (bench/records/narrated.md). It validates with the keyframes left null."""
    from dw.validation import workflow_errors
    from dw.workflow import workflow_from_definition
    from tests.test_narrated_take_template import ARGUMENTS

    definition = json.loads(TEMPLATE.read_text())
    shots = definition["presets"]["shot"]
    assert shots["standing"] == {
        "continuity": "guide",
        "keystone": "segment_1",
        "keystone_slot": "last",
    }
    assert "keyframes" not in shots["standing"]
    assert list(shots) == ["dialogue", "standing", "walk_keyframed"]
    workflow = workflow_from_definition(definition, str(tmp_path))
    assert (
        workflow_errors(workflow, copy.deepcopy(ARGUMENTS) | {"shot": "standing"}) == []
    )
