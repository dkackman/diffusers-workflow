"""The MiniMax templates draw their cast and boards through
`portrait_workflow` (#1626 stage C, #1722): a `variable:` path whose default
is `preference:text_to_image`, pinned at tier standard on every step so the
run's own `tier` (the video's resolution) never moves the portrait model.

Runs the real resolver over the shipped store on a fixed host. No GPU, no
model download, no network.
"""

import json
import os

import pytest

from dw import preferences
from dw.preferences import Host
from dw.workflow import workflow_from_file

from tests.test_examples import REPO_ROOT

MINIMAX = os.path.join(REPO_ROOT, "workflows", "templates", "minimax")

# template -> {portrait step: (width, height)}
PORTRAIT_STEPS = {
    "dialogue-short": {
        "draw_character_a": (768, 768),
        "draw_character_b": (768, 768),
    },
    "generated-subject-reference": {"draw_subject": (768, 768)},
    "music-video": {"draw_singer": (768, 768)},
    "storyboard": {
        "board_1_launch": (960, 544),
        "board_2_gutter": (960, 544),
        "board_3_shore": (960, 544),
    },
}
# The templates whose own `tier` variable sets the video's resolution
RUN_TIERED = ["dialogue-short", "music-video"]
# The step that held the portrait pipeline last, before the video loads
RELEASES = {
    "dialogue-short": "draw_character_b",
    "generated-subject-reference": "draw_subject",
    "music-video": "draw_singer",
    "storyboard": "board_3_shore",
}
LINE = "preference:text_to_image (tier standard) -> models/z-image"


@pytest.fixture
def cuda_session():
    session = preferences.Session(
        host=Host(device_type="cuda", smallest_card_gb=48, host_ram_gb=128)
    )
    token = preferences.activate(session)
    yield session
    preferences.deactivate(token)


def path(name):
    return os.path.join(MINIMAX, f"{name}.json")


def definition(name):
    with open(path(name), encoding="utf-8") as file:
        return json.load(file)


def steps_by_name(steps):
    return {step["name"]: step for step in steps}


def expanded(name, tmp_path, arguments=None):
    flow = workflow_from_file(path(name), str(tmp_path))
    return flow, steps_by_name(flow.expanded_definition(arguments)["steps"])


@pytest.mark.parametrize("name", sorted(PORTRAIT_STEPS))
def test_portrait_steps_compose_the_variable(name):
    doc = definition(name)
    assert doc["variables"]["portrait_workflow"] == "preference:text_to_image"
    steps = steps_by_name(doc["steps"])
    for step_name, (width, height) in PORTRAIT_STEPS[name].items():
        step = steps[step_name]
        assert "pipeline" not in step
        assert step["workflow"]["path"] == "variable:portrait_workflow"
        arguments = step["workflow"]["arguments"]
        assert arguments["seed"] == "variable:seed"
        assert arguments["num_images_per_prompt"] == 1
        assert (arguments["width"], arguments["height"]) == (width, height)
        assert arguments["tier"] == "standard"
        # steps and guidance are the target's own (Q5)
        assert "num_inference_steps" not in arguments
        assert "guidance_scale" not in arguments


@pytest.mark.parametrize("name", sorted(PORTRAIT_STEPS))
def test_no_inline_z_image_pipeline_remains(name):
    assert "Z-Image-Turbo" not in json.dumps(definition(name)["steps"])


@pytest.mark.parametrize("name", sorted(RELEASES))
def test_the_last_portrait_step_still_releases_its_pipeline(name):
    assert steps_by_name(definition(name)["steps"])[RELEASES[name]].get(
        "release_pipeline"
    )


@pytest.mark.parametrize("name", sorted(PORTRAIT_STEPS))
def test_each_template_validates(name, tmp_path, cuda_session):
    workflow_from_file(path(name), str(tmp_path)).validate()


@pytest.mark.parametrize("name", sorted(PORTRAIT_STEPS))
def test_every_portrait_step_resolves_to_z_image_at_standard(
    name, tmp_path, cuda_session
):
    flow, steps = expanded(name, tmp_path)
    for step_name in PORTRAIT_STEPS[name]:
        reference = steps[step_name]["workflow"]
        assert reference["path"] == "models/z-image"
        # models/z-image declares no tier: the resolver keeps it out
        assert "tier" not in reference["arguments"]
    lines = flow.preference_lines()
    assert lines and all(line.startswith(LINE) for line in lines)


@pytest.mark.parametrize("name", RUN_TIERED)
@pytest.mark.parametrize("tier", ["draft", "best"])
def test_the_run_tier_does_not_move_the_portrait(name, tier, tmp_path, cuda_session):
    flow, steps = expanded(name, tmp_path, {"tier": tier})
    for step_name in PORTRAIT_STEPS[name]:
        assert steps[step_name]["workflow"]["path"] == "models/z-image"
    lines = flow.preference_lines()
    assert lines and all(line.startswith(LINE) for line in lines)


@pytest.mark.parametrize("name", sorted(PORTRAIT_STEPS))
def test_a_named_workflow_overrides_with_no_preference_line(
    name, tmp_path, cuda_session
):
    arguments = {"portrait_workflow": "models/qwen-image-2.1"}
    flow, steps = expanded(name, tmp_path, arguments)
    for step_name in PORTRAIT_STEPS[name]:
        reference = steps[step_name]["workflow"]
        assert reference["path"] == "models/qwen-image-2.1"
        # qwen-image-2.1 declares no tier: the step's pin is not handed on
        assert "tier" not in reference["arguments"]
    assert flow.preference_lines() == []
    assert not [
        error
        for error in flow.validation_errors(arguments)
        if "tier" in error["message"] or "portrait" in error["path"]
    ]


def test_a_tier_is_still_handed_to_a_child_that_declares_one(tmp_path, cuda_session):
    """`templates/preferred/text-to-image` declares `tier`, so the step's pin
    reaches it as an argument (and its own preference resolves at it)."""
    arguments = {"portrait_workflow": "templates/preferred/text-to-image"}
    _, steps = expanded("music-video", tmp_path, arguments)
    reference = steps["draw_singer"]["workflow"]
    assert reference["path"] == "templates/preferred/text-to-image"
    assert reference["arguments"]["tier"] == "standard"


def test_a_composing_step_is_elided_only_when_it_says_save_false():
    """A supplied cast still skips the portraits now that they compose a
    workflow: an unread composing step with `save: false` is dropped, and one
    without it is kept, since its child saves on its own terms."""
    from dw.elision import elide_unreferenced_steps

    def steps(result):
        portrait = {"name": "portrait", "workflow": {"path": "models/z-image"}}
        if result is not None:
            portrait["result"] = result
        return [portrait, {"name": "last", "pipeline": {}}]

    unsaved = {"content_type": "image/jpeg", "save": False}
    kept, elided = elide_unreferenced_steps(steps(unsaved))
    assert [entry["step"] for entry in elided] == ["portrait"]
    for result in (None, {"content_type": "image/jpeg"}):
        kept, elided = elide_unreferenced_steps(steps(result))
        assert elided == [] and len(kept) == 2
