"""The composing catalog templates hand their child's run-affecting values
down as parent variables (#1642 stage A): an override given to the parent
reaches the child, and a default run hands the child what it would have used
on its own. compose-workflows also hands down its seed, so the image child's
declared `seed: 42` no longer beats the parent's."""

import copy
import json
import os

import pytest

from dw.workflow import workflow_from_file
from dw.workflow_run import prepare_definition, run_base_dir

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(REPO_ROOT, "workflows", "templates")


def declared(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)["variables"]


def compose(template, step_name, tmp_path, arguments=None, seed=1234):
    """The child Workflow the parent's `step_name` step composes, from the
    parent's definition realized with `arguments`, as the run builds it."""
    parent = workflow_from_file(os.path.join(TEMPLATES, template), str(tmp_path))
    expanded = parent.expanded_definition(arguments or {})
    (step,) = [s for s in expanded["steps"] if s["name"] == step_name]
    return parent._sub_workflow_action(copy.deepcopy(step), seed)


def child_seed(child):
    definition = copy.deepcopy(child.workflow_definition)
    # Step.run hands the child its argument_template as the run's arguments
    _, seed, _ = prepare_definition(
        child, definition, dict(child.argument_template), run_base_dir(child)
    )
    return seed


# describe-and-regenerate -> ../models/flux-dev.json

FLUX_DEV = os.path.join(REPO_ROOT, "workflows", "models", "flux-dev.json")
DESCRIBE = "describe-and-regenerate.json"
DESCRIBE_KNOBS = ["num_images_per_prompt", "num_inference_steps", "width", "height"]


def test_describe_and_regenerate_defaults_hand_flux_dev_its_own_defaults(tmp_path):
    child = compose(DESCRIBE, "flux", tmp_path)
    assert os.path.samefile(child.file_spec, FLUX_DEV)
    flux_dev = declared(FLUX_DEV)
    handed = child._handed_arguments
    for name in DESCRIBE_KNOBS:
        assert handed[name] == flux_dev[name], name


def test_describe_and_regenerate_overrides_reach_the_child(tmp_path):
    overrides = {
        "num_images_per_prompt": 2,
        "num_inference_steps": 8,
        "width": 512,
        "height": 512,
    }
    child = compose(DESCRIBE, "flux", tmp_path, overrides)
    handed = child._handed_arguments
    for name, value in overrides.items():
        assert handed[name] == value, name
    assert handed["prompt"] == "previous_result:augment_prompt"


def test_describe_and_regenerate_image_workflow_names_the_child(tmp_path):
    child = compose(
        DESCRIBE, "flux", tmp_path, {"image_workflow": "../models/z-image.json"}
    )
    assert os.path.basename(child.file_spec) == "z-image.json"


# compose-workflows -> variable: paths (z-image, minimax/image-to-video)

COMPOSE = "compose-workflows.json"


def test_compose_workflows_hands_num_images_per_prompt_defaulting_to_one(tmp_path):
    # Q3: the one deliberate departure from the child's default (z-image's
    # 4) - one still feeds the one video step
    child = compose(COMPOSE, "image_generation", tmp_path)
    assert child._handed_arguments["num_images_per_prompt"] == 1
    child = compose(COMPOSE, "image_generation", tmp_path, {"num_images_per_prompt": 3})
    assert child._handed_arguments["num_images_per_prompt"] == 3


@pytest.mark.parametrize("step_name", ["image_generation", "video_generation"])
def test_compose_workflows_hands_its_seed_to_both_children(step_name, tmp_path):
    child = compose(COMPOSE, step_name, tmp_path, {"seed": 7})
    assert child._handed_arguments["seed"] == 7
    assert child_seed(child) == 7


def test_a_null_parent_seed_beats_the_image_childs_declared_42(tmp_path):
    # z-image declares seed: 42. Handed the parent's null, its variable
    # resolves to None and the run falls back to the parent's drawn seed
    # (#1641) rather than 42
    assert (
        declared(os.path.join(REPO_ROOT, "workflows", "models", "z-image.json"))["seed"]
        == 42
    )
    child = compose(COMPOSE, "image_generation", tmp_path, seed=98765)
    assert child._handed_arguments["seed"] is None
    assert child_seed(child) == 98765


def _inline_parent(tmp_path, seed_argument, pass_seed=True):
    """A parent composing z-image (declared seed: 42) whose own seed variable
    is null, handing `seed_argument` as the child's seed."""
    arguments = {"prompt": "a cat"}
    if pass_seed:
        arguments["seed"] = seed_argument
    definition = {
        "id": "parent",
        "variables": {"seed": None, "fixed": 5},
        "steps": [
            {
                "name": "image",
                "workflow": {"path": "../models/z-image.json", "arguments": arguments},
            }
        ],
    }
    path = os.path.join(TEMPLATES, "__seed_probe__.json")
    from dw.workflow import Workflow

    return Workflow(definition, str(tmp_path), path)


def _child_of(parent, seed):
    expanded = parent.expanded_definition({})
    (step,) = expanded["steps"]
    return parent._sub_workflow_action(copy.deepcopy(step), seed)


def test_a_literal_child_seed_still_wins_over_the_parents(tmp_path):
    child = _child_of(_inline_parent(tmp_path, 11), 98765)
    assert child._handed_arguments["seed"] == 11
    assert child_seed(child) == 11


def test_a_set_reference_hands_its_value(tmp_path):
    child = _child_of(_inline_parent(tmp_path, "variable:fixed"), 98765)
    assert child_seed(child) == 5


def test_a_step_passing_no_seed_keeps_the_childs_own(tmp_path):
    child = _child_of(_inline_parent(tmp_path, None, pass_seed=False), 98765)
    assert "seed" not in child._handed_arguments
    assert child_seed(child) == 42


def test_a_null_reference_under_validation_hands_null_not_a_drawn_seed(tmp_path):
    # Validation expands the same way: the child is handed null, and only
    # the run supplies a seed for it to fall back to
    expanded = _inline_parent(tmp_path, "variable:seed").expanded_definition({})
    assert expanded["steps"][0]["workflow"]["arguments"]["seed"] is None


# best-of-n-to-video -> ./ltx2/image-to-video.json

LTX_I2V = os.path.join(TEMPLATES, "ltx2", "image-to-video.json")
BEST_OF_N = "best-of-n-to-video.json"
VIDEO_KNOBS = {
    "video_width": "width",
    "video_height": "height",
    "num_frames": "num_frames",
}


def test_best_of_n_video_defaults_equal_the_ltx_childs(tmp_path):
    child = compose(BEST_OF_N, "video", tmp_path)
    ltx = declared(LTX_I2V)
    handed = child._handed_arguments
    for parent_name, child_name in VIDEO_KNOBS.items():
        assert handed[child_name] == ltx[child_name], parent_name


def test_best_of_n_video_overrides_reach_the_child(tmp_path):
    child = compose(
        BEST_OF_N,
        "video",
        tmp_path,
        {"num_frames": 97, "video_width": 512, "video_height": 512},
    )
    handed = child._handed_arguments
    assert (handed["num_frames"], handed["width"], handed["height"]) == (97, 512, 512)
