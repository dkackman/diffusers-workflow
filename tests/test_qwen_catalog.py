"""The Qwen-Image-2.1 transparent template (#1681 stage B, #1709): the vendor's
RGBA prompt wrapper is built by an existing composition (compose_text), and the
result is a PNG so the alpha survives."""

import json
import os

from dw.server.catalog_shape import derive_catalog_metadata
from dw.workflow import Workflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "workflows", "templates", "qwen", "transparent.json")

WRAPPER_HEAD = "This is an RGBA image with transparency."
WRAPPER_TAIL = "The image has alpha channel and the background is transparent."


def load():
    with open(TEMPLATE) as f:
        return json.load(f)


def steps_by_name(definition):
    return {s["name"]: s for s in definition["steps"]}


def test_the_result_is_a_png_so_the_alpha_survives():
    main = steps_by_name(load())["main"]
    assert main["result"]["content_type"] == "image/png"


def test_the_prompt_is_the_composed_wrapper():
    steps = steps_by_name(load())
    assert steps["main"]["pipeline"]["arguments"]["prompt"] == "previous_result:wrap"
    assert steps["wrap"]["task"]["command"] == "compose_text"


def test_the_composed_prompt_carries_both_halves_of_the_wrapper(tmp_path):
    """The template's own wrap step, run through the engine with a caller's
    subject: the prompt the pipeline receives is the vendor's wrapper around
    it. Only the pipeline step is left off (it needs the weights)."""
    definition = load()
    definition["steps"] = [steps_by_name(definition)["wrap"]]

    (prompt,) = Workflow(definition, str(tmp_path), "").run({"subject": "a blue mug"})

    assert prompt == f"{WRAPPER_HEAD} a blue mug. {WRAPPER_TAIL}"


def test_it_validates_with_defaults_and_refuses_an_off_grid_width(tmp_path):
    workflow = Workflow(load(), str(tmp_path), "")
    assert workflow.validation_errors() == []
    errors = workflow.validation_errors(arguments={"width": 1000})
    assert errors and any("width" in e["message"] for e in errors)


def test_defaults_are_40_steps_at_1024_on_the_32_grid():
    definition = load()
    variables = definition["variables"]
    assert variables["num_inference_steps"] == 40
    assert (variables["width"], variables["height"]) == (1024, 1024)
    for side in ("width", "height"):
        assert definition["variable_constraints"][side]["modulus"] == 32


def test_it_runs_on_the_base_with_no_new_download():
    main = steps_by_name(load())["main"]
    assert main["pipeline"]["from_pretrained_arguments"]["model_name"] == (
        "Qwen/Qwen-Image-2.1"
    )


def test_it_lists_as_an_image_that_says_transparent_png():
    meta = derive_catalog_metadata(load())
    assert meta["shape"] == "image"
    assert "transparent" in meta["summary"].lower()
    assert "png" in meta["summary"].lower()


def test_the_description_names_the_jpeg_flattening_and_the_licence():
    description = load()["description"]
    assert "jpeg" in description and "flatten" in description
    assert "Qwen Research License" in description
    assert "non-commercial" in description
